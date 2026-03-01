#!/usr/bin/env python3
"""
Multi-Source DINOv2 Authentication Pipeline

Data sources: Rijksmuseum, Met Open Access, Wikidata SPARQL, NGA, CMA, AIC.
Three-tier cache: metadata → images → embeddings.
Each stage checks cache before running; re-runs skip completed work.

Usage:
    python scan.py              # full pipeline
    python scan.py --stage 1    # metadata only
    python scan.py --stage 4    # analysis only (needs prior stages cached)
    python scan.py --refetch    # clear cache and re-fetch from all sources
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_META = CACHE_DIR / "metadata"
CACHE_IMG = CACHE_DIR / "images"
CACHE_IMG_HIRES = CACHE_DIR / "images_hires"
CACHE_EMB = CACHE_DIR / "embeddings"
CACHE_PLOTS = CACHE_DIR / "plots"

INVENTORY_CSV = CACHE_META / "inventory.csv"
EMBEDDINGS_NPZ = CACHE_EMB / "embeddings.npz"
EMBEDDINGS_V2_NPZ = CACHE_EMB / "embeddings_v2.npz"
EMBEDDINGS_VITL_NPZ = CACHE_EMB / "embeddings_vitl.npz"
RESULTS_JSON = CACHE_DIR / "results.json"
RESULTS_V2_JSON = CACHE_DIR / "results_v2.json"
RESULTS_VITL_JSON = CACHE_DIR / "results_vitl.json"
EMBEDDINGS_ENTROPY_NPZ = CACHE_EMB / "embeddings_entropy.npz"
EMBEDDINGS_ENTROPY_VITL_NPZ = CACHE_EMB / "embeddings_entropy_vitl.npz"
RESULTS_ENTROPY_JSON = CACHE_DIR / "results_entropy.json"
RESULTS_ENTROPY_VITL_JSON = CACHE_DIR / "results_entropy_vitl.json"
RESULTS_PROBE_JSON = CACHE_DIR / "results_probe.json"

TILE_SIZE = 224
IMG_MAX_PX = 2000        # v1 low-res cap
IMG_HIRES_MAX_PX = 8000  # v2 cap (avoids OOM on Night Watch 14K)
BATCH_SIZE_VITB = 16     # conservative for 8 GB RAM
BATCH_SIZE_VITL = 8      # ViT-L is ~1.2 GB vs ~350 MB
CHUNK_SIZE = 5           # paintings per chunk (smaller for hires — more tiles per painting)
EMBED_DIM = 768          # DINOv2 ViT-B/14=768, ViT-L/14=1024
RATE_LIMIT = 0.3     # seconds between API calls

# Rijksmuseum
RK_SEARCH_URL = "https://data.rijksmuseum.nl/search/collection"
RK_DATA_URL = "https://data.rijksmuseum.nl"
IIIF_BASE = "https://iiif.micr.io"

# Met
MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# Wikidata
WD_SPARQL_URL = "https://query.wikidata.org/sparql"
WD_UA = "AuthScanner/1.0 (art authentication research; contact: github.com/david-fitzgerald)"
WD_REMBRANDT = "Q5598"
WD_CONTROL_ARTISTS = {
    "Ferdinand Bol": ("Q374039", "rembrandt_pupil"),
    "Govert Flinck": ("Q550401", "rembrandt_pupil"),
    "Jan Lievens": ("Q430783", "rembrandt_pupil"),
    "Frans Hals": ("Q167654", "dutch_other"),
    "Johannes Vermeer": ("Q41264", "dutch_other"),
}
# P1774=workshop, P1775=follower, P1776=circle, P1777=manner, P1780=school
WD_CIRCLE_QUALIFIERS = ["P1774", "P1775", "P1776", "P1777", "P1780"]
WD_QUALIFIER_LABELS = {
    "P1774": "workshop", "P1775": "style", "P1776": "circle",
    "P1777": "style", "P1780": "school",
}

# Museum APIs (supplementary)
NGA_API_BASE = "https://api.nga.gov/art/search"
CMA_API_BASE = "https://openaccess-api.clevelandart.org/api/artworks"
AIC_API_BASE = "https://api.artic.edu/api/v1/artworks/search"
AIC_IIIF_BASE = "https://www.artic.edu/iiif/2"

# Attribution classification
ATTRIBUTION_PATTERNS = {
    "workshop": [r"\bworkshop\s+of\b", r"\batelier\s+van\b"],
    "circle":   [r"\bcircle\s+of\b", r"\bomgeving\s+van\b", r"\bkring\s+van\b"],
    "school":   [r"\bschool\s+of\b", r"\bschool\s+van\b"],
    "style":    [r"\bstyle\s+of\b", r"\bstijl\s+van\b", r"\bmanner\s+of\b",
                 r"\bmanier\s+van\b", r"\bnavolger\b", r"\bfollower\b"],
    "after":    [r"\bafter\b", r"\bnaar\b", r"\bcopy\s+after\b", r"\bkopie\s+naar\b"],
    "attributed": [r"\battributed\s+to\b", r"\btoegeschreven\s+aan\b"],
}

# Previous Rijksmuseum-only results (from prototype-results.md)
PROTOTYPE_METRICS = {
    "source": "Rijksmuseum only",
    "n_paintings": 70,
    "n_autograph": 19,
    "n_circle": 2,
    "n_pupil": 36,
    "n_other": 13,
    "auto_auto_sim": 0.8470,
    "auto_circle_sim": 0.8314,
    "auto_pupil_sim": 0.8255,
    "auto_other_sim": 0.8235,
    "mw_p_circle": 8.79e-2,
    "mw_p_pupil": 1.01e-3,
    "mw_p_other": 1.98e-3,
    "knn_k3": 0.829,
    "knn_k5": 0.757,
    "knn_k7": 0.829,
    "baseline": 0.729,
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_with_retry(url, params=None, max_retries=3, backoff=0.3):
    """GET with exponential backoff. Returns response or None."""
    for attempt in range(max_retries):
        try:
            delay = backoff * (2 ** attempt) if attempt > 0 else RATE_LIMIT
            time.sleep(delay)
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                print(f"  Rate limited, backing off...")
                continue
            print(f"  HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            print(f"  Request failed (attempt {attempt+1}): {e}")
    print(f"  All retries exhausted for {url}")
    return None


def fetch_json(url, params=None):
    """Fetch and parse JSON. Returns dict or None."""
    resp = fetch_with_retry(url, params=params)
    if resp is None:
        return None
    try:
        return resp.json()
    except json.JSONDecodeError:
        print(f"  Invalid JSON from {url}")
        return None


# ---------------------------------------------------------------------------
# Rijksmuseum helpers (from prototype)
# ---------------------------------------------------------------------------

def classify_attribution(text):
    """Classify attribution level from a creator/title string."""
    t = text.lower()
    for level, patterns in ATTRIBUTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t):
                return level
    return "autograph"


def rk_search_all_pages(params):
    """Fetch all pages from Rijksmuseum search API. Returns list of object IDs."""
    ids = []
    page_params = dict(params)
    while True:
        data = fetch_json(RK_SEARCH_URL, params=page_params)
        if not data:
            break
        for item in data.get("orderedItems", []):
            match = re.search(r'/(\d+)$', item.get("id", ""))
            if match:
                ids.append(match.group(1))
        next_page = data.get("next", {})
        next_url = next_page.get("id", "") if isinstance(next_page, dict) else ""
        if not next_url:
            break
        token_match = re.search(r'pageToken=([^&]+)', next_url)
        if token_match:
            page_params["pageToken"] = token_match.group(1)
        else:
            break
    return ids


def parse_la_metadata(la_json):
    """Extract title, creator, date from Linked Art JSON-LD."""
    if not la_json:
        return {}
    result = {}

    if "_label" in la_json:
        result["title"] = la_json["_label"]
    if "title" not in result:
        for ident in la_json.get("identified_by", []):
            if isinstance(ident, dict) and ident.get("type") == "Name":
                content = ident.get("content", "")
                if content:
                    result["title"] = content
                    break

    produced = la_json.get("produced_by", {})
    if not isinstance(produced, dict):
        return result

    for ref in produced.get("referred_to_by", []):
        if not isinstance(ref, dict):
            continue
        for cls in ref.get("classified_as", []):
            if isinstance(cls, dict) and "300435416" in cls.get("id", ""):
                result["creator"] = ref.get("content", "")
                break
        if "creator" in result:
            break

    if "creator" not in result:
        for part in produced.get("part", []):
            if not isinstance(part, dict):
                continue
            for person in part.get("carried_out_by", []):
                if isinstance(person, dict) and "_label" in person:
                    result["creator"] = person["_label"]
                    break
            if "creator" in result:
                break

    ts = produced.get("timespan", {})
    if isinstance(ts, dict):
        for ident in ts.get("identified_by", []):
            if isinstance(ident, dict) and ident.get("type") == "Name":
                content = ident.get("content", "")
                if content:
                    result["date"] = content

    return result


def extract_iiif_id(edm_json):
    """Extract Micrio IIIF identifier from EDM JSON-LD."""
    if not edm_json:
        return None
    text = json.dumps(edm_json)
    match = re.search(r'iiif\.micr\.io/([a-zA-Z0-9]+)', text)
    return match.group(1) if match else None


def extract_title_from_edm(edm_json):
    """Fallback: extract title from EDM JSON-LD."""
    if not edm_json:
        return None
    text = json.dumps(edm_json, ensure_ascii=False)
    for key in ["dc:title", "dcterms:title", "title"]:
        match = re.search(rf'"{key}"[^"]*"@value"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    return None


def classify_rembrandt_group(creator_str):
    """Classify Rembrandt-related painting. Returns (artist_group, attribution) or (None, None)."""
    if "rembrandt" not in creator_str.lower():
        return None, None
    attrib = classify_attribution(creator_str)
    if attrib == "autograph":
        return "rembrandt_autograph", "autograph"
    return "rembrandt_circle", attrib


# ---------------------------------------------------------------------------
# Met helpers
# ---------------------------------------------------------------------------

MET_PREFIX_MAP = {
    "": "autograph",
    "Style of": "style",
    "Follower of": "style",
    "Copy after": "after",
    "After": "after",
    "Imitator of": "style",
    "Workshop of": "workshop",
    "Circle of": "circle",
    "School of": "school",
    "Attributed to": "attributed",
    "Formerly attributed to": "style",
}


def met_search(query, department_id):
    """Search Met API. Returns list of object IDs."""
    data = fetch_json(f"{MET_BASE}/search", params={
        "departmentId": department_id,
        "artistOrCulture": "true",
        "hasImages": "true",
        "q": query,
    })
    if not data:
        return []
    return data.get("objectIDs", []) or []


def met_get_object(object_id):
    """Fetch Met object detail. Returns dict or None."""
    return fetch_json(f"{MET_BASE}/objects/{object_id}")


def met_classify_attribution(artist_prefix):
    """Map Met artistPrefix to attribution level."""
    prefix = (artist_prefix or "").strip()
    return MET_PREFIX_MAP.get(prefix, "style" if prefix else "autograph")


def met_artist_group(artist_name, artist_prefix, query_group=None):
    """Determine artist_group for a Met object."""
    if query_group is not None:
        return query_group
    name_lower = (artist_name or "").lower()
    if "rembrandt" in name_lower:
        attrib = met_classify_attribution(artist_prefix)
        if attrib == "autograph":
            return "rembrandt_autograph"
        return "rembrandt_circle"
    return None


# ---------------------------------------------------------------------------
# Wikidata helpers
# ---------------------------------------------------------------------------

def wd_sparql(query):
    """POST SPARQL query to Wikidata. Returns list of result bindings."""
    for attempt in range(3):
        try:
            delay = RATE_LIMIT * (2 ** attempt) if attempt > 0 else RATE_LIMIT
            time.sleep(delay)
            resp = requests.post(WD_SPARQL_URL,
                data={"query": query},
                headers={"User-Agent": WD_UA, "Accept": "application/sparql-results+json"},
                timeout=120)
            if resp.status_code == 429:
                print("  Wikidata rate limited, backing off...")
                continue
            if resp.status_code != 200:
                print(f"  Wikidata SPARQL HTTP {resp.status_code}")
                return []
            # Wikidata sometimes includes control chars in labels
            data = json.loads(resp.text, strict=False)
            return data.get("results", {}).get("bindings", [])
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  Wikidata SPARQL error (attempt {attempt+1}): {e}")
    return []


def wd_image_url(commons_url, width=2000):
    """Convert Wikimedia Commons file URL to a sized thumbnail URL.

    Input:  http://commons.wikimedia.org/wiki/Special:FilePath/Foo.jpg
    Output: https://upload.wikimedia.org/wikipedia/commons/thumb/hash/Foo.jpg/{width}px-Foo.jpg
    """
    # Commons P18 URLs are Special:FilePath redirects — just append ?width=
    if "Special:FilePath" in commons_url:
        return f"{commons_url}?width={width}"
    # Direct upload URL — add /thumb/ variant
    return f"{commons_url}?width={width}"


def wd_fetch_circle_paintings():
    """Fetch circle/workshop/follower/manner/school Rembrandt paintings from Wikidata."""
    query = f"""SELECT ?painting ?paintingLabel ?image ?qualLabel WHERE {{
  ?painting p:P170 ?stmt .
  ?painting wdt:P31 wd:Q3305213 .
  ?painting wdt:P18 ?image .
  {{
    ?stmt pq:P1774 wd:{WD_REMBRANDT} . BIND("workshop" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1775 wd:{WD_REMBRANDT} . BIND("follower" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1776 wd:{WD_REMBRANDT} . BIND("circle" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1777 wd:{WD_REMBRANDT} . BIND("manner" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1780 wd:{WD_REMBRANDT} . BIND("school" AS ?qualLabel)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}"""
    bindings = wd_sparql(query)
    # Deduplicate by QID (UNION can produce duplicate rows)
    seen = set()
    results = []
    for b in bindings:
        qid = b["painting"]["value"].split("/")[-1]
        if qid in seen:
            continue
        seen.add(qid)
        qual = b.get("qualLabel", {}).get("value", "circle")
        # Map Wikidata qualifier to our attribution levels
        attrib_map = {"workshop": "workshop", "follower": "style",
                      "circle": "circle", "manner": "style", "school": "school"}
        results.append({
            "qid": qid,
            "title": b.get("paintingLabel", {}).get("value", ""),
            "image": b.get("image", {}).get("value", ""),
            "attribution": attrib_map.get(qual, "circle"),
            "wd_qualifier": qual,
        })
    return results


def wd_fetch_autograph_paintings():
    """Fetch autograph Rembrandt paintings from Wikidata (direct P170, no qualifier)."""
    query = f"""SELECT ?painting ?paintingLabel ?image WHERE {{
  ?painting wdt:P170 wd:{WD_REMBRANDT} .
  ?painting wdt:P31 wd:Q3305213 .
  ?painting wdt:P18 ?image .
  FILTER NOT EXISTS {{
    ?painting p:P170 ?stmt .
    {{ ?stmt pq:P1774 wd:{WD_REMBRANDT} }} UNION {{ ?stmt pq:P1775 wd:{WD_REMBRANDT} }}
    UNION {{ ?stmt pq:P1776 wd:{WD_REMBRANDT} }} UNION {{ ?stmt pq:P1777 wd:{WD_REMBRANDT} }}
    UNION {{ ?stmt pq:P1780 wd:{WD_REMBRANDT} }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}"""
    bindings = wd_sparql(query)
    seen = set()
    results = []
    for b in bindings:
        qid = b["painting"]["value"].split("/")[-1]
        if qid in seen:
            continue
        seen.add(qid)
        results.append({
            "qid": qid,
            "title": b.get("paintingLabel", {}).get("value", ""),
            "image": b.get("image", {}).get("value", ""),
        })
    return results


def wd_fetch_control_paintings():
    """Fetch paintings by control artists (pupils + other Dutch masters) from Wikidata."""
    all_results = []
    for artist_name, (qid, group) in WD_CONTROL_ARTISTS.items():
        query = f"""SELECT ?painting ?paintingLabel ?image WHERE {{
  ?painting wdt:P170 wd:{qid} .
  ?painting wdt:P31 wd:Q3305213 .
  ?painting wdt:P18 ?image .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}"""
        bindings = wd_sparql(query)
        seen = set()
        count = 0
        for b in bindings:
            painting_qid = b["painting"]["value"].split("/")[-1]
            if painting_qid in seen:
                continue
            seen.add(painting_qid)
            all_results.append({
                "qid": painting_qid,
                "title": b.get("paintingLabel", {}).get("value", ""),
                "image": b.get("image", {}).get("value", ""),
                "artist_name": artist_name,
                "artist_group": group,
            })
            count += 1
        print(f"    {artist_name}: {count} paintings")
    return all_results


# ---------------------------------------------------------------------------
# Museum API helpers (supplementary — fill gaps not on Wikidata)
# ---------------------------------------------------------------------------

def nga_fetch_paintings(query="Rembrandt"):
    """Search NGA API for paintings. Returns list of row dicts."""
    results = []
    resp = fetch_with_retry(NGA_API_BASE, params={
        "exactArtist": query,
        "artType": "painting",
        "hasImage": "true",
        "limit": 100,
    })
    if resp is None:
        return results
    try:
        data = json.loads(resp.text, strict=False)
    except json.JSONDecodeError:
        return results
    for item in data.get("results", []):
        title = item.get("title", "")
        creator = item.get("artist", "")
        image_url = item.get("iiifUrl", "") or item.get("imageUrl", "")
        if not image_url:
            continue
        attrib = classify_attribution(creator)
        if "rembrandt" in creator.lower():
            group = "rembrandt_autograph" if attrib == "autograph" else "rembrandt_circle"
        else:
            continue  # Only Rembrandt-related for now
        obj_id = item.get("id", item.get("objectId", ""))
        results.append({
            "obj_id": f"nga_{obj_id}",
            "source": "nga",
            "title": title,
            "creator": creator,
            "date": item.get("date", ""),
            "image_url": image_url,
            "artist_group": group,
            "attribution": attrib,
        })
    return results


def cma_search(query="Rembrandt"):
    """Search Cleveland Museum of Art API. Returns list of row dicts."""
    results = []
    data = fetch_json(CMA_API_BASE, params={
        "q": query,
        "type": "painting",
        "has_image": 1,
        "limit": 100,
    })
    if not data:
        return results
    for item in data.get("data", []):
        # CMA uses creators array with qualifier field
        creators = item.get("creators", [])
        if not creators:
            continue
        creator_obj = creators[0]
        creator_name = creator_obj.get("description", "")
        qualifier = creator_obj.get("qualifier", "")
        if "rembrandt" not in creator_name.lower():
            continue
        image_url = item.get("images", {}).get("web", {}).get("url", "")
        if not image_url:
            continue
        attrib = classify_attribution(f"{qualifier} {creator_name}" if qualifier else creator_name)
        group = "rembrandt_autograph" if attrib == "autograph" else "rembrandt_circle"
        obj_id = item.get("id", "")
        results.append({
            "obj_id": f"cma_{obj_id}",
            "source": "cma",
            "title": item.get("title", ""),
            "creator": f"{qualifier} {creator_name}".strip() if qualifier else creator_name,
            "date": item.get("creation_date", ""),
            "image_url": image_url,
            "artist_group": group,
            "attribution": attrib,
        })
    return results


def aic_search(query="Rembrandt"):
    """Search Art Institute of Chicago API. Returns list of row dicts."""
    results = []
    data = fetch_json(AIC_API_BASE, params={
        "q": query,
        "query[term][classification_titles]": "painting",
        "fields": "id,title,artist_display,image_id,date_display",
        "limit": 100,
    })
    if not data:
        return results
    for item in data.get("data", []):
        artist = item.get("artist_display", "")
        if "rembrandt" not in artist.lower():
            continue
        image_id = item.get("image_id")
        if not image_id:
            continue
        image_url = f"{AIC_IIIF_BASE}/{image_id}/full/!2000,2000/0/default.jpg"
        attrib = classify_attribution(artist)
        group = "rembrandt_autograph" if attrib == "autograph" else "rembrandt_circle"
        obj_id = item.get("id", "")
        results.append({
            "obj_id": f"aic_{obj_id}",
            "source": "aic",
            "title": item.get("title", ""),
            "creator": artist,
            "date": item.get("date_display", ""),
            "image_url": image_url,
            "artist_group": group,
            "attribution": attrib,
        })
    return results


# ---------------------------------------------------------------------------
# Stage 1: Fetch Painting Inventory
# ---------------------------------------------------------------------------

def stage1_metadata():
    """Fetch and cache painting inventory from Rijksmuseum + Met."""
    if INVENTORY_CSV.exists():
        rows = []
        with open(INVENTORY_CSV) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"[Stage 1] Loaded cached inventory: {len(rows)} paintings")
        return rows

    CACHE_META.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_titles = set()  # for cross-source dedup

    # --- Rijksmuseum ---
    print("[Stage 1] Fetching Rijksmuseum inventory...")
    rk_queries = [
        ({"description": "Rembrandt", "type": "painting", "imageAvailable": "true"}, None, None),
        ({"creator": "Ferdinand Bol", "type": "painting", "imageAvailable": "true"}, "rembrandt_pupil", "autograph"),
        ({"creator": "Govert Flinck", "type": "painting", "imageAvailable": "true"}, "rembrandt_pupil", "autograph"),
        ({"creator": "Jan Lievens", "type": "painting", "imageAvailable": "true"}, "rembrandt_pupil", "autograph"),
        ({"creator": "Frans Hals", "type": "painting", "imageAvailable": "true"}, "dutch_other", "autograph"),
        ({"creator": "Johannes Vermeer", "type": "painting", "imageAvailable": "true"}, "dutch_other", "autograph"),
    ]

    rk_seen_ids = set()
    rk_records = []
    for params, artist_group, force_attrib in rk_queries:
        label = params.get("creator") or f"description={params.get('description')}"
        ids = rk_search_all_pages(params)
        new_ids = [i for i in ids if i not in rk_seen_ids]
        rk_seen_ids.update(new_ids)
        print(f"  {label}: {len(ids)} found, {len(new_ids)} new")
        for obj_id in new_ids:
            rk_records.append({"obj_id": obj_id, "artist_group": artist_group, "force_attribution": force_attrib})

    print(f"  Resolving {len(rk_records)} Rijksmuseum paintings...")
    skipped = 0
    for i, rec in enumerate(rk_records):
        obj_id = rec["obj_id"]
        cache_file = CACHE_META / f"rk_{obj_id}.json"

        if cache_file.exists():
            row = json.loads(cache_file.read_text())
            rows.append(row)
            seen_titles.add(row.get("title", "").lower().strip())
            continue

        la = fetch_json(f"{RK_DATA_URL}/{obj_id}", params={"_profile": "la-framed"})
        meta = parse_la_metadata(la)
        edm = fetch_json(f"{RK_DATA_URL}/{obj_id}", params={"_profile": "edm-framed"})
        iiif_id = extract_iiif_id(edm)
        if "title" not in meta:
            edm_title = extract_title_from_edm(edm)
            if edm_title:
                meta["title"] = edm_title

        creator = meta.get("creator", "")

        if rec["artist_group"] is not None:
            artist_group = rec["artist_group"]
            attribution = rec["force_attribution"]
        else:
            artist_group, attribution = classify_rembrandt_group(creator)
            if artist_group is None:
                skipped += 1
                continue

        if not iiif_id:
            skipped += 1
            continue

        image_url = f"{IIIF_BASE}/{iiif_id}/full/!{IMG_MAX_PX},{IMG_MAX_PX}/0/default.jpg"
        row = {
            "obj_id": f"rk_{obj_id}",
            "source": "rijksmuseum",
            "title": meta.get("title", ""),
            "creator": creator,
            "date": meta.get("date", ""),
            "image_url": image_url,
            "artist_group": artist_group,
            "attribution": attribution,
        }
        rows.append(row)
        seen_titles.add(row["title"].lower().strip())
        cache_file.write_text(json.dumps(row, ensure_ascii=False))

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(rk_records)} resolved")

    print(f"  Rijksmuseum: {len([r for r in rows if r.get('source') == 'rijksmuseum'])} paintings ({skipped} skipped)")

    # --- Met Open Access ---
    print("[Stage 1] Fetching Met inventory...")
    met_queries = [
        # Rembrandt across European Paintings (dept 11) and Lehman (dept 15)
        ("Rembrandt", 11, None),
        ("Rembrandt", 15, None),
        # Control: pupils
        ("Ferdinand Bol", 11, "rembrandt_pupil"),
        ("Govert Flinck", 11, "rembrandt_pupil"),
        ("Jan Lievens", 11, "rembrandt_pupil"),
        # Control: other Dutch
        ("Frans Hals", 11, "dutch_other"),
        ("Johannes Vermeer", 11, "dutch_other"),
    ]

    met_seen_ids = set()
    met_records = []
    for query, dept_id, force_group in met_queries:
        ids = met_search(query, dept_id)
        new_ids = [i for i in ids if i not in met_seen_ids]
        met_seen_ids.update(new_ids)
        print(f"  Met dept {dept_id} '{query}': {len(ids)} found, {len(new_ids)} new")
        for obj_id in new_ids:
            met_records.append({"obj_id": obj_id, "force_group": force_group})

    print(f"  Resolving {len(met_records)} Met objects...")
    met_added = 0
    met_skipped = 0
    for i, rec in enumerate(met_records):
        obj_id = rec["obj_id"]
        cache_file = CACHE_META / f"met_{obj_id}.json"

        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("_skip"):
                met_skipped += 1
                continue
            # Dedup against Rijksmuseum
            title_key = cached.get("title", "").lower().strip()
            if title_key and title_key in seen_titles:
                met_skipped += 1
                continue
            rows.append(cached)
            seen_titles.add(title_key)
            met_added += 1
            continue

        obj = met_get_object(obj_id)
        if not obj:
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue

        # Filter: paintings only, public domain, has image
        if obj.get("classification") != "Paintings":
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue
        if not obj.get("isPublicDomain"):
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue
        if not obj.get("primaryImage"):
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue

        artist_name = obj.get("artistDisplayName", "")
        artist_prefix = obj.get("artistPrefix", "")
        attribution = met_classify_attribution(artist_prefix)

        artist_group = met_artist_group(artist_name, artist_prefix, rec["force_group"])
        if artist_group is None:
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue

        # Force attribution for control groups
        if rec["force_group"] is not None:
            attribution = "autograph"

        title = obj.get("title", "")
        # Dedup: skip if same title already from Rijksmuseum
        title_key = title.lower().strip()
        if title_key and title_key in seen_titles:
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue

        row = {
            "obj_id": f"met_{obj_id}",
            "source": "met",
            "title": title,
            "creator": f"{artist_prefix} {artist_name}".strip() if artist_prefix else artist_name,
            "date": obj.get("objectDate", ""),
            "image_url": obj.get("primaryImage", ""),
            "artist_group": artist_group,
            "attribution": attribution,
        }
        rows.append(row)
        seen_titles.add(title_key)
        met_added += 1
        cache_file.write_text(json.dumps(row, ensure_ascii=False))

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(met_records)} resolved")

    print(f"  Met: {met_added} paintings ({met_skipped} skipped)")

    # --- Wikidata: circle/workshop paintings ---
    print("[Stage 1] Fetching Wikidata circle/workshop paintings...")
    wd_circle = wd_fetch_circle_paintings()
    wd_circle_added = 0
    for p in wd_circle:
        cache_file = CACHE_META / f"wd_{p['qid']}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("_skip"):
                continue
            title_key = cached.get("title", "").lower().strip()
            if title_key and title_key in seen_titles:
                continue
            rows.append(cached)
            seen_titles.add(title_key)
            wd_circle_added += 1
            continue

        title_key = p["title"].lower().strip()
        if title_key and title_key in seen_titles:
            cache_file.write_text(json.dumps({"_skip": True}))
            continue

        image_url = wd_image_url(p["image"])
        row = {
            "obj_id": f"wd_{p['qid']}",
            "source": "wikidata",
            "title": p["title"],
            "creator": f"{p['wd_qualifier']} of Rembrandt",
            "date": "",
            "image_url": image_url,
            "artist_group": "rembrandt_circle",
            "attribution": p["attribution"],
        }
        rows.append(row)
        seen_titles.add(title_key)
        wd_circle_added += 1
        cache_file.write_text(json.dumps(row, ensure_ascii=False))
    print(f"  Wikidata circle: {wd_circle_added} paintings (from {len(wd_circle)} SPARQL results)")

    # --- Wikidata: autograph paintings ---
    print("[Stage 1] Fetching Wikidata autograph paintings...")
    wd_auto = wd_fetch_autograph_paintings()
    wd_auto_added = 0
    for p in wd_auto:
        cache_file = CACHE_META / f"wd_{p['qid']}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("_skip"):
                continue
            title_key = cached.get("title", "").lower().strip()
            if title_key and title_key in seen_titles:
                continue
            rows.append(cached)
            seen_titles.add(title_key)
            wd_auto_added += 1
            continue

        title_key = p["title"].lower().strip()
        if title_key and title_key in seen_titles:
            cache_file.write_text(json.dumps({"_skip": True}))
            continue

        image_url = wd_image_url(p["image"])
        row = {
            "obj_id": f"wd_{p['qid']}",
            "source": "wikidata",
            "title": p["title"],
            "creator": "Rembrandt",
            "date": "",
            "image_url": image_url,
            "artist_group": "rembrandt_autograph",
            "attribution": "autograph",
        }
        rows.append(row)
        seen_titles.add(title_key)
        wd_auto_added += 1
        cache_file.write_text(json.dumps(row, ensure_ascii=False))
    print(f"  Wikidata autograph: {wd_auto_added} paintings (from {len(wd_auto)} SPARQL results)")

    # --- Wikidata: control artists (pupils + other Dutch) ---
    print("[Stage 1] Fetching Wikidata control artists...")
    wd_ctrl = wd_fetch_control_paintings()
    wd_ctrl_added = 0
    for p in wd_ctrl:
        cache_file = CACHE_META / f"wd_{p['qid']}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            if cached.get("_skip"):
                continue
            title_key = cached.get("title", "").lower().strip()
            if title_key and title_key in seen_titles:
                continue
            rows.append(cached)
            seen_titles.add(title_key)
            wd_ctrl_added += 1
            continue

        title_key = p["title"].lower().strip()
        if title_key and title_key in seen_titles:
            cache_file.write_text(json.dumps({"_skip": True}))
            continue

        image_url = wd_image_url(p["image"])
        row = {
            "obj_id": f"wd_{p['qid']}",
            "source": "wikidata",
            "title": p["title"],
            "creator": p["artist_name"],
            "date": "",
            "image_url": image_url,
            "artist_group": p["artist_group"],
            "attribution": "autograph",
        }
        rows.append(row)
        seen_titles.add(title_key)
        wd_ctrl_added += 1
        cache_file.write_text(json.dumps(row, ensure_ascii=False))
    print(f"  Wikidata control: {wd_ctrl_added} paintings (from {len(wd_ctrl)} SPARQL results)")

    # --- Museum APIs (supplementary) ---
    print("[Stage 1] Fetching museum API paintings (NGA, CMA, AIC)...")
    museum_added = 0
    for label, fetch_fn in [("NGA", nga_fetch_paintings), ("CMA", cma_search), ("AIC", aic_search)]:
        try:
            museum_rows = fetch_fn()
        except Exception as e:
            print(f"  {label}: error — {e}")
            continue
        source_added = 0
        for row in museum_rows:
            title_key = row["title"].lower().strip()
            if title_key and title_key in seen_titles:
                continue
            cache_file = CACHE_META / f"{row['obj_id']}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text())
                if cached.get("_skip"):
                    continue
                rows.append(cached)
                seen_titles.add(title_key)
                source_added += 1
                continue
            rows.append(row)
            seen_titles.add(title_key)
            source_added += 1
            cache_file.write_text(json.dumps(row, ensure_ascii=False))
        museum_added += source_added
        print(f"  {label}: {source_added} paintings added")
    print(f"  Museum APIs total: {museum_added} paintings")

    # --- Deduplicate and save ---
    seen_obj_ids = set()
    deduped = []
    for row in rows:
        if row["obj_id"] not in seen_obj_ids:
            seen_obj_ids.add(row["obj_id"])
            deduped.append(row)
    rows = deduped

    fieldnames = ["obj_id", "source", "title", "creator", "date", "image_url", "artist_group", "attribution"]
    with open(INVENTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Stage 1] Saved {len(rows)} paintings to {INVENTORY_CSV}")
    _print_inventory_summary(rows)
    return rows


def _print_inventory_summary(rows):
    """Print inventory breakdown."""
    from collections import Counter
    groups = Counter(r["artist_group"] for r in rows)
    sources = Counter(r["source"] for r in rows)
    attribs = Counter(r["attribution"] for r in rows)

    print(f"\n  By source: {dict(sources)}")
    print(f"  By group:  {dict(groups)}")
    print(f"  By attrib: {dict(attribs)}")

    circle = [r for r in rows if r["artist_group"] == "rembrandt_circle"]
    if circle:
        print(f"\n  Circle/disputed works ({len(circle)}):")
        for r in circle:
            print(f"    {r['attribution']:12s} | {r['source']:12s} | {r['creator'][:40]:<40s} | {r['title'][:50]}")


# ---------------------------------------------------------------------------
# Stage 2: Download Images
# ---------------------------------------------------------------------------

def _hires_url(row):
    """Compute high-res download URL from inventory row."""
    if row["source"] == "rijksmuseum":
        # Extract IIIF ID from stored URL, request max resolution
        m = re.search(r'micr\.io/([a-zA-Z0-9]+)/', row["image_url"])
        if m:
            return f"{IIIF_BASE}/{m.group(1)}/full/max/0/default.jpg"
    # Met: primaryImage is already full-res
    return row["image_url"]


def stage2_images(rows, hires=False):
    """Download images. Rijksmuseum via IIIF, Met via primaryImage URL."""
    cache_dir = CACHE_IMG_HIRES if hires else CACHE_IMG
    cache_dir.mkdir(parents=True, exist_ok=True)
    label = "high-res" if hires else "standard"
    print(f"\n[Stage 2] Downloading {label} images for {len(rows)} paintings...")

    success, cached, failed = 0, 0, 0
    for i, row in enumerate(rows):
        img_path = cache_dir / f"{row['obj_id']}.jpg"
        if img_path.exists():
            cached += 1
            continue

        url = _hires_url(row) if hires else row["image_url"]
        resp = fetch_with_retry(url)
        if resp is None:
            print(f"  FAIL: {row['obj_id']} — {row['title'][:40]}")
            failed += 1
            continue

        try:
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            if hires:
                # Cap at IMG_HIRES_MAX_PX to avoid OOM (Night Watch is 14K)
                if max(w, h) > IMG_HIRES_MAX_PX:
                    scale = IMG_HIRES_MAX_PX / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            else:
                # v1 behavior: resize non-Rijksmuseum to 2000px
                if row["source"] != "rijksmuseum" and max(w, h) > IMG_MAX_PX:
                    scale = IMG_MAX_PX / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(img_path, "JPEG", quality=95)
            success += 1
        except Exception as e:
            print(f"  FAIL decode: {row['obj_id']} — {e}")
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(rows)} ({success} new, {cached} cached, {failed} failed)")

    total_ok = success + cached
    print(f"[Stage 2] Done: {total_ok} available ({cached} cached, {success} new), {failed} failed")
    return total_ok


# ---------------------------------------------------------------------------
# Stage 3: Tile & Embed
# ---------------------------------------------------------------------------

def _tile_batches(img, tile_size, batch_size, transform):
    """Generator: yields batches of transformed tile tensors from an image.

    Memory-safe — never holds all tiles in memory. Critical for high-res
    images (8000px = ~1200 tiles vs 2000px = ~60 tiles).
    """
    import torch
    w, h = img.size
    batch = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            batch.append(transform(tile))
            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []
    if batch:
        yield torch.stack(batch)


def stage3_embed(rows, hires=False, model_name="vitb14", entropy=False):
    """Tile images and embed with DINOv2.

    v1 (hires=False): mean-only aggregation. Cache: embeddings.npz
    v2 (hires=True):  mean+std aggregation. Cache: embeddings_v2.npz
    vitl14:           ViT-L model (1024d vs 768d). Cache: embeddings_vitl.npz
    entropy:          entropy-weighted tile aggregation. Cache: embeddings_entropy[_vitl].npz
    """
    CACHE_EMB.mkdir(parents=True, exist_ok=True)
    if entropy:
        cache_file = EMBEDDINGS_ENTROPY_VITL_NPZ if model_name == "vitl14" else EMBEDDINGS_ENTROPY_NPZ
    elif model_name == "vitl14":
        cache_file = EMBEDDINGS_VITL_NPZ
    elif hires:
        cache_file = EMBEDDINGS_V2_NPZ
    else:
        cache_file = EMBEDDINGS_NPZ
    img_dir = CACHE_IMG_HIRES if hires else CACHE_IMG

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        print(f"[Stage 3] Loaded cached embeddings: {len(data['painting_ids'])} paintings")
        keys = list(data.keys())
        if hires and "cls_std" in keys:
            return (data["painting_ids"], data["cls_mean"], data["cls_std"],
                    data["patch_mean"], data["patch_std"],
                    data["artist_groups"], data["attributions"])
        return (data["painting_ids"], data["cls_embeddings"],
                data["patch_embeddings"], data["artist_groups"], data["attributions"])

    import torch
    from torchvision import transforms

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Stage 3] Device: {device}")

    dino_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dino_hub_name = f"dinov2_{model_name}"
    model_label = "ViT-L/14" if model_name == "vitl14" else "ViT-B/14"
    batch_size = BATCH_SIZE_VITL if model_name == "vitl14" else BATCH_SIZE_VITB
    print(f"[Stage 3] Loading DINOv2 {model_label}...")
    model = torch.hub.load("facebookresearch/dinov2", dino_hub_name)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")

    painting_ids = []
    cls_mean_list = []
    cls_std_list = []
    patch_mean_list = []
    patch_std_list = []
    groups_list = []
    attribs_list = []
    skipped = 0

    chunk_starts = list(range(0, len(rows), CHUNK_SIZE))
    print(f"  Embedding {len(rows)} paintings in {len(chunk_starts)} chunks of {CHUNK_SIZE}...")

    for ci, start in enumerate(chunk_starts):
        chunk = rows[start:start + CHUNK_SIZE]

        for row in chunk:
            img_path = img_dir / f"{row['obj_id']}.jpg"
            if not img_path.exists():
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  Skip bad image {row['obj_id']}: {e}")
                skipped += 1
                continue

            w, h = img.size
            n_tiles = (w // TILE_SIZE) * (h // TILE_SIZE)
            if n_tiles == 0:
                skipped += 1
                continue

            # Stream tiles through model — never hold all in memory
            all_cls = []
            all_patch = []
            all_tile_var = []
            for batch_tensor in _tile_batches(img, TILE_SIZE, batch_size, dino_transform):
                batch_tensor = batch_tensor.to(device)
                with torch.no_grad():
                    out = model.forward_features(batch_tensor)
                    all_cls.append(out["x_norm_clstoken"].cpu().numpy())
                    patch_tokens = out["x_norm_patchtokens"]
                    all_patch.append(patch_tokens.mean(dim=1).cpu().numpy())
                    if entropy:
                        # Patch token variance = visual complexity proxy
                        tile_var = patch_tokens.var(dim=1).mean(dim=1)  # (batch,)
                        all_tile_var.append(tile_var.cpu().numpy())

            all_cls = np.concatenate(all_cls, axis=0)    # (N_tiles, embed_dim)
            all_patch = np.concatenate(all_patch, axis=0)  # (N_tiles, embed_dim)

            painting_ids.append(row["obj_id"])

            if entropy:
                tile_vars = np.concatenate(all_tile_var)
                var_std = tile_vars.std()
                if var_std > 1e-6:
                    z = (tile_vars - tile_vars.mean()) / var_std
                    weights = np.exp(z)
                    weights /= weights.sum()
                else:
                    weights = np.ones(len(tile_vars)) / len(tile_vars)
                cls_mean_list.append((all_cls * weights[:, None]).sum(axis=0))
                patch_mean_list.append((all_patch * weights[:, None]).sum(axis=0))
                # Diagnostic for first 3 paintings
                if len(painting_ids) <= 3:
                    uniform = 1.0 / len(tile_vars)
                    print(f"      {row['obj_id']}: {len(tile_vars)} tiles, "
                          f"var range [{tile_vars.min():.4f}, {tile_vars.max():.4f}], "
                          f"weight range [{weights.min():.4f}, {weights.max():.4f}], "
                          f"max/uniform {weights.max()/uniform:.1f}x")
            else:
                cls_mean_list.append(all_cls.mean(axis=0))
                patch_mean_list.append(all_patch.mean(axis=0))

            if hires:
                # Distribution features: std captures style consistency
                cls_std_list.append(all_cls.std(axis=0))
                patch_std_list.append(all_patch.std(axis=0))

            groups_list.append(row["artist_group"])
            attribs_list.append(row["attribution"])

            if hires and (len(painting_ids) % 10 == 0 or len(painting_ids) == 1):
                print(f"      {row['obj_id']}: {img.size[0]}x{img.size[1]}, {n_tiles} tiles")

            del all_cls, all_patch, img

        print(f"    Chunk {ci+1}/{len(chunk_starts)} done ({len(painting_ids)} embedded, {skipped} skipped)")

    painting_ids = np.array(painting_ids)
    artist_groups = np.array(groups_list)
    attributions = np.array(attribs_list)

    if hires:
        cls_mean = np.array(cls_mean_list)
        cls_std = np.array(cls_std_list)
        patch_mean = np.array(patch_mean_list)
        patch_std = np.array(patch_std_list)
        np.savez(cache_file,
                 painting_ids=painting_ids,
                 cls_mean=cls_mean, cls_std=cls_std,
                 patch_mean=patch_mean, patch_std=patch_std,
                 artist_groups=artist_groups, attributions=attributions)
        print(f"[Stage 3] Saved {len(painting_ids)} embeddings ({cls_mean.shape[1]*4}d) to {cache_file}")
        del model
        return painting_ids, cls_mean, cls_std, patch_mean, patch_std, artist_groups, attributions
    else:
        cls_embeddings = np.array(cls_mean_list)
        patch_embeddings = np.array(patch_mean_list)
        np.savez(cache_file,
                 painting_ids=painting_ids,
                 cls_embeddings=cls_embeddings,
                 patch_embeddings=patch_embeddings,
                 artist_groups=artist_groups, attributions=attributions)
        print(f"[Stage 3] Saved {len(painting_ids)} embeddings to {cache_file}")
        del model
        return painting_ids, cls_embeddings, patch_embeddings, artist_groups, attributions


# ---------------------------------------------------------------------------
# Stage 4: Analysis
# ---------------------------------------------------------------------------

def stage4_analysis(painting_ids, artist_groups, attributions, rows,
                    cls_mean=None, cls_std=None, patch_mean=None, patch_std=None,
                    cls_embeddings=None, patch_embeddings=None, hires=False,
                    model_name="vitb14", entropy=False):
    """Run five analyses. Save plots and metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from scipy.stats import mannwhitneyu
    from sklearn.metrics.pairwise import cosine_similarity

    CACHE_PLOTS.mkdir(parents=True, exist_ok=True)

    # Build full embedding: v2 = mean+std (3072d), v1 = mean only (1536d)
    if hires and cls_std is not None:
        full_embeddings = np.concatenate([cls_mean, cls_std, patch_mean, patch_std], axis=1)
    else:
        cm = cls_mean if cls_mean is not None else cls_embeddings
        pm = patch_mean if patch_mean is not None else patch_embeddings
        full_embeddings = np.concatenate([cm, pm], axis=1)

    n = len(painting_ids)
    if entropy:
        results_file = RESULTS_ENTROPY_VITL_JSON if model_name == "vitl14" else RESULTS_ENTROPY_JSON
    elif model_name == "vitl14":
        results_file = RESULTS_VITL_JSON
    elif hires:
        results_file = RESULTS_V2_JSON
    else:
        results_file = RESULTS_JSON
    print(f"\n[Stage 4] Analysing {n} paintings ({full_embeddings.shape[1]}d embeddings)")

    GROUP_COLORS = {
        "rembrandt_autograph": "#e41a1c",
        "rembrandt_circle": "#ff7f00",
        "rembrandt_pupil": "#984ea3",
        "dutch_other": "#4daf4a",
    }
    GROUP_LABELS = {
        "rembrandt_autograph": "Rembrandt (autograph)",
        "rembrandt_circle": "Circle/Workshop/Style",
        "rembrandt_pupil": "Pupils (Bol, Flinck, Lievens)",
        "dutch_other": "Other Dutch (Hals, Vermeer)",
    }

    # Build source lookup
    id_to_source = {}
    for row in rows:
        id_to_source[row["obj_id"]] = row.get("source", "unknown")

    # Masks
    auto_mask = artist_groups == "rembrandt_autograph"
    circle_mask = artist_groups == "rembrandt_circle"
    pupil_mask = artist_groups == "rembrandt_pupil"
    other_mask = artist_groups == "dutch_other"

    sim_matrix = cosine_similarity(full_embeddings)

    def pairwise_sims(mask_a, mask_b, sim_mat):
        idx_a = np.where(mask_a)[0]
        idx_b = np.where(mask_b)[0]
        sims = []
        for i in idx_a:
            for j in idx_b:
                if i < j or not np.array_equal(mask_a, mask_b):
                    sims.append(sim_mat[i, j])
        return np.array(sims)

    # -------------------------------------------------------
    # 4a: UMAP scatter
    # -------------------------------------------------------
    print("  4a: UMAP projection...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(full_embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for group, color in GROUP_COLORS.items():
        mask = artist_groups == group
        if mask.sum() == 0:
            continue
        # Mark source with different markers
        for source, marker in [("rijksmuseum", "o"), ("met", "s")]:
            source_mask = np.array([id_to_source.get(str(pid), "") == source
                                    for pid in painting_ids])
            combined = mask & source_mask
            if combined.sum() == 0:
                continue
            label = f"{GROUP_LABELS[group]} ({source[:3].upper()})" if combined.sum() > 0 else None
            ax.scatter(umap_coords[combined, 0], umap_coords[combined, 1],
                       c=color, marker=marker, label=label,
                       s=60, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_title("DINOv2 Embedding Space — Rijksmuseum + Met", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(CACHE_PLOTS / "umap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved umap.png")

    # -------------------------------------------------------
    # 4b: Cosine distributions + Mann-Whitney
    # -------------------------------------------------------
    print("  4b: Cosine distributions...")
    auto_auto = pairwise_sims(auto_mask, auto_mask, sim_matrix)
    auto_circle = pairwise_sims(auto_mask, circle_mask, sim_matrix)
    auto_pupil = pairwise_sims(auto_mask, pupil_mask, sim_matrix)
    auto_other = pairwise_sims(auto_mask, other_mask, sim_matrix)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    bins = np.linspace(0, 1, 50)
    for sims, label, color in [
        (auto_auto, "Autograph↔Autograph", "#e41a1c"),
        (auto_circle, "Autograph↔Circle", "#ff7f00"),
        (auto_pupil, "Autograph↔Pupils", "#984ea3"),
        (auto_other, "Autograph↔Other", "#4daf4a"),
    ]:
        if len(sims) > 0:
            ax.hist(sims, bins=bins, alpha=0.6, label=f"{label} (n={len(sims)})", color=color)
    ax.set_title("Cosine Similarity Distributions", fontsize=14)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(CACHE_PLOTS / "cosine.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved cosine.png")

    # Mann-Whitney U tests
    p_values = {}
    print("    Mann-Whitney U (one-tailed: intra > inter):")
    for name, inter_sims in [("circle", auto_circle), ("pupils", auto_pupil), ("other", auto_other)]:
        if len(auto_auto) > 0 and len(inter_sims) > 0:
            stat, p = mannwhitneyu(auto_auto, inter_sims, alternative="greater")
            effect = auto_auto.mean() - inter_sims.mean()
            p_values[name] = p
            print(f"      vs {name}: U={stat:.0f}, p={p:.2e}, Δmean={effect:+.4f}")

    # -------------------------------------------------------
    # 4c: Similarity heatmap
    # -------------------------------------------------------
    print("  4c: Heatmap...")
    group_order = ["rembrandt_autograph", "rembrandt_circle", "rembrandt_pupil", "dutch_other"]
    sort_idx = np.argsort([group_order.index(g) for g in artist_groups])
    sorted_sim = sim_matrix[sort_idx][:, sort_idx]
    sorted_groups = artist_groups[sort_idx]

    boundaries = [(sorted_groups == g).sum() for g in group_order]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(sorted_sim, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    cumsum = np.cumsum(boundaries[:-1])
    for b in cumsum:
        ax.axhline(y=b - 0.5, color="black", linewidth=1)
        ax.axvline(x=b - 0.5, color="black", linewidth=1)

    positions = []
    pos = 0
    for g, count in zip(group_order, boundaries):
        if count > 0:
            positions.append((pos + count / 2, GROUP_LABELS.get(g, g)))
        pos += count
    ax.set_xticks([p for p, _ in positions])
    ax.set_xticklabels([l for _, l in positions], rotation=45, ha="right", fontsize=9)
    ax.set_yticks([p for p, _ in positions])
    ax.set_yticklabels([l for _, l in positions], fontsize=9)
    ax.set_title("Cosine Similarity Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(CACHE_PLOTS / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved heatmap.png")

    # -------------------------------------------------------
    # 4d: KNN leave-one-out
    # -------------------------------------------------------
    print("  4d: KNN leave-one-out...")
    labels = (artist_groups == "rembrandt_autograph").astype(int)
    baseline = max(labels.mean(), 1 - labels.mean())

    knn_results = {}
    for K in [3, 5, 7]:
        correct = 0
        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            neighbors = np.argsort(sims)[-K:]
            vote = labels[neighbors].mean() > 0.5
            if vote == labels[i]:
                correct += 1
        acc = correct / n
        knn_results[K] = acc
        print(f"      K={K}: {acc:.1%} ({correct}/{n})")
    print(f"      Baseline (majority): {baseline:.1%}")

    # -------------------------------------------------------
    # 4e: Top candidates closest to autograph centroid
    # -------------------------------------------------------
    print("  4e: Top candidates...")
    auto_centroid = full_embeddings[auto_mask].mean(axis=0, keepdims=True)
    circle_idx = np.where(circle_mask)[0]

    candidates = []
    if len(circle_idx) > 0:
        circle_sims = cosine_similarity(full_embeddings[circle_idx], auto_centroid).flatten()
        top_idx = np.argsort(circle_sims)[::-1]

        print(f"\n    {'Rank':<5} {'Sim':>6} {'Source':<5} {'Title':<45} {'Creator'}")
        print(f"    {'-'*95}")
        id_to_row = {r["obj_id"]: r for r in rows}
        for rank, idx in enumerate(top_idx, 1):
            global_idx = circle_idx[idx]
            pid = str(painting_ids[global_idx])
            sim = circle_sims[idx]
            row_data = id_to_row.get(pid, {})
            title = row_data.get("title", "?")[:43]
            creator = row_data.get("creator", "?")
            source = row_data.get("source", "?")[:3].upper()
            print(f"    {rank:<5} {sim:>6.4f} {source:<5} {title:<45} {creator}")
            candidates.append({
                "rank": rank, "sim": float(sim), "obj_id": pid,
                "source": row_data.get("source", ""), "title": row_data.get("title", ""),
                "creator": row_data.get("creator", ""),
            })

    # -------------------------------------------------------
    # Build metrics dict
    # -------------------------------------------------------
    metrics = {
        "source": "Rijksmuseum + Met",
        "n_paintings": n,
        "n_autograph": int(auto_mask.sum()),
        "n_circle": int(circle_mask.sum()),
        "n_pupil": int(pupil_mask.sum()),
        "n_other": int(other_mask.sum()),
        "auto_auto_sim": float(auto_auto.mean()) if len(auto_auto) > 0 else None,
        "auto_circle_sim": float(auto_circle.mean()) if len(auto_circle) > 0 else None,
        "auto_pupil_sim": float(auto_pupil.mean()) if len(auto_pupil) > 0 else None,
        "auto_other_sim": float(auto_other.mean()) if len(auto_other) > 0 else None,
        "mw_p_circle": float(p_values.get("circle", 1.0)),
        "mw_p_pupil": float(p_values.get("pupils", 1.0)),
        "mw_p_other": float(p_values.get("other", 1.0)),
        "knn_k3": knn_results[3],
        "knn_k5": knn_results[5],
        "knn_k7": knn_results[7],
        "baseline": float(baseline),
        "candidates": candidates,
    }

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n    Metrics saved to {results_file}")

    return metrics


# ---------------------------------------------------------------------------
# Stage 5: Before/After Comparison
# ---------------------------------------------------------------------------

def stage5_comparison(metrics, hires=False, model_name="vitb14", entropy=False):
    """Print side-by-side comparison."""
    # Compare against v1 combined results if hires/vitl/entropy, else against prototype
    if entropy and RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            old = json.load(f)
        model_suffix = " ViT-L/14" if model_name == "vitl14" else " ViT-B/14"
        old_label = f"v1{model_suffix} (mean)"
        new_label = f"v1-D Entropy-weighted"
    elif model_name == "vitl14" and RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            old = json.load(f)
        old_label = "v1 ViT-B/14"
        new_label = "v1 ViT-L/14"
    elif hires and RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            old = json.load(f)
        old_label = "v1 (low-res, mean)"
        new_label = "v2 (high-res, mean+std)"
    else:
        old = PROTOTYPE_METRICS
        old_label = "Rijksmuseum"
        new_label = "Combined"
    new = metrics

    print(f"\n{'='*70}")
    print(f"RESULTS COMPARISON: {old_label} vs {new_label}")
    print(f"{'='*70}")

    def row(label, old_val, new_val, fmt=".4f", better="auto"):
        old_s = f"{old_val:{fmt}}" if old_val is not None else "N/A"
        new_s = f"{new_val:{fmt}}" if new_val is not None else "N/A"
        delta = ""
        if old_val is not None and new_val is not None:
            diff = new_val - old_val
            if better == "lower":
                arrow = "+" if diff < 0 else "-" if diff > 0 else "="
            else:
                arrow = "+" if diff > 0 else "-" if diff < 0 else "="
            delta = f" ({arrow}{abs(diff):{fmt}})"
        print(f"  {label:<35} {old_s:>12} → {new_s:>12}{delta}")

    print(f"\n  {'Metric':<35} {old_label:>12}   {new_label:>12}")
    print(f"  {'-'*70}")
    row("Paintings", old["n_paintings"], new["n_paintings"], "d")
    row("Autograph", old["n_autograph"], new["n_autograph"], "d")
    row("Circle/disputed", old["n_circle"], new["n_circle"], "d")
    row("Pupils", old["n_pupil"], new["n_pupil"], "d")
    row("Other Dutch", old["n_other"], new["n_other"], "d")
    print()
    row("Auto↔Auto sim", old["auto_auto_sim"], new["auto_auto_sim"])
    row("Auto↔Circle sim", old["auto_circle_sim"], new["auto_circle_sim"])
    row("Auto↔Pupil sim", old["auto_pupil_sim"], new["auto_pupil_sim"])
    row("Auto↔Other sim", old["auto_other_sim"], new["auto_other_sim"])
    print()
    row("MW p-value (circle)", old["mw_p_circle"], new["mw_p_circle"], ".2e", "lower")
    row("MW p-value (pupils)", old["mw_p_pupil"], new["mw_p_pupil"], ".2e", "lower")
    row("MW p-value (other)", old["mw_p_other"], new["mw_p_other"], ".2e", "lower")
    print()
    row("KNN K=3 accuracy", old["knn_k3"], new["knn_k3"], ".1%")
    row("KNN K=5 accuracy", old["knn_k5"], new["knn_k5"], ".1%")
    row("KNN K=7 accuracy", old["knn_k7"], new["knn_k7"], ".1%")
    row("Baseline (majority)", old["baseline"], new["baseline"], ".1%")

    # Signal assessment
    min_p = min(new["mw_p_circle"], new["mw_p_pupil"], new["mw_p_other"])
    best_knn = max(new["knn_k3"], new["knn_k5"], new["knn_k7"])
    if min_p < 0.01 and best_knn > 0.85:
        signal = "STRONG"
    elif min_p < 0.05 and best_knn > new["baseline"]:
        signal = "WEAK"
    else:
        signal = "NONE"

    circle_p_improved = new["mw_p_circle"] < old["mw_p_circle"]
    print(f"\n  Signal assessment: {signal}")
    print(f"  Circle p-value improved: {'YES' if circle_p_improved else 'NO'} "
          f"({old['mw_p_circle']:.2e} → {new['mw_p_circle']:.2e})")
    print(f"  Circle N: {old['n_circle']} → {new['n_circle']}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Stage 4b: Linear Probe (Option F)
# ---------------------------------------------------------------------------

def stage4b_probe(full_embeddings, artist_groups, painting_ids, rows):
    """PCA + multi-classifier LOO-CV + permutation test on autograph vs circle."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    # Filter to autograph + circle only
    mask = np.array([(g in ("rembrandt_autograph", "rembrandt_circle")) for g in artist_groups])
    X = full_embeddings[mask]
    y = np.array([1 if g == "rembrandt_autograph" else 0 for g in artist_groups[mask]])
    ids = painting_ids[mask]
    n = len(y)
    n_auto = int(y.sum())
    n_circle = n - n_auto
    print(f"\n[Stage 4b] Probe: {n_auto} autograph + {n_circle} circle = {n} paintings")
    print(f"  Features: {X.shape[1]}d → PCA reduction\n")

    pca_dims = [d for d in [10, 20] if d < n]

    # Classifier configs: (name, make_clf, param_name, param_values)
    classifiers = [
        ("Logistic", lambda v: LogisticRegression(C=v, solver="lbfgs", max_iter=1000),
         "C", [0.001, 0.01, 0.1, 1.0, 10.0]),
        ("SVM RBF", lambda v: SVC(C=v, kernel="rbf", gamma="scale"),
         "C", [0.001, 0.01, 0.1, 1.0, 10.0]),
        ("MLP (32)", lambda v: MLPClassifier(hidden_layer_sizes=(32,), alpha=v, max_iter=1000, random_state=42),
         "alpha", [0.01, 0.1, 1.0, 10.0]),
    ]

    def loo_accuracy(X_data, y_data, make_clf, param_val):
        """Leave-one-out CV accuracy for a given classifier."""
        correct = 0
        for i in range(len(y_data)):
            X_train = np.delete(X_data, i, axis=0)
            y_train = np.delete(y_data, i)
            X_test = X_data[i:i+1]
            clf = make_clf(param_val)
            clf.fit(X_train, y_train)
            if clf.predict(X_test)[0] == y_data[i]:
                correct += 1
        return correct / len(y_data)

    # Grid search per classifier
    all_results = []
    overall_best_acc = 0
    overall_best = None  # (name, dims, param_name, param_val, make_clf)

    for clf_name, make_clf, param_name, param_values in classifiers:
        print(f"  --- {clf_name} ---")
        print(f"  {'PCA dims':<10} " + " ".join(f"{param_name}={v:<7}" for v in param_values))
        print(f"  {'-'*10} " + " ".join(f"{'-'*10}" for _ in param_values))
        best_acc, best_dims, best_param = 0, 0, 0
        for dims in pca_dims:
            pca = PCA(n_components=dims, random_state=42)
            X_pca = pca.fit_transform(X)
            row_str = f"  {dims:<10}"
            for v in param_values:
                acc = loo_accuracy(X_pca, y, make_clf, v)
                row_str += f" {acc:<10.3f}"
                if acc > best_acc:
                    best_acc, best_dims, best_param = acc, dims, v
            print(row_str)
        print(f"  Best: PCA={best_dims}, {param_name}={best_param} → {best_acc:.3f}\n")
        all_results.append({
            "classifier": clf_name,
            "best_pca_dims": best_dims,
            "best_param_name": param_name,
            "best_param_value": best_param,
            "loo_accuracy": round(best_acc, 4),
        })
        if best_acc > overall_best_acc:
            overall_best_acc = best_acc
            overall_best = (clf_name, best_dims, param_name, best_param, make_clf)

    # Comparison table
    print(f"  COMPARISON")
    print(f"  {'Classifier':<14} {'Best PCA':<10} {'Best param':<14} {'LOO acc':<8}")
    print(f"  {'-'*14} {'-'*10} {'-'*14} {'-'*8}")
    for r in all_results:
        param_str = f"{r['best_param_name']}={r['best_param_value']}"
        print(f"  {r['classifier']:<14} {r['best_pca_dims']:<10} {param_str:<14} {r['loo_accuracy']:<8.3f}")

    best_name, best_dims, best_pname, best_pval, best_make_clf = overall_best
    print(f"\n  Overall best: {best_name} (PCA={best_dims}, {best_pname}={best_pval}) → {overall_best_acc:.3f}")

    # Permutation test on overall best only
    n_perms = 1000
    print(f"\n  Permutation test on {best_name} ({n_perms} shuffles)...")
    pca = PCA(n_components=best_dims, random_state=42)
    X_best = pca.fit_transform(X)
    null_accs = np.zeros(n_perms)
    rng = np.random.RandomState(42)
    for p in range(n_perms):
        y_shuf = rng.permutation(y)
        null_accs[p] = loo_accuracy(X_best, y_shuf, best_make_clf, best_pval)
        if (p + 1) % 100 == 0:
            print(f"    {p+1}/{n_perms} done")

    p_value = (np.sum(null_accs >= overall_best_acc) + 1) / (n_perms + 1)
    null_mean = null_accs.mean()
    null_std = null_accs.std()

    print(f"\n{'='*60}")
    print(f"  PROBE RESULTS (Option H — Non-linear)")
    print(f"{'='*60}")
    print(f"  N autograph:          {n_auto}")
    print(f"  N circle:             {n_circle}")
    print(f"  Best classifier:      {best_name}")
    print(f"  Best PCA dims:        {best_dims}")
    print(f"  Best {best_pname}:{'':>{13-len(best_pname)}}{best_pval}")
    print(f"  LOO accuracy:         {overall_best_acc:.3f}")
    print(f"  Permutation p-value:  {p_value:.4f}")
    print(f"  Null mean +/- std:    {null_mean:.3f} +/- {null_std:.3f}")
    print(f"  Signal:               {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}")
    print(f"{'='*60}")

    results = {
        "n_autograph": n_auto,
        "n_circle": n_circle,
        "n_total": n,
        "classifier": best_name,
        "best_pca_dims": best_dims,
        "best_param_name": best_pname,
        "best_param_value": best_pval,
        "loo_accuracy": round(overall_best_acc, 4),
        "permutation_p_value": round(p_value, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "n_permutations": n_perms,
        "all_classifiers": all_results,
    }
    with open(RESULTS_PROBE_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_PROBE_JSON}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DINOv2 authentication pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only this stage (requires prior stages cached)")
    parser.add_argument("--hires", action="store_true",
                        help="v2: high-res images + distribution features (mean+std)")
    parser.add_argument("--model", choices=["vitb14", "vitl14"], default="vitb14",
                        help="DINOv2 backbone: vitb14 (768d, default) or vitl14 (1024d)")
    parser.add_argument("--entropy", action="store_true",
                        help="Option D: entropy-weighted tile aggregation")
    parser.add_argument("--probe", action="store_true",
                        help="Option F: linear probe on frozen embeddings (autograph vs circle)")
    parser.add_argument("--refetch", action="store_true",
                        help="Delete cached inventory + embeddings to force re-fetch from all sources")
    args = parser.parse_args()
    hires = args.hires
    model_name = args.model
    entropy = args.entropy
    probe = args.probe

    for d in [CACHE_META, CACHE_IMG, CACHE_IMG_HIRES, CACHE_EMB, CACHE_PLOTS]:
        d.mkdir(parents=True, exist_ok=True)

    if args.refetch:
        for f in [INVENTORY_CSV, EMBEDDINGS_NPZ, EMBEDDINGS_V2_NPZ,
                  EMBEDDINGS_VITL_NPZ, EMBEDDINGS_ENTROPY_NPZ,
                  EMBEDDINGS_ENTROPY_VITL_NPZ]:
            if f.exists():
                f.unlink()
                print(f"  Deleted {f.name}")
        # Delete per-source metadata caches so Wikidata/museum results re-fetch
        for f in CACHE_META.glob("wd_*.json"):
            f.unlink()
        for f in CACHE_META.glob("nga_*.json"):
            f.unlink()
        for f in CACHE_META.glob("cma_*.json"):
            f.unlink()
        for f in CACHE_META.glob("aic_*.json"):
            f.unlink()
        print("[Refetch] Cleared cached data — will re-fetch from all sources")

    # --probe: load cached v1 embeddings, run linear probe, exit
    if probe:
        print("[Mode] v1-H — probe: Logistic + SVM RBF + MLP (autograph vs circle)")
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run full pipeline first.")
            sys.exit(1)
        if not EMBEDDINGS_NPZ.exists():
            print("ERROR: No cached v1 embeddings. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
        painting_ids = data["painting_ids"]
        cls_emb = data["cls_embeddings"]
        patch_emb = data["patch_embeddings"]
        artist_groups = data["artist_groups"]
        full_embeddings = np.concatenate([cls_emb, patch_emb], axis=1)
        stage4b_probe(full_embeddings, artist_groups, painting_ids, rows)
        return

    start_stage = args.stage or 1
    end_stage = args.stage or 5

    model_label = "ViT-L/14" if model_name == "vitl14" else "ViT-B/14"
    if entropy:
        print(f"[Mode] v1-D — entropy-weighted tile aggregation, {model_label}")
    elif hires:
        print(f"[Mode] v2 — high-res images, mean+std features, {model_label}")
    else:
        print(f"[Mode] v1 — standard images, mean-only features, {model_label}")

    # Stage 1 (same for both modes)
    if start_stage <= 1:
        rows = stage1_metadata()
    else:
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run stage 1 first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        print(f"[Stage 1] Loaded {len(rows)} paintings from cache")

    if end_stage < 2:
        return

    # Stage 2
    if start_stage <= 2:
        stage2_images(rows, hires=hires)

    if end_stage < 3:
        return

    # Stage 3
    if entropy:
        emb_file = EMBEDDINGS_ENTROPY_VITL_NPZ if model_name == "vitl14" else EMBEDDINGS_ENTROPY_NPZ
    elif model_name == "vitl14":
        emb_file = EMBEDDINGS_VITL_NPZ
    elif hires:
        emb_file = EMBEDDINGS_V2_NPZ
    else:
        emb_file = EMBEDDINGS_NPZ
    if start_stage <= 3:
        emb_result = stage3_embed(rows, hires=hires, model_name=model_name, entropy=entropy)
    else:
        if not emb_file.exists():
            print(f"ERROR: No cached embeddings at {emb_file}. Run stage 3 first.")
            sys.exit(1)
        data = np.load(emb_file, allow_pickle=True)
        if hires and "cls_std" in data:
            emb_result = (data["painting_ids"], data["cls_mean"], data["cls_std"],
                          data["patch_mean"], data["patch_std"],
                          data["artist_groups"], data["attributions"])
        else:
            emb_result = (data["painting_ids"], data["cls_embeddings"],
                          data["patch_embeddings"], data["artist_groups"],
                          data["attributions"])

    if end_stage < 4:
        return

    # Stage 4 — unpack based on mode
    if hires and len(emb_result) == 7:
        pids, cls_m, cls_s, patch_m, patch_s, groups, attribs = emb_result
        analysis_kwargs = dict(cls_mean=cls_m, cls_std=cls_s,
                               patch_mean=patch_m, patch_std=patch_s, hires=True)
    else:
        pids, cls_emb, patch_emb, groups, attribs = emb_result
        analysis_kwargs = dict(cls_embeddings=cls_emb, patch_embeddings=patch_emb)

    analysis_kwargs["model_name"] = model_name
    analysis_kwargs["entropy"] = entropy
    if start_stage <= 4:
        metrics = stage4_analysis(pids, groups, attribs, rows, **analysis_kwargs)
    else:
        if entropy:
            results_file = RESULTS_ENTROPY_VITL_JSON if model_name == "vitl14" else RESULTS_ENTROPY_JSON
        elif model_name == "vitl14":
            results_file = RESULTS_VITL_JSON
        elif hires:
            results_file = RESULTS_V2_JSON
        else:
            results_file = RESULTS_JSON
        if not results_file.exists():
            print(f"ERROR: No cached results at {results_file}. Run stage 4 first.")
            sys.exit(1)
        with open(results_file) as f:
            metrics = json.load(f)

    if end_stage < 5:
        return

    # Stage 5
    stage5_comparison(metrics, hires=hires, model_name=model_name, entropy=entropy)


if __name__ == "__main__":
    main()
