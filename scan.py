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
INVENTORY_TRANSFER_CSV = CACHE_META / "inventory_transfer.csv"
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
RESULTS_CONFOUNDER_JSON = CACHE_DIR / "results_confounder.json"
EMBEDDINGS_TRANSFER_NPZ = CACHE_EMB / "embeddings_transfer.npz"
RESULTS_TRANSFER_JSON = CACHE_DIR / "results_transfer.json"
RESULTS_LORA_JSON = CACHE_DIR / "results_lora.json"
RESULTS_LORA_TRANSFER_JSON = CACHE_DIR / "results_lora_transfer.json"
RESULTS_LORA_CURRICULUM_JSON = CACHE_DIR / "results_lora_curriculum.json"
RESULTS_ROBUSTNESS_JSON = CACHE_DIR / "results_robustness.json"

TILE_SIZE = 224
IMG_MAX_PX = 2000        # v1 low-res cap
IMG_HIRES_MAX_PX = 8000  # v2 cap (avoids OOM on Night Watch 14K)
BATCH_SIZE_VITB = 16     # conservative for 8 GB RAM
BATCH_SIZE_VITL = 8      # ViT-L is ~1.2 GB vs ~350 MB
CHUNK_SIZE = 5           # paintings per chunk (smaller for hires — more tiles per painting)
EMBED_DIM = 768          # DINOv2 ViT-B/14=768, ViT-L/14=1024
RATE_LIMIT = 5.0     # seconds between image downloads (Wikimedia rate limit)

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
# Transfer learning artists: name → (QID, short_label)
WD_TRANSFER_ARTISTS = {
    "Rubens": ("Q5599", "rubens"),
    "Cranach": ("Q191748", "cranach"),
    "Van Dyck": ("Q150679", "vandyck"),
    "Titian": ("Q47551", "titian"),
    "Frans Hals": ("Q167654", "hals"),
    "Rembrandt": ("Q5598", "rembrandt"),
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

def fetch_with_retry(url, params=None, max_retries=5, backoff=0.5):
    """GET with exponential backoff. Returns response or None."""
    for attempt in range(max_retries):
        try:
            delay = backoff * (2 ** attempt) if attempt > 0 else RATE_LIMIT
            time.sleep(delay)
            resp = requests.get(url, params=params, timeout=30,
                                headers={"User-Agent": WD_UA})
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10)) + 1
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
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
    """Classify attribution level from a creator/title string.

    Returns (attribution, label_confidence):
      - high: Wikidata qualifier match or explicit museum prefix
      - medium: regex match on creator string (ATTRIBUTION_PATTERNS)
      - low: default fallback (no match → assumed autograph)
    """
    t = text.lower()
    for level, patterns in ATTRIBUTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t):
                return level, "medium"
    return "autograph", "low"


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
    """Classify Rembrandt-related painting. Returns (artist_group, attribution, label_confidence) or (None, None, None)."""
    if "rembrandt" not in creator_str.lower():
        return None, None, None
    attrib, confidence = classify_attribution(creator_str)
    if attrib == "autograph":
        return "rembrandt_autograph", "autograph", confidence
    return "rembrandt_circle", attrib, confidence


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
    """Map Met artistPrefix to attribution level.

    Returns (attribution, label_confidence):
      - high: explicit museum prefix (Workshop of, Circle of, etc.)
      - low: empty prefix → assumed autograph
    """
    prefix = (artist_prefix or "").strip()
    attrib = MET_PREFIX_MAP.get(prefix, "style" if prefix else "autograph")
    confidence = "high" if prefix else "low"
    return attrib, confidence


def met_artist_group(artist_name, artist_prefix, query_group=None):
    """Determine artist_group for a Met object."""
    if query_group is not None:
        return query_group
    name_lower = (artist_name or "").lower()
    if "rembrandt" in name_lower:
        attrib, _conf = met_classify_attribution(artist_prefix)
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
    """Convert Wikimedia Commons file URL to an upload.wikimedia.org thumbnail URL.

    Input:  http://commons.wikimedia.org/wiki/Special:FilePath/Foo.jpg
    Output: https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Foo.jpg/2000px-Foo.jpg
    """
    import hashlib
    import urllib.parse
    if "Special:FilePath/" in commons_url:
        filename = commons_url.split("Special:FilePath/")[-1]
    else:
        filename = commons_url.rsplit("/", 1)[-1]
    filename = urllib.parse.unquote(filename).replace(" ", "_")
    md5 = hashlib.md5(filename.encode()).hexdigest()
    encoded = urllib.parse.quote(filename)
    thumb_name = f"{width}px-{encoded}"
    # Non-JPEG formats: Commons generates .jpg thumbnails
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("tiff", "tif", "png", "svg"):
        thumb_name += ".jpg"
    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{md5[0]}/{md5[:2]}/{encoded}/{thumb_name}"


def wd_fetch_artist_circle(qid, artist_name="artist"):
    """Fetch circle/workshop/follower/manner/school paintings for any artist from Wikidata."""
    query = f"""SELECT ?painting ?paintingLabel ?image ?qualLabel WHERE {{
  ?painting p:P170 ?stmt .
  ?painting wdt:P31 wd:Q3305213 .
  ?painting wdt:P18 ?image .
  {{
    ?stmt pq:P1774 wd:{qid} . BIND("workshop" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1775 wd:{qid} . BIND("follower" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1776 wd:{qid} . BIND("circle" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1777 wd:{qid} . BIND("manner" AS ?qualLabel)
  }} UNION {{
    ?stmt pq:P1780 wd:{qid} . BIND("school" AS ?qualLabel)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}"""
    bindings = wd_sparql(query)
    seen = set()
    results = []
    attrib_map = {"workshop": "workshop", "follower": "style",
                  "circle": "circle", "manner": "style", "school": "school"}
    for b in bindings:
        painting_qid = b["painting"]["value"].split("/")[-1]
        if painting_qid in seen:
            continue
        seen.add(painting_qid)
        qual = b.get("qualLabel", {}).get("value", "circle")
        results.append({
            "qid": painting_qid,
            "title": b.get("paintingLabel", {}).get("value", ""),
            "image": b.get("image", {}).get("value", ""),
            "attribution": attrib_map.get(qual, "circle"),
            "wd_qualifier": qual,
        })
    return results


def wd_fetch_artist_autographs(qid, artist_name="artist"):
    """Fetch autograph paintings for any artist from Wikidata (direct P170, no qualifier)."""
    query = f"""SELECT ?painting ?paintingLabel ?image WHERE {{
  ?painting wdt:P170 wd:{qid} .
  ?painting wdt:P31 wd:Q3305213 .
  ?painting wdt:P18 ?image .
  FILTER NOT EXISTS {{
    ?painting p:P170 ?stmt .
    {{ ?stmt pq:P1774 wd:{qid} }} UNION {{ ?stmt pq:P1775 wd:{qid} }}
    UNION {{ ?stmt pq:P1776 wd:{qid} }} UNION {{ ?stmt pq:P1777 wd:{qid} }}
    UNION {{ ?stmt pq:P1780 wd:{qid} }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}"""
    bindings = wd_sparql(query)
    seen = set()
    results = []
    for b in bindings:
        painting_qid = b["painting"]["value"].split("/")[-1]
        if painting_qid in seen:
            continue
        seen.add(painting_qid)
        results.append({
            "qid": painting_qid,
            "title": b.get("paintingLabel", {}).get("value", ""),
            "image": b.get("image", {}).get("value", ""),
        })
    return results


def wd_fetch_circle_paintings():
    """Fetch circle/workshop Rembrandt paintings. Backward-compat wrapper."""
    return wd_fetch_artist_circle(WD_REMBRANDT, "Rembrandt")


def wd_fetch_autograph_paintings():
    """Fetch autograph Rembrandt paintings. Backward-compat wrapper."""
    return wd_fetch_artist_autographs(WD_REMBRANDT, "Rembrandt")


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
        attrib, confidence = classify_attribution(creator)
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
            "label_confidence": confidence,
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
        attrib, confidence = classify_attribution(f"{qualifier} {creator_name}" if qualifier else creator_name)
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
            "label_confidence": confidence,
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
        attrib, confidence = classify_attribution(artist)
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
            "label_confidence": confidence,
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
            label_confidence = "low"  # forced control group, no explicit prefix
        else:
            artist_group, attribution, label_confidence = classify_rembrandt_group(creator)
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
            "label_confidence": label_confidence,
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
        attribution, label_confidence = met_classify_attribution(artist_prefix)

        artist_group = met_artist_group(artist_name, artist_prefix, rec["force_group"])
        if artist_group is None:
            cache_file.write_text(json.dumps({"_skip": True}))
            met_skipped += 1
            continue

        # Force attribution for control groups
        if rec["force_group"] is not None:
            attribution = "autograph"
            label_confidence = "low"  # forced control group

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
            "label_confidence": label_confidence,
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
            "label_confidence": "high",  # Wikidata qualifier match (P1774-P1780)
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
            "label_confidence": "high",  # Wikidata direct P170, no qualifier
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
            "label_confidence": "high",  # Wikidata direct P170
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

    fieldnames = ["obj_id", "source", "title", "creator", "date", "image_url", "artist_group", "attribution", "label_confidence"]
    with open(INVENTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
# Stage 1T: Transfer Metadata (multi-artist)
# ---------------------------------------------------------------------------

def stage1_transfer_metadata():
    """Fetch painting inventory for all transfer artists from Wikidata."""
    if INVENTORY_TRANSFER_CSV.exists():
        rows = []
        with open(INVENTORY_TRANSFER_CSV) as f:
            rows = list(csv.DictReader(f))
        print(f"[Stage 1T] Loaded cached transfer inventory: {len(rows)} paintings")
        return rows

    CACHE_META.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_qids = set()

    for artist_name, (qid, label) in WD_TRANSFER_ARTISTS.items():
        print(f"\n[Stage 1T] Fetching {artist_name} (QID={qid})...")

        # Autograph paintings
        auto = wd_fetch_artist_autographs(qid, artist_name)
        auto_added = 0
        for p in auto:
            if p["qid"] in seen_qids:
                continue
            seen_qids.add(p["qid"])
            image_url = wd_image_url(p["image"])
            row = {
                "obj_id": f"wd_{p['qid']}",
                "source": "wikidata",
                "title": p["title"],
                "creator": artist_name,
                "date": "",
                "image_url": image_url,
                "artist_group": f"{label}_autograph",
                "attribution": "autograph",
                "artist": label,
                "label_confidence": "high",  # Wikidata direct P170
            }
            rows.append(row)
            auto_added += 1
        print(f"  Autograph: {auto_added} paintings")

        # Circle/workshop paintings
        circle = wd_fetch_artist_circle(qid, artist_name)
        circle_added = 0
        for p in circle:
            if p["qid"] in seen_qids:
                continue
            seen_qids.add(p["qid"])
            image_url = wd_image_url(p["image"])
            row = {
                "obj_id": f"wd_{p['qid']}",
                "source": "wikidata",
                "title": p["title"],
                "creator": f"{p['wd_qualifier']} of {artist_name}",
                "date": "",
                "image_url": image_url,
                "artist_group": f"{label}_circle",
                "attribution": p["attribution"],
                "artist": label,
                "label_confidence": "high",  # Wikidata qualifier match
            }
            rows.append(row)
            circle_added += 1
        print(f"  Circle/workshop: {circle_added} paintings")

    # Save
    fieldnames = ["obj_id", "source", "title", "creator", "date", "image_url",
                   "artist_group", "attribution", "artist", "label_confidence"]
    with open(INVENTORY_TRANSFER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Stage 1T] Saved {len(rows)} paintings to {INVENTORY_TRANSFER_CSV}")
    _print_transfer_summary(rows)
    return rows


def _print_transfer_summary(rows):
    """Print transfer inventory breakdown by artist."""
    from collections import Counter
    artists = Counter(r["artist"] for r in rows)
    print("\n  By artist:")
    for artist in sorted(artists):
        artist_rows = [r for r in rows if r["artist"] == artist]
        n_auto = sum(1 for r in artist_rows if r["artist_group"].endswith("_autograph"))
        n_circle = sum(1 for r in artist_rows if r["artist_group"].endswith("_circle"))
        print(f"    {artist:12s}: {n_auto:5d} autograph, {n_circle:4d} circle/workshop = {len(artist_rows):5d} total")
    total_auto = sum(1 for r in rows if r["artist_group"].endswith("_autograph"))
    total_circle = sum(1 for r in rows if r["artist_group"].endswith("_circle"))
    print(f"    {'TOTAL':12s}: {total_auto:5d} autograph, {total_circle:4d} circle/workshop = {len(rows):5d} total")


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
                # v1 behavior: resize ALL images to IMG_MAX_PX for uniform preprocessing
                if max(w, h) > IMG_MAX_PX:
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
# Stage 2b: Perceptual dedup
# ---------------------------------------------------------------------------

def compute_phashes(rows, cache_dir=None):
    """Compute perceptual hashes for all downloaded images. Returns {obj_id: phash}."""
    import imagehash
    if cache_dir is None:
        cache_dir = CACHE_IMG
    phashes = {}
    for row in rows:
        img_path = cache_dir / f"{row['obj_id']}.jpg"
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path)
            phashes[row["obj_id"]] = imagehash.average_hash(img, hash_size=16)
        except Exception:
            continue
    return phashes


def dedup_by_phash(rows, phashes, threshold=10):
    """Remove near-duplicate images based on perceptual hash hamming distance.

    Keeps the higher-resolution version when a pair is found.
    Returns (deduped_rows, n_removed).
    """
    from itertools import combinations

    # Build list of (obj_id, phash) pairs
    items = [(r["obj_id"], phashes[r["obj_id"]]) for r in rows if r["obj_id"] in phashes]

    # Find near-duplicate pairs
    to_remove = set()
    for (id_a, hash_a), (id_b, hash_b) in combinations(items, 2):
        if id_a in to_remove or id_b in to_remove:
            continue
        if hash_a - hash_b < threshold:
            # Keep higher-resolution: check file size as proxy
            path_a = CACHE_IMG / f"{id_a}.jpg"
            path_b = CACHE_IMG / f"{id_b}.jpg"
            size_a = path_a.stat().st_size if path_a.exists() else 0
            size_b = path_b.stat().st_size if path_b.exists() else 0
            to_remove.add(id_b if size_a >= size_b else id_a)

    deduped = [r for r in rows if r["obj_id"] not in to_remove]
    return deduped, len(to_remove)


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


def _embed_single_image(img, model, device, batch_size, dino_transform,
                        entropy=False, return_raw=False):
    """Embed one PIL Image -> (cls_vec, patch_vec) numpy arrays.

    Returns None if image has zero tiles.
    If return_raw=True, returns (all_cls, all_patch) raw tile arrays instead of aggregated.
    """
    import torch

    w, h = img.size
    n_tiles = (w // TILE_SIZE) * (h // TILE_SIZE)
    if n_tiles == 0:
        return None

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
                tile_var = patch_tokens.var(dim=1).mean(dim=1)
                all_tile_var.append(tile_var.cpu().numpy())

    all_cls = np.concatenate(all_cls, axis=0)
    all_patch = np.concatenate(all_patch, axis=0)

    if return_raw:
        return all_cls, all_patch

    if entropy:
        tile_vars = np.concatenate(all_tile_var)
        var_std = tile_vars.std()
        if var_std > 1e-6:
            z = (tile_vars - tile_vars.mean()) / var_std
            weights = np.exp(z)
            weights /= weights.sum()
        else:
            weights = np.ones(len(tile_vars)) / len(tile_vars)
        cls_vec = (all_cls * weights[:, None]).sum(axis=0)
        patch_vec = (all_patch * weights[:, None]).sum(axis=0)
    else:
        cls_vec = all_cls.mean(axis=0)
        patch_vec = all_patch.mean(axis=0)

    return cls_vec, patch_vec


def stage3_embed(rows, hires=False, model_name="vitb14", entropy=False, emb_file_override=None):
    """Tile images and embed with DINOv2.

    v1 (hires=False): mean-only aggregation. Cache: embeddings.npz
    v2 (hires=True):  mean+std aggregation. Cache: embeddings_v2.npz
    vitl14:           ViT-L model (1024d vs 768d). Cache: embeddings_vitl.npz
    entropy:          entropy-weighted tile aggregation. Cache: embeddings_entropy[_vitl].npz
    emb_file_override: override cache file path (for transfer corpus)
    """
    CACHE_EMB.mkdir(parents=True, exist_ok=True)
    if emb_file_override:
        cache_file = emb_file_override
    elif entropy:
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
    artists_list = []
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

            if hires:
                result = _embed_single_image(img, model, device, batch_size,
                                             dino_transform, return_raw=True)
            else:
                result = _embed_single_image(img, model, device, batch_size,
                                             dino_transform, entropy=entropy)
            if result is None:
                skipped += 1
                del img
                continue

            if hires:
                all_cls, all_patch = result
                cls_mean_list.append(all_cls.mean(axis=0))
                patch_mean_list.append(all_patch.mean(axis=0))
                cls_std_list.append(all_cls.std(axis=0))
                patch_std_list.append(all_patch.std(axis=0))
                if len(painting_ids) % 10 == 0 or len(painting_ids) == 0:
                    w, h = img.size
                    n_tiles = (w // TILE_SIZE) * (h // TILE_SIZE)
                    print(f"      {row['obj_id']}: {img.size[0]}x{img.size[1]}, {n_tiles} tiles")
            else:
                cls_vec, patch_vec = result
                cls_mean_list.append(cls_vec)
                patch_mean_list.append(patch_vec)

            painting_ids.append(row["obj_id"])
            groups_list.append(row["artist_group"])
            attribs_list.append(row["attribution"])
            artists_list.append(row.get("artist", ""))

            del img

        print(f"    Chunk {ci+1}/{len(chunk_starts)} done ({len(painting_ids)} embedded, {skipped} skipped)")

    painting_ids = np.array(painting_ids)
    artist_groups = np.array(groups_list)
    attributions = np.array(attribs_list)
    artists = np.array(artists_list)

    if hires:
        cls_mean = np.array(cls_mean_list)
        cls_std = np.array(cls_std_list)
        patch_mean = np.array(patch_mean_list)
        patch_std = np.array(patch_std_list)
        np.savez(cache_file,
                 painting_ids=painting_ids,
                 cls_mean=cls_mean, cls_std=cls_std,
                 patch_mean=patch_mean, patch_std=patch_std,
                 artist_groups=artist_groups, attributions=attributions,
                 artists=artists)
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
                 artist_groups=artist_groups, attributions=attributions,
                 artists=artists)
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
        new_label = "v1-D Entropy-weighted"
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

def stage4b_probe(full_embeddings, artist_groups, painting_ids, rows,
                  strict_labels=False, legacy_cv=False, holdout_source=None):
    """PCA + multi-classifier CV + permutation test on autograph vs circle."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    # Filter to autograph + circle only
    mask = np.array([(g in ("rembrandt_autograph", "rembrandt_circle")) for g in artist_groups])

    # --strict-labels: exclude low-confidence labels
    if strict_labels:
        confidence_by_id = {r["obj_id"]: r.get("label_confidence", "low") for r in rows}
        conf_mask = np.array([confidence_by_id.get(pid, "low") != "low" for pid in painting_ids])
        n_excluded = int(mask.sum()) - int((mask & conf_mask).sum())
        mask = mask & conf_mask
        print(f"  [strict-labels] Excluded {n_excluded} low-confidence rows")

    # --holdout-source: hold out one institution for domain-shift evaluation
    if holdout_source:
        source_by_id = {r["obj_id"]: r.get("source", "") for r in rows}
        holdout_mask = np.array([source_by_id.get(pid, "") == holdout_source for pid in painting_ids])
        holdout_mask = holdout_mask & mask
        n_holdout = int(holdout_mask.sum())
        print(f"  [holdout-source] Holding out {n_holdout} rows from '{holdout_source}'")

    X = full_embeddings[mask]
    y = np.array([1 if g == "rembrandt_autograph" else 0 for g in artist_groups[mask]])
    _ids = painting_ids[mask]
    n = len(y)
    n_auto = int(y.sum())
    n_circle = n - n_auto
    print(f"\n[Stage 4b] Probe: {n_auto} autograph + {n_circle} circle = {n} paintings")
    print(f"  Features: {X.shape[1]}d → PCA reduction\n")

    pca_dims = [d for d in [10, 20] if d < n]

    # Classifier configs: (name, make_clf, param_name, param_values)
    classifiers = [
        ("Logistic", lambda v: LogisticRegression(C=v, solver="lbfgs", max_iter=1000, class_weight="balanced"),
         "C", [0.001, 0.01, 0.1, 1.0, 10.0]),
        ("SVM RBF", lambda v: SVC(C=v, kernel="rbf", gamma="scale", class_weight="balanced"),
         "C", [0.001, 0.01, 0.1, 1.0, 10.0]),
        ("MLP (32)", lambda v: MLPClassifier(hidden_layer_sizes=(32,), alpha=v, max_iter=1000, random_state=42),
         "alpha", [0.01, 0.1, 1.0, 10.0]),
    ]

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import balanced_accuracy_score

    if legacy_cv:
        # --- Legacy (non-nested) CV: grid search on same folds used for reporting ---
        use_loo = n <= 100
        cv_label = "LOO (legacy)" if use_loo else "10-fold (legacy)"
        print("  [legacy-cv] Using non-nested CV for comparison")

        def cv_accuracy(X_data, y_data, make_clf, param_val):
            clf = make_clf(param_val)
            if use_loo:
                correct = 0
                for i in range(len(y_data)):
                    X_train = np.delete(X_data, i, axis=0)
                    y_train = np.delete(y_data, i)
                    X_test = X_data[i:i+1]
                    c = make_clf(param_val)
                    c.fit(X_train, y_train)
                    if c.predict(X_test)[0] == y_data[i]:
                        correct += 1
                return correct / len(y_data)
            else:
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_data, y_data, cv=skf, scoring="balanced_accuracy")
                return scores.mean()

        all_results = []
        overall_best_acc = 0
        overall_best = None
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
                    acc = cv_accuracy(X_pca, y, make_clf, v)
                    row_str += f" {acc:<10.3f}"
                    if acc > best_acc:
                        best_acc, best_dims, best_param = acc, dims, v
                print(row_str)
            print(f"  Best: PCA={best_dims}, {param_name}={best_param} → {best_acc:.3f} ({cv_label})\n")
            all_results.append({
                "classifier": clf_name, "best_pca_dims": best_dims,
                "best_param_name": param_name, "best_param_value": best_param,
                "cv_accuracy": round(best_acc, 4), "cv_method": cv_label,
            })
            if best_acc > overall_best_acc:
                overall_best_acc = best_acc
                overall_best = (clf_name, best_dims, param_name, best_param, make_clf)

        best_name, best_dims, best_pname, best_pval, best_make_clf = overall_best

        # Permutation test (same non-nested approach)
        n_perms = 1000
        print(f"\n  Permutation test on {best_name} ({n_perms} shuffles)...")
        pca = PCA(n_components=best_dims, random_state=42)
        X_best = pca.fit_transform(X)
        null_accs = np.zeros(n_perms)
        rng = np.random.RandomState(42)
        for p in range(n_perms):
            y_shuf = rng.permutation(y)
            null_accs[p] = cv_accuracy(X_best, y_shuf, best_make_clf, best_pval)
            if (p + 1) % 100 == 0:
                print(f"    {p+1}/{n_perms} done")
    else:
        # --- Nested CV: outer 10-fold for reporting, inner 5-fold for model selection ---
        cv_label = "nested 10×5-fold"
        print("  Using nested CV (outer=10, inner=5)")

        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Build config grid: (clf_name, make_clf, param_name, param_val, dims)
        configs = []
        for clf_name, make_clf, param_name, param_values in classifiers:
            for v in param_values:
                for dims in pca_dims:
                    configs.append((clf_name, make_clf, param_name, v, dims))

        outer_scores = []
        outer_configs = []

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]

            # Inner CV: select best config
            best_inner_acc = 0
            best_config = None
            for clf_name, make_clf, param_name, param_val, dims in configs:
                pca = PCA(n_components=dims, random_state=42)
                X_inner = pca.fit_transform(X_train_outer)
                inner_scores_list = []
                for inner_train, inner_val in inner_cv.split(X_inner, y_train_outer):
                    clf = make_clf(param_val)
                    clf.fit(X_inner[inner_train], y_train_outer[inner_train])
                    preds = clf.predict(X_inner[inner_val])
                    inner_scores_list.append(balanced_accuracy_score(y_train_outer[inner_val], preds))
                mean_inner = np.mean(inner_scores_list)
                if mean_inner > best_inner_acc:
                    best_inner_acc = mean_inner
                    best_config = (clf_name, make_clf, param_name, param_val, dims)

            # Evaluate best config on outer test fold
            clf_name, make_clf, param_name, param_val, dims = best_config
            pca = PCA(n_components=dims, random_state=42)
            X_train_pca = pca.fit_transform(X_train_outer)
            X_test_pca = pca.transform(X_test_outer)
            clf = make_clf(param_val)
            clf.fit(X_train_pca, y_train_outer)
            preds = clf.predict(X_test_pca)
            fold_acc = balanced_accuracy_score(y_test_outer, preds)
            outer_scores.append(fold_acc)
            outer_configs.append(best_config)

            print(f"    Fold {fold_i+1}/10: {fold_acc:.3f} ({clf_name}, PCA={dims}, {param_name}={param_val})")

        overall_best_acc = np.mean(outer_scores)
        overall_std = np.std(outer_scores)

        # Most frequently selected config across folds
        from collections import Counter
        config_keys = [(c[0], c[4], c[2], c[3]) for c in outer_configs]
        most_common = Counter(config_keys).most_common(1)[0][0]
        best_name, best_dims, best_pname, best_pval = most_common
        best_make_clf = next(mc for cn, mc, pn, pv, d in configs
                            if cn == best_name and d == best_dims and pn == best_pname and pv == best_pval)

        all_results = [{
            "classifier": best_name, "best_pca_dims": best_dims,
            "best_param_name": best_pname, "best_param_value": best_pval,
            "cv_accuracy": round(overall_best_acc, 4), "cv_std": round(overall_std, 4),
            "cv_method": cv_label,
        }]

        print(f"\n  Nested CV mean: {overall_best_acc:.3f} ± {overall_std:.3f}")
        print(f"  Most selected: {best_name} (PCA={best_dims}, {best_pname}={best_pval})")

        # Permutation test: full nested CV per permutation (correct but expensive)
        n_perms = 200  # Reduced from 1000 — full nested CV per permutation
        print(f"\n  Permutation test ({n_perms} shuffles, full nested CV each)...")
        null_accs = np.zeros(n_perms)
        rng = np.random.RandomState(42)
        for p in range(n_perms):
            y_shuf = rng.permutation(y)
            perm_scores = []
            for train_idx, test_idx in outer_cv.split(X, y_shuf):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y_shuf[train_idx], y_shuf[test_idx]
                best_inner = 0
                best_cfg = configs[0]
                for cn, mc, pn, pv, dims in configs:
                    pca = PCA(n_components=dims, random_state=42)
                    X_inn = pca.fit_transform(X_tr)
                    inn_scores = []
                    for i_tr, i_val in inner_cv.split(X_inn, y_tr):
                        c = mc(pv)
                        c.fit(X_inn[i_tr], y_tr[i_tr])
                        inn_scores.append(balanced_accuracy_score(y_tr[i_val], c.predict(X_inn[i_val])))
                    if np.mean(inn_scores) > best_inner:
                        best_inner = np.mean(inn_scores)
                        best_cfg = (cn, mc, pn, pv, dims)
                _, mc, _, pv, dims = best_cfg
                pca = PCA(n_components=dims, random_state=42)
                X_tr_pca = pca.fit_transform(X_tr)
                X_te_pca = pca.transform(X_te)
                c = mc(pv)
                c.fit(X_tr_pca, y_tr)
                perm_scores.append(balanced_accuracy_score(y_te, c.predict(X_te_pca)))
            null_accs[p] = np.mean(perm_scores)
            if (p + 1) % 50 == 0:
                print(f"    {p+1}/{n_perms} done")

    p_value = (np.sum(null_accs >= overall_best_acc) + 1) / (n_perms + 1)
    null_mean = null_accs.mean()
    null_std = null_accs.std()

    print(f"\n{'='*60}")
    print(f"  PROBE RESULTS (balanced, {cv_label})")
    print(f"{'='*60}")
    print(f"  N autograph:          {n_auto}")
    print(f"  N circle:             {n_circle}")
    print(f"  Best classifier:      {best_name}")
    print(f"  Best PCA dims:        {best_dims}")
    print(f"  Best {best_pname}:{'':>{13-len(best_pname)}}{best_pval}")
    print(f"  {cv_label} accuracy:  {overall_best_acc:.3f}")
    print(f"  Permutation p-value:  {p_value:.4f} ({n_perms} perms)")
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
        "cv_accuracy": round(overall_best_acc, 4),
        "cv_method": cv_label,
        "permutation_p_value": round(p_value, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "n_permutations": n_perms,
        "all_classifiers": all_results,
    }

    # --- Institution holdout evaluation (Fix 4) ---
    if holdout_source:
        from sklearn.metrics import balanced_accuracy_score
        source_by_id = {r["obj_id"]: r.get("source", "") for r in rows}
        holdout_idx = np.array([source_by_id.get(pid, "") == holdout_source for pid in _ids])
        if holdout_idx.sum() > 0 and (~holdout_idx).sum() > 0:
            pca = PCA(n_components=best_dims, random_state=42)
            X_train_h = pca.fit_transform(X[~holdout_idx])
            X_test_h = pca.transform(X[holdout_idx])
            clf = best_make_clf(best_pval)
            clf.fit(X_train_h, y[~holdout_idx])
            preds = clf.predict(X_test_h)
            holdout_acc = balanced_accuracy_score(y[holdout_idx], preds)
            n_holdout_auto = int(y[holdout_idx].sum())
            n_holdout_circle = int((y[holdout_idx] == 0).sum())
            print(f"\n  --- Institution holdout: {holdout_source} ---")
            print(f"  Train: {int((~holdout_idx).sum())} paintings (other sources)")
            print(f"  Test:  {int(holdout_idx.sum())} paintings ({n_holdout_auto} auto + {n_holdout_circle} circle)")
            print(f"  Balanced accuracy: {holdout_acc:.3f}")
            results["holdout_source"] = holdout_source
            results["holdout_bal_acc"] = round(holdout_acc, 4)
            results["holdout_n"] = int(holdout_idx.sum())
        else:
            print(f"\n  [holdout] No {holdout_source} rows in filtered data, skipping")

    with open(RESULTS_PROBE_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_PROBE_JSON}")


# ---------------------------------------------------------------------------
# Confounder Audit — source classifier + within-source probe
# ---------------------------------------------------------------------------

def confounder_audit():
    """Test whether embeddings encode source (museum) rather than brushwork style.

    Three tests:
    1. Source classifier — can we predict which museum from embeddings?
    2. Wikidata-only probe — autograph vs circle within largest single source
    3. Source-stratified probe — CV folds preserve source proportions
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    print("\n" + "=" * 60)
    print("  CONFOUNDER AUDIT")
    print("=" * 60)

    # Load data
    if not INVENTORY_CSV.exists():
        print("ERROR: No cached inventory. Run full pipeline first.")
        sys.exit(1)
    if not EMBEDDINGS_ENTROPY_NPZ.exists():
        print("ERROR: No cached entropy embeddings. Run --entropy pipeline first.")
        sys.exit(1)

    with open(INVENTORY_CSV) as f:
        rows = list(csv.DictReader(f))
    source_by_id = {r["obj_id"]: r["source"] for r in rows}

    data = np.load(EMBEDDINGS_ENTROPY_NPZ, allow_pickle=True)
    painting_ids = data["painting_ids"]
    artist_groups = data["artist_groups"]
    cls_emb = data["cls_embeddings"]
    patch_emb = data["patch_embeddings"]
    X_all = np.concatenate([cls_emb, patch_emb], axis=1)

    # Filter to autograph + circle
    ac_mask = np.array([(g in ("rembrandt_autograph", "rembrandt_circle"))
                        for g in artist_groups])
    X = X_all[ac_mask]
    y_attr = np.array([1 if g == "rembrandt_autograph" else 0
                       for g in artist_groups[ac_mask]])
    ids = painting_ids[ac_mask]
    sources = np.array([source_by_id.get(pid, "unknown") for pid in ids])

    n = len(y_attr)
    print(f"\n  Dataset: {n} paintings (autograph+circle)")
    from collections import Counter
    src_counts = Counter(sources)
    for s, c in src_counts.most_common():
        print(f"    {s}: {c}")

    results = {"n_total": n, "source_counts": dict(src_counts)}

    # --- Test 1: Source classifier ---
    print("\n  --- Test 1: Source Classifier ---")
    unique_sources = sorted(set(sources))
    y_source = np.array([unique_sources.index(s) for s in sources])

    pca = PCA(n_components=min(20, n - 1), random_state=42)
    X_pca = pca.fit_transform(X)

    clf_src = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 class_weight="balanced")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    src_scores = cross_val_score(clf_src, X_pca, y_source, cv=skf,
                                 scoring="balanced_accuracy")
    src_acc = src_scores.mean()
    src_std = src_scores.std()
    chance = 1.0 / len(unique_sources)
    print(f"  Sources: {unique_sources}")
    print(f"  Balanced accuracy: {src_acc:.3f} ± {src_std:.3f}")
    print(f"  Chance level: {chance:.3f}")
    results["test1_source_acc"] = round(src_acc, 4)
    results["test1_source_std"] = round(src_std, 4)
    results["test1_chance"] = round(chance, 4)

    # --- Test 2: Wikidata-only probe ---
    print("\n  --- Test 2: Wikidata-Only Probe ---")
    wd_mask = sources == "wikidata"
    X_wd = X[wd_mask]
    y_wd = y_attr[wd_mask]
    n_wd = len(y_wd)
    n_wd_auto = int(y_wd.sum())
    n_wd_circle = n_wd - n_wd_auto
    print(f"  N={n_wd} ({n_wd_auto} autograph + {n_wd_circle} circle)")

    if n_wd < 20:
        print("  SKIP: too few wikidata paintings for meaningful CV")
        results["test2_wikidata_acc"] = None
        results["test2_wikidata_n"] = n_wd
    else:
        pca_wd = PCA(n_components=min(20, n_wd - 1), random_state=42)
        X_wd_pca = pca_wd.fit_transform(X_wd)

        # Run same classifiers as main probe
        best_wd_acc = 0
        best_wd_name = ""
        for name, make_clf in [
            ("Logistic", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                     max_iter=1000, class_weight="balanced")),
            ("SVM RBF", lambda: SVC(C=1.0, kernel="rbf", gamma="scale",
                                     class_weight="balanced")),
        ]:
            skf_wd = StratifiedKFold(n_splits=min(10, min(n_wd_auto, n_wd_circle)),
                                     shuffle=True, random_state=42)
            scores = cross_val_score(make_clf(), X_wd_pca, y_wd, cv=skf_wd,
                                     scoring="balanced_accuracy")
            acc = scores.mean()
            print(f"  {name}: {acc:.3f} ± {scores.std():.3f}")
            if acc > best_wd_acc:
                best_wd_acc = acc
                best_wd_name = name

        results["test2_wikidata_acc"] = round(best_wd_acc, 4)
        results["test2_wikidata_clf"] = best_wd_name
        results["test2_wikidata_n"] = n_wd

    # --- Test 3: Source-stratified probe ---
    print("\n  --- Test 3: Source-Stratified Probe ---")
    # Create composite stratification label: source × attribution
    strat_labels = np.array([f"{s}_{a}" for s, a in zip(sources, y_attr)])

    # Check all strat groups have >= 2 members (needed for stratified CV)
    strat_counts = Counter(strat_labels)
    min_group = min(strat_counts.values())
    if min_group < 2:
        print("  Some strat groups have <2 members, falling back to source-aware grouping")
        # Fall back: just use source as group, stratify on attribution
        from sklearn.model_selection import GroupKFold
        groups = y_source
        pca_s = PCA(n_components=min(20, n - 1), random_state=42)
        X_s_pca = pca_s.fit_transform(X)
        n_groups = len(unique_sources)
        gkf = GroupKFold(n_splits=min(n_groups, 3))
        best_strat_acc = 0
        best_strat_name = ""
        for name, make_clf in [
            ("Logistic", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                     max_iter=1000, class_weight="balanced")),
            ("SVM RBF", lambda: SVC(C=1.0, kernel="rbf", gamma="scale",
                                     class_weight="balanced")),
        ]:
            scores = cross_val_score(make_clf(), X_s_pca, y_attr, cv=gkf,
                                     groups=groups, scoring="balanced_accuracy")
            acc = scores.mean()
            print(f"  {name} (GroupKFold by source): {acc:.3f} ± {scores.std():.3f}")
            if acc > best_strat_acc:
                best_strat_acc = acc
                best_strat_name = name
    else:
        pca_s = PCA(n_components=min(20, n - 1), random_state=42)
        X_s_pca = pca_s.fit_transform(X)
        skf_s = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        best_strat_acc = 0
        best_strat_name = ""
        for name, make_clf in [
            ("Logistic", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                     max_iter=1000, class_weight="balanced")),
            ("SVM RBF", lambda: SVC(C=1.0, kernel="rbf", gamma="scale",
                                     class_weight="balanced")),
        ]:
            scores = cross_val_score(make_clf(), X_s_pca, y_attr, cv=skf_s,
                                     scoring="balanced_accuracy")
            acc = scores.mean()
            print(f"  {name} (stratified): {acc:.3f} ± {scores.std():.3f}")
            if acc > best_strat_acc:
                best_strat_acc = acc
                best_strat_name = name

    results["test3_stratified_acc"] = round(best_strat_acc, 4)
    results["test3_stratified_clf"] = best_strat_name

    # --- Verdict ---
    src_acc_val = results["test1_source_acc"]
    wd_acc_val = results.get("test2_wikidata_acc")

    if src_acc_val < 0.50:
        verdict = "CLEAN"
    elif src_acc_val >= 0.70 and (wd_acc_val is None or wd_acc_val < 0.55):
        verdict = "DIRTY"
    elif src_acc_val >= 0.70 and wd_acc_val is not None and wd_acc_val >= 0.58:
        verdict = "MIXED"
    else:
        verdict = "MIXED"

    results["verdict"] = verdict

    print(f"\n{'='*60}")
    print("  CONFOUNDER AUDIT RESULTS")
    print(f"{'='*60}")
    print(f"  Test 1 — Source classifier:    {src_acc_val:.3f} (chance={results['test1_chance']:.3f})")
    if wd_acc_val is not None:
        print(f"  Test 2 — Wikidata-only probe:  {wd_acc_val:.3f} (N={results['test2_wikidata_n']})")
    else:
        print(f"  Test 2 — Wikidata-only probe:  SKIPPED (N={results.get('test2_wikidata_n', 0)})")
    print(f"  Test 3 — Source-stratified:    {results['test3_stratified_acc']:.3f}")
    print(f"  Verdict:                       {verdict}")
    print(f"{'='*60}")

    with open(RESULTS_CONFOUNDER_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_CONFOUNDER_JSON}")


# ---------------------------------------------------------------------------
# Robustness test: augmentation flip rate
# ---------------------------------------------------------------------------

def _aug_jpeg_q30(img):
    """JPEG compression at quality 30."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=30)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _aug_gaussian_blur(img):
    """Gaussian blur with sigma=2."""
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def _aug_brightness_120(img):
    """Brightness increase by 20%."""
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(img).enhance(1.2)


def _aug_center_crop_10(img):
    """10% center crop, resized back to original dimensions."""
    w, h = img.size
    crop_x = int(w * 0.05)
    crop_y = int(h * 0.05)
    cropped = img.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    return cropped.resize((w, h), Image.LANCZOS)


def _aug_horizontal_flip(img):
    """Horizontal flip (mirror)."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


AUGMENTATIONS = [
    ("JPEG q=30", _aug_jpeg_q30),
    ("Gaussian blur s=2", _aug_gaussian_blur),
    ("Brightness +20%", _aug_brightness_120),
    ("Center crop 10%", _aug_center_crop_10),
    ("Horizontal flip", _aug_horizontal_flip),
]

FLIP_RATE_THRESHOLD = 0.05  # 5%


def robustness_test():
    """Test prediction stability under benign image augmentations.

    Train SVM RBF on clean entropy embeddings (autograph+circle),
    re-embed under 5 augmentations, measure prediction flip rate.
    Pass criterion: <5% flip rate per augmentation.
    """
    import torch
    from torchvision import transforms
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    print("\n" + "=" * 60)
    print("  ROBUSTNESS TEST")
    print("=" * 60)

    # --- Load clean data ---
    if not EMBEDDINGS_ENTROPY_NPZ.exists():
        print("ERROR: No cached entropy embeddings. Run --entropy pipeline first.")
        sys.exit(1)

    data = np.load(EMBEDDINGS_ENTROPY_NPZ, allow_pickle=True)
    painting_ids = data["painting_ids"]
    artist_groups = data["artist_groups"]
    cls_emb = data["cls_embeddings"]
    patch_emb = data["patch_embeddings"]

    # Filter to autograph + circle
    mask = np.isin(artist_groups, ["rembrandt_autograph", "rembrandt_circle"])
    painting_ids = painting_ids[mask]
    artist_groups = artist_groups[mask]
    full_emb = np.concatenate([cls_emb[mask], patch_emb[mask]], axis=1)
    labels = (artist_groups == "rembrandt_autograph").astype(int)  # 1=autograph, 0=circle

    N = len(painting_ids)
    print(f"  Paintings: {N} (autograph={labels.sum()}, circle={N - labels.sum()})")

    # --- Train fixed classifier on clean data ---
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(full_emb)
    clf = SVC(kernel="rbf", C=1.0, random_state=42)
    clf.fit(X_pca, labels)
    clean_preds = clf.predict(X_pca)
    print(f"  Clean train accuracy: {(clean_preds == labels).mean():.3f}")

    # --- Load DINOv2 ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    dino_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("  Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device)
    model.eval()
    print("  Model loaded")

    # --- Run augmentations ---
    results = {}
    img_dir = CACHE_IMG

    for aug_name, aug_fn in AUGMENTATIONS:
        print(f"\n  Augmentation: {aug_name}")
        aug_cls_list = []
        aug_patch_list = []
        aug_ids = []
        skipped = 0

        for i, pid in enumerate(painting_ids):
            img_path = img_dir / f"{pid}.jpg"
            if not img_path.exists():
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            aug_img = aug_fn(img)
            result = _embed_single_image(aug_img, model, device, BATCH_SIZE_VITB,
                                         dino_transform, entropy=True)
            del img, aug_img

            if result is None:
                skipped += 1
                continue

            cls_vec, patch_vec = result
            aug_cls_list.append(cls_vec)
            aug_patch_list.append(patch_vec)
            aug_ids.append(pid)

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{N} embedded")

        if not aug_cls_list:
            print("    SKIP — no images embedded")
            results[aug_name] = {"flip_rate": None, "flips": 0, "total": 0, "verdict": "SKIP"}
            continue

        aug_emb = np.concatenate([
            np.array(aug_cls_list), np.array(aug_patch_list)
        ], axis=1)
        aug_pca = pca.transform(aug_emb)
        aug_preds = clf.predict(aug_pca)

        # Match predictions to clean predictions by painting ID
        clean_pred_by_id = dict(zip(painting_ids, clean_preds))
        flips = sum(1 for pid, pred in zip(aug_ids, aug_preds)
                    if clean_pred_by_id.get(pid) != pred)
        total = len(aug_ids)
        flip_rate = flips / total if total > 0 else 0.0
        verdict = "PASS" if flip_rate < FLIP_RATE_THRESHOLD else "FAIL"

        results[aug_name] = {
            "flip_rate": round(flip_rate, 4),
            "flips": flips,
            "total": total,
            "skipped": skipped,
            "verdict": verdict,
        }
        print(f"    {flips}/{total} flipped ({flip_rate:.1%}) — {verdict}")

    del model

    # --- Summary ---
    all_pass = all(r["verdict"] == "PASS" for r in results.values() if r["verdict"] != "SKIP")
    overall = "PASS" if all_pass else "FAIL"

    print(f"\n{'='*60}")
    print("  ROBUSTNESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"  {'Augmentation':<22s} {'Flip Rate':>10s}   {'Flips/Total':>12s}   Verdict")
    for aug_name, r in results.items():
        if r["verdict"] == "SKIP":
            print(f"  {aug_name:<22s} {'N/A':>10s}   {'N/A':>12s}   SKIP")
        else:
            ft = f"{r['flips']}/{r['total']}"
            print(f"  {aug_name:<22s} {r['flip_rate']:>9.1%}   {ft:>12s}   {r['verdict']}")
    print(f"{'='*60}")
    print(f"  Overall:              {overall} ({'all <5%' if all_pass else 'some >=5%'})")
    print(f"{'='*60}")

    output = {"augmentations": results, "overall_verdict": overall, "n_paintings": N,
              "threshold": FLIP_RATE_THRESHOLD}
    with open(RESULTS_ROBUSTNESS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {RESULTS_ROBUSTNESS_JSON}")


# ---------------------------------------------------------------------------
# Option I: Fine-tune DINOv2
# ---------------------------------------------------------------------------

RESULTS_FINETUNE_JSON = CACHE_DIR / "results_finetune.json"
EMBEDDINGS_TILES_NPZ = CACHE_EMB / "embeddings_tiles.npz"
RESULTS_TILES_JSON = CACHE_DIR / "results_tiles.json"
RESULTS_CLIP_JSON = CACHE_DIR / "results_clip.json"
EMBEDDINGS_CLIP_NPZ = CACHE_EMB / "embeddings_clip.npz"

def stage_finetune(rows):
    """Fine-tune DINOv2 ViT-B/14 last 2 blocks + linear head for autograph vs circle.

    Stratified 5-fold CV, balanced class weights, early stopping.
    Whole paintings resized to 518×518 (DINOv2 native resolution).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Fine-tune] Device: {device}")

    # Filter to autograph + circle
    auto_circle = [(r, 1 if r["artist_group"] == "rembrandt_autograph" else 0)
                   for r in rows if r["artist_group"] in ("rembrandt_autograph", "rembrandt_circle")]
    print(f"[Fine-tune] {sum(y for _, y in auto_circle)} autograph + "
          f"{sum(1 - y for _, y in auto_circle)} circle = {len(auto_circle)} paintings")

    # Resolve image paths
    img_paths = []
    labels = []
    for r, y in auto_circle:
        pid = r["obj_id"]
        path = CACHE_IMG / f"{pid}.jpg"
        if path.exists():
            img_paths.append(path)
            labels.append(y)
    labels = np.array(labels)
    print(f"[Fine-tune] {len(img_paths)} images found")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class PaintingDataset(Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            i = self.indices[idx]
            img = Image.open(img_paths[i]).convert("RGB")
            return self.transform(img), labels[i]

    class FineTuneModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
            # Freeze all but last 2 blocks
            for param in self.backbone.parameters():
                param.requires_grad = False
            n_blocks = len(self.backbone.blocks)
            for block in self.backbone.blocks[n_blocks - 2:]:
                for param in block.parameters():
                    param.requires_grad = True
            self.head = nn.Linear(768, 1)
        def forward(self, x):
            features = self.backbone(x)  # CLS token, shape (B, 768)
            return self.head(features).squeeze(-1)

    # Hyperparameters
    lr = 5e-5
    weight_decay = 0.01
    epochs = 30
    patience = 7
    batch_size = 4  # MPS memory limited during backprop with 518x518

    n_auto = int(labels.sum())
    n_circle = len(labels) - n_auto

    # Check param counts on a throwaway model
    _m = FineTuneModel()
    trainable = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in _m.parameters())
    n_blocks = len(_m.backbone.blocks)
    print(f"  Unfroze last 2/{n_blocks} blocks: {trainable:,} / {total:,} params trainable")
    del _m

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        print(f"\n  --- Fold {fold + 1}/5 ---")
        print(f"  Train: {sum(labels[train_idx])} auto + {sum(1 - labels[train_idx])} circle = {len(train_idx)}")
        print(f"  Val:   {sum(labels[val_idx])} auto + {sum(1 - labels[val_idx])} circle = {len(val_idx)}")

        train_ds = PaintingDataset(train_idx, train_transform)
        val_ds = PaintingDataset(val_idx, val_transform)

        # Weighted sampler for class balance
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Fresh model per fold (pretrained weights reloaded)
        model = FineTuneModel()
        model.to(device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_val_bal_acc = 0
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_idx)
            scheduler.step()

            # Validate
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    preds = (logits > 0).long().cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_batch.numpy())
            val_bal_acc = balanced_accuracy_score(val_true, val_preds)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:2d}: loss={train_loss:.4f}, val_bal_acc={val_bal_acc:.3f}")

            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stop at epoch {epoch + 1} (best: {best_val_bal_acc:.3f} at epoch {best_epoch})")
                    break

        print(f"  Fold {fold + 1} best: {best_val_bal_acc:.3f} (epoch {best_epoch})")
        fold_results.append({
            "fold": fold + 1,
            "best_val_bal_acc": round(best_val_bal_acc, 4),
            "best_epoch": best_epoch,
        })

    # Summary
    accs = [r["best_val_bal_acc"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{'='*60}")
    print("  FINE-TUNE RESULTS (Option I)")
    print(f"{'='*60}")
    print("  Model:            DINOv2 ViT-B/14 (last 2 blocks unfrozen)")
    print("  Image size:       518×518")
    print(f"  N autograph:      {n_auto}")
    print(f"  N circle:         {n_circle}")
    print("  CV:               5-fold stratified")
    print(f"  Folds:            {', '.join(f'{a:.3f}' for a in accs)}")
    print(f"  Mean bal acc:     {mean_acc:.3f} ± {std_acc:.3f}")
    print("  vs frozen best:   63.7% (entropy SVM RBF)")
    print(f"{'='*60}")

    results = {
        "model": "DINOv2 ViT-B/14 (last 2 blocks unfrozen)",
        "image_size": 518,
        "n_autograph": n_auto,
        "n_circle": n_circle,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_max": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "cv_folds": 5,
        "fold_results": fold_results,
        "mean_bal_acc": round(float(mean_acc), 4),
        "std_bal_acc": round(float(std_acc), 4),
    }
    with open(RESULTS_FINETUNE_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_FINETUNE_JSON}")


# ---------------------------------------------------------------------------
# Option E: Per-tile classification
# ---------------------------------------------------------------------------

def stage_tiles(rows):
    """Embed tiles individually (no averaging), then probe with painting-level voting."""
    import torch
    from torchvision import transforms
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    # Check cache
    if EMBEDDINGS_TILES_NPZ.exists():
        print("[Tiles] Loading cached per-tile embeddings...")
        data = np.load(EMBEDDINGS_TILES_NPZ, allow_pickle=True)
        tile_embs = data["tile_embeddings"]        # list of arrays, one per painting
        tile_counts = data["tile_counts"]           # number of tiles per painting
        painting_ids = data["painting_ids"]
        artist_groups = data["artist_groups"]
    else:
        # Re-embed, saving per-tile features
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Tiles] Embedding per-tile features on {device}...")

        dino_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model = model.to(device).eval()

        all_tile_embs = []  # flat array of all tile embeddings
        tile_counts = []
        painting_ids = []
        artist_groups = []

        ac_rows = [r for r in rows if r["artist_group"] in ("rembrandt_autograph", "rembrandt_circle")]
        print(f"  {len(ac_rows)} autograph+circle paintings to embed")

        for i, row in enumerate(ac_rows):
            img_path = CACHE_IMG / f"{row['obj_id']}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            tile_cls = []
            for batch_tensor in _tile_batches(img, TILE_SIZE, BATCH_SIZE_VITB, dino_transform):
                batch_tensor = batch_tensor.to(device)
                with torch.no_grad():
                    out = model.forward_features(batch_tensor)
                    tile_cls.append(out["x_norm_clstoken"].cpu().numpy())
            del img
            tile_cls = np.concatenate(tile_cls, axis=0)  # (N_tiles, 768)
            all_tile_embs.append(tile_cls)
            tile_counts.append(len(tile_cls))
            painting_ids.append(row["obj_id"])
            artist_groups.append(row["artist_group"])
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(ac_rows)} done")

        painting_ids = np.array(painting_ids)
        artist_groups = np.array(artist_groups)
        tile_counts = np.array(tile_counts)
        # Save as flat array + counts for reconstruction
        tile_embs = np.concatenate(all_tile_embs, axis=0)
        np.savez_compressed(EMBEDDINGS_TILES_NPZ,
                            tile_embeddings=tile_embs,
                            tile_counts=tile_counts,
                            painting_ids=painting_ids,
                            artist_groups=artist_groups)
        print(f"  Saved {len(tile_embs)} tiles from {len(painting_ids)} paintings → {EMBEDDINGS_TILES_NPZ}")

    # Reconstruct per-painting tile arrays
    y = np.array([1 if g == "rembrandt_autograph" else 0 for g in artist_groups])
    n = len(y)
    n_auto = int(y.sum())
    n_circle = n - n_auto
    total_tiles = int(tile_counts.sum()) if isinstance(tile_counts, np.ndarray) else sum(tile_counts)
    print(f"\n[Tiles] Probe: {n_auto} autograph + {n_circle} circle = {n} paintings, {total_tiles} tiles")

    # Reconstruct tile arrays per painting from flat array
    if isinstance(tile_embs, np.ndarray) and tile_embs.ndim == 2:
        # Flat array — reconstruct
        per_painting_tiles = []
        offset = 0
        for count in tile_counts:
            per_painting_tiles.append(tile_embs[offset:offset + int(count)])
            offset += int(count)
    else:
        per_painting_tiles = list(tile_embs)

    # Stratified 10-fold CV with tile-level SVM + painting-level majority vote
    pca_dims = 20
    C_values = [0.01, 0.1, 1.0, 10.0]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print(f"  PCA={pca_dims}, SVM RBF with class_weight=balanced")
    print(f"  {'C':<10} {'Tile acc':<12} {'Vote acc':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")

    best_vote_acc = 0
    best_C = 0

    for C in C_values:
        tile_accs = []
        vote_accs = []
        for train_idx, val_idx in skf.split(np.zeros(n), y):
            # Gather train tiles
            train_tiles = np.concatenate([per_painting_tiles[i] for i in train_idx])
            train_labels = np.concatenate([np.full(len(per_painting_tiles[i]), y[i]) for i in train_idx])
            # PCA on train tiles
            pca = PCA(n_components=pca_dims, random_state=42)
            train_pca = pca.fit_transform(train_tiles)
            # Train SVM
            clf = SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced")
            clf.fit(train_pca, train_labels)
            # Predict val tiles
            painting_preds = []
            painting_true = []
            tile_correct = 0
            tile_total = 0
            for i in val_idx:
                tiles_pca = pca.transform(per_painting_tiles[i])
                tile_preds = clf.predict(tiles_pca)
                # Majority vote
                vote = int(tile_preds.sum() > len(tile_preds) / 2)
                painting_preds.append(vote)
                painting_true.append(y[i])
                tile_correct += (tile_preds == y[i]).sum()
                tile_total += len(tile_preds)
            tile_accs.append(tile_correct / tile_total)
            vote_accs.append(balanced_accuracy_score(painting_true, painting_preds))

        mean_tile = np.mean(tile_accs)
        mean_vote = np.mean(vote_accs)
        print(f"  {C:<10} {mean_tile:<12.3f} {mean_vote:<12.3f}")
        if mean_vote > best_vote_acc:
            best_vote_acc = mean_vote
            best_C = C

    print(f"\n  Best: C={best_C} → {best_vote_acc:.3f} vote balanced accuracy")

    # Permutation test on best
    n_perms = 200
    print(f"\n  Permutation test ({n_perms} shuffles)...")
    null_accs = np.zeros(n_perms)
    rng = np.random.RandomState(42)
    for p in range(n_perms):
        y_shuf = rng.permutation(y)
        fold_accs = []
        for train_idx, val_idx in skf.split(np.zeros(n), y_shuf):
            train_tiles = np.concatenate([per_painting_tiles[i] for i in train_idx])
            train_labels = np.concatenate([np.full(len(per_painting_tiles[i]), y_shuf[i]) for i in train_idx])
            pca = PCA(n_components=pca_dims, random_state=42)
            train_pca = pca.fit_transform(train_tiles)
            clf = SVC(C=best_C, kernel="rbf", gamma="scale", class_weight="balanced")
            clf.fit(train_pca, train_labels)
            painting_preds, painting_true = [], []
            for i in val_idx:
                tiles_pca = pca.transform(per_painting_tiles[i])
                tile_preds = clf.predict(tiles_pca)
                vote = int(tile_preds.sum() > len(tile_preds) / 2)
                painting_preds.append(vote)
                painting_true.append(y_shuf[i])
            fold_accs.append(balanced_accuracy_score(painting_true, painting_preds))
        null_accs[p] = np.mean(fold_accs)
        if (p + 1) % 100 == 0:
            print(f"    {p + 1}/{n_perms} done")

    p_value = (np.sum(null_accs >= best_vote_acc) + 1) / (n_perms + 1)
    null_mean = null_accs.mean()
    null_std = null_accs.std()

    print(f"\n{'='*60}")
    print("  TILE PROBE RESULTS (Option E)")
    print(f"{'='*60}")
    print(f"  N paintings:       {n} ({n_auto} auto + {n_circle} circle)")
    print(f"  Total tiles:       {total_tiles}")
    print(f"  Best C:            {best_C}")
    print(f"  Vote bal acc:      {best_vote_acc:.3f}")
    print(f"  Perm p-value:      {p_value:.4f}")
    print(f"  Null mean ± std:   {null_mean:.3f} ± {null_std:.3f}")
    print(f"{'='*60}")

    results = {
        "n_paintings": n, "n_autograph": n_auto, "n_circle": n_circle,
        "total_tiles": int(total_tiles), "pca_dims": pca_dims,
        "best_C": best_C, "vote_bal_acc": round(best_vote_acc, 4),
        "perm_p_value": round(p_value, 4),
        "null_mean": round(null_mean, 4), "null_std": round(null_std, 4),
    }
    with open(RESULTS_TILES_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_TILES_JSON}")


# ---------------------------------------------------------------------------
# Option J: CLIP features
# ---------------------------------------------------------------------------

def stage_clip(rows):
    """Embed paintings with CLIP ViT-L/14 and run probe."""
    import torch
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if EMBEDDINGS_CLIP_NPZ.exists():
        print("[CLIP] Loading cached embeddings...")
        data = np.load(EMBEDDINGS_CLIP_NPZ, allow_pickle=True)
        painting_ids = data["painting_ids"]
        embeddings = data["embeddings"]
        artist_groups = data["artist_groups"]
    else:
        import open_clip

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[CLIP] Device: {device}")
        print("[CLIP] Loading CLIP ViT-L/14...")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        model = model.to(device).eval()

        ac_rows = [r for r in rows if r["artist_group"] in ("rembrandt_autograph", "rembrandt_circle")]
        print(f"  {len(ac_rows)} autograph+circle paintings to embed")

        painting_ids = []
        embeddings_list = []
        artist_groups = []

        for i, row in enumerate(ac_rows):
            img_path = CACHE_IMG / f"{row['obj_id']}.jpg"
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
            embeddings_list.append(features.cpu().numpy().squeeze())
            painting_ids.append(row["obj_id"])
            artist_groups.append(row["artist_group"])
            del img
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(ac_rows)} done")

        painting_ids = np.array(painting_ids)
        artist_groups = np.array(artist_groups)
        embeddings = np.array(embeddings_list)
        np.savez_compressed(EMBEDDINGS_CLIP_NPZ,
                            painting_ids=painting_ids,
                            embeddings=embeddings,
                            artist_groups=artist_groups)
        print(f"  Saved {len(embeddings)} × {embeddings.shape[1]}d → {EMBEDDINGS_CLIP_NPZ}")

    y = np.array([1 if g == "rembrandt_autograph" else 0 for g in artist_groups])
    n = len(y)
    n_auto = int(y.sum())
    n_circle = n - n_auto
    print(f"\n[CLIP] Probe: {n_auto} autograph + {n_circle} circle = {n} paintings, {embeddings.shape[1]}d")

    pca_dims_list = [10, 20]
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    classifiers = [
        ("Logistic", lambda v: LogisticRegression(C=v, solver="lbfgs", max_iter=1000, class_weight="balanced"),
         "C", C_values),
        ("SVM RBF", lambda v: SVC(C=v, kernel="rbf", gamma="scale", class_weight="balanced"),
         "C", C_values),
    ]

    overall_best_acc = 0
    overall_best = None
    all_results = []

    for clf_name, make_clf, param_name, param_values in classifiers:
        print(f"\n  --- {clf_name} ---")
        print(f"  {'PCA dims':<10} " + " ".join(f"{param_name}={v:<7}" for v in param_values))
        print(f"  {'-'*10} " + " ".join(f"{'-'*10}" for _ in param_values))
        best_acc, best_dims, best_param = 0, 0, 0
        for dims in pca_dims_list:
            pca = PCA(n_components=dims, random_state=42)
            X_pca = pca.fit_transform(embeddings)
            row_str = f"  {dims:<10}"
            for v in param_values:
                clf = make_clf(v)
                scores = cross_val_score(clf, X_pca, y, cv=skf, scoring="balanced_accuracy")
                acc = scores.mean()
                row_str += f" {acc:<10.3f}"
                if acc > best_acc:
                    best_acc, best_dims, best_param = acc, dims, v
            print(row_str)
        print(f"  Best: PCA={best_dims}, {param_name}={best_param} → {best_acc:.3f}")
        all_results.append({
            "classifier": clf_name, "best_pca": best_dims,
            "best_param": best_param, "bal_acc": round(best_acc, 4),
        })
        if best_acc > overall_best_acc:
            overall_best_acc = best_acc
            overall_best = (clf_name, best_dims, best_param, make_clf)

    best_name, best_dims, best_param, best_make_clf = overall_best
    print(f"\n  Overall best: {best_name} (PCA={best_dims}, C={best_param}) → {overall_best_acc:.3f}")

    # Permutation test
    n_perms = 1000
    print(f"\n  Permutation test ({n_perms} shuffles)...")
    pca = PCA(n_components=best_dims, random_state=42)
    X_best = pca.fit_transform(embeddings)
    null_accs = np.zeros(n_perms)
    rng = np.random.RandomState(42)
    for p in range(n_perms):
        y_shuf = rng.permutation(y)
        clf = best_make_clf(best_param)
        scores = cross_val_score(clf, X_best, y_shuf, cv=skf, scoring="balanced_accuracy")
        null_accs[p] = scores.mean()
        if (p + 1) % 100 == 0:
            print(f"    {p + 1}/{n_perms} done")

    p_value = (np.sum(null_accs >= overall_best_acc) + 1) / (n_perms + 1)
    null_mean = null_accs.mean()
    null_std = null_accs.std()

    print(f"\n{'='*60}")
    print("  CLIP PROBE RESULTS (Option J)")
    print(f"{'='*60}")
    print("  Model:          CLIP ViT-L/14 (OpenAI)")
    print(f"  N paintings:    {n} ({n_auto} auto + {n_circle} circle)")
    print(f"  Embed dim:      {embeddings.shape[1]}")
    print(f"  Best:           {best_name} PCA={best_dims} C={best_param}")
    print(f"  Bal acc:        {overall_best_acc:.3f}")
    print(f"  Perm p-value:   {p_value:.4f}")
    print(f"  Null mean±std:  {null_mean:.3f} ± {null_std:.3f}")
    print(f"{'='*60}")

    results = {
        "model": "CLIP ViT-L/14 (OpenAI)", "embed_dim": int(embeddings.shape[1]),
        "n_autograph": n_auto, "n_circle": n_circle,
        "best_classifier": best_name, "best_pca": best_dims,
        "best_param": best_param, "bal_acc": round(overall_best_acc, 4),
        "perm_p_value": round(p_value, 4),
        "null_mean": round(null_mean, 4), "null_std": round(null_std, 4),
        "all_classifiers": all_results,
    }
    with open(RESULTS_CLIP_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_CLIP_JSON}")


# ---------------------------------------------------------------------------
# Experiment A: Pooled frozen probe on transfer corpus
# ---------------------------------------------------------------------------

def stage_transfer_probe(rows):
    """Frozen-feature probe on multi-artist transfer corpus.

    A1: Pooled 10-fold (all 6 artists, binary autograph vs circle)
    A2: Leave-artist-out — train on 5, test on 1, repeat 6x
    """
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import balanced_accuracy_score

    if not EMBEDDINGS_TRANSFER_NPZ.exists():
        print("ERROR: No transfer embeddings. Run --corpus transfer first.")
        sys.exit(1)

    data = np.load(EMBEDDINGS_TRANSFER_NPZ, allow_pickle=True)
    _painting_ids = data["painting_ids"]
    cls_emb = data["cls_embeddings"]
    patch_emb = data["patch_embeddings"]
    artist_groups = data["artist_groups"]
    artists = data["artists"]
    full_emb = np.concatenate([cls_emb, patch_emb], axis=1)

    # Binary label: autograph=1, circle=0
    y = np.array([1 if g.endswith("_autograph") else 0 for g in artist_groups])
    n = len(y)
    n_auto = int(y.sum())
    n_circle = n - n_auto
    unique_artists = sorted(set(artists))

    print(f"\n[Exp A] Transfer probe: {n_auto} autograph + {n_circle} circle = {n} paintings")
    print(f"  Artists: {', '.join(unique_artists)}")
    print(f"  Features: {full_emb.shape[1]}d")

    # --- A1: Pooled 10-fold (upper bound — includes artist-identity signal) ---
    print("\n  === A1: Pooled 10-fold (upper bound — includes artist-identity signal) ===")
    pca_dims = [10, 20]
    C_values = [0.01, 0.1, 1.0, 10.0]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    best_acc, best_dims, best_C = 0, 0, 0
    print(f"  {'PCA':<6} " + " ".join(f"C={v:<6}" for v in C_values))
    for dims in pca_dims:
        pca = PCA(n_components=dims, random_state=42)
        X_pca = pca.fit_transform(full_emb)
        row_str = f"  {dims:<6}"
        for C in C_values:
            clf = SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced")
            scores = cross_val_score(clf, X_pca, y, cv=skf, scoring="balanced_accuracy")
            acc = scores.mean()
            row_str += f" {acc:<8.3f}"
            if acc > best_acc:
                best_acc, best_dims, best_C = acc, dims, C
        print(row_str)
    print(f"  A1 best: PCA={best_dims}, C={best_C} → {best_acc:.3f} balanced accuracy")

    # --- A2: Leave-artist-out ---
    print("\n  === A2: Leave-artist-out (train 5, test 1) ===")
    a2_results = []
    for test_artist in unique_artists:
        test_mask = artists == test_artist
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        n_test_auto = int((y[test_mask] == 1).sum())
        n_test_circle = n_test - n_test_auto

        if n_test_auto == 0 or n_test_circle == 0:
            print(f"    {test_artist:12s}: SKIP (only one class in test set)")
            a2_results.append({"artist": test_artist, "bal_acc": None, "n": n_test})
            continue

        pca = PCA(n_components=best_dims, random_state=42)
        X_train = pca.fit_transform(full_emb[train_mask])
        X_test = pca.transform(full_emb[test_mask])
        clf = SVC(C=best_C, kernel="rbf", gamma="scale", class_weight="balanced")
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(y[test_mask], preds)
        print(f"    {test_artist:12s}: {bal_acc:.3f} ({n_test_auto} auto + {n_test_circle} circle = {n_test})")
        a2_results.append({"artist": test_artist, "bal_acc": round(bal_acc, 4),
                           "n": n_test, "n_auto": n_test_auto, "n_circle": n_test_circle})

    # Summary
    valid = [r for r in a2_results if r["bal_acc"] is not None]
    mean_a2 = np.mean([r["bal_acc"] for r in valid])
    rembrandt_a2 = next((r["bal_acc"] for r in a2_results if r["artist"] == "rembrandt"), None)

    print(f"\n{'='*60}")
    print("  EXPERIMENT A RESULTS")
    print(f"{'='*60}")
    # A2 (leave-artist-out) is the primary metric — tests generalization across artists
    print(f"  A2 leave-artist-out mean: {mean_a2:.3f}  ← PRIMARY")
    print(f"  A2 Rembrandt (zero-shot): {rembrandt_a2:.3f}" if rembrandt_a2 else "  A2 Rembrandt: N/A")
    # A1 (pooled) includes artist-identity signal, not just attribution signal
    print(f"  A1 (pooled, upper bound): {best_acc:.3f}")
    print(f"  Cross-artist signal:     {'YES (>55%)' if rembrandt_a2 and rembrandt_a2 > 0.55 else 'WEAK/NO'}")
    print(f"{'='*60}")

    results = {
        "a1_pooled_bal_acc": round(best_acc, 4),
        "a1_best_pca": best_dims,
        "a1_best_C": best_C,
        "a2_leave_artist_out": a2_results,
        "a2_mean": round(mean_a2, 4),
        "a2_rembrandt": rembrandt_a2,
        "n_total": n,
        "n_autograph": n_auto,
        "n_circle": n_circle,
    }
    with open(RESULTS_TRANSFER_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_TRANSFER_JSON}")


# ---------------------------------------------------------------------------
# Experiment B: LoRA Rembrandt-only
# ---------------------------------------------------------------------------

def _build_lora_model(rank=8, alpha=16, dropout=0.1):
    """Build DINOv2 ViT-B/14 with LoRA adapters on last 4 attention blocks.

    Returns (model, n_trainable_params).
    Falls back to manual LoRA if peft is not available.
    """
    import torch
    import torch.nn as nn

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    for param in backbone.parameters():
        param.requires_grad = False

    try:
        from peft import LoraConfig, get_peft_model
        # Apply LoRA to last 4 blocks' attention qkv and projection
        target_modules = []
        n_blocks = len(backbone.blocks)
        for i in range(n_blocks - 4, n_blocks):
            target_modules.extend([
                f"blocks.{i}.attn.qkv",
                f"blocks.{i}.attn.proj",
            ])
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        backbone = get_peft_model(backbone, config)
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"  LoRA via peft: {trainable:,} trainable params (rank={rank}, alpha={alpha})")
    except ImportError:
        print("  peft not available, using manual LoRA")
        trainable = _apply_manual_lora(backbone, rank, alpha, dropout)

    class LoRAModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(768, 1)
        def forward(self, x):
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features["x_norm_clstoken"]
            return self.head(features).squeeze(-1)

    model = LoRAModel(backbone)
    # Head is trainable
    trainable += sum(p.numel() for p in model.head.parameters())
    return model, trainable


def _apply_manual_lora(backbone, rank, alpha, dropout):
    """Manual LoRA injection — fallback when peft is unavailable."""
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, original, rank, alpha, dropout):
            super().__init__()
            self.original = original
            in_f = original.in_features
            out_f = original.out_features
            self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
            self.scale = alpha / rank
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            base = self.original(x)
            lora = self.dropout(x) @ self.lora_A @ self.lora_B * self.scale
            return base + lora

    trainable = 0
    n_blocks = len(backbone.blocks)
    for i in range(n_blocks - 4, n_blocks):
        block = backbone.blocks[i]
        # Replace qkv and proj with LoRA versions
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha, dropout)
        block.attn.proj = LoRALinear(block.attn.proj, rank, alpha, dropout)
        trainable += sum(p.numel() for p in block.attn.qkv.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in block.attn.proj.parameters() if p.requires_grad)

    print(f"  Manual LoRA: {trainable:,} trainable params (rank={rank}, alpha={alpha})")
    return trainable


def stage_lora(rows, label="Rembrandt", results_file=None):
    """LoRA fine-tune DINOv2 ViT-B/14 on autograph vs circle.

    5-fold stratified CV, balanced class weights, early stopping.
    Whole paintings resized to 518×518.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    if results_file is None:
        results_file = RESULTS_LORA_JSON

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[LoRA {label}] Device: {device}")

    # Filter to autograph + circle
    auto_circle = [(r, 1 if r["artist_group"].endswith("_autograph") else 0)
                   for r in rows if r["artist_group"].endswith("_autograph") or r["artist_group"].endswith("_circle")]
    print(f"[LoRA {label}] {sum(y for _, y in auto_circle)} autograph + "
          f"{sum(1 - y for _, y in auto_circle)} circle = {len(auto_circle)} paintings")

    # Resolve image paths
    img_paths = []
    labels = []
    for r, y in auto_circle:
        path = CACHE_IMG / f"{r['obj_id']}.jpg"
        if path.exists():
            img_paths.append(path)
            labels.append(y)
    labels = np.array(labels)
    print(f"[LoRA {label}] {len(img_paths)} images found")

    train_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class PaintingDataset(Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            i = self.indices[idx]
            img = Image.open(img_paths[i]).convert("RGB")
            return self.transform(img), labels[i]

    # Hyperparameters
    lr = 5e-5
    weight_decay = 0.01
    epochs = 50
    patience = 10
    batch_size = 4

    n_auto = int(labels.sum())
    n_circle = len(labels) - n_auto

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # Build model ONCE, reset weights between folds to avoid CUDA memory fragmentation
    model, trainable = _build_lora_model()
    model.to(device)
    print(f"  Trainable params: {trainable:,}")
    _initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        print(f"\n  --- Fold {fold + 1}/5 ---")
        print(f"  Train: {sum(labels[train_idx])} auto + {sum(1 - labels[train_idx])} circle = {len(train_idx)}")
        print(f"  Val:   {sum(labels[val_idx])} auto + {sum(1 - labels[val_idx])} circle = {len(val_idx)}")

        train_ds = PaintingDataset(train_idx, train_transform)
        val_ds = PaintingDataset(val_idx, val_transform)

        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Reset to initial weights (no GPU reallocation)
        model.load_state_dict(_initial_state)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_val_bal_acc = 0
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_idx)
            scheduler.step()

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    preds = (logits > 0).long().cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_batch.numpy())
            val_bal_acc = balanced_accuracy_score(val_true, val_preds)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:2d}: loss={train_loss:.4f}, val_bal_acc={val_bal_acc:.3f}")

            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stop at epoch {epoch + 1} (best: {best_val_bal_acc:.3f} at epoch {best_epoch})")
                    break

        print(f"  Fold {fold + 1} best: {best_val_bal_acc:.3f} (epoch {best_epoch})")
        fold_results.append({
            "fold": fold + 1,
            "best_val_bal_acc": round(best_val_bal_acc, 4),
            "best_epoch": best_epoch,
        })

    # Clean up after all folds
    del model, _initial_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accs = [r["best_val_bal_acc"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{'='*60}")
    print(f"  LORA RESULTS ({label})")
    print(f"{'='*60}")
    print("  Model:            DINOv2 ViT-B/14 + LoRA (last 4 blocks)")
    print("  Image size:       518×518")
    print(f"  N autograph:      {n_auto}")
    print(f"  N circle:         {n_circle}")
    print("  CV:               5-fold stratified")
    print(f"  Folds:            {', '.join(f'{a:.3f}' for a in accs)}")
    print(f"  Mean bal acc:     {mean_acc:.3f} ± {std_acc:.3f}")
    print("  vs frozen best:   63.7% (entropy SVM RBF)")
    print("  vs fine-tune (I): 60.0% ± 5.0%")
    print(f"{'='*60}")

    results = {
        "model": "DINOv2 ViT-B/14 + LoRA (last 4 blocks)",
        "image_size": 518,
        "n_autograph": n_auto,
        "n_circle": n_circle,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_max": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lora_rank": 8,
        "lora_alpha": 16,
        "cv_folds": 5,
        "fold_results": fold_results,
        "mean_bal_acc": round(float(mean_acc), 4),
        "std_bal_acc": round(float(std_acc), 4),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {results_file}")
    return results


# ---------------------------------------------------------------------------
# Experiment C: LoRA leave-artist-out
# ---------------------------------------------------------------------------

def stage_lora_transfer(rows):
    """LoRA leave-artist-out on transfer corpus.

    C1: Pooled 10-fold (all artists mixed)
    C2: Leave-artist-out — train on 5, test on 1, repeat 6x
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from sklearn.metrics import balanced_accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Exp C] LoRA leave-artist-out. Device: {device}")

    # Filter to autograph + circle, resolve images
    ac_rows = [r for r in rows
                if r["artist_group"].endswith("_autograph") or r["artist_group"].endswith("_circle")]
    img_paths = []
    labels = []
    artists = []
    for r in ac_rows:
        path = CACHE_IMG / f"{r['obj_id']}.jpg"
        if path.exists():
            img_paths.append(path)
            labels.append(1 if r["artist_group"].endswith("_autograph") else 0)
            artists.append(r.get("artist", ""))
    labels = np.array(labels)
    artists = np.array(artists)
    n = len(labels)
    n_auto = int(labels.sum())
    n_circle = n - n_auto
    unique_artists = sorted(set(artists))
    print(f"  {n_auto} autograph + {n_circle} circle = {n} paintings across {len(unique_artists)} artists")

    train_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class PaintingDataset(Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            i = self.indices[idx]
            img = Image.open(img_paths[i]).convert("RGB")
            return self.transform(img), labels[i]

    lr = 5e-5
    weight_decay = 0.01
    epochs = 50
    patience = 10
    batch_size = 4

    # Build model ONCE, save initial state for resetting between folds.
    # Rebuilding via torch.hub.load each fold causes CUDA memory fragmentation → crash.
    shared_model, _ = _build_lora_model()
    shared_model.to(device)
    initial_state = {k: v.clone() for k, v in shared_model.state_dict().items()}

    def _train_lora_fold(model, init_state, train_idx, val_idx, fold_label=""):
        """Train one LoRA fold, return best validation balanced accuracy."""
        train_ds = PaintingDataset(train_idx, train_transform)
        val_ds = PaintingDataset(val_idx, val_transform)

        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=2)
        if class_counts[0] == 0 or class_counts[1] == 0:
            print(f"    {fold_label}: SKIP (single class in train)")
            return None
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Reset model weights to initial state (no reallocation)
        model.load_state_dict(init_state)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_val_bal_acc = 0
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    preds = (logits > 0).long().cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_batch.numpy())
            val_bal_acc = balanced_accuracy_score(val_true, val_preds)

            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        print(f"    {fold_label}: {best_val_bal_acc:.3f} (epoch {best_epoch})")
        return best_val_bal_acc

    # --- C2: Leave-artist-out ---
    print("\n  === C2: Leave-artist-out ===")
    c2_results = []
    for test_artist in unique_artists:
        test_mask = artists == test_artist
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        n_test_auto = int((labels[test_idx] == 1).sum())
        n_test_circle = len(test_idx) - n_test_auto
        if n_test_auto == 0 or n_test_circle == 0:
            print(f"    {test_artist}: SKIP (only one class)")
            c2_results.append({"artist": test_artist, "bal_acc": None})
            continue

        bal_acc = _train_lora_fold(shared_model, initial_state, train_idx, test_idx, fold_label=test_artist)
        c2_results.append({"artist": test_artist, "bal_acc": round(bal_acc, 4) if bal_acc else None,
                           "n_auto": n_test_auto, "n_circle": n_test_circle})

    # Clean up shared model after all folds
    del shared_model, initial_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_c2 = [r for r in c2_results if r["bal_acc"] is not None]
    mean_c2 = np.mean([r["bal_acc"] for r in valid_c2]) if valid_c2 else 0
    rembrandt_c2 = next((r["bal_acc"] for r in c2_results if r["artist"] == "rembrandt"), None)

    print(f"\n{'='*60}")
    print("  EXPERIMENT C RESULTS")
    print(f"{'='*60}")
    print(f"  C2 leave-artist-out mean: {mean_c2:.3f}")
    print(f"  C2 Rembrandt:             {rembrandt_c2:.3f}" if rembrandt_c2 else "  C2 Rembrandt: N/A")
    print("  vs Exp B (LoRA Rembrandt): compare manually")
    print(f"{'='*60}")

    results = {
        "c2_leave_artist_out": c2_results,
        "c2_mean": round(mean_c2, 4),
        "c2_rembrandt": rembrandt_c2,
        "n_total": n, "n_autograph": n_auto, "n_circle": n_circle,
    }
    with open(RESULTS_LORA_TRANSFER_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_LORA_TRANSFER_JSON}")


# ---------------------------------------------------------------------------
# Experiment D: Two-phase LoRA transfer
# ---------------------------------------------------------------------------

def stage_lora_curriculum(transfer_rows, rembrandt_rows):
    """Two-phase LoRA: pre-train on 5 non-Rembrandt artists, fine-tune on Rembrandt.

    Phase 1: Train LoRA on non-Rembrandt (autograph vs circle). Save weights.
    Phase 2: Load phase 1 weights, fine-tune on Rembrandt with lower LR.
    Evaluate: 5-fold CV on Rembrandt for phase 2 only.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Exp D] Two-phase LoRA curriculum. Device: {device}")

    # Phase 1 data: non-Rembrandt transfer artists
    phase1_rows = [r for r in transfer_rows
                   if r.get("artist", "") != "rembrandt"
                   and (r["artist_group"].endswith("_autograph") or r["artist_group"].endswith("_circle"))]
    phase1_paths = []
    phase1_labels = []
    for r in phase1_rows:
        path = CACHE_IMG / f"{r['obj_id']}.jpg"
        if path.exists():
            phase1_paths.append(path)
            phase1_labels.append(1 if r["artist_group"].endswith("_autograph") else 0)
    phase1_labels = np.array(phase1_labels)
    print(f"  Phase 1 data: {int(phase1_labels.sum())} auto + "
          f"{len(phase1_labels) - int(phase1_labels.sum())} circle = {len(phase1_labels)} non-Rembrandt")

    # Phase 2 data: Rembrandt only
    phase2_ac = [(r, 1 if r["artist_group"] in ("rembrandt_autograph",) else 0)
                 for r in rembrandt_rows
                 if r["artist_group"] in ("rembrandt_autograph", "rembrandt_circle")]
    phase2_paths = []
    phase2_labels = []
    for r, y in phase2_ac:
        path = CACHE_IMG / f"{r['obj_id']}.jpg"
        if path.exists():
            phase2_paths.append(path)
            phase2_labels.append(y)
    phase2_labels = np.array(phase2_labels)
    n_auto = int(phase2_labels.sum())
    n_circle = len(phase2_labels) - n_auto
    print(f"  Phase 2 data: {n_auto} auto + {n_circle} circle = {len(phase2_labels)} Rembrandt")

    train_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class PathDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]

    class IndexDataset(Dataset):
        def __init__(self, indices, paths, labels, transform):
            self.indices = indices
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            i = self.indices[idx]
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), self.labels[i]

    # --- Phase 1: Pre-train on 5 artists ---
    print("\n  === Phase 1: Pre-train on non-Rembrandt artists ===")
    phase1_epochs = 30
    phase1_patience = 7
    phase1_lr = 5e-5
    batch_size = 4

    model, trainable = _build_lora_model()
    print(f"  Trainable params: {trainable:,}")
    model.to(device)

    phase1_ds = PathDataset(phase1_paths, phase1_labels, train_transform)
    class_counts = np.bincount(phase1_labels, minlength=2)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[phase1_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    phase1_loader = DataLoader(phase1_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=phase1_lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    no_improve = 0
    for epoch in range(phase1_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for X_batch, y_batch in phase1_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:2d}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= phase1_patience:
                print(f"    Early stop at epoch {epoch + 1}")
                break

    # Save phase 1 weights
    phase1_weights_path = CACHE_DIR / "lora_phase1_weights.pt"
    torch.save({k: v for k, v in model.state_dict().items()}, phase1_weights_path)
    print(f"  Phase 1 complete. Saved weights → {phase1_weights_path}")
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Phase 2: Fine-tune on Rembrandt (5-fold CV) ---
    print("\n  === Phase 2: Fine-tune on Rembrandt (5-fold CV) ===")
    phase2_lr = 1e-5  # Lower LR for fine-tuning
    phase2_epochs = 50
    phase2_patience = 10

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # Build model ONCE, reset to phase1 weights between folds
    _phase2_model, _ = _build_lora_model()
    _phase1_state = torch.load(phase1_weights_path, weights_only=True)
    _phase2_model.to(device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(phase2_paths, phase2_labels)):
        print(f"\n  --- Fold {fold + 1}/5 ---")
        train_ds = IndexDataset(train_idx, phase2_paths, phase2_labels, train_transform)
        val_ds = IndexDataset(val_idx, phase2_paths, phase2_labels, val_transform)

        train_labels = phase2_labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Reset model to phase 1 weights (no GPU reallocation)
        _phase2_model.load_state_dict(_phase1_state)

        optimizer = torch.optim.AdamW(
            [p for p in _phase2_model.parameters() if p.requires_grad],
            lr=phase2_lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
        criterion = nn.BCEWithLogitsLoss()

        best_val_bal_acc = 0
        best_epoch = 0
        no_improve = 0

        for epoch in range(phase2_epochs):
            _phase2_model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                optimizer.zero_grad()
                logits = _phase2_model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            _phase2_model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    logits = _phase2_model(X_batch)
                    preds = (logits > 0).long().cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_batch.numpy())
            val_bal_acc = balanced_accuracy_score(val_true, val_preds)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:2d}: val_bal_acc={val_bal_acc:.3f}")

            if val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = val_bal_acc
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= phase2_patience:
                    print(f"    Early stop at epoch {epoch + 1} (best: {best_val_bal_acc:.3f} at epoch {best_epoch})")
                    break

        print(f"  Fold {fold + 1} best: {best_val_bal_acc:.3f} (epoch {best_epoch})")
        fold_results.append({
            "fold": fold + 1,
            "best_val_bal_acc": round(best_val_bal_acc, 4),
            "best_epoch": best_epoch,
        })

    del _phase2_model, _phase1_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accs = [r["best_val_bal_acc"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n{'='*60}")
    print("  EXPERIMENT D RESULTS (Two-phase LoRA)")
    print(f"{'='*60}")
    print(f"  Phase 1: {len(phase1_labels)} non-Rembrandt paintings")
    print(f"  Phase 2: {len(phase2_labels)} Rembrandt paintings")
    print(f"  Phase 2 LR: {phase2_lr}")
    print(f"  Folds:            {', '.join(f'{a:.3f}' for a in accs)}")
    print(f"  Mean bal acc:     {mean_acc:.3f} ± {std_acc:.3f}")
    print("  vs Exp B (LoRA):  compare manually")
    print("  vs frozen best:   63.7% (entropy SVM RBF)")
    print(f"{'='*60}")

    results = {
        "model": "DINOv2 ViT-B/14 + LoRA (two-phase curriculum)",
        "phase1_n": len(phase1_labels),
        "phase2_n": len(phase2_labels),
        "phase1_lr": phase1_lr,
        "phase2_lr": phase2_lr,
        "n_autograph": n_auto,
        "n_circle": n_circle,
        "fold_results": fold_results,
        "mean_bal_acc": round(float(mean_acc), 4),
        "std_bal_acc": round(float(std_acc), 4),
    }
    with open(RESULTS_LORA_CURRICULUM_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {RESULTS_LORA_CURRICULUM_JSON}")


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
    parser.add_argument("--concat", action="store_true",
                        help="Option K: concatenate ViT-B + ViT-L + entropy embeddings for probe")
    parser.add_argument("--finetune", action="store_true",
                        help="Option I: fine-tune DINOv2 last 2 blocks on autograph vs circle")
    parser.add_argument("--tiles", action="store_true",
                        help="Option E: per-tile classification with painting-level voting")
    parser.add_argument("--clip", action="store_true",
                        help="Option J: CLIP ViT-L/14 features for probe")
    parser.add_argument("--refetch", action="store_true",
                        help="Delete cached inventory + embeddings to force re-fetch from all sources")
    parser.add_argument("--corpus", choices=["rembrandt", "transfer"], default="rembrandt",
                        help="Corpus: rembrandt (default) or transfer (multi-artist)")
    parser.add_argument("--lora", action="store_true",
                        help="Experiment B: LoRA fine-tune DINOv2 on autograph vs circle")
    parser.add_argument("--transfer-probe", action="store_true",
                        help="Experiment A: frozen probe on transfer corpus")
    parser.add_argument("--lora-transfer", action="store_true",
                        help="Experiment C: LoRA leave-artist-out on transfer corpus")
    parser.add_argument("--lora-curriculum", action="store_true",
                        help="Experiment D: LoRA pre-train on 5 artists, fine-tune on Rembrandt")
    parser.add_argument("--strict-labels", action="store_true",
                        help="Exclude low-confidence labels from training/eval")
    parser.add_argument("--dedup-threshold", type=int, default=10,
                        help="Perceptual hash hamming distance threshold for dedup (default: 10)")
    parser.add_argument("--holdout-source", type=str, default=None,
                        help="Hold out all rows from this source for domain-shift evaluation")
    parser.add_argument("--legacy-cv", action="store_true",
                        help="Use legacy non-nested CV (for comparison with old results)")
    parser.add_argument("--confounder-audit", action="store_true",
                        help="Run confounder audit: source classifier + within-source probes")
    parser.add_argument("--robustness-test", action="store_true",
                        help="Test prediction stability under benign image augmentations")
    args = parser.parse_args()
    hires = args.hires
    model_name = args.model
    entropy = args.entropy
    probe = args.probe
    corpus = args.corpus

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

    # --confounder-audit: source confound analysis, exit
    if args.confounder_audit:
        confounder_audit()
        return

    # --robustness-test: augmentation flip rate test, exit
    if args.robustness_test:
        robustness_test()
        return

    # --concat --probe: concatenate all three embedding sets, run probe, exit
    if args.concat:
        print("[Mode] Option K: concatenated embeddings (ViT-B + ViT-L + entropy)")
        needed = {
            "ViT-B": EMBEDDINGS_NPZ,
            "ViT-L": EMBEDDINGS_VITL_NPZ,
            "entropy": EMBEDDINGS_ENTROPY_NPZ,
        }
        for label, path in needed.items():
            if not path.exists():
                print(f"ERROR: No cached {label} embeddings at {path}. Run that pipeline first.")
                sys.exit(1)
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))

        parts = []
        for label, path in needed.items():
            data = np.load(path, allow_pickle=True)
            emb = np.concatenate([data["cls_embeddings"], data["patch_embeddings"]], axis=1)
            print(f"  {label}: {emb.shape[1]}d")
            parts.append(emb)
        # Use painting_ids/artist_groups from ViT-B (all should match)
        data0 = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
        painting_ids = data0["painting_ids"]
        artist_groups = data0["artist_groups"]
        full_embeddings = np.concatenate(parts, axis=1)
        print(f"  Concatenated: {full_embeddings.shape[1]}d ({full_embeddings.shape[0]} paintings)")
        stage4b_probe(full_embeddings, artist_groups, painting_ids, rows,
                      strict_labels=args.strict_labels, legacy_cv=args.legacy_cv,
                      holdout_source=args.holdout_source)
        return

    # --finetune: load inventory, fine-tune DINOv2, exit
    if args.finetune:
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        stage_finetune(rows)
        return

    # --tiles: per-tile classification (Option E)
    if args.tiles:
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        stage_tiles(rows)
        return

    # --clip: CLIP features (Option J)
    if args.clip:
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        stage_clip(rows)
        return

    # --lora: LoRA fine-tune on Rembrandt (Experiment B)
    if args.lora:
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        stage_lora(rows)
        return

    # --transfer-probe: frozen probe on transfer corpus (Experiment A)
    if args.transfer_probe:
        if not INVENTORY_TRANSFER_CSV.exists():
            print("ERROR: No transfer inventory. Run --corpus transfer --stage 1 first.")
            sys.exit(1)
        if not EMBEDDINGS_TRANSFER_NPZ.exists():
            print("ERROR: No transfer embeddings. Run --corpus transfer first.")
            sys.exit(1)
        with open(INVENTORY_TRANSFER_CSV) as f:
            rows = list(csv.DictReader(f))
        stage_transfer_probe(rows)
        return

    # --lora-transfer: LoRA leave-artist-out (Experiment C)
    if args.lora_transfer:
        if not INVENTORY_TRANSFER_CSV.exists():
            print("ERROR: No transfer inventory. Run --corpus transfer --stage 1 first.")
            sys.exit(1)
        with open(INVENTORY_TRANSFER_CSV) as f:
            rows = list(csv.DictReader(f))
        stage_lora_transfer(rows)
        return

    # --lora-curriculum: Two-phase transfer (Experiment D)
    if args.lora_curriculum:
        if not INVENTORY_TRANSFER_CSV.exists():
            print("ERROR: No transfer inventory. Run --corpus transfer --stage 1 first.")
            sys.exit(1)
        if not INVENTORY_CSV.exists():
            print("ERROR: No Rembrandt inventory. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_TRANSFER_CSV) as f:
            transfer_rows = list(csv.DictReader(f))
        with open(INVENTORY_CSV) as f:
            rembrandt_rows = list(csv.DictReader(f))
        stage_lora_curriculum(transfer_rows, rembrandt_rows)
        return

    # --probe: load cached embeddings, run probe, exit
    if probe:
        if entropy and model_name == "vitl14":
            emb_path = EMBEDDINGS_ENTROPY_VITL_NPZ
            mode_label = "entropy ViT-L/14"
        elif entropy:
            emb_path = EMBEDDINGS_ENTROPY_NPZ
            mode_label = "entropy ViT-B/14"
        elif model_name == "vitl14":
            emb_path = EMBEDDINGS_VITL_NPZ
            mode_label = "ViT-L/14"
        else:
            emb_path = EMBEDDINGS_NPZ
            mode_label = "ViT-B/14"
        print(f"[Mode] probe ({mode_label}): Logistic + SVM RBF + MLP (autograph vs circle)")
        if not INVENTORY_CSV.exists():
            print("ERROR: No cached inventory. Run full pipeline first.")
            sys.exit(1)
        if not emb_path.exists():
            print(f"ERROR: No cached {mode_label} embeddings. Run full pipeline first.")
            sys.exit(1)
        with open(INVENTORY_CSV) as f:
            rows = list(csv.DictReader(f))
        data = np.load(emb_path, allow_pickle=True)
        painting_ids = data["painting_ids"]
        cls_emb = data["cls_embeddings"]
        patch_emb = data["patch_embeddings"]
        artist_groups = data["artist_groups"]
        full_embeddings = np.concatenate([cls_emb, patch_emb], axis=1)
        stage4b_probe(full_embeddings, artist_groups, painting_ids, rows,
                      strict_labels=args.strict_labels, legacy_cv=args.legacy_cv,
                      holdout_source=args.holdout_source)
        return

    start_stage = args.stage or 1
    end_stage = args.stage or 5

    # Transfer corpus: simplified pipeline (stages 1-3 only, entropy mode)
    if corpus == "transfer":
        print("[Mode] Transfer corpus — multi-artist Wikidata pipeline")
        inv_csv = INVENTORY_TRANSFER_CSV
        emb_file = EMBEDDINGS_TRANSFER_NPZ

        if start_stage <= 1:
            rows = stage1_transfer_metadata()
        else:
            if not inv_csv.exists():
                print("ERROR: No transfer inventory. Run --corpus transfer --stage 1 first.")
                sys.exit(1)
            with open(inv_csv) as f:
                rows = list(csv.DictReader(f))
            print(f"[Stage 1T] Loaded {len(rows)} paintings from cache")

        if end_stage < 2:
            return

        if start_stage <= 2:
            stage2_images(rows)

        if end_stage < 3:
            return

        if start_stage <= 3:
            stage3_embed(rows, entropy=True, emb_file_override=emb_file)
        return

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

    # Stage 2b: perceptual dedup
    dedup_threshold = args.dedup_threshold
    if dedup_threshold > 0 and start_stage <= 2:
        cache_dir = CACHE_IMG_HIRES if hires else CACHE_IMG
        print(f"\n[Stage 2b] Computing perceptual hashes (threshold={dedup_threshold})...")
        phashes = compute_phashes(rows, cache_dir=cache_dir)
        rows, n_removed = dedup_by_phash(rows, phashes, threshold=dedup_threshold)
        if n_removed > 0:
            print(f"  Removed {n_removed} near-duplicate images")
            # Re-save inventory without duplicates
            fieldnames = ["obj_id", "source", "title", "creator", "date", "image_url",
                          "artist_group", "attribution", "label_confidence"]
            with open(INVENTORY_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Updated inventory: {len(rows)} paintings")
        else:
            print(f"  No near-duplicates found ({len(phashes)} images hashed)")

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
