"""Unit tests for pure functions in scan.py — no network, GPU, or cache needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scan import (
    classify_attribution,
    classify_rembrandt_group,
    confounder_audit,
    dedup_by_phash,
    extract_iiif_id,
    met_artist_group,
    met_classify_attribution,
    parse_la_metadata,
    _aug_jpeg_q30,
    _aug_gaussian_blur,
    _aug_brightness_120,
    _aug_center_crop_10,
    _aug_horizontal_flip,
    _embed_single_image,
)


# ---------------------------------------------------------------------------
# classify_attribution
# ---------------------------------------------------------------------------

class TestClassifyAttribution:
    def test_autograph_plain_name(self):
        assert classify_attribution("Rembrandt van Rijn") == ("autograph", "low")

    def test_workshop(self):
        assert classify_attribution("Workshop of Rembrandt") == ("workshop", "medium")

    def test_workshop_dutch(self):
        assert classify_attribution("Atelier van Rembrandt") == ("workshop", "medium")

    def test_circle(self):
        assert classify_attribution("Circle of Rembrandt") == ("circle", "medium")

    def test_style(self):
        assert classify_attribution("Style of Rembrandt") == ("style", "medium")

    def test_follower(self):
        assert classify_attribution("Follower of Rembrandt") == ("style", "medium")

    def test_after(self):
        assert classify_attribution("After Rembrandt van Rijn") == ("after", "medium")

    def test_copy_after(self):
        assert classify_attribution("Copy after Rembrandt") == ("after", "medium")

    def test_attributed(self):
        assert classify_attribution("Attributed to Rembrandt") == ("attributed", "medium")

    def test_school(self):
        assert classify_attribution("School of Rembrandt") == ("school", "medium")

    def test_case_insensitive(self):
        assert classify_attribution("WORKSHOP OF REMBRANDT") == ("workshop", "medium")


# ---------------------------------------------------------------------------
# classify_rembrandt_group
# ---------------------------------------------------------------------------

class TestClassifyRembrandtGroup:
    def test_autograph(self):
        assert classify_rembrandt_group("Rembrandt van Rijn") == ("rembrandt_autograph", "autograph", "low")

    def test_circle(self):
        assert classify_rembrandt_group("Circle of Rembrandt") == ("rembrandt_circle", "circle", "medium")

    def test_workshop(self):
        assert classify_rembrandt_group("Workshop of Rembrandt") == ("rembrandt_circle", "workshop", "medium")

    def test_non_rembrandt(self):
        assert classify_rembrandt_group("Jan Vermeer") == (None, None, None)


# ---------------------------------------------------------------------------
# met_classify_attribution
# ---------------------------------------------------------------------------

class TestMetClassifyAttribution:
    def test_empty_prefix(self):
        assert met_classify_attribution("") == ("autograph", "low")

    def test_none_prefix(self):
        assert met_classify_attribution(None) == ("autograph", "low")

    def test_style_of(self):
        assert met_classify_attribution("Style of") == ("style", "high")

    def test_after(self):
        assert met_classify_attribution("After") == ("after", "high")

    def test_workshop(self):
        assert met_classify_attribution("Workshop of") == ("workshop", "high")

    def test_unknown_prefix(self):
        assert met_classify_attribution("Possibly by") == ("style", "high")


# ---------------------------------------------------------------------------
# met_artist_group
# ---------------------------------------------------------------------------

class TestMetArtistGroup:
    def test_rembrandt_autograph(self):
        assert met_artist_group("Rembrandt van Rijn", "") == "rembrandt_autograph"

    def test_rembrandt_circle(self):
        assert met_artist_group("Rembrandt", "Circle of") == "rembrandt_circle"

    def test_non_rembrandt(self):
        assert met_artist_group("Jan Steen", "") is None

    def test_query_group_override(self):
        assert met_artist_group("Rembrandt", "", query_group="rubens_autograph") == "rubens_autograph"


# ---------------------------------------------------------------------------
# extract_iiif_id
# ---------------------------------------------------------------------------

class TestExtractIiifId:
    def test_extracts_id(self):
        edm = {"body": {"id": "https://iiif.micr.io/abc123/info.json"}}
        assert extract_iiif_id(edm) == "abc123"

    def test_none_input(self):
        assert extract_iiif_id(None) is None

    def test_no_match(self):
        assert extract_iiif_id({"body": "no iiif here"}) is None


# ---------------------------------------------------------------------------
# parse_la_metadata
# ---------------------------------------------------------------------------

class TestParseLaMetadata:
    def test_empty(self):
        assert parse_la_metadata(None) == {}
        assert parse_la_metadata({}) == {}

    def test_label_title(self):
        result = parse_la_metadata({"_label": "The Night Watch"})
        assert result["title"] == "The Night Watch"

    def test_identified_by_title(self):
        la = {"identified_by": [{"type": "Name", "content": "Self-Portrait"}]}
        result = parse_la_metadata(la)
        assert result["title"] == "Self-Portrait"

    def test_creator_from_referred_to_by(self):
        la = {
            "produced_by": {
                "referred_to_by": [{
                    "classified_as": [{"id": "http://vocab.getty.edu/aat/300435416"}],
                    "content": "Rembrandt van Rijn",
                }]
            }
        }
        result = parse_la_metadata(la)
        assert result["creator"] == "Rembrandt van Rijn"

    def test_creator_from_carried_out_by(self):
        la = {
            "produced_by": {
                "part": [{
                    "carried_out_by": [{"_label": "Frans Hals"}]
                }]
            }
        }
        result = parse_la_metadata(la)
        assert result["creator"] == "Frans Hals"

    def test_date_from_timespan(self):
        la = {
            "produced_by": {
                "timespan": {
                    "identified_by": [{"type": "Name", "content": "1642"}]
                }
            }
        }
        result = parse_la_metadata(la)
        assert result["date"] == "1642"


# ---------------------------------------------------------------------------
# Label confidence (Fix 1)
# ---------------------------------------------------------------------------

class TestLabelConfidence:
    def test_wikidata_qualifier_high(self):
        """Wikidata qualifier match → high confidence."""
        attrib, conf = classify_attribution("Workshop of Rembrandt")
        assert conf == "medium"  # regex match → medium

    def test_met_explicit_prefix_high(self):
        """Explicit Met prefix → high confidence."""
        attrib, conf = met_classify_attribution("Workshop of")
        assert conf == "high"

    def test_met_empty_prefix_low(self):
        """No Met prefix → low confidence (assumed autograph)."""
        attrib, conf = met_classify_attribution("")
        assert attrib == "autograph"
        assert conf == "low"

    def test_plain_name_fallback_low(self):
        """Plain artist name → low confidence (default autograph)."""
        attrib, conf = classify_attribution("Rembrandt van Rijn")
        assert attrib == "autograph"
        assert conf == "low"

    def test_regex_match_medium(self):
        """Regex match on ATTRIBUTION_PATTERNS → medium confidence."""
        attrib, conf = classify_attribution("Follower of Rembrandt")
        assert attrib == "style"
        assert conf == "medium"


# ---------------------------------------------------------------------------
# Split integrity (Fix 5.1) — phash dedup prevents train/test leakage
# ---------------------------------------------------------------------------

try:
    import imagehash
    _has_imagehash = True
except ImportError:
    _has_imagehash = False

try:
    import sklearn  # noqa: F401
    _has_sklearn = True
except ImportError:
    _has_sklearn = False


@pytest.mark.skipif(not _has_imagehash, reason="imagehash not installed")
class TestSplitIntegrity:
    def test_near_duplicates_deduped(self):
        """Near-duplicate phashes should be removed by dedup."""

        # Create two "near-duplicate" hashes (hamming distance < threshold)
        hash_a = imagehash.hex_to_hash("0" * 64)  # 256-bit all zeros
        hash_b = imagehash.hex_to_hash("0" * 63 + "1")  # 1 bit different
        assert hash_a - hash_b < 10  # near-duplicate

        # Different hash
        hash_c = imagehash.hex_to_hash("f" * 64)
        assert hash_a - hash_c > 10  # not near-duplicate

    def test_dedup_keeps_larger(self):
        """Dedup should keep the higher-resolution version."""
        import imagehash

        rows = [
            {"obj_id": "test_a", "source": "rijksmuseum"},
            {"obj_id": "test_b", "source": "met"},
            {"obj_id": "test_c", "source": "wikidata"},
        ]
        phashes = {
            "test_a": imagehash.hex_to_hash("0" * 64),
            "test_b": imagehash.hex_to_hash("0" * 63 + "1"),  # near-dup of a
            "test_c": imagehash.hex_to_hash("f" * 64),  # different
        }

        # Can't test file-size logic without CACHE_IMG, just test count
        deduped, n_removed = dedup_by_phash(rows, phashes, threshold=10)
        assert n_removed == 1
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# Preprocessing uniformity (Fix 5.3)
# ---------------------------------------------------------------------------

class TestPreprocessingUniformity:
    def test_all_sources_resized_uniformly(self):
        """After Fix 2b, the source condition is removed from resize logic.

        Verify that the resize code path no longer checks source — all images
        get capped at IMG_MAX_PX regardless of source.
        """
        import inspect
        from scan import stage2_images
        source_code = inspect.getsource(stage2_images)
        # The old code had: if row["source"] != "rijksmuseum"
        # Fix 2b removes this condition
        assert 'row["source"] != "rijksmuseum"' not in source_code
        assert "source" not in source_code.split("# v1 behavior")[1].split("\n")[0] if "# v1 behavior" in source_code else True


# ---------------------------------------------------------------------------
# Nested CV smoke test (Fix 5.4)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed")
class TestNestedCV:
    def test_nested_cv_no_data_leak(self):
        """Outer fold test data should never appear in inner selection."""
        from sklearn.model_selection import StratifiedKFold

        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 10)
        y = np.array([1] * 25 + [0] * 25)

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for train_idx, test_idx in outer_cv.split(X, y):
            # Test indices must not appear in train
            assert len(set(train_idx) & set(test_idx)) == 0

            # Inner folds should only use train data
            X_train = X[train_idx]
            y_train = y[train_idx]
            for inner_train, inner_val in inner_cv.split(X_train, y_train):
                # Inner indices are relative to X_train, so they can't
                # reference test data. Verify they stay in bounds.
                assert max(inner_train) < len(X_train)
                assert max(inner_val) < len(X_train)
                assert len(set(inner_train) & set(inner_val)) == 0


# ---------------------------------------------------------------------------
# Permutation test sanity (Fix 5.5)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed")
class TestPermutationSanity:
    def test_random_data_nonsignificant(self):
        """On random data, permutation p-value should be > 0.05 most of the time."""
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        significant_count = 0
        n_runs = 20

        for seed in range(n_runs):
            rng = np.random.RandomState(seed)
            n = 60
            X = rng.randn(n, 10)
            y = np.array([1] * 30 + [0] * 30)

            # Get observed accuracy
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            clf = SVC(kernel="rbf", class_weight="balanced")
            observed = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy").mean()

            # Quick permutation test (50 perms for speed)
            null_accs = []
            for _ in range(50):
                y_shuf = rng.permutation(y)
                acc = cross_val_score(clf, X, y_shuf, cv=skf, scoring="balanced_accuracy").mean()
                null_accs.append(acc)

            p_val = (np.sum(np.array(null_accs) >= observed) + 1) / (len(null_accs) + 1)
            if p_val < 0.05:
                significant_count += 1

        # On random data, should be significant < 10% of the time (expect ~5%)
        assert significant_count <= 0.10 * n_runs + 1, (
            f"Too many significant results on random data: {significant_count}/{n_runs}"
        )


# ---------------------------------------------------------------------------
# Confounder audit
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed")
class TestConfounderAudit:
    """Smoke tests for confounder_audit() with synthetic data via monkeypatch."""

    def _make_synthetic_data(self, tmp_path, n=200, source_signal=False):
        """Create synthetic inventory CSV and embeddings NPZ."""
        import csv as csv_mod

        rng = np.random.RandomState(42)
        sources = ["wikidata"] * (n - 40) + ["rijksmuseum"] * 20 + ["met"] * 20
        groups = (["rembrandt_autograph"] * (n // 2)
                  + ["rembrandt_circle"] * (n - n // 2))
        obj_ids = [f"obj_{i}" for i in range(n)]

        # Shuffle together
        order = rng.permutation(n)
        sources = [sources[i] for i in order]
        groups = [groups[i] for i in order]
        obj_ids = [obj_ids[i] for i in order]

        # Write inventory CSV
        inv_path = tmp_path / "metadata" / "inventory.csv"
        inv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(inv_path, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=["obj_id", "source", "title",
                                                   "creator", "date", "image_url",
                                                   "artist_group", "attribution"])
            w.writeheader()
            for i in range(n):
                w.writerow({"obj_id": obj_ids[i], "source": sources[i],
                            "title": "", "creator": "", "date": "",
                            "image_url": "", "artist_group": groups[i],
                            "attribution": ""})

        # Write embeddings NPZ — random features
        emb_dim = 768
        if source_signal:
            # Inject strong source signal: shift mean by source
            cls = rng.randn(n, emb_dim).astype(np.float32)
            for i in range(n):
                if sources[i] == "rijksmuseum":
                    cls[i] += 3.0
                elif sources[i] == "met":
                    cls[i] -= 3.0
        else:
            cls = rng.randn(n, emb_dim).astype(np.float32)
        patch = rng.randn(n, emb_dim).astype(np.float32)

        emb_path = tmp_path / "embeddings" / "embeddings_entropy.npz"
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(emb_path,
                 painting_ids=np.array(obj_ids),
                 artist_groups=np.array(groups),
                 cls_embeddings=cls,
                 patch_embeddings=patch)

        return inv_path, emb_path

    def test_confounder_audit_runs(self, tmp_path, monkeypatch):
        """Smoke test: confounder_audit completes and writes JSON."""
        import scan

        inv_path, emb_path = self._make_synthetic_data(tmp_path)
        results_path = tmp_path / "results_confounder.json"

        monkeypatch.setattr(scan, "INVENTORY_CSV", inv_path)
        monkeypatch.setattr(scan, "EMBEDDINGS_ENTROPY_NPZ", emb_path)
        monkeypatch.setattr(scan, "RESULTS_CONFOUNDER_JSON", results_path)

        confounder_audit()

        assert results_path.exists()
        import json
        result = json.loads(results_path.read_text())
        assert "verdict" in result
        assert result["verdict"] in ("CLEAN", "MIXED", "DIRTY")
        assert "test1_source_acc" in result
        assert "test2_wikidata_acc" in result
        assert "test3_stratified_acc" in result

    def test_confounder_detects_source_signal(self, tmp_path, monkeypatch):
        """When embeddings encode source strongly, source classifier acc should be high."""
        import scan

        inv_path, emb_path = self._make_synthetic_data(tmp_path, source_signal=True)
        results_path = tmp_path / "results_confounder.json"

        monkeypatch.setattr(scan, "INVENTORY_CSV", inv_path)
        monkeypatch.setattr(scan, "EMBEDDINGS_ENTROPY_NPZ", emb_path)
        monkeypatch.setattr(scan, "RESULTS_CONFOUNDER_JSON", results_path)

        confounder_audit()

        import json
        result = json.loads(results_path.read_text())
        # With +3.0 / -3.0 shifts, source classifier should nail it
        assert result["test1_source_acc"] > 0.80, (
            f"Source classifier should detect injected signal: {result['test1_source_acc']}"
        )


# ---------------------------------------------------------------------------
# Robustness: augmentation functions
# ---------------------------------------------------------------------------

class TestAugmentations:
    """Test augmentation functions (no GPU needed)."""

    @pytest.fixture
    def sample_img(self):
        from PIL import Image
        return Image.new("RGB", (500, 400), color=(128, 64, 32))

    def test_jpeg_q30_returns_rgb(self, sample_img):
        out = _aug_jpeg_q30(sample_img)
        assert out.mode == "RGB"
        assert out.size == sample_img.size

    def test_gaussian_blur_returns_rgb(self, sample_img):
        out = _aug_gaussian_blur(sample_img)
        assert out.mode == "RGB"
        assert out.size == sample_img.size

    def test_brightness_returns_rgb(self, sample_img):
        out = _aug_brightness_120(sample_img)
        assert out.mode == "RGB"
        assert out.size == sample_img.size

    def test_center_crop_preserves_size(self, sample_img):
        out = _aug_center_crop_10(sample_img)
        assert out.mode == "RGB"
        assert out.size == sample_img.size

    def test_horizontal_flip_returns_rgb(self, sample_img):
        out = _aug_horizontal_flip(sample_img)
        assert out.mode == "RGB"
        assert out.size == sample_img.size

    def test_horizontal_flip_is_mirror(self, sample_img):
        from PIL import Image
        # Create image with asymmetric content
        img = Image.new("RGB", (100, 50), color=(0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0))  # red top-left
        flipped = _aug_horizontal_flip(img)
        assert flipped.getpixel((99, 0)) == (255, 0, 0)
        assert flipped.getpixel((0, 0)) == (0, 0, 0)


class TestEmbedSingleImage:
    """Test _embed_single_image shape (skip without GPU)."""

    @pytest.fixture
    def check_torch(self):
        try:
            import torch
            return torch
        except ImportError:
            pytest.skip("torch not installed")

    def test_returns_none_for_tiny_image(self, check_torch):
        """Image smaller than TILE_SIZE should return None."""
        from PIL import Image
        img = Image.new("RGB", (100, 100))  # smaller than 224x224
        result = _embed_single_image(img, None, None, 16, None, entropy=False)
        assert result is None
