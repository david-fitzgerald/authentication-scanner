"""Unit tests for pure functions in scan.py — no network, GPU, or cache needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scan import (
    classify_attribution,
    classify_rembrandt_group,
    extract_iiif_id,
    met_artist_group,
    met_classify_attribution,
    parse_la_metadata,
)


# ---------------------------------------------------------------------------
# classify_attribution
# ---------------------------------------------------------------------------

class TestClassifyAttribution:
    def test_autograph_plain_name(self):
        assert classify_attribution("Rembrandt van Rijn") == "autograph"

    def test_workshop(self):
        assert classify_attribution("Workshop of Rembrandt") == "workshop"

    def test_workshop_dutch(self):
        assert classify_attribution("Atelier van Rembrandt") == "workshop"

    def test_circle(self):
        assert classify_attribution("Circle of Rembrandt") == "circle"

    def test_style(self):
        assert classify_attribution("Style of Rembrandt") == "style"

    def test_follower(self):
        assert classify_attribution("Follower of Rembrandt") == "style"

    def test_after(self):
        assert classify_attribution("After Rembrandt van Rijn") == "after"

    def test_copy_after(self):
        assert classify_attribution("Copy after Rembrandt") == "after"

    def test_attributed(self):
        assert classify_attribution("Attributed to Rembrandt") == "attributed"

    def test_school(self):
        assert classify_attribution("School of Rembrandt") == "school"

    def test_case_insensitive(self):
        assert classify_attribution("WORKSHOP OF REMBRANDT") == "workshop"


# ---------------------------------------------------------------------------
# classify_rembrandt_group
# ---------------------------------------------------------------------------

class TestClassifyRembrandtGroup:
    def test_autograph(self):
        assert classify_rembrandt_group("Rembrandt van Rijn") == ("rembrandt_autograph", "autograph")

    def test_circle(self):
        assert classify_rembrandt_group("Circle of Rembrandt") == ("rembrandt_circle", "circle")

    def test_workshop(self):
        assert classify_rembrandt_group("Workshop of Rembrandt") == ("rembrandt_circle", "workshop")

    def test_non_rembrandt(self):
        assert classify_rembrandt_group("Jan Vermeer") == (None, None)


# ---------------------------------------------------------------------------
# met_classify_attribution
# ---------------------------------------------------------------------------

class TestMetClassifyAttribution:
    def test_empty_prefix(self):
        assert met_classify_attribution("") == "autograph"

    def test_none_prefix(self):
        assert met_classify_attribution(None) == "autograph"

    def test_style_of(self):
        assert met_classify_attribution("Style of") == "style"

    def test_after(self):
        assert met_classify_attribution("After") == "after"

    def test_workshop(self):
        assert met_classify_attribution("Workshop of") == "workshop"

    def test_unknown_prefix(self):
        assert met_classify_attribution("Possibly by") == "style"


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
