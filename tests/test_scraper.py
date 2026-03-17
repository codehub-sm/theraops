"""
Tests for agents/scraper.py — fully mocked, no real API keys needed.

Run:  python3 tests/test_scraper.py
      python3 -m pytest tests/test_scraper.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Fake env vars — must be set BEFORE importing scraper (which calls load_dotenv)
# ---------------------------------------------------------------------------
os.environ.update({
    "GOOGLE_PLACES_API_KEY": "fake-google-key",
    "AIRTABLE_API_KEY": "fake-airtable-key",
    "AIRTABLE_BASE_ID": "appFAKEBASE",
    "AIRTABLE_LEADS_TABLE": "Leads",
})

# Add project root to path so `agents.scraper` resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agents.scraper as scraper


# ---------------------------------------------------------------------------
# Fake Google Places responses
# ---------------------------------------------------------------------------

FAKE_PLACES = [
    {"place_id": "pid_001", "name": "Sunshine Autism Center"},
    {"place_id": "pid_002", "name": "Little Stars Speech Therapy Clinic"},
    {"place_id": "pid_003", "name": "BrightMinds ABA Therapy"},
]

FAKE_DETAILS = {
    "pid_001": {
        "place_id": "pid_001",
        "name": "Sunshine Autism Center",
        "formatted_address": "123 MG Road, Bangalore",
        "formatted_phone_number": "+91 80 1234 5678",
        "website": "https://sunshineautism.in",
        "rating": 4.5,
        "user_ratings_total": 87,
        "url": "https://maps.google.com/?cid=111",
    },
    "pid_002": {
        "place_id": "pid_002",
        "name": "Little Stars Speech Therapy Clinic",
        "formatted_address": "456 Park St, Bangalore",
        "formatted_phone_number": "+91 80 9876 5432",
        "website": "https://littlestars.in",
        "rating": 4.2,
        "user_ratings_total": 42,
        "url": "https://maps.google.com/?cid=222",
    },
    "pid_003": {
        "place_id": "pid_003",
        "name": "BrightMinds ABA Therapy",
        "formatted_address": "789 Residency Rd, Bangalore",
        "formatted_phone_number": "+91 80 5555 1234",
        "website": "https://brightmindsaba.in",
        "rating": 4.8,
        "user_ratings_total": 120,
        "url": "https://maps.google.com/?cid=333",
    },
}


def _mock_places(query, page_token=None):
    """Simulate gmaps.places() — returns all 3 fakes, no pagination."""
    return {"results": FAKE_PLACES}


def _mock_place(place_id, fields=None):
    """Simulate gmaps.place() — returns details for known IDs."""
    return {"result": FAKE_DETAILS.get(place_id, {})}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestInferTherapyTypes(unittest.TestCase):
    """Unit tests for the keyword→therapy-type mapper."""

    def test_autism_keyword(self):
        types = scraper._infer_therapy_types("Sunshine Autism Center")
        self.assertIn("Autism Therapy", types)

    def test_speech_keyword(self):
        types = scraper._infer_therapy_types("Little Stars Speech Therapy Clinic")
        self.assertIn("Speech Therapy", types)

    def test_aba_keyword(self):
        types = scraper._infer_therapy_types("BrightMinds ABA Therapy")
        self.assertIn("ABA Therapy", types)

    def test_no_match_returns_general(self):
        types = scraper._infer_therapy_types("Some Random Clinic")
        self.assertEqual(types, ["General Therapy"])

    def test_multiple_keywords(self):
        types = scraper._infer_therapy_types("Autism & Speech Therapy Hub")
        self.assertIn("Autism Therapy", types)
        self.assertIn("Speech Therapy", types)


class TestDeduplication(unittest.TestCase):
    """Feed the same place_id twice — only 1 lead should be produced."""

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    def test_duplicate_place_id_scraped_once(self, mock_retry, mock_sleep):
        # _retry just calls the function it wraps; we control the mock client
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

        gmaps = MagicMock()
        # Return pid_001 twice across two "search terms"
        gmaps.places.return_value = {
            "results": [{"place_id": "pid_001", "name": "Sunshine Autism Center"}]
        }
        gmaps.place.return_value = {"result": FAKE_DETAILS["pid_001"]}

        # Patch SEARCH_TERMS to just 2 entries so both return the same place
        with patch.object(scraper, "SEARCH_TERMS", ["autism center", "ABA therapy"]):
            seen = set()
            leads = scraper._scrape_city(gmaps, "Bangalore", 1, seen)

        self.assertEqual(len(leads), 1, "Duplicate place_id should be scraped only once")
        self.assertEqual(leads[0]["place_id"], "pid_001")

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    def test_airtable_existing_ids_skipped(self, mock_retry, mock_sleep):
        """place_ids already in Airtable are pre-loaded into seen_ids."""
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

        gmaps = MagicMock()
        gmaps.places.return_value = {
            "results": [
                {"place_id": "pid_001", "name": "Sunshine Autism Center"},
                {"place_id": "pid_NEW", "name": "New Therapy Spot"},
            ]
        }
        gmaps.place.return_value = {
            "result": {
                "place_id": "pid_NEW",
                "name": "New Therapy Spot",
                "formatted_address": "1 New Rd",
                "formatted_phone_number": "+91 99 0000 1111",
                "rating": 3.9,
                "user_ratings_total": 10,
                "url": "https://maps.google.com/?cid=999",
            }
        }

        # pid_001 already in Airtable
        seen = {"pid_001"}
        with patch.object(scraper, "SEARCH_TERMS", ["autism center"]):
            leads = scraper._scrape_city(gmaps, "Bangalore", 1, seen)

        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["place_id"], "pid_NEW")


class TestCityTier(unittest.TestCase):
    """Verify tier assignment for known and unknown cities."""

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    @patch("agents.scraper._get_airtable_table")
    @patch("agents.scraper._fetch_existing_place_ids", return_value=set())
    @patch("agents.scraper._push_to_airtable", return_value=0)
    @patch("agents.scraper._save_raw_json")
    @patch("agents.scraper._save_run_log")
    @patch("agents.scraper.googlemaps.Client")
    def test_bangalore_is_tier1(self, MockClient, mock_log, mock_raw,
                                mock_push, mock_fetch, mock_table,
                                mock_retry, mock_sleep):
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        client = MockClient.return_value
        client.places.return_value = {"results": FAKE_PLACES[:1]}
        client.place.return_value = {"result": FAKE_DETAILS["pid_001"]}

        with patch.object(scraper, "SEARCH_TERMS", ["autism center"]):
            leads = scraper.run(city_filter="Bangalore", dry_run=False)

        self.assertTrue(len(leads) > 0)
        self.assertEqual(leads[0]["city_tier"], 1)
        self.assertEqual(leads[0]["city"], "Bangalore")

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    @patch("agents.scraper._get_airtable_table")
    @patch("agents.scraper._fetch_existing_place_ids", return_value=set())
    @patch("agents.scraper._push_to_airtable", return_value=0)
    @patch("agents.scraper._save_raw_json")
    @patch("agents.scraper._save_run_log")
    @patch("agents.scraper.googlemaps.Client")
    def test_jaipur_is_tier2(self, MockClient, mock_log, mock_raw,
                             mock_push, mock_fetch, mock_table,
                             mock_retry, mock_sleep):
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        client = MockClient.return_value
        client.places.return_value = {"results": FAKE_PLACES[:1]}
        client.place.return_value = {"result": FAKE_DETAILS["pid_001"]}

        with patch.object(scraper, "SEARCH_TERMS", ["autism center"]):
            leads = scraper.run(city_filter="Jaipur", dry_run=False)

        self.assertTrue(len(leads) > 0)
        self.assertEqual(leads[0]["city_tier"], 2)


class TestDryRun(unittest.TestCase):
    """--dry-run should NOT call Airtable insert or save files."""

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    @patch("agents.scraper._save_raw_json")
    @patch("agents.scraper._save_run_log")
    @patch("agents.scraper._push_to_airtable")
    @patch("agents.scraper._get_airtable_table")
    @patch("agents.scraper._fetch_existing_place_ids")
    @patch("agents.scraper.googlemaps.Client")
    def test_dry_run_skips_persistence(self, MockClient, mock_fetch, mock_table,
                                       mock_push, mock_log, mock_raw,
                                       mock_retry, mock_sleep):
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        client = MockClient.return_value
        client.places.return_value = {"results": FAKE_PLACES}
        client.place.side_effect = lambda place_id, fields=None: {
            "result": FAKE_DETAILS.get(place_id, {})
        }

        with patch.object(scraper, "SEARCH_TERMS", ["autism center"]):
            leads = scraper.run(city_filter="Bangalore", dry_run=True)

        # Should still return leads
        self.assertEqual(len(leads), 3)

        # Should NOT have called any persistence functions
        mock_table.assert_not_called()
        mock_fetch.assert_not_called()
        mock_push.assert_not_called()
        mock_raw.assert_not_called()
        mock_log.assert_not_called()


class TestScraperLog(unittest.TestCase):
    """Verify scraper_log.json is written with correct summary shape."""

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    @patch("agents.scraper._get_airtable_table")
    @patch("agents.scraper._fetch_existing_place_ids", return_value=set())
    @patch("agents.scraper._push_to_airtable", return_value=3)
    @patch("agents.scraper.googlemaps.Client")
    def test_log_written_with_summary(self, MockClient, mock_push,
                                      mock_fetch, mock_table,
                                      mock_retry, mock_sleep):
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        client = MockClient.return_value
        client.places.return_value = {"results": FAKE_PLACES}
        client.place.side_effect = lambda place_id, fields=None: {
            "result": FAKE_DETAILS.get(place_id, {})
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            with patch.object(scraper, "DATA_DIR", tmp_data), \
                 patch.object(scraper, "SEARCH_TERMS", ["autism center"]):
                leads = scraper.run(city_filter="Bangalore", dry_run=False)

            # Check that scraper_log.json was created
            log_path = tmp_data / "scraper_log.json"
            self.assertTrue(log_path.exists(), "scraper_log.json should be created")

            log_data = json.loads(log_path.read_text())
            self.assertIsInstance(log_data, list)
            self.assertEqual(len(log_data), 1)

            entry = log_data[0]
            self.assertIn("date", entry)
            self.assertIn("cities_searched", entry)
            self.assertIn("total_found", entry)
            self.assertIn("new_added", entry)
            self.assertIn("duplicates_skipped", entry)
            self.assertIn("errors", entry)
            self.assertEqual(entry["cities_searched"], ["Bangalore"])
            self.assertEqual(entry["total_found"], 3)
            self.assertEqual(entry["new_added"], 3)
            self.assertIsInstance(entry["errors"], list)


class TestGracefulCityFailure(unittest.TestCase):
    """A failed city should not crash the entire run."""

    @patch("agents.scraper._sleep")
    @patch("agents.scraper._retry")
    @patch("agents.scraper._get_airtable_table")
    @patch("agents.scraper._fetch_existing_place_ids", return_value=set())
    @patch("agents.scraper._push_to_airtable", return_value=1)
    @patch("agents.scraper._save_raw_json")
    @patch("agents.scraper._save_run_log")
    @patch("agents.scraper.googlemaps.Client")
    def test_failed_city_continues(self, MockClient, mock_log, mock_raw,
                                   mock_push, mock_fetch, mock_table,
                                   mock_retry, mock_sleep):
        mock_retry.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
        client = MockClient.return_value

        call_count = {"n": 0}

        def places_side_effect(query, page_token=None):
            call_count["n"] += 1
            # First city's search raises; second city works
            if "Mumbai" in query:
                raise RuntimeError("Simulated network failure for Mumbai")
            return {"results": FAKE_PLACES[:1]}

        client.places.side_effect = places_side_effect
        client.place.return_value = {"result": FAKE_DETAILS["pid_001"]}

        # Run with 2 cities — Mumbai (fails) then Bangalore (succeeds)
        with patch.object(scraper, "CITIES", {1: ["Mumbai", "Bangalore"]}), \
             patch.object(scraper, "SEARCH_TERMS", ["autism center"]):
            leads = scraper.run(city_filter=None, dry_run=False)

        # Bangalore should still produce leads despite Mumbai failing
        self.assertTrue(len(leads) >= 1, "Should have leads from Bangalore")
        self.assertEqual(leads[0]["city"], "Bangalore")


class TestRawJsonBackup(unittest.TestCase):
    """Verify data/leads_raw.json is written correctly."""

    def test_save_raw_json_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            with patch.object(scraper, "DATA_DIR", tmp_data):
                fake_leads = [{"place_id": "pid_001", "name": "Test Center"}]
                scraper._save_raw_json(fake_leads)

            raw_path = tmp_data / "leads_raw.json"
            self.assertTrue(raw_path.exists())
            data = json.loads(raw_path.read_text())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["place_id"], "pid_001")

    def test_save_raw_json_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            # Pre-seed with 1 record
            raw_path = tmp_data / "leads_raw.json"
            raw_path.write_text(json.dumps([{"place_id": "old"}]))

            with patch.object(scraper, "DATA_DIR", tmp_data):
                scraper._save_raw_json([{"place_id": "new"}])

            data = json.loads(raw_path.read_text())
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["place_id"], "old")
            self.assertEqual(data[1]["place_id"], "new")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
