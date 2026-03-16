"""
Google Places lead scraper for therapy centers in India.

Searches Google Places API for autism/therapy centers across Indian cities,
deduplicates against Airtable, and appends new leads.

Usage:
    python -m agents.scraper                  # scrape all cities
    python -m agents.scraper --city Bangalore # single city
    python -m agents.scraper --dry-run        # preview, don't save
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import googlemaps
from dotenv import load_dotenv
from pyairtable import Api as AirtableApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SEARCH_TERMS = [
    "autism center",
    "early intervention center",
    "child development center",
    "ABA therapy",
    "speech therapy clinic",
    "occupational therapy clinic",
    "special needs center",
    "sensory integration therapy",
]

CITIES = {
    1: ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune"],
    2: ["Ahmedabad", "Jaipur", "Kochi", "Chandigarh", "Nagpur", "Coimbatore"],
}

# Keyword → therapy type mapping used to infer services from the place name
THERAPY_KEYWORDS = {
    "autism": "Autism Therapy",
    "aba": "ABA Therapy",
    "speech": "Speech Therapy",
    "occupational": "Occupational Therapy",
    "sensory": "Sensory Integration",
    "early intervention": "Early Intervention",
    "special needs": "Special Needs",
    "child development": "Child Development",
    "behavioral": "Behavioral Therapy",
    "neuro": "Neurodevelopmental Therapy",
}

API_CALL_DELAY = 0.5  # seconds between API calls
MAX_RETRIES = 4
INITIAL_BACKOFF = 1.0  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str) -> str:
    """Return an env var or raise with a clear message."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


def _sleep():
    """Polite delay between API calls."""
    time.sleep(API_CALL_DELAY)


def _retry(fn, *args, **kwargs):
    """Call *fn* with exponential backoff on googlemaps exceptions."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (googlemaps.exceptions.ApiError,
                googlemaps.exceptions.TransportError,
                googlemaps.exceptions.Timeout) as exc:
            if attempt == MAX_RETRIES:
                raise
            log.warning("API error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, MAX_RETRIES, exc, backoff)
            time.sleep(backoff)
            backoff *= 2


def _infer_therapy_types(name: str) -> list[str]:
    """Guess therapy types from the place name."""
    name_lower = name.lower()
    return sorted({label for kw, label in THERAPY_KEYWORDS.items()
                   if kw in name_lower}) or ["General Therapy"]


def _build_maps_url(place_id: str) -> str:
    return f"https://www.google.com/maps/place/?q=place_id:{place_id}"


# ---------------------------------------------------------------------------
# Airtable helpers
# ---------------------------------------------------------------------------


def _get_airtable_table():
    api = AirtableApi(_env("AIRTABLE_API_KEY"))
    return api.table(_env("AIRTABLE_BASE_ID"), _env("AIRTABLE_LEADS_TABLE"))


def _fetch_existing_place_ids(table) -> set[str]:
    """Return the set of place_ids already in Airtable."""
    log.info("Fetching existing place_ids from Airtable …")
    existing: set[str] = set()
    for record in table.all(fields=["place_id"]):
        pid = record.get("fields", {}).get("place_id")
        if pid:
            existing.add(pid)
    log.info("Found %d existing leads in Airtable.", len(existing))
    return existing


# ---------------------------------------------------------------------------
# Google Places scraping
# ---------------------------------------------------------------------------


def _search_places(gmaps: googlemaps.Client,
                   query: str,
                   city: str) -> list[dict]:
    """Run a Text Search and return raw result dicts (handles pagination)."""
    full_query = f"{query} in {city}, India"
    results = []

    _sleep()
    response = _retry(gmaps.places, query=full_query)
    results.extend(response.get("results", []))

    # Follow up to 2 pages of next_page_token
    for _ in range(2):
        token = response.get("next_page_token")
        if not token:
            break
        # Google requires a short wait before the token becomes valid
        time.sleep(2)
        _sleep()
        response = _retry(gmaps.places, query=full_query, page_token=token)
        results.extend(response.get("results", []))

    return results


def _get_place_details(gmaps: googlemaps.Client, place_id: str) -> dict:
    """Fetch detailed info for a single place."""
    _sleep()
    resp = _retry(
        gmaps.place,
        place_id=place_id,
        fields=[
            "place_id",
            "name",
            "formatted_address",
            "formatted_phone_number",
            "website",
            "rating",
            "user_ratings_total",
            "url",
        ],
    )
    return resp.get("result", {})


def _scrape_city(gmaps: googlemaps.Client,
                 city: str,
                 city_tier: int,
                 seen_ids: set[str]) -> list[dict]:
    """Scrape all search terms for one city, deduplicate in-run."""
    city_leads: list[dict] = []

    for term in SEARCH_TERMS:
        log.info("  Searching: '%s' in %s", term, city)
        try:
            raw_results = _search_places(gmaps, term, city)
        except Exception:
            log.exception("    Failed search '%s' in %s — skipping term", term, city)
            continue

        for place in raw_results:
            pid = place.get("place_id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            # Fetch details
            try:
                details = _get_place_details(gmaps, pid)
            except Exception:
                log.exception("    Failed details for %s — skipping", pid)
                continue

            name = details.get("name", place.get("name", ""))
            lead = {
                "place_id": pid,
                "name": name,
                "address": details.get("formatted_address", ""),
                "phone": details.get("formatted_phone_number", ""),
                "website": details.get("website", ""),
                "rating": details.get("rating", 0),
                "review_count": details.get("user_ratings_total", 0),
                "google_maps_url": details.get("url", _build_maps_url(pid)),
                "therapy_types": ", ".join(_infer_therapy_types(name)),
                "city": city,
                "city_tier": city_tier,
                "scraped_date": date.today().isoformat(),
            }
            city_leads.append(lead)

    log.info("  %s: %d unique leads found.", city, len(city_leads))
    return city_leads


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_raw_json(leads: list[dict]):
    """Append leads to data/leads_raw.json (create if missing)."""
    path = DATA_DIR / "leads_raw.json"
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            log.warning("Corrupt leads_raw.json — overwriting.")
    existing.extend(leads)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    log.info("Saved %d total records to %s", len(existing), path)


def _push_to_airtable(table, leads: list[dict]):
    """Batch-create records in Airtable (batches of 10)."""
    BATCH = 10
    created = 0
    for i in range(0, len(leads), BATCH):
        batch = leads[i : i + BATCH]
        records = [{"fields": lead} for lead in batch]
        try:
            table.batch_create(records)
            created += len(batch)
            log.info("  Airtable: pushed %d/%d", created, len(leads))
        except Exception:
            log.exception("  Airtable batch_create failed at offset %d", i)
    return created


def _save_run_log(summary: dict):
    """Append run summary to data/scraper_log.json."""
    path = DATA_DIR / "scraper_log.json"
    history: list[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    history.append(summary)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    log.info("Run log saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(city_filter: str | None = None, dry_run: bool = False):
    """Execute the scraper pipeline."""
    gmaps = googlemaps.Client(key=_env("GOOGLE_PLACES_API_KEY"))

    # Build city list
    if city_filter:
        # Find the tier for the requested city
        tier = None
        for t, cities in CITIES.items():
            if city_filter in cities:
                tier = t
                break
        if tier is None:
            log.warning("City '%s' not in predefined list — defaulting to tier 2.",
                        city_filter)
            tier = 2
        city_plan = [(city_filter, tier)]
    else:
        city_plan = []
        for tier, cities in CITIES.items():
            for c in cities:
                city_plan.append((c, tier))

    # Pre-fetch existing place_ids from Airtable (skip in dry-run)
    existing_ids: set[str] = set()
    table = None
    if not dry_run:
        try:
            table = _get_airtable_table()
            existing_ids = _fetch_existing_place_ids(table)
        except Exception:
            log.exception("Could not connect to Airtable — will save raw only.")

    # Scrape
    seen_ids: set[str] = set(existing_ids)  # merge to dedupe in-run too
    all_leads: list[dict] = []
    errors: list[str] = []

    for city, tier in city_plan:
        log.info("=== Scraping %s (Tier %d) ===", city, tier)
        try:
            leads = _scrape_city(gmaps, city, tier, seen_ids)
            all_leads.extend(leads)
        except Exception as exc:
            msg = f"{city}: {exc}"
            log.exception("City-level failure: %s", msg)
            errors.append(msg)

    total_found = len(all_leads)
    dupes_skipped = len(existing_ids & seen_ids)  # ids we skipped from Airtable
    new_count = total_found

    log.info("--- Scrape complete: %d new leads across %d cities ---",
             total_found, len(city_plan))

    if dry_run:
        log.info("[DRY RUN] Would save %d leads. Preview:", total_found)
        for lead in all_leads[:10]:
            log.info("  %s | %s | %s", lead["name"], lead["city"], lead["phone"])
        if total_found > 10:
            log.info("  ... and %d more.", total_found - 10)
    else:
        # Save raw JSON backup
        if all_leads:
            _save_raw_json(all_leads)

        # Push to Airtable
        pushed = 0
        if table and all_leads:
            pushed = _push_to_airtable(table, all_leads)

        # Run log
        summary = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "cities_searched": [c for c, _ in city_plan],
            "total_found": total_found,
            "new_added": pushed,
            "duplicates_skipped": len(existing_ids),
            "errors": errors,
        }
        _save_run_log(summary)

    return all_leads


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Scrape therapy centers from Google Places → Airtable"
    )
    parser.add_argument("--city", type=str, default=None,
                        help="Scrape a single city (e.g. --city Bangalore)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview results without saving to Airtable or files")
    args = parser.parse_args()

    run(city_filter=args.city, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
