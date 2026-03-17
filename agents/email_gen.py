"""
Email sequence generator for TheraFlow marketing automation.

Reads new leads from Airtable, matches pain points to their therapy types,
generates a 5-email cold outreach sequence via Claude API, and saves each
sequence to email_sequences/<place_id>.json.

Usage:
    python -m agents.email_gen               # process all new leads
    python -m agents.email_gen --limit 5     # first 5 leads only
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pyairtable import Api as AirtableApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEQUENCES_DIR = PROJECT_ROOT / "email_sequences"

BATCH_SIZE = 10
BATCH_DELAY = 2.0  # seconds between batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("email_gen")


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


# ---------------------------------------------------------------------------
# Pain-point matching
# ---------------------------------------------------------------------------

# Map therapy type keywords to search terms for matching pain point text
THERAPY_SEARCH_TERMS: dict[str, list[str]] = {
    "speech":              ["speech", "communication"],
    "occupational":        ["ot", "occupational", "fine motor", "sensory"],
    "aba":                 ["aba", "behavioral", "behaviour"],
    "autism":              ["autism"],
    "special needs":       ["special needs"],
    "child development":   ["child development", "developmental"],
    "sensory integration": ["sensory"],
    "early intervention":  ["early intervention"],
    "general therapy":     [],
}


def load_pain_points() -> list[dict]:
    """Load pain_points.json from data/."""
    path = DATA_DIR / "pain_points.json"
    if not path.exists():
        raise FileNotFoundError(f"Pain points file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("pain_points.json is empty or not a list")
    return data


def match_pain_points(therapy_types_str: str,
                      pain_points: list[dict],
                      top_n: int = 5) -> list[dict]:
    """Return the most relevant pain points for a lead's therapy types.

    Strategy:
    1. Score each pain point by how many therapy keywords appear in its text.
    2. Use severity as a tiebreaker (higher severity first).
    3. Return the top N results.  If fewer than N match, pad with highest-
       severity general pain points.
    """
    therapy_lower = therapy_types_str.lower()
    # Collect all search terms relevant to this lead's therapies
    search_terms: list[str] = []
    for therapy_key, terms in THERAPY_SEARCH_TERMS.items():
        if therapy_key in therapy_lower:
            search_terms.extend(terms)

    scored: list[tuple[int, int, dict]] = []
    for pp in pain_points:
        text = pp["pain"].lower()
        hits = sum(1 for t in search_terms if t in text)
        scored.append((hits, pp.get("severity", 0), pp))

    # Sort: most keyword hits first, then highest severity
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # If nothing matched by keyword, just take highest severity
    results = [pp for _, _, pp in scored[:top_n]]
    return results


# ---------------------------------------------------------------------------
# Airtable
# ---------------------------------------------------------------------------


def _get_airtable_table():
    api = AirtableApi(_env("AIRTABLE_API_KEY"))
    return api.table(_env("AIRTABLE_BASE_ID"), _env("AIRTABLE_LEADS_TABLE"))


def fetch_new_leads(table, limit: int | None = None) -> list[dict]:
    """Fetch leads with status='New' and sequence_step=0 from Airtable."""
    formula = "AND({status}='New', {sequence_step}=0)"
    records = table.all(formula=formula)

    leads = []
    for rec in records:
        fields = rec.get("fields", {})
        leads.append({
            "place_id":      fields.get("place_id", ""),
            "name":          fields.get("name", ""),
            "city":          fields.get("city", ""),
            "city_tier":     fields.get("city_tier", 2),
            "therapy_types": fields.get("therapy_types", ""),
            "phone":         fields.get("phone", ""),
            "website":       fields.get("website", ""),
            "rating":        fields.get("rating", 0),
        })

    if limit and limit > 0:
        leads = leads[:limit]

    log.info("Fetched %d new leads from Airtable.", len(leads))
    return leads


# ---------------------------------------------------------------------------
# Claude API — email generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You write cold emails for the founder of {product_name}, a practice "
    "management SaaS built specifically for early intervention and autism "
    "therapy centers in India. Voice: direct, warm, never corporate. "
    "Emails are short (under 100 words), specific to the center, "
    "always lead with their pain not your product. Never use: "
    "'I hope this email finds you well', 'streamline', 'revolutionize', "
    "'game-changer', 'delve'. Sign every email as {founder_name}, "
    "Founder, {product_name}"
)

USER_PROMPT_TEMPLATE = """\
Generate a 5-email cold outreach sequence for this therapy center.

CENTER INFO:
- Name: {center_name}
- City: {city} (Tier {city_tier})
- Therapy types offered: {therapy_types}

RELEVANT PAIN POINTS (sorted by relevance):
{pain_points_text}

DEMO BOOKING URL: {demo_url}

EMAIL SPECS:
- Email 1 (Day 0): Reference center name + city + their specific therapy mix \
+ one sharp pain point. End with a soft question, not a pitch. Subject: 2 variants (A/B)
- Email 2 (Day 3): Share one insight about managing {therapy_types} centers in \
India. Link to {product_url}/blog. No pitch. Subject: 2 variants
- Email 3 (Day 6): 2-line casual demo offer. "Would a 20-min call make sense?" \
style. Include {demo_url}. Subject: 2 variants
- Email 4 (Day 10): One specific result from a pilot center (anonymised as \
"a center in {same_tier_city}"). Keep under 60 words. Subject: 2 variants
- Email 5 (Day 14): Breakup email. Closing their file. Leave door open. \
Under 50 words. Subject: 2 variants

Return ONLY a valid JSON array of 5 objects. No markdown. No explanation.
Each object must have these exact keys:
{{
  "day": number,
  "subject_a": string,
  "subject_b": string,
  "body": string,
  "plain_text": string
}}
"""


def _pick_same_tier_city(city: str, tier: int) -> str:
    """Pick another city from the same tier for the social-proof email."""
    from agents.scraper import CITIES
    tier_cities = CITIES.get(tier, CITIES.get(1, []))
    for c in tier_cities:
        if c != city:
            return c
    return city  # fallback if only one city


def _format_pain_points(matched: list[dict]) -> str:
    lines = []
    for i, pp in enumerate(matched, 1):
        lines.append(f"{i}. (severity {pp['severity']}/5) {pp['pain']}")
    return "\n".join(lines)


def generate_sequence_for_lead(
    client: anthropic.Anthropic,
    lead: dict,
    pain_points: list[dict],
    model: str,
    founder_name: str,
    product_name: str,
    product_url: str,
    demo_url: str,
) -> dict:
    """Call Claude API to generate a 5-email sequence for one lead."""

    matched = match_pain_points(lead["therapy_types"], pain_points, top_n=5)
    same_tier_city = _pick_same_tier_city(lead["city"],
                                          lead.get("city_tier", 2))

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        product_name=product_name,
        founder_name=founder_name,
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        center_name=lead["name"],
        city=lead["city"],
        city_tier=lead.get("city_tier", 2),
        therapy_types=lead["therapy_types"],
        pain_points_text=_format_pain_points(matched),
        demo_url=demo_url,
        product_url=product_url,
        same_tier_city=same_tier_city,
    )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = message.content[0].text.strip()

    # Strip markdown fences if the model wraps them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0]

    emails = json.loads(raw_text)
    if not isinstance(emails, list) or len(emails) != 5:
        raise ValueError(f"Expected 5 emails, got {type(emails).__name__} "
                         f"len={len(emails) if isinstance(emails, list) else 'N/A'}")

    return {
        "lead_id": lead["place_id"],
        "center_name": lead["name"],
        "city": lead["city"],
        "therapy_types": lead["therapy_types"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "emails": emails,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _sequence_path(place_id: str) -> Path:
    return SEQUENCES_DIR / f"{place_id}.json"


def sequence_exists(place_id: str) -> bool:
    return _sequence_path(place_id).exists()


def save_sequence(sequence: dict):
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = _sequence_path(sequence["lead_id"])
    path.write_text(json.dumps(sequence, indent=2, ensure_ascii=False))


def save_run_log(summary: dict):
    path = DATA_DIR / "email_gen_log.json"
    history: list[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    history.append(summary)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    log.info("Run log saved to %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(limit: int | None = None) -> dict:
    """Execute the email generation pipeline. Returns the run summary."""

    pain_points = load_pain_points()
    model       = _env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    founder     = _env("FOUNDER_NAME", "Founder")
    product     = _env("PRODUCT_NAME", "TheraFlow")
    product_url = _env("PRODUCT_URL", "https://theraflow.in")
    demo_url    = _env("DEMO_BOOKING_URL", "https://theraflow.in/demo")

    client = anthropic.Anthropic(api_key=_env("ANTHROPIC_API_KEY"))
    table = _get_airtable_table()
    leads = fetch_new_leads(table, limit=limit)

    generated = 0
    skipped = 0
    errors: list[str] = []
    details: list[dict] = []

    for batch_start in range(0, len(leads), BATCH_SIZE):
        batch = leads[batch_start:batch_start + BATCH_SIZE]

        if batch_start > 0:
            log.info("Batch delay: %.1fs …", BATCH_DELAY)
            time.sleep(BATCH_DELAY)

        for lead in batch:
            pid = lead["place_id"]
            name = lead["name"]
            city = lead["city"]

            if sequence_exists(pid):
                log.info("  SKIP  %s (%s) — sequence exists", name, city)
                skipped += 1
                details.append({"center": name, "city": city, "status": "skipped"})
                continue

            try:
                seq = generate_sequence_for_lead(
                    client, lead, pain_points, model,
                    founder, product, product_url, demo_url,
                )
                save_sequence(seq)
                generated += 1
                log.info("  OK    %s (%s) — 5 emails generated", name, city)
                details.append({"center": name, "city": city, "status": "generated"})
            except Exception as exc:
                msg = f"{name} ({city}): {exc}"
                log.exception("  ERROR %s", msg)
                errors.append(msg)
                details.append({"center": name, "city": city, "status": "error",
                                "error": str(exc)})

    summary = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "leads_fetched": len(leads),
        "generated": generated,
        "skipped": skipped,
        "errors": len(errors),
        "error_details": errors,
        "details": details,
    }

    save_run_log(summary)

    log.info("--- Done: %d generated, %d skipped, %d errors ---",
             generated, skipped, len(errors))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate email sequences for new leads"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N leads (for testing)")
    args = parser.parse_args()

    run(limit=args.limit)


if __name__ == "__main__":
    main()
