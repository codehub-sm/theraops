"""
Content generation agent for TheraFlow marketing automation.

Generates structured pain points, content ideas, and marketing copy
using the Claude API. Output feeds email, LinkedIn, and blog agents.

Usage:
    python -m agents.content_gen --generate-pain-points
    python -m agents.content_gen --validate
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_PAIN_POINT_KEYS = {
    "id", "pain", "emotional_trigger", "who_feels_it_most",
    "theraflow_hook", "content_angle", "severity",
}
VALID_CONTENT_ANGLES = {"story", "stat", "question", "comparison", "myth_bust"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("content_gen")


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


# ---------------------------------------------------------------------------
# Pain-point generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a B2B SaaS content strategist specializing in Indian "
    "healthcare. You write with empathy and specificity. No generic fluff. "
    "You deeply understand how early intervention therapy centers in India "
    "operate — the chaos of managing multiple therapists, anxious parents, "
    "paper-based workflows, and no software built for them."
)

USER_PROMPT = """\
Generate exactly 25 pain points for early intervention center owners \
in India. Each pain point must be specific to their daily operations — \
not generic 'billing is hard' but 'your front desk calls 12 parents \
every morning to confirm appointments because there is no automated \
reminder system'.

Return ONLY a valid JSON array. No markdown. No explanation. No preamble.

Each item must have these exact keys:
{
  "id": number (1-25),
  "pain": string (specific 1-2 line description of the pain),
  "emotional_trigger": string (what the owner feels: frustrated/anxious/\
embarrassed/overwhelmed/helpless),
  "who_feels_it_most": string (owner/therapist/front_desk/parent),
  "theraflow_hook": string (one sentence — how TheraFlow solves this),
  "content_angle": string (best post format: story/stat/question/\
comparison/myth_bust),
  "severity": number (1-5, how painful this is for the center)
}

Cover these areas across the 25 pain points:
- Appointment scheduling and no-shows (3 pain points)
- Parent communication and progress reporting (4 pain points)
- Multi-therapist coordination and handoffs (4 pain points)
- SOAP notes and clinical documentation (3 pain points)
- Billing across different therapy types (3 pain points)
- Staff management and attendance (2 pain points)
- Center owner visibility into operations (3 pain points)
- Government/insurance documentation (3 pain points)\
"""


def generate_pain_points() -> list[dict]:
    """Call Claude API to generate 25 pain points, save to data/pain_points.json.

    Skips generation if the file already contains a non-empty JSON array.
    Returns the pain points list.
    """
    path = DATA_DIR / "pain_points.json"

    # Guard: don't overwrite existing content
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, list) and len(existing) > 0:
                log.info("pain_points.json already has %d items — skipping generation.", len(existing))
                return existing
        except (json.JSONDecodeError, ValueError):
            pass  # file is corrupt or empty object — regenerate

    model = _env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    log.info("Generating pain points with model=%s …", model)

    client = anthropic.Anthropic(api_key=_env("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
    )

    raw_text = message.content[0].text.strip()

    # Strip markdown fences if the model wraps them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0]

    pain_points = json.loads(raw_text)

    if not isinstance(pain_points, list):
        raise ValueError(f"Expected JSON array, got {type(pain_points).__name__}")

    # Write
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pain_points, indent=2, ensure_ascii=False))
    log.info("Saved %d pain points to %s", len(pain_points), path)

    return pain_points


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_pain_points() -> bool:
    """Read data/pain_points.json and run structural checks.

    Prints a validation report and returns True if all checks pass.
    """
    path = DATA_DIR / "pain_points.json"
    if not path.exists():
        log.error("File not found: %s", path)
        return False

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        log.error("Invalid JSON: %s", exc)
        return False

    if not isinstance(data, list):
        log.error("Top-level value is not a list (got %s).", type(data).__name__)
        return False

    errors: list[str] = []

    # Check count
    if len(data) != 25:
        errors.append(f"Expected 25 items, found {len(data)}.")

    seen_ids: set[int] = set()

    for idx, item in enumerate(data):
        prefix = f"Item [{idx}]"

        # Required keys
        missing = REQUIRED_PAIN_POINT_KEYS - set(item.keys())
        if missing:
            errors.append(f"{prefix}: missing keys {missing}")
            continue

        # Duplicate ids
        item_id = item["id"]
        if item_id in seen_ids:
            errors.append(f"{prefix}: duplicate id={item_id}")
        seen_ids.add(item_id)

        # Severity 1-5
        sev = item["severity"]
        if not isinstance(sev, (int, float)) or not (1 <= sev <= 5):
            errors.append(f"{prefix} (id={item_id}): severity={sev!r} not in 1-5")

        # content_angle
        angle = item["content_angle"]
        if angle not in VALID_CONTENT_ANGLES:
            errors.append(f"{prefix} (id={item_id}): content_angle={angle!r} not in {VALID_CONTENT_ANGLES}")

    # Report
    print("=" * 60)
    print("PAIN POINTS VALIDATION REPORT")
    print("=" * 60)
    print(f"  File:       {path}")
    print(f"  Items:      {len(data)}")
    print(f"  Unique IDs: {len(seen_ids)}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        print("\n  RESULT: FAIL")
    else:
        print("\n  RESULT: PASS")

    print("=" * 60)
    return len(errors) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TheraFlow content generation agent"
    )
    parser.add_argument("--generate-pain-points", action="store_true",
                        help="Generate 25 pain points via Claude API")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing pain_points.json")
    args = parser.parse_args()

    if args.generate_pain_points:
        generate_pain_points()
        validate_pain_points()
    elif args.validate:
        validate_pain_points()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
