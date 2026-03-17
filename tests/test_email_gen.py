"""
Tests for agents/email_gen.py — fully mocked, no real API keys needed.

Run:  python3 tests/test_email_gen.py
      python3 -m pytest tests/test_email_gen.py -v
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Fake env vars — set BEFORE importing the module
# ---------------------------------------------------------------------------
os.environ.update({
    "ANTHROPIC_API_KEY":    "sk-ant-fake-key",
    "ANTHROPIC_MODEL":      "claude-sonnet-4-20250514",
    "AIRTABLE_API_KEY":     "fake-airtable-key",
    "AIRTABLE_BASE_ID":     "appFAKE",
    "AIRTABLE_LEADS_TABLE": "Leads",
    "FOUNDER_NAME":         "Arjun",
    "PRODUCT_NAME":         "TheraFlow",
    "PRODUCT_URL":          "https://theraflow.in",
    "DEMO_BOOKING_URL":     "https://theraflow.in/demo",
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agents.email_gen as email_gen

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FAKE_PAIN_POINTS = [
    {
        "id": 1,
        "pain": "Your front desk calls 12 parents every morning to confirm speech therapy appointments.",
        "emotional_trigger": "frustrated",
        "who_feels_it_most": "front_desk",
        "theraflow_hook": "TheraFlow sends automated reminders.",
        "content_angle": "stat",
        "severity": 5,
    },
    {
        "id": 2,
        "pain": "Your speech therapist and OT set conflicting goals for the same child.",
        "emotional_trigger": "frustrated",
        "who_feels_it_most": "therapist",
        "theraflow_hook": "TheraFlow aligns therapy goals.",
        "content_angle": "myth_bust",
        "severity": 5,
    },
    {
        "id": 3,
        "pain": "Billing across OT, speech, and ABA is done in one Excel sheet.",
        "emotional_trigger": "overwhelmed",
        "who_feels_it_most": "owner",
        "theraflow_hook": "TheraFlow separates billing per therapy type.",
        "content_angle": "comparison",
        "severity": 4,
    },
    {
        "id": 4,
        "pain": "Insurance claims rejected because SOAP notes are incomplete.",
        "emotional_trigger": "frustrated",
        "who_feels_it_most": "owner",
        "theraflow_hook": "TheraFlow templates meet insurance standards.",
        "content_angle": "story",
        "severity": 5,
    },
    {
        "id": 5,
        "pain": "No dashboard to see center utilization rates.",
        "emotional_trigger": "anxious",
        "who_feels_it_most": "owner",
        "theraflow_hook": "TheraFlow shows real-time utilization.",
        "content_angle": "stat",
        "severity": 4,
    },
    {
        "id": 6,
        "pain": "Managing sensory integration therapy schedules is chaotic.",
        "emotional_trigger": "overwhelmed",
        "who_feels_it_most": "front_desk",
        "theraflow_hook": "TheraFlow auto-schedules sensory sessions.",
        "content_angle": "question",
        "severity": 3,
    },
]

FAKE_LEADS = [
    {
        "place_id": "pid_A",
        "name": "Sunshine Speech Therapy Center",
        "city": "Bangalore",
        "city_tier": 1,
        "therapy_types": "Speech Therapy, Occupational Therapy",
        "phone": "+91 80 1111 2222",
        "website": "https://sunshine.in",
        "rating": 4.5,
    },
    {
        "place_id": "pid_B",
        "name": "BrightMinds ABA Center",
        "city": "Jaipur",
        "city_tier": 2,
        "therapy_types": "ABA Therapy, Autism Therapy",
        "phone": "+91 99 3333 4444",
        "website": "https://brightminds.in",
        "rating": 4.2,
    },
]


def _fake_airtable_records(leads: list[dict]) -> list[dict]:
    """Convert our flat lead dicts into Airtable-style records."""
    return [{"id": f"rec_{l['place_id']}", "fields": l} for l in leads]


FAKE_EMAILS = [
    {"day": 0,  "subject_a": "Subj A1", "subject_b": "Subj B1",
     "body": "Body 1", "plain_text": "Plain 1"},
    {"day": 3,  "subject_a": "Subj A2", "subject_b": "Subj B2",
     "body": "Body 2", "plain_text": "Plain 2"},
    {"day": 6,  "subject_a": "Subj A3", "subject_b": "Subj B3",
     "body": "Body 3", "plain_text": "Plain 3"},
    {"day": 10, "subject_a": "Subj A4", "subject_b": "Subj B4",
     "body": "Body 4", "plain_text": "Plain 4"},
    {"day": 14, "subject_a": "Subj A5", "subject_b": "Subj B5",
     "body": "Body 5", "plain_text": "Plain 5"},
]


def _make_mock_message(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMatchPainPoints(unittest.TestCase):
    """Verify pain-point matching logic by therapy type."""

    def test_speech_therapy_gets_speech_pain_points_first(self):
        matched = email_gen.match_pain_points(
            "Speech Therapy", FAKE_PAIN_POINTS, top_n=3
        )
        # The top results should include pain points mentioning "speech"
        texts = " ".join(pp["pain"].lower() for pp in matched)
        self.assertIn("speech", texts)

    def test_aba_therapy_gets_aba_pain_points(self):
        matched = email_gen.match_pain_points(
            "ABA Therapy", FAKE_PAIN_POINTS, top_n=3
        )
        texts = " ".join(pp["pain"].lower() for pp in matched)
        self.assertIn("aba", texts)

    def test_severity_used_as_tiebreaker(self):
        """When keywords don't match, highest severity should come first."""
        matched = email_gen.match_pain_points(
            "General Therapy", FAKE_PAIN_POINTS, top_n=3
        )
        severities = [pp["severity"] for pp in matched]
        self.assertEqual(severities, sorted(severities, reverse=True))

    def test_returns_requested_count(self):
        matched = email_gen.match_pain_points(
            "Speech Therapy", FAKE_PAIN_POINTS, top_n=5
        )
        self.assertEqual(len(matched), 5)


class TestGenerateSequenceForLead(unittest.TestCase):
    """Verify the Claude API call produces a correctly shaped sequence."""

    @patch("agents.email_gen.anthropic.Anthropic")
    def test_generates_5_emails_with_required_keys(self, MockAnthropic):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(FAKE_EMAILS)
        )

        seq = email_gen.generate_sequence_for_lead(
            client=mock_client,
            lead=FAKE_LEADS[0],
            pain_points=FAKE_PAIN_POINTS,
            model="claude-sonnet-4-20250514",
            founder_name="Arjun",
            product_name="TheraFlow",
            product_url="https://theraflow.in",
            demo_url="https://theraflow.in/demo",
        )

        self.assertEqual(seq["lead_id"], "pid_A")
        self.assertEqual(seq["center_name"], "Sunshine Speech Therapy Center")
        self.assertEqual(seq["city"], "Bangalore")
        self.assertIn("generated_at", seq)
        self.assertEqual(len(seq["emails"]), 5)

        required_keys = {"day", "subject_a", "subject_b", "body", "plain_text"}
        for email in seq["emails"]:
            self.assertEqual(set(email.keys()), required_keys)

    @patch("agents.email_gen.anthropic.Anthropic")
    def test_handles_markdown_fenced_response(self, MockAnthropic):
        mock_client = MockAnthropic.return_value
        fenced = "```json\n" + json.dumps(FAKE_EMAILS) + "\n```"
        mock_client.messages.create.return_value = _make_mock_message(fenced)

        seq = email_gen.generate_sequence_for_lead(
            client=mock_client,
            lead=FAKE_LEADS[0],
            pain_points=FAKE_PAIN_POINTS,
            model="claude-sonnet-4-20250514",
            founder_name="Arjun",
            product_name="TheraFlow",
            product_url="https://theraflow.in",
            demo_url="https://theraflow.in/demo",
        )
        self.assertEqual(len(seq["emails"]), 5)


class TestExistingSequenceSkipped(unittest.TestCase):
    """Leads that already have a file in email_sequences/ should be skipped."""

    @patch("agents.email_gen.time.sleep")
    @patch("agents.email_gen._get_airtable_table")
    @patch("agents.email_gen.anthropic.Anthropic")
    def test_skip_existing_sequence(self, MockAnthropic, mock_table_fn, mock_sleep):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(FAKE_EMAILS)
        )

        table = MagicMock()
        table.all.return_value = _fake_airtable_records(FAKE_LEADS)
        mock_table_fn.return_value = table

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_seq = Path(tmpdir) / "sequences"
            tmp_data = Path(tmpdir) / "data"
            tmp_data.mkdir()

            # Write pain_points.json
            (tmp_data / "pain_points.json").write_text(json.dumps(FAKE_PAIN_POINTS))

            # Pre-create sequence for pid_A so it gets skipped
            tmp_seq.mkdir()
            (tmp_seq / "pid_A.json").write_text('{"already": "exists"}')

            with patch.object(email_gen, "SEQUENCES_DIR", tmp_seq), \
                 patch.object(email_gen, "DATA_DIR", tmp_data):
                summary = email_gen.run(limit=None)

        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["generated"], 1)


class TestBatchDelay(unittest.TestCase):
    """Verify that time.sleep is called between batches."""

    @patch("agents.email_gen.time.sleep")
    @patch("agents.email_gen._get_airtable_table")
    @patch("agents.email_gen.anthropic.Anthropic")
    def test_sleep_called_between_batches(self, MockAnthropic, mock_table_fn,
                                          mock_sleep):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(FAKE_EMAILS)
        )

        # Create 12 leads to trigger 2 batches (10 + 2)
        many_leads = []
        for i in range(12):
            lead = dict(FAKE_LEADS[0])
            lead["place_id"] = f"pid_{i:03d}"
            lead["name"] = f"Center {i}"
            many_leads.append(lead)

        table = MagicMock()
        table.all.return_value = _fake_airtable_records(many_leads)
        mock_table_fn.return_value = table

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_seq = Path(tmpdir) / "sequences"
            tmp_data = Path(tmpdir) / "data"
            tmp_seq.mkdir()
            tmp_data.mkdir()
            (tmp_data / "pain_points.json").write_text(json.dumps(FAKE_PAIN_POINTS))

            with patch.object(email_gen, "SEQUENCES_DIR", tmp_seq), \
                 patch.object(email_gen, "DATA_DIR", tmp_data), \
                 patch.object(email_gen, "BATCH_SIZE", 10):
                summary = email_gen.run(limit=None)

        # 2 batches → 1 sleep call between them
        sleep_calls = [c for c in mock_sleep.call_args_list
                       if c == call(email_gen.BATCH_DELAY)]
        self.assertEqual(len(sleep_calls), 1,
                         "Should sleep once between batch 1 and batch 2")
        self.assertEqual(summary["generated"], 12)


class TestRunLogWritten(unittest.TestCase):
    """Verify email_gen_log.json is written after a run."""

    @patch("agents.email_gen.time.sleep")
    @patch("agents.email_gen._get_airtable_table")
    @patch("agents.email_gen.anthropic.Anthropic")
    def test_log_file_created(self, MockAnthropic, mock_table_fn, mock_sleep):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(FAKE_EMAILS)
        )

        table = MagicMock()
        table.all.return_value = _fake_airtable_records(FAKE_LEADS[:1])
        mock_table_fn.return_value = table

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_seq = Path(tmpdir) / "sequences"
            tmp_data = Path(tmpdir) / "data"
            tmp_seq.mkdir()
            tmp_data.mkdir()
            (tmp_data / "pain_points.json").write_text(json.dumps(FAKE_PAIN_POINTS))

            with patch.object(email_gen, "SEQUENCES_DIR", tmp_seq), \
                 patch.object(email_gen, "DATA_DIR", tmp_data):
                email_gen.run(limit=1)

            log_path = tmp_data / "email_gen_log.json"
            self.assertTrue(log_path.exists())

            log_data = json.loads(log_path.read_text())
            self.assertIsInstance(log_data, list)
            self.assertEqual(len(log_data), 1)

            entry = log_data[0]
            self.assertIn("date", entry)
            self.assertIn("leads_fetched", entry)
            self.assertIn("generated", entry)
            self.assertIn("skipped", entry)
            self.assertIn("errors", entry)


class TestSequenceFileSaved(unittest.TestCase):
    """Verify the per-lead JSON file is written to email_sequences/."""

    @patch("agents.email_gen.time.sleep")
    @patch("agents.email_gen._get_airtable_table")
    @patch("agents.email_gen.anthropic.Anthropic")
    def test_sequence_file_written(self, MockAnthropic, mock_table_fn, mock_sleep):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(FAKE_EMAILS)
        )

        table = MagicMock()
        table.all.return_value = _fake_airtable_records(FAKE_LEADS[:1])
        mock_table_fn.return_value = table

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_seq = Path(tmpdir) / "sequences"
            tmp_data = Path(tmpdir) / "data"
            tmp_data.mkdir()
            (tmp_data / "pain_points.json").write_text(json.dumps(FAKE_PAIN_POINTS))

            with patch.object(email_gen, "SEQUENCES_DIR", tmp_seq), \
                 patch.object(email_gen, "DATA_DIR", tmp_data):
                email_gen.run(limit=1)

            seq_path = tmp_seq / "pid_A.json"
            self.assertTrue(seq_path.exists())

            seq = json.loads(seq_path.read_text())
            self.assertEqual(seq["lead_id"], "pid_A")
            self.assertEqual(len(seq["emails"]), 5)


class TestLimitArg(unittest.TestCase):
    """--limit should restrict how many leads are processed."""

    @patch("agents.email_gen.time.sleep")
    @patch("agents.email_gen._get_airtable_table")
    @patch("agents.email_gen.anthropic.Anthropic")
    def test_limit_restricts_leads(self, MockAnthropic, mock_table_fn, mock_sleep):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(FAKE_EMAILS)
        )

        table = MagicMock()
        table.all.return_value = _fake_airtable_records(FAKE_LEADS)
        mock_table_fn.return_value = table

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_seq = Path(tmpdir) / "sequences"
            tmp_data = Path(tmpdir) / "data"
            tmp_data.mkdir()
            (tmp_data / "pain_points.json").write_text(json.dumps(FAKE_PAIN_POINTS))

            with patch.object(email_gen, "SEQUENCES_DIR", tmp_seq), \
                 patch.object(email_gen, "DATA_DIR", tmp_data):
                summary = email_gen.run(limit=1)

        self.assertEqual(summary["leads_fetched"], 1)
        self.assertEqual(summary["generated"], 1)


class TestErrorHandling(unittest.TestCase):
    """A failed lead should not crash the entire run."""

    @patch("agents.email_gen.time.sleep")
    @patch("agents.email_gen._get_airtable_table")
    @patch("agents.email_gen.anthropic.Anthropic")
    def test_one_failure_does_not_stop_run(self, MockAnthropic, mock_table_fn,
                                           mock_sleep):
        mock_client = MockAnthropic.return_value

        call_count = {"n": 0}

        def messages_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated API failure")
            return _make_mock_message(json.dumps(FAKE_EMAILS))

        mock_client.messages.create.side_effect = messages_side_effect

        table = MagicMock()
        table.all.return_value = _fake_airtable_records(FAKE_LEADS)
        mock_table_fn.return_value = table

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_seq = Path(tmpdir) / "sequences"
            tmp_data = Path(tmpdir) / "data"
            tmp_data.mkdir()
            (tmp_data / "pain_points.json").write_text(json.dumps(FAKE_PAIN_POINTS))

            with patch.object(email_gen, "SEQUENCES_DIR", tmp_seq), \
                 patch.object(email_gen, "DATA_DIR", tmp_data):
                summary = email_gen.run(limit=None)

        self.assertEqual(summary["generated"], 1)
        self.assertEqual(summary["errors"], 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
