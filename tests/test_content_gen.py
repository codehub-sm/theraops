"""
Tests for agents/content_gen.py — fully mocked, no real API keys needed.

Run:  python3 tests/test_content_gen.py
      python3 -m pytest tests/test_content_gen.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Fake env vars before importing the module
os.environ.update({
    "ANTHROPIC_API_KEY": "sk-ant-fake-key-for-testing",
    "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agents.content_gen as content_gen

# ---------------------------------------------------------------------------
# Fake Claude API response — valid 25-item pain points array
# ---------------------------------------------------------------------------

VALID_PAIN_POINTS = [
    {
        "id": i,
        "pain": f"Pain point {i}: specific operational pain for therapy centers.",
        "emotional_trigger": ["frustrated", "anxious", "embarrassed", "overwhelmed", "helpless"][i % 5],
        "who_feels_it_most": ["owner", "therapist", "front_desk", "parent"][i % 4],
        "theraflow_hook": f"TheraFlow automates this with feature {i}.",
        "content_angle": ["story", "stat", "question", "comparison", "myth_bust"][i % 5],
        "severity": (i % 5) + 1,
    }
    for i in range(1, 26)
]


def _make_mock_message(text: str) -> MagicMock:
    """Build a mock anthropic Message with .content[0].text."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGeneratePainPoints(unittest.TestCase):
    """Generation via mocked Claude API."""

    @patch("agents.content_gen.anthropic.Anthropic")
    def test_generates_and_saves_25_items(self, MockAnthropic):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(VALID_PAIN_POINTS)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            with patch.object(content_gen, "DATA_DIR", tmp_data):
                result = content_gen.generate_pain_points()

            self.assertEqual(len(result), 25)

            saved = json.loads((tmp_data / "pain_points.json").read_text())
            self.assertEqual(len(saved), 25)
            self.assertEqual(saved[0]["id"], 1)

    @patch("agents.content_gen.anthropic.Anthropic")
    def test_strips_markdown_fences(self, MockAnthropic):
        """Model sometimes wraps JSON in ```json ... ```."""
        mock_client = MockAnthropic.return_value
        fenced = "```json\n" + json.dumps(VALID_PAIN_POINTS) + "\n```"
        mock_client.messages.create.return_value = _make_mock_message(fenced)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            with patch.object(content_gen, "DATA_DIR", tmp_data):
                result = content_gen.generate_pain_points()

            self.assertEqual(len(result), 25)

    @patch("agents.content_gen.anthropic.Anthropic")
    def test_does_not_overwrite_existing_content(self, MockAnthropic):
        """If pain_points.json already has a non-empty list, skip generation."""
        existing = [{"id": 1, "pain": "already here", "emotional_trigger": "frustrated",
                     "who_feels_it_most": "owner", "theraflow_hook": "hook",
                     "content_angle": "story", "severity": 3}]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            (tmp_data / "pain_points.json").write_text(json.dumps(existing))

            with patch.object(content_gen, "DATA_DIR", tmp_data):
                result = content_gen.generate_pain_points()

            # Should return existing data, NOT call Claude
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["pain"], "already here")
            MockAnthropic.assert_not_called()

    @patch("agents.content_gen.anthropic.Anthropic")
    def test_regenerates_if_file_is_empty_object(self, MockAnthropic):
        """An empty {} should trigger regeneration."""
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = _make_mock_message(
            json.dumps(VALID_PAIN_POINTS)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            (tmp_data / "pain_points.json").write_text("{}")

            with patch.object(content_gen, "DATA_DIR", tmp_data):
                result = content_gen.generate_pain_points()

            self.assertEqual(len(result), 25)


class TestValidatePainPoints(unittest.TestCase):
    """Validation logic."""

    def _write_and_validate(self, data) -> bool:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            (tmp_data / "pain_points.json").write_text(json.dumps(data))
            with patch.object(content_gen, "DATA_DIR", tmp_data):
                return content_gen.validate_pain_points()

    def test_valid_25_items_passes(self):
        self.assertTrue(self._write_and_validate(VALID_PAIN_POINTS))

    def test_wrong_count_fails(self):
        self.assertFalse(self._write_and_validate(VALID_PAIN_POINTS[:10]))

    def test_missing_keys_fails(self):
        bad = [dict(VALID_PAIN_POINTS[0])]  # copy first item
        del bad[0]["theraflow_hook"]
        # Pad to 25 items
        bad.extend(VALID_PAIN_POINTS[1:])
        self.assertFalse(self._write_and_validate(bad))

    def test_severity_out_of_range_fails(self):
        bad = [dict(item) for item in VALID_PAIN_POINTS]
        bad[5]["severity"] = 9  # invalid
        self.assertFalse(self._write_and_validate(bad))

    def test_severity_zero_fails(self):
        bad = [dict(item) for item in VALID_PAIN_POINTS]
        bad[0]["severity"] = 0
        self.assertFalse(self._write_and_validate(bad))

    def test_invalid_content_angle_fails(self):
        bad = [dict(item) for item in VALID_PAIN_POINTS]
        bad[3]["content_angle"] = "meme"  # not in valid set
        self.assertFalse(self._write_and_validate(bad))

    def test_duplicate_ids_fails(self):
        bad = [dict(item) for item in VALID_PAIN_POINTS]
        bad[1]["id"] = bad[0]["id"]  # duplicate
        self.assertFalse(self._write_and_validate(bad))

    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(content_gen, "DATA_DIR", Path(tmpdir)):
                self.assertFalse(content_gen.validate_pain_points())

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            (tmp_data / "pain_points.json").write_text("not json at all {{{")
            with patch.object(content_gen, "DATA_DIR", tmp_data):
                self.assertFalse(content_gen.validate_pain_points())

    def test_not_a_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_data = Path(tmpdir)
            (tmp_data / "pain_points.json").write_text('{"key": "value"}')
            with patch.object(content_gen, "DATA_DIR", tmp_data):
                self.assertFalse(content_gen.validate_pain_points())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
