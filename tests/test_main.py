"""Offline integration tests for the CSV-to-report entry point."""

import csv
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from src.schema_new import AnalysisResults, PADERReport


class FakeQwen:
    model = "fake-qwen"

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        packet = json.loads(messages[1][1])
        ids = [observation["observation_id"] for observation in packet["observations"][:3]]
        return SimpleNamespace(content=json.dumps({"selected_observation_ids": ids}), response_metadata={})


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = self.directory / "data"
        self.data.mkdir()
        self.csv_path = self.data / "synthetic.csv"
        self.output = self.directory / "output"
        rows = [
            {"safetyreportid": "0001", "safetyreportversion": "1", "serious": "serious",
             "receivedate": "20250101", "fulfillexpeditecriteria": "yes",
             "patient_reaction_reactionmeddrapt": "Headache,Nausea",
             "patient_reaction_reactionoutcome": "unknown,recovering/resolving"},
            {"safetyreportid": "0002", "safetyreportversion": "1", "serious": "not serious",
             "receivedate": "20250301", "fulfillexpeditecriteria": "no",
             "patient_reaction_reactionmeddrapt": "Headache",
             "patient_reaction_reactionoutcome": "recovered/resolved"},
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def generate(self):
        return main.run_pipeline(self.csv_path, self.output, product="Synthetic product", llm=FakeQwen())

    def test_pipeline_writes_complete_draft_and_resolving_evidence_links(self):
        report = self.generate()
        self.assertEqual(report.review.status, "draft")
        self.assertTrue(all((self.output / name).is_file() for name in main.OUTPUT_FILES))
        content = (self.output / "report_output.md").read_text(encoding="utf-8")
        for title, _ in main.SECTION_SPECS.values():
            self.assertIn(f"## {title}\n", content)
        for title in ("Age groups", "Sex", "Country", "Outcomes", "Monthly received-case counts"):
            self.assertIn(f"### {title}\n", content)
        self.assertIn("| Unique cases | 2 |", content)
        self.assertIn("| Case-reaction pairs | 3 |", content)
        self.assertIn("| Application identifier | Not supplied |", content)
        self.assertIn("DRAFT", content)
        evidence = (self.output / "evidence.md").read_text(encoding="utf-8")
        for number in re.findall(r"\[E\d+\]\(evidence.md#e(\d+)\)", content):
            self.assertIn(f"## E{number}\n", evidence)
        self.assertIn("(case_listing.csv)", content)
        self.assertIn("0001", evidence)
        analysis = AnalysisResults.model_validate_json((self.output / "analysis_results.json").read_text(encoding="utf-8"))
        report.validate_against(analysis)

    def test_generation_failure_preserves_previous_successful_outputs(self):
        self.generate()
        before = {name: (self.output / name).read_bytes() for name in main.OUTPUT_FILES}
        broken = FakeQwen()
        broken.invoke = lambda messages: (_ for _ in ()).throw(ConnectionError("Ollama unavailable"))
        with self.assertRaises(RuntimeError):
            main.run_pipeline(self.csv_path, self.output, llm=broken)
        for name, original in before.items():
            self.assertEqual((self.output / name).read_bytes(), original)
        self.assertTrue((self.output / "last_generation_failure.json").is_file())

    def test_wrong_analysis_hash_is_rejected_before_rendering(self):
        report = self.generate()
        analysis = AnalysisResults.model_validate_json((self.output / "analysis_results.json").read_text(encoding="utf-8"))
        report.analysis_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "analysis hash"):
            main.generate_markdown_report(report, analysis)

    def test_csv_discovery_requires_one_unambiguous_source(self):
        self.assertEqual(main.find_input_csv(self.data), self.csv_path)
        second = self.data / "another.CSV"
        second.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            main.find_input_csv(self.data)
        with self.assertRaises(ValueError):
            main.find_input_csv(self.directory / "missing")

    def test_explicit_review_does_not_call_qwen(self):
        self.generate()
        with patch.object(main, "generate_report", side_effect=AssertionError("Must not generate during review")):
            reviewed = main.review_report(self.output, "approve", "Test reviewer", "Reviewed synthetic evidence")
        self.assertEqual(reviewed.review.status, "approved")
        self.assertIn("APPROVED", (self.output / "report_output.md").read_text(encoding="utf-8"))
        stored = PADERReport.model_validate_json((self.output / "report_draft.json").read_text(encoding="utf-8"))
        self.assertEqual(stored.review.reviewer, "Test reviewer")

    def test_review_refuses_unreconciled_markdown_edits(self):
        self.generate()
        (self.output / "report_output.md").write_text("Manually changed figures", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Markdown differs"):
            main.review_report(self.output, "approve", "Test reviewer")

    def test_source_csv_cannot_be_replaced_by_an_output(self):
        source = self.directory / "case_listing.csv"
        source.write_bytes(self.csv_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "source CSV"):
            main.run_pipeline(source, self.directory, llm=FakeQwen())


if __name__ == "__main__":
    unittest.main()
