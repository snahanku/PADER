"""Validate the CSV contracts using synthetic cases; no dataset is embedded."""

from copy import deepcopy
import csv
from datetime import datetime, timezone
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
schema = SimpleNamespace(**runpy.run_path(str(ROOT / "src" / "schema_new.py")))
analyzer = SimpleNamespace(**runpy.run_path(str(ROOT / "src" / "analyzer.py")))


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "synthetic.csv"
        rows = [
            {"safetyreportid": "001", "safetyreportversion": "1", "serious": "serious",
             "receivedate": "20250101", "fulfillexpeditecriteria": "yes",
             "patient_reaction_reactionmeddrapt": "Headache,Nausea",
             "patient_reaction_reactionoutcome": "fatal,unknown"},
            {"safetyreportid": "002", "safetyreportversion": "1", "serious": "not serious",
             "receivedate": "20250301", "fulfillexpeditecriteria": "no",
             "patient_reaction_reactionmeddrapt": "Headache",
             "patient_reaction_reactionoutcome": "recovered/resolved"},
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.result = analyzer.analyze_csv(path)

    def report_data(self, analysis):
        return {
            "metadata": {"product": "Synthetic product",
                         "reporting_period": analysis.reporting_period.model_dump()},
            "analysis_sha256": analysis.content_sha256(),
            "sections": [{"section_id": "case_summary", "title": "Case summary",
                          "statements": [{"text": "Two cases were received.",
                                          "evidence": [{"analysis_pointer": "/summary/total_cases"}]}]}],
        }

    def test_analyzer_contract_preserves_overlaps_unknowns_and_leading_zeros(self):
        analysis = schema.AnalysisResults.model_validate(self.result)
        self.assertEqual(analysis.case_ids, ["001", "002"])
        self.assertEqual(analysis.summary.total_cases, 2)
        self.assertEqual(sum(bucket.case_count for bucket in analysis.outcomes), 3)
        self.assertIsNone(analysis.case_listing[0].age_years)
        self.assertEqual(analysis.trends.monthly[1].change_from_previous_month, -1)
        self.assertIsNone(analysis.trends.monthly[2].percent_change)
        self.assertEqual(schema.AnalysisResults.model_validate_json(analysis.model_dump_json()), analysis)

    def test_strict_counts_reject_missing_negative_boolean_and_inconsistent_values(self):
        for value in (-1, True, "2", 3):
            with self.subTest(value=value):
                result = deepcopy(self.result)
                result["summary"]["total_cases"] = value
                with self.assertRaises(ValidationError):
                    schema.AnalysisResults.model_validate(result)
        result = deepcopy(self.result)
        del result["summary"]["serious_cases"]
        with self.assertRaises(ValidationError):
            schema.AnalysisResults.model_validate(result)

    def test_tampered_bucket_labels_percentages_and_unknown_ids_fail(self):
        for key, value in (("category", "Invented term"), ("percent_of_cases", 0),
                           ("case_ids", ["001", "999"])):
            with self.subTest(key=key):
                result = deepcopy(self.result)
                result["reactions"]["all"][0][key] = value
                with self.assertRaises(ValidationError):
                    schema.AnalysisResults.model_validate(result)

    def test_evidence_records_must_belong_to_the_case(self):
        result = deepcopy(self.result)
        result["case_listing"][0]["reactions"][0]["source_records"] = [2]
        with self.assertRaises(ValidationError):
            schema.AnalysisResults.model_validate(result)

    def test_date_validation_allows_unknowns_but_rejects_invalid_ranges(self):
        base = {"start": None, "end": None, "derived_from": "receivedate", "scope": "Observed"}
        self.assertIsNone(schema.ReportingPeriod.model_validate(base).start)
        for start, end in (("2025-02-30", "2025-03-01"),
                           ("2025-04-01", "2025-03-01"), (None, "2025-01-01")):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValidationError):
                    schema.ReportingPeriod.model_validate({**base, "start": start, "end": end})

    def test_draft_report_resolves_references_to_exact_analysis(self):
        analysis = schema.AnalysisResults.model_validate(self.result)
        data = self.report_data(analysis)
        report = schema.PADERReport.model_validate(data)
        self.assertEqual(report.review.status, "draft")
        self.assertIs(report.validate_against(analysis), report)
        for pointer in ("/summary/invented", "/case_listing/999/case_id",
                        "/case_listing/-1", "/case_listing/00", "/methods/bad~~0"):
            with self.subTest(pointer=pointer):
                bad = deepcopy(data)
                bad["sections"][0]["statements"][0]["evidence"][0]["analysis_pointer"] = pointer
                with self.assertRaisesRegex(ValueError, "Unresolved evidence"):
                    schema.PADERReport.model_validate(bad).validate_against(analysis)

    def test_analysis_hash_changes_when_configuration_changes(self):
        analysis = schema.AnalysisResults.model_validate(self.result)
        report = schema.PADERReport.model_validate(self.report_data(analysis))
        result = deepcopy(self.result)
        result["methods"]["country_field"] = "primarysource_reportercountry"
        changed = schema.AnalysisResults.model_validate(result)
        self.assertEqual(changed.provenance.sha256, analysis.provenance.sha256)
        self.assertNotEqual(changed.content_sha256(), analysis.content_sha256())
        with self.assertRaisesRegex(ValueError, "analysis hash"):
            report.validate_against(changed)

    def test_approval_requires_review_metadata_and_complete_sections(self):
        with self.assertRaises(ValidationError):
            schema.ReportReview(status="approved")
        analysis = schema.AnalysisResults.model_validate(self.result)
        data = self.report_data(analysis)
        data["review"] = {"status": "approved", "reviewer": "Test reviewer",
                          "reviewed_at": datetime.now(timezone.utc)}
        with self.assertRaisesRegex(ValidationError, "eight required sections"):
            schema.PADERReport.model_validate(data)
        template = data["sections"][0]
        data["sections"] = [{**deepcopy(template), "section_id": section_id} for section_id in (
            "reporting_period", "narrative_summary", "case_summary", "reaction_analysis",
            "serious_cases", "trends", "history_of_actions", "case_listing")]
        self.assertEqual(schema.PADERReport.model_validate(data).review.status, "approved")

    def test_generated_sections_cannot_include_approval_or_uncited_statements(self):
        section = self.report_data(schema.AnalysisResults.model_validate(self.result))["sections"][0]
        with self.assertRaises(ValidationError):
            schema.GeneratedSection.model_validate({**section, "review": {"status": "approved"}})
        section["statements"][0]["evidence"] = []
        with self.assertRaises(ValidationError):
            schema.GeneratedSection.model_validate(section)

    def test_legacy_models_remain_available_for_staged_migration(self):
        self.assertEqual(schema.ReactionTotals().total_reactions, 0)
        self.assertEqual(schema.AlertTotals().total_15_day_alerts, 0)
        self.assertEqual(schema.PADERTermsOnly(unlabelled_terms=[]).unlabelled_terms, [])


if __name__ == "__main__":
    unittest.main()
