"""Synthetic tests only; the evaluation dataset is not embedded in tests."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "src" / "analyzer.py"
SPEC = importlib.util.spec_from_file_location("analyzer", MODULE)
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def record(case_id, **changes):
    row = {
        "safetyreportid": case_id, "safetyreportversion": "1",
        "serious": "serious", "fulfillexpeditecriteria": "yes",
        "receivedate": "20250101", "occurcountry": "UK",
        "patient_patientonsetage": "65", "patient_patientonsetageunit": "year",
        "patient_patientsex": "female", "primarysource_qualification": "physician",
        "patient_reaction_reactionmeddrapt": "Headache",
        "patient_reaction_reactionoutcome": "recovered/resolved",
    }
    row.update({flag: "no" for flag in analyzer.SERIOUSNESS_FLAGS})
    row.update(changes)
    return row


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.csv"

    def analyze(self, rows, **options):
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with self.path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return analyzer.analyze_csv(self.path, **options)

    def test_latest_version_and_multi_reactions_are_not_extra_cases(self):
        result = self.analyze([
            record("A", patient_reaction_reactionmeddrapt="Old term"),
            record("A", safetyreportversion="2", occurcountry="France",
                   patient_reaction_reactionmeddrapt="Headache,Nausea,Headache",
                   patient_reaction_reactionoutcome="fatal,unknown,fatal"),
            record("B", serious="not serious", fulfillexpeditecriteria="no"),
        ])
        self.assertEqual(result["summary"]["total_cases"], 2)
        self.assertEqual(result["summary"]["serious_cases"], 1)
        self.assertEqual(result["summary"]["case_reaction_pairs"], 3)
        self.assertEqual(result["data_quality"]["excluded_older_source_records"], [1])
        counts = {row["category"]: row["case_count"] for row in result["reactions"]["all"]}
        self.assertEqual(counts, {"Headache": 2, "Nausea": 1})
        self.assertEqual(result["case_listing"][0]["country"], "france")
        self.assertTrue(all(result["data_quality"]["checks"].values()))

    def test_same_version_duplicate_and_conflicting_fields(self):
        original = record("A")
        result = self.analyze([original, original, record("A", serious="not serious", occurcountry="France")])
        self.assertEqual(result["summary"]["total_cases"], 1)
        self.assertEqual(result["summary"]["case_reaction_pairs"], 1)
        self.assertEqual(result["summary"]["unknown_seriousness_cases"], 1)
        self.assertEqual(result["case_listing"][0]["country"], "unknown")

    def test_missing_and_invalid_values_stay_unknown(self):
        result = self.analyze([record("A", serious="unexpected", receivedate="20250230",
                                     fulfillexpeditecriteria="", patient_patientsex="",
                                     patient_patientonsetageunit="800")])
        self.assertEqual(result["summary"]["unknown_seriousness_cases"], 1)
        self.assertEqual(result["summary"]["unknown_date_cases"], 1)
        self.assertEqual(result["case_listing"][0]["age_group"], "unknown")
        self.assertEqual(result["trends"]["monthly"], [])
        self.assertTrue(result["data_quality"]["issues"])

    def test_age_units_and_boundaries(self):
        rows = [record("A", patient_patientonsetage="18"),
                record("B", patient_patientonsetage="11", patient_patientonsetageunit="month"),
                record("C", patient_patientonsetage="75"),
                record("D", patient_patientonsetage="NaN"),
                record("E", patient_patientonsetage="-1")]
        result = self.analyze(rows)
        groups = {case["case_id"]: case["age_group"] for case in result["case_listing"]}
        self.assertEqual(groups, {"A": "18-44", "B": "0-17", "C": "75+", "D": "unknown", "E": "unknown"})
        json.dumps(result, allow_nan=False)

    def test_month_gaps_zero_baseline_and_year_rollover(self):
        result = self.analyze([record("A", receivedate="20241227"), record("B", receivedate="20250201")])
        months = result["trends"]["monthly"]
        self.assertEqual([row["month"] for row in months], ["2024-12", "2025-01", "2025-02"])
        self.assertEqual([row["case_count"] for row in months], [1, 0, 1])
        self.assertIsNone(months[2]["percent_change"])

    def test_outcome_alignment_does_not_broadcast_one_outcome(self):
        result = self.analyze([record("A", patient_reaction_reactionmeddrapt="Headache,Nausea",
                                     patient_reaction_reactionoutcome="fatal")])
        self.assertEqual(result["outcomes"][0]["category"], "unknown")
        self.assertEqual(result["outcomes"][0]["case_count"], 1)
        self.assertIn("outcome_alignment", {issue["code"] for issue in result["data_quality"]["issues"]})

    def test_seriousness_flags_overlap_and_do_not_override_classification(self):
        result = self.analyze([record("A", serious="not serious", seriousnessdeath="yes", seriousnesshospitalization="yes")])
        self.assertEqual(result["summary"]["non_serious_cases"], 1)
        for field in ("seriousnessdeath", "seriousnesshospitalization"):
            self.assertEqual(result["seriousness_criteria"][field][0]["case_count"], 1)
        self.assertIn("seriousness_disagreement", {issue["code"] for issue in result["data_quality"]["issues"]})

    def test_case_id_and_required_columns_are_validated(self):
        with self.assertRaisesRegex(ValueError, "no safetyreportid"):
            self.analyze([record("")])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            self.analyze([{"safetyreportid": "A"}])
        with self.assertRaisesRegex(ValueError, "invalid safetyreportversion"):
            self.analyze([record("A", safetyreportversion="invalid")])

    def test_output_preserves_report_and_contains_traceable_listing(self):
        result = self.analyze([record("001")])
        output = Path(self.temp.name) / "output"
        output.mkdir()
        report = output / "report_output.md"
        report.write_text("Existing report", encoding="utf-8")
        analysis_path, listing_path = analyzer.write_outputs(result, output)
        self.assertEqual(report.read_text(encoding="utf-8"), "Existing report")
        self.assertEqual(json.loads(analysis_path.read_text(encoding="utf-8"))["case_ids"], ["001"])
        with listing_path.open(encoding="utf-8", newline="") as stream:
            listing = list(csv.DictReader(stream))
        self.assertEqual(listing[0]["case_id"], "001")
        self.assertEqual(json.loads(listing[0]["source_records"]), [1])


if __name__ == "__main__":
    unittest.main()
