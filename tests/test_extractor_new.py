"""Offline extraction tests using synthetic cases and a fake Ollama client."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzer import analyze_csv
from src import extractor_new as extractor
from src.schema_new import AnalysisResults, GeneratedSection


record = runpy.run_path(str(ROOT / "tests" / "test_analyzer.py"))["record"]
AI_SECTIONS = (
    "narrative_summary", "case_summary", "reaction_analysis", "serious_cases", "trends",
)
ALL_SECTIONS = {
    "reporting_period", *AI_SECTIONS, "history_of_actions", "case_listing",
}


class FakeOllama:
    """Implements only the bound model interface; never accesses a server."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.bindings = []

    def bind(self, **options):
        self.bindings.append(options)
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected extra model invocation")
        return SimpleNamespace(content=self.responses.pop(0))


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "synthetic.csv"
        rows = [
            record("CASE_ONLY_ALPHA", receivedate="20250101",
                   patient_reaction_reactionmeddrapt="Headache,Nausea",
                   patient_reaction_reactionoutcome="fatal,unknown"),
            record("CASE_ONLY_BETA", receivedate="20250301",
                   serious="not serious", fulfillexpeditecriteria="no"),
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.analysis = AnalysisResults.model_validate(analyze_csv(path))

    def response(self, packet, *, text="The supplied evidence is available for human review.",
                 pointer=None):
        return {
            "section_id": packet["section_id"], "title": packet["title"],
            "statements": [{"text": text, "evidence": [{
                "analysis_pointer": pointer or packet["evidence"][0]["analysis_pointer"],
            }]}],
        }

    def responses(self):
        packets = extractor.build_evidence_packets(self.analysis)
        return [json.dumps(self.selection_response(packets[key])) for key in AI_SECTIONS]

    def selection_response(self, packet):
        return {"selected_observation_ids": [
            item["observation_id"] for item in packet["observations"][:3]
        ]}

    def count_packet(self):
        return {
            "section_id": "case_summary", "title": "Case summary",
            "instructions": "Describe only the supplied case counts.",
            "evidence": [
                {"analysis_pointer": "/summary/total_cases", "value": 2},
                {"analysis_pointer": "/summary/case_reaction_pairs", "value": 3},
            ],
        }

    def test_packets_cover_all_sections_and_omit_case_payloads(self):
        packets = extractor.build_evidence_packets(self.analysis)
        self.assertEqual(set(packets), ALL_SECTIONS)

        def check_compact(value):
            if isinstance(value, dict):
                self.assertFalse({"case_ids", "source_records", "all_source_records"} & set(value))
                for child in value.values():
                    check_compact(child)
            elif isinstance(value, list):
                for child in value:
                    check_compact(child)

        for key, packet in packets.items():
            with self.subTest(section=key):
                self.assertEqual(packet["section_id"], key)
                self.assertTrue(packet["title"])
                self.assertTrue(packet["instructions"])
                self.assertTrue(packet["evidence"])
                for evidence in packet["evidence"]:
                    check_compact(evidence["value"])
                serialized = json.dumps(packet)
                self.assertNotIn("CASE_ONLY_ALPHA", serialized)
                self.assertNotIn("CASE_ONLY_BETA", serialized)

    def test_packet_references_resolve_to_the_original_analysis(self):
        document = self.analysis.model_dump(mode="json")
        for packet in extractor.build_evidence_packets(self.analysis).values():
            for evidence in packet["evidence"]:
                pointer = evidence["analysis_pointer"]
                with self.subTest(pointer=pointer):
                    self.assertTrue(pointer.startswith("/"))
                    value = document
                    for token in pointer[1:].split("/"):
                        key = token.replace("~1", "/").replace("~0", "~")
                        value = value[int(key)] if isinstance(value, list) else value[key]

    def test_valid_cited_number_is_accepted(self):
        packet = self.count_packet()
        section = GeneratedSection.model_validate(self.response(packet, text="There are 2 cases."))
        self.assertEqual(extractor.validate_generated_section(section, packet), section)

    def test_number_must_be_supported_by_cited_evidence_not_other_packet_entries(self):
        packet = self.count_packet()
        # 3 exists in this packet but belongs to reaction pairs, not the cited total.
        for text in ("There are 3 cases.", "There are 999 cases."):
            with self.subTest(text=text):
                section = GeneratedSection.model_validate(self.response(packet, text=text))
                with self.assertRaises(ValueError):
                    extractor.validate_generated_section(section, packet)

    def test_alternative_numeric_formats_cannot_bypass_evidence_checks(self):
        packet = self.count_packet()
        for text in ("There are 9e9 cases.", "There are eleven cases."):
            with self.subTest(text=text):
                section = GeneratedSection.model_validate(self.response(packet, text=text))
                with self.assertRaises(ValueError):
                    extractor.validate_generated_section(section, packet)

    def test_pointer_must_be_supplied_to_the_current_section(self):
        packet = self.count_packet()
        for pointer in ("/summary/serious_cases", "/summary/invented"):
            with self.subTest(pointer=pointer):
                section = GeneratedSection.model_validate(self.response(packet, pointer=pointer))
                with self.assertRaises(ValueError):
                    extractor.validate_generated_section(section, packet)

    def test_wrong_section_and_title_are_rejected(self):
        packet = self.count_packet()
        for field, value in (("section_id", "trends"), ("title", "Invented title")):
            with self.subTest(field=field):
                data = self.response(packet)
                data[field] = value
                with self.assertRaises(ValueError):
                    extractor.validate_generated_section(GeneratedSection.model_validate(data), packet)

    def test_generation_uses_five_model_calls_and_returns_bound_draft(self):
        llm = FakeOllama(self.responses())
        report = extractor.generate_report(self.analysis, product="Synthetic product", llm=llm)
        self.assertEqual(len(llm.calls), 5)
        self.assertTrue(llm.bindings)
        self.assertTrue(all("format" in binding for binding in llm.bindings))
        self.assertEqual({section.section_id for section in report.sections}, ALL_SECTIONS)
        self.assertEqual(report.metadata.product, "Synthetic product")
        self.assertEqual(report.metadata.reporting_period, self.analysis.reporting_period)
        self.assertIsNone(report.metadata.application_number)
        self.assertIsNone(report.metadata.applicant_sponsor)
        self.assertEqual(report.analysis_sha256, self.analysis.content_sha256())
        self.assertEqual(report.review.status, "draft")
        self.assertIsNone(report.review.reviewer)
        report.validate_against(self.analysis)

    def test_malformed_reply_is_retried_without_restarting_completed_sections(self):
        replies = self.responses()
        replies.insert(1, "This is not a JSON response")
        llm = FakeOllama(replies)
        report = extractor.generate_report(self.analysis, llm=llm, max_retries=1)
        self.assertEqual(len(llm.calls), 6)
        self.assertFalse(llm.responses)
        self.assertEqual(len(report.sections), 8)

    def test_repeated_invalid_reply_fails_after_bounded_attempts(self):
        llm = FakeOllama(["invalid", "still invalid"])
        with self.assertRaises((ValueError, RuntimeError)):
            extractor.generate_report(self.analysis, llm=llm, max_retries=1)
        self.assertEqual(len(llm.calls), 2)

    def test_unknown_observation_selection_is_retried(self):
        invalid = {"selected_observation_ids": ["O01", "O02", "INVENTED"]}
        llm = FakeOllama([json.dumps(invalid), *self.responses()])
        report = extractor.generate_report(self.analysis, llm=llm, max_retries=1)
        self.assertEqual(len(llm.calls), 6)
        report.validate_against(self.analysis)

    def test_invalid_analysis_never_reaches_the_model(self):
        invalid = deepcopy(self.analysis.model_dump(mode="json"))
        invalid["summary"]["total_cases"] = 999
        llm = FakeOllama()
        with self.assertRaises(ValueError):
            extractor.generate_report(invalid, llm=llm)
        self.assertEqual(llm.calls, [])

    def test_oversized_full_request_fails_before_model_invocation(self):
        llm = FakeOllama()
        audit = {}
        with patch.object(extractor, "MAX_REQUEST_CHARS", 1):
            with self.assertRaises((ValueError, RuntimeError)):
                extractor.generate_report(self.analysis, llm=llm, audit=audit)
        self.assertEqual(llm.calls, [])
        self.assertEqual(audit["status"], "failed")

    def test_ai_cannot_supply_a_review_decision(self):
        packets = extractor.build_evidence_packets(self.analysis)
        invalid = self.selection_response(packets[AI_SECTIONS[0]])
        invalid["review"] = {"status": "approved"}
        llm = FakeOllama([json.dumps(invalid)])
        with self.assertRaises((ValueError, RuntimeError)):
            extractor.generate_report(self.analysis, llm=llm, max_retries=0)
        self.assertEqual(len(llm.calls), 1)

    def test_selected_observations_expand_to_exact_supplied_statements(self):
        packets = extractor.build_evidence_packets(self.analysis)
        llm = FakeOllama(self.responses())
        report = extractor.generate_report(self.analysis, llm=llm)
        for section in report.sections:
            if section.section_id not in AI_SECTIONS:
                continue
            with self.subTest(section=section.section_id):
                expected = packets[section.section_id]["observations"][:3]
                self.assertEqual([statement.text for statement in section.statements],
                                 [item["text"] for item in expected])
                self.assertEqual([statement.model_dump()["evidence"] for statement in section.statements],
                                 [item["evidence"] for item in expected])

    def test_duplicate_or_empty_selection_is_rejected(self):
        packets = extractor.build_evidence_packets(self.analysis)
        first_id = packets[AI_SECTIONS[0]]["observations"][0]["observation_id"]
        for selected in ([], [first_id, first_id, first_id]):
            with self.subTest(selected=selected):
                llm = FakeOllama([json.dumps({"selected_observation_ids": selected})])
                with self.assertRaises((ValueError, RuntimeError)):
                    extractor.generate_report(self.analysis, llm=llm, max_retries=0)
                self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
