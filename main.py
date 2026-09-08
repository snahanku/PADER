"""CSV -> validated analysis -> Qwen observation selection -> draft PADER report.

    python main.py
    python main.py --input "data/Bisoprolol_icsr_sample_1068rows (1).csv"
    python main.py --review approve --reviewer "Snahanku Karar"

The reference PDF supplies no figures or clinical narratives to this workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import logging
import os
from pathlib import Path
import re
import tempfile

from src.analyzer import analyze_csv, write_outputs
from src.extractor_new import DEFAULT_MODEL, SECTION_SPECS, generate_report
from src.schema_new import AnalysisResults, PADERReport, ReportReview


ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
OUTPUT_FILES = (
    "analysis_results.json", "case_listing.csv", "report_draft.json",
    "generation_audit.json", "report_output.md", "evidence.md",
)


def find_input_csv(data_dir: Path) -> Path:
    files = sorted(path for path in Path(data_dir).glob("*") if path.is_file() and path.suffix.lower() == ".csv")
    if len(files) != 1:
        raise ValueError(f"Expected one CSV in {data_dir}; found {len(files)}. Specify --input with the intended source CSV.")
    return files[0]


def _escape(value) -> str:
    """Keep source categories and reviewer text from changing Markdown structure."""
    text = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]<>|#])", r"\\\1", text)


def _count(value: int) -> str:
    return f"{value:,}"


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(_escape(item) for item in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_escape(item) for item in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


class EvidenceIndex:
    def __init__(self):
        self.pointers = {}

    def link(self, pointer: str) -> str:
        number = self.pointers.setdefault(pointer, len(self.pointers) + 1)
        return f"[E{number}](evidence.md#e{number})"


def _validated_pair(report, analysis):
    analysis = AnalysisResults.model_validate(analysis.model_dump(mode="json"))
    report = PADERReport.model_validate(report.model_dump(mode="json"))
    report.validate_against(analysis)
    if {section.section_id for section in report.sections} != set(SECTION_SPECS):
        raise ValueError("Report generation requires all eight sections")
    return report, analysis


def _render_report(report: PADERReport, analysis: AnalysisResults):
    report, analysis = _validated_pair(report, analysis)
    evidence = EvidenceIndex()
    summary = analysis.summary
    metadata = report.metadata
    period = analysis.reporting_period
    status = {"draft": "DRAFT — requires human review", "flagged": "FLAGGED — requires revision",
              "approved": "APPROVED — reviewer decision recorded"}[report.review.status]
    parts = ["# Pharmacovigilance Periodic Adverse Drug Experience Report (PADER)",
             f"**Status: {status}**",
             "Prepared for the GenAR engineering evaluation. This is an evidence-based exercise report, not a regulatory submission."]
    if report.review.status != "draft":
        parts.append(f"Reviewer: {_escape(report.review.reviewer)}\n\nReviewed at: {report.review.reviewed_at.isoformat()}")
        if report.review.notes:
            parts.append(f"Review notes: {_escape(report.review.notes)}")
    parts.append("Qwen selected and ordered calculated observations. Python retained their wording, citations and all tables. Review source-data quality and selection coverage before approving.")

    def note(text, pointer):
        parts.append(f"{text} {evidence.link(pointer)}")

    def distribution(title, buckets, pointer, denominator, *, ordered=None):
        parts.append(f"### {title}")
        selected = buckets if ordered is None else sorted(buckets, key=lambda item: ordered.index(item.category) if item.category in ordered else len(ordered))
        if selected:
            parts.append(_table(("Category", "Cases", "% of denominator"),
                                ((item.category, _count(item.case_count), f"{item.percent_of_cases:.2f}%") for item in selected)))
        else:
            parts.append("No populated categories are available.")
        note(f"Denominator: {denominator}.", pointer)

    sections = {section.section_id: section for section in report.sections}
    for section_id, (title, _) in SECTION_SPECS.items():
        parts.append(f"## {title}")
        if section_id == "reporting_period":
            parts.append(_table(("Field", "Value"), (
                ("Product", metadata.product), ("Report type", metadata.report_type),
                ("Application identifier", metadata.application_number or "Not supplied"),
                ("Applicant / sponsor", metadata.applicant_sponsor or "Not supplied"),
                ("Observed receivedate range", f"{period.start} to {period.end}" if period.start else "Unavailable"),
                ("Latest observed receivedate", period.end or "Unavailable"),
                ("Input file", analysis.provenance.source_file),
            )))
            parts.append("Product is supplied as the reporting task's configuration. The application identifier and sponsor are not inferred from the reference PDF. An externally defined reporting window or data cut-off was not supplied.")
        for statement in sections[section_id].statements:
            links = " ".join(evidence.link(ref.analysis_pointer) for ref in statement.evidence)
            parts.append(f"{_escape(statement.text)} {links}")

        if section_id == "narrative_summary":
            parts.append(_table(("Metric", "Count"), (
                ("Input CSV rows", _count(summary.input_rows)),
                ("Unique cases", _count(summary.total_cases)),
                ("Serious cases", _count(summary.serious_cases)),
                ("Non-serious cases", _count(summary.non_serious_cases)),
                ("Unknown seriousness", _count(summary.unknown_seriousness_cases)),
                ("Case-reaction pairs", _count(summary.case_reaction_pairs)),
                ("Selected-version rows", _count(summary.selected_version_rows)),
                ("Excluded older-version rows", _count(summary.excluded_older_version_rows)),
            )))
            note("Rows, unique cases and case-reaction pairs are different counting units.", "/summary")
            note(_escape(analysis.methods.case_selection), "/methods/case_selection")

        elif section_id == "case_summary":
            denominator = f"all {_count(summary.total_cases)} unique cases"
            distribution("Age groups", analysis.demographics.age_groups, "/demographics/age_groups", denominator,
                         ordered=["0-17", "18-44", "45-64", "65-74", "75+", "unknown"])
            note(_escape(analysis.methods.age_groups), "/methods/age_groups")
            distribution("Sex", analysis.demographics.sex, "/demographics/sex", denominator)
            distribution("Country", analysis.demographics.country, "/demographics/country", denominator)
            note(f"Country field: `{analysis.methods.country_field}`. Source category codes are retained.", "/methods/country_field")
            distribution("Reporter type", analysis.demographics.reporter_type, "/demographics/reporter_type", denominator)
            distribution("Outcomes", analysis.outcomes, "/outcomes", denominator)
            note(_escape(analysis.methods.outcomes), "/methods/outcomes")
            parts.append("A case may appear in multiple outcome categories. These counts and percentages should not be added to obtain the total number of cases.")

        elif section_id == "reaction_analysis":
            distribution("Most common reactions", analysis.reactions.top, "/reactions/top", f"all {_count(summary.total_cases)} unique cases")
            distribution("Most common reactions among serious cases", analysis.reactions.top_serious, "/reactions/top_serious", f"{_count(summary.serious_cases)} serious cases")
            note(_escape(analysis.methods.reaction_counting), "/methods/reaction_counting")
            parts.append("Reaction categories overlap. A reaction's count is the number of distinct selected-version cases containing that preferred term. Full rankings and reaction-by-age, sex and month breakdowns are available in the analysis artifact.")
            note("Expectedness and System Organ Class classification are out of scope because the required label and SOC mapping were not supplied.", "/methods/limitations")

        elif section_id == "serious_cases":
            distribution("Expedited flag", analysis.expedited.status, "/expedited/status", f"all {_count(summary.total_cases)} unique cases")
            distribution("Seriousness among expedited cases", analysis.expedited.seriousness, "/expedited/seriousness", f"{_count(summary.expedited_cases)} expedited cases")
            distribution("Outcomes among expedited cases", analysis.expedited.outcomes, "/expedited/outcomes", f"{_count(summary.expedited_cases)} expedited cases")
            parts.append("### Independent seriousness criteria")
            flag_labels = {"seriousnessdeath": "Death", "seriousnesslifethreatening": "Life-threatening",
                           "seriousnesshospitalization": "Hospitalization", "seriousnessdisabling": "Disability",
                           "seriousnesscongenitalanomali": "Congenital anomaly", "seriousnessother": "Other"}
            rows = []
            for key, label in flag_labels.items():
                values = {item.category: item.case_count for item in analysis.seriousness_criteria[key]}
                rows.append((label, _count(values.get("yes", 0)), _count(values.get("no", 0)), _count(values.get("unknown", 0))))
            parts.append(_table(("Criterion", "Yes", "No", "Unknown"), rows))
            note("Criteria are independent and may overlap; they are not a partition of the case population.", "/seriousness_criteria")
            note(_escape(analysis.expedited.classification), "/expedited/classification")
            parts.append("The expedited flag is the exercise's basis for alert analysis. It does not establish unlabelled status, causality, or submission within 15 days. No submission-timeliness assessment is performed.")

        elif section_id == "trends":
            parts.append("### Monthly received-case counts")
            parts.append(_table(("Month", "Cases", "Change from previous month", "% change"), (
                (month.month, _count(month.case_count),
                 f"{month.change_from_previous_month:+,}" if month.change_from_previous_month is not None else "Not applicable",
                 f"{month.percent_change:+.2f}%" if month.percent_change is not None else "Not applicable")
                for month in analysis.trends.monthly)))
            note("Months use receivedate from the selected case version. Percentage change is undefined for the first month or a zero previous-month count.", "/trends/monthly")
            note(f"Cases with unknown receivedate: {_count(summary.unknown_date_cases)}.", "/summary/unknown_date_cases")
            note("Peak observed month(s): " + (", ".join(analysis.trends.peak_months) or "Unavailable") + ".", "/trends/peak_months")
            note(_escape(analysis.trends.interpretation), "/trends/interpretation")

        elif section_id == "case_listing":
            parts.append("[Open the case listing](case_listing.csv) · [Full analysis and source references](analysis_results.json) · [Evidence index](evidence.md) · [Qwen prompts and responses](generation_audit.json)")
            parts.append("The listing includes case ID, report version, receivedate, seriousness, expedited status, age, sex, country, reactions, outcomes and source-record references. Nested reaction/outcome details are JSON within CSV cells. Source records refer to data records in the local CSV, excluding its header.")

    parts.append("## Data quality, limitations and review")
    issue_counts = Counter(issue.code for issue in analysis.data_quality.issues)
    if issue_counts:
        parts.append(_table(("Review flag", "Occurrences"), ((key, _count(value)) for key, value in sorted(issue_counts.items()))))
        parts.append("Flag occurrences are not a count of unique affected cases; one case can have multiple flags.")
    else:
        parts.append("No analyzer review flags were recorded.")
    note("Inspect case-level details before interpreting missing or inconsistent data.", "/data_quality")
    for limitation in analysis.methods.limitations:
        parts.append(f"- {_escape(limitation)}")
    parts.append("Before approval, review the data-quality flags, denominators, excluded versions, cited evidence and selected observations. The review command records a local reviewer decision; it does not authenticate the reviewer or certify regulatory compliance.")
    parts.append(f"Dataset SHA-256: `{analysis.provenance.sha256}`\n\nAnalysis SHA-256: `{report.analysis_sha256}`")
    return "\n\n".join(parts).rstrip() + "\n", evidence


def generate_markdown_report(report: PADERReport, analysis: AnalysisResults) -> str:
    return _render_report(report, analysis)[0]


def _render_evidence(analysis: AnalysisResults, index: EvidenceIndex) -> str:
    data = analysis.model_dump(mode="json")
    parts = ["# Evidence index", f"Analysis SHA-256: `{analysis.content_sha256()}`",
             "Each entry shows the cited value from [analysis_results.json](analysis_results.json). Aggregate case_ids link the count to [case_listing.csv](case_listing.csv); case records retain the original CSV record numbers. For summary counts, use the full case_listing and case_ids fields in the analysis. This index validates traceability, not causal interpretation."]
    for pointer, number in index.pointers.items():
        value = data
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
        text = json.dumps(value, indent=2, ensure_ascii=False)
        fence = "`" * max(3, 1 + max((len(match) for match in re.findall(r"`+", text)), default=0))
        parts.extend((f"## E{number}", f"JSON Pointer: `{pointer}`", f"{fence}json\n{text}\n{fence}"))
    return "\n\n".join(parts) + "\n"


def save_outputs(report: PADERReport, analysis: AnalysisResults, audit: dict, output_dir="output"):
    """Prepare all files before replacing results from a previous successful run."""
    report, analysis = _validated_pair(report, analysis)
    markdown, index = _render_report(report, analysis)
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pader-run-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        write_outputs(analysis.model_dump(mode="json"), staging)
        (staging / "report_draft.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (staging / "generation_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (staging / "report_output.md").write_text(markdown, encoding="utf-8")
        (staging / "evidence.md").write_text(_render_evidence(analysis, index), encoding="utf-8")
        directory.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_FILES:
            os.replace(staging / name, directory / name)
    return directory / "report_output.md"


def run_pipeline(csv_path, output_dir="output", *, product="Bisoprolol", model=DEFAULT_MODEL, llm=None) -> PADERReport:
    source = Path(csv_path).resolve()
    directory = Path(output_dir).resolve()
    if source.suffix.lower() != ".csv":
        raise ValueError("The input must be the case-level CSV, not the reference PDF")
    if source in {directory / name for name in OUTPUT_FILES}:
        raise ValueError("The source CSV must not be one of the generated output files")
    logger.info("Analyzing %s", source.name)
    analysis = AnalysisResults.model_validate(analyze_csv(source))
    logger.info("Validated %s unique cases from %s rows", analysis.summary.total_cases, analysis.summary.input_rows)
    audit = {}
    try:
        report = generate_report(analysis, product=product, model=model, llm=llm, audit=audit)
        destination = save_outputs(report, analysis, audit, directory)
    except Exception as error:
        if audit:
            directory.mkdir(parents=True, exist_ok=True)
            failure = {"failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": str(error), "generation": audit}
            (directory / "last_generation_failure.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    logger.info("Saved draft report: %s", destination)
    return report


def review_report(output_dir, decision, reviewer, notes="") -> PADERReport:
    """Record an explicit local human decision without invoking Qwen."""
    directory = Path(output_dir)
    analysis = AnalysisResults.model_validate_json((directory / "analysis_results.json").read_text(encoding="utf-8"))
    report = PADERReport.model_validate_json((directory / "report_draft.json").read_text(encoding="utf-8"))
    canonical = generate_markdown_report(report, analysis)
    if (directory / "report_output.md").read_text(encoding="utf-8") != canonical:
        raise ValueError("Markdown differs from the generated report. Reconcile edits with report_draft.json before recording review.")
    if decision not in {"approve", "flag"}:
        raise ValueError("Review decision must be approve or flag")
    report.review = ReportReview(status="approved" if decision == "approve" else "flagged",
                                 reviewer=reviewer, reviewed_at=datetime.now(timezone.utc), notes=notes)
    report = PADERReport.model_validate(report.model_dump(mode="json"))
    audit = json.loads((directory / "generation_audit.json").read_text(encoding="utf-8"))
    save_outputs(report, analysis, audit, directory)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local source CSV; defaults to the only CSV in data/")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--product", default="Bisoprolol")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--review", choices=("approve", "flag"))
    parser.add_argument("--reviewer")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        if args.review:
            if not args.reviewer or not args.reviewer.strip():
                raise ValueError("--review requires a nonempty --reviewer")
            if args.input:
                raise ValueError("Review existing outputs separately from generating a new report")
            report = review_report(args.output_dir, args.review, args.reviewer, args.notes)
            print(f"Review recorded: {report.review.status}")
        else:
            if args.reviewer or args.notes:
                raise ValueError("--reviewer and --notes require --review")
            run_pipeline(args.input or find_input_csv(ROOT / "data"), args.output_dir,
                         product=args.product, model=args.model)
            print(f"Draft ready for review: {args.output_dir / 'report_output.md'}")
    except Exception as error:
        logger.debug("Pipeline error", exc_info=True)
        parser.exit(1, f"Pipeline failed: {error}\n")


if __name__ == "__main__":
    main()
