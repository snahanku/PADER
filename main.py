import logging
import os
import re

import pandas as pd

from src.extractor_new import extract_pader_data
from src.pdf_loader import load_pdf_text

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("pader_pipeline")


def post_process_data(pader_data, raw_pdf_text: str):
    """Recovery for issues the schema's own validators can't fix on their own
    (mainly: recovering real dates when the LLM echoes a title-page
    placeholder). Note: the reaction/alert-totals arithmetic is now
    self-correcting inside schema_new.py's model_validators (triggered
    automatically via validate_assignment=True), so it no longer needs to be
    duplicated here."""

    # 1. DYNAMIC DATE RECOVERY (title page sometimes shows literal
    # placeholders like "<START_DATE>"; the real dates are in the
    # Introduction paragraph, e.g. "...from 2024-12-27 to 2025-12-26").
    meta = pader_data.metadata
    if "<" in meta.reporting_interval_start or "START" in meta.reporting_interval_start.upper():
        iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", raw_pdf_text)
        if len(iso_dates) >= 2:
            logger.info(
                "Recovered placeholder start date from body text: %s", iso_dates[0]
            )
            meta.reporting_interval_start = iso_dates[0]
            meta.reporting_interval_end = iso_dates[1]

    # NOTE: removed the old "swap serious/non-serious if serious < non-serious"
    # heuristic. That silently assumed serious counts are always >= non-serious
    # counts and rewrote data to match — an undocumented domain assumption
    # that could corrupt genuinely correct data. If that pattern shows up,
    # it's now surfaced as a warning by IntervalReactionTotals for a human to
    # check, not auto-corrected.

    # 2. Report any terms flagged incomplete/inconsistent by the schema, so
    # they're visible in pipeline logs, not just buried in Python warnings.
    if (
        pader_data.expected_unlabelled_term_count
        and len(pader_data.unlabelled_terms) != pader_data.expected_unlabelled_term_count
    ):
        logger.warning(
            "Document states %s unlabelled terms; extraction returned %s. "
            "Review report_output.md for completeness.",
            pader_data.expected_unlabelled_term_count,
            len(pader_data.unlabelled_terms),
        )

    return pader_data


def generate_markdown_report(pader_data) -> str:
    """Generates a structured report_output.md from extracted data."""
    meta = pader_data.metadata
    rx = pader_data.reaction_totals
    al = pader_data.alert_totals

    md = f"""# Pharmacovigilance Periodic Adverse Drug Experience Report (PADER)

## Executive Summary
- **Drug Name**: {pader_data.drug_name}
- **PADER Control Number**: {meta.pader_control_number}
- **Application Number**: {meta.application_number}
- **Applicant / Sponsor**: {meta.applicant_sponsor}
- **Reporting Period**: {meta.reporting_interval_start} to {meta.reporting_interval_end}

---

## Reaction & Alert Summary

### Interval Reaction Totals
| Metric | Count |
| :--- | :--- |
| **Total Serious Interval Reactions** | {rx.total_serious_interval:,} |
| **Total Non-Serious Interval Reactions** | {rx.total_nonserious_interval:,} |
| **Total Interval Reactions** | {rx.total_reactions:,} |

### 15-Day Alert Totals
| Category | Solicited (Study) | Solicited (Other) | Spontaneous | Total |
| :--- | ---: | ---: | ---: | ---: |
| **Serious, Unlabelled — Non-Fatal** | {al.serious_unlabelled_non_fatal_solicited_study:,} | {al.serious_unlabelled_non_fatal_solicited_other:,} | {al.serious_unlabelled_non_fatal_spontaneous:,} | {al.serious_unlabelled_non_fatal_total:,} |
| **Serious, Unlabelled — Fatal** | {al.serious_unlabelled_fatal_solicited_study:,} | {al.serious_unlabelled_fatal_solicited_other:,} | {al.serious_unlabelled_fatal_spontaneous:,} | {al.serious_unlabelled_fatal_total:,} |

**Total 15-Day Alerts: {al.total_15_day_alerts:,}**

---

## Unlabelled Adverse Event Summaries
_Document states {pader_data.expected_unlabelled_term_count or 'an unspecified number of'} unlabelled terms; {len(pader_data.unlabelled_terms)} extracted._

"""
    for term in pader_data.unlabelled_terms:
        md += f"### {term.preferred_term} (Cases: {term.case_count})\n"
        md += f"{term.clinical_narrative}\n\n"

    return md


def save_outputs(pader_data, output_dir="output"):
    """Creates output directory if not exists and writes JSON, CSV, and MD files."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "extracted_pader_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(pader_data.model_dump_json(indent=4))
    logger.info("JSON generated: %s", json_path)

    csv_path = os.path.join(output_dir, "extracted_unlabelled_terms.csv")
    df = pd.DataFrame([t.model_dump() for t in pader_data.unlabelled_terms])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("CSV generated: %s", csv_path)

    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_output.md")
    md_content = generate_markdown_report(pader_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report generated: %s", md_path)


def run_pipeline(pdf_path: str):
    if not os.path.exists(pdf_path):
        logger.error("Path not found: %s", pdf_path)
        return

    logger.info("Processing PDF: %s", pdf_path)
    raw_text = load_pdf_text(pdf_path)

    logger.info("Executing section-anchored extraction...")
    pader_data = extract_pader_data(raw_text)

    logger.info("Applying post-processing (date recovery, review flags)...")
    pader_data = post_process_data(pader_data, raw_text)

    logger.info("Saving all data to output directory...")
    save_outputs(pader_data)
    logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pdf")
        ]
        if files:
            run_pipeline(files[0])
        else:
            logger.error("No PDF files found in 'data/' directory.")
    else:
        logger.error("'data/' directory does not exist.")
