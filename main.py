import json
import os
import re
import pandas as pd
from src.extractor_new import extract_pader_data
from src.pdf_loader import load_pdf_text


def post_process_data(pader_data, raw_pdf_text: str):
  """Dynamic mathematical assertions & regex pattern recovery."""

  # 1. Date Recovery via Regex
  if (
      "<" in pader_data.metadata.reporting_interval_start
      or "YYYY" in pader_data.metadata.reporting_interval_start
  ):
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", raw_pdf_text)
    if len(iso_dates) >= 2:
      pader_data.metadata.reporting_interval_start = iso_dates[0]
      pader_data.metadata.reporting_interval_end = iso_dates[1]

  # 2. Mathematical Assertions: Reaction Totals
  rx = pader_data.reaction_totals
  if rx.total_serious_interval < rx.total_nonserious_interval:
    rx.total_serious_interval, rx.total_nonserious_interval = (
        rx.total_nonserious_interval,
        rx.total_serious_interval,
    )
  calculated_rx_total = rx.total_serious_interval + rx.total_nonserious_interval
  if rx.total_reactions != calculated_rx_total and calculated_rx_total > 0:
    rx.total_reactions = calculated_rx_total

  # 3. Mathematical Assertions: 15-Day Alerts
  al = pader_data.alert_totals
  if (
      al.serious_unlabelled_non_fatal < al.serious_unlabelled_fatal
      and al.serious_unlabelled_fatal > 0
  ):
    al.serious_unlabelled_non_fatal, al.serious_unlabelled_fatal = (
        al.serious_unlabelled_fatal,
        al.serious_unlabelled_non_fatal,
    )
  calculated_al_total = (
      al.serious_unlabelled_non_fatal + al.serious_unlabelled_fatal
  )
  if al.total_15_day_alerts != calculated_al_total and calculated_al_total > 0:
    al.total_15_day_alerts = calculated_al_total

  # 4. Deduplicate Preferred Terms
  unique_terms = []
  seen = set()
  for t in pader_data.unlabelled_terms:
    clean_name = t.preferred_term.strip()
    if clean_name.lower() not in seen and len(clean_name) > 0:
      seen.add(clean_name.lower())
      t.preferred_term = clean_name
      unique_terms.append(t)

  pader_data.unlabelled_terms = unique_terms
  return pader_data


def generate_markdown_report(pader_data) -> str:
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
| Category | Count |
| :--- | :--- |
| **Serious Unlabelled Non-Fatal Alerts** | {al.serious_unlabelled_non_fatal:,} |
| **Serious Unlabelled Fatal Alerts** | {al.serious_unlabelled_fatal:,} |
| **Total 15-Day Alerts** | {al.total_15_day_alerts:,} |

---

## Unlabelled Adverse Event Summaries

"""
  for term in pader_data.unlabelled_terms:
    md += f"### {term.preferred_term} (Cases: {term.case_count})\n"
    md += f"{term.clinical_narrative}\n\n"

  return md


def save_outputs(pader_data, output_dir="output"):
  os.makedirs(output_dir, exist_ok=True)

  # Save JSON
  json_path = os.path.join(output_dir, "extracted_pader_summary.json")
  with open(json_path, "w", encoding="utf-8") as f:
    f.write(pader_data.model_dump_json(indent=4))
  print(f"[SUCCESS] JSON generated: {json_path}")

  # Save CSV
  csv_path = os.path.join(output_dir, "extracted_unlabelled_terms.csv")
  df = pd.DataFrame([t.model_dump() for t in pader_data.unlabelled_terms])
  df.to_csv(csv_path, index=False, encoding="utf-8")
  print(f"[SUCCESS] CSV generated: {csv_path}")

  # Save Markdown Report
  md_path = os.path.join(output_dir, "report_output.md")
  md_content = generate_markdown_report(pader_data)
  with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
  print(f"[SUCCESS] Markdown report generated: {md_path}")


def run_pipeline(pdf_path: str):
  if not os.path.exists(pdf_path):
    print(f"[ERROR] PDF not found at {pdf_path}")
    return

  print(f"[INFO] Processing PDF: {pdf_path}")
  raw_text = load_pdf_text(pdf_path)

  print("[INFO] Executing Hybrid Extraction (pdfplumber + Ollama)...")
  pader_data = extract_pader_data(raw_text, pdf_path)

  print("[INFO] Applying Post-Processing & Mathematical Invariants...")
  pader_data = post_process_data(pader_data, raw_text)

  print("[INFO] Saving files to output/ directory...")
  save_outputs(pader_data)
  print("[COMPLETE] Pipeline execution successful.")


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
      print("[ERROR] No PDF found in 'data/' folder.")
  else:
    print("[ERROR] 'data/' folder missing.")