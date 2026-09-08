"""Deterministic case analysis for the GenAR challenge (Python standard library).

Run from the project root:
    python -m src.analyzer --input "data/Bisoprolol_icsr_sample_1068rows (1).csv"

Returns evidence-backed aggregates, not generated clinical conclusions. Case-level
statistics use the highest numeric safetyreportversion for each safetyreportid.
Reaction statistics count distinct (case ID, preferred term) pairs from those
versions. The input CSV is never copied to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "safetyreportid", "serious", "receivedate",
    "patient_reaction_reactionmeddrapt",
}
OPTIONAL_COLUMNS = {
    "safetyreportversion", "fulfillexpeditecriteria", "occurcountry",
    "primarysource_reportercountry", "patient_patientonsetage",
    "patient_patientonsetageunit", "patient_patientsex",
    "patient_reaction_reactionoutcome", "primarysource_qualification",
}
SERIOUSNESS_FLAGS = (
    "seriousnessdeath", "seriousnesslifethreatening",
    "seriousnesshospitalization", "seriousnessdisabling",
    "seriousnesscongenitalanomali", "seriousnessother",
)
AGE_GROUPS = ("0-17", "18-44", "45-64", "65-74", "75+", "unknown")
UNKNOWN = "unknown"


def _text(value: str) -> str:
    return value.strip().casefold() or UNKNOWN


def _date(value: str) -> str | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _age_years(age: str, unit: str) -> float | None:
    # Unrecognized units (including undocumented numeric codes) stay unknown.
    factors = {"year": 1, "month": 1 / 12, "week": 7 / 365.25,
               "day": 1 / 365.25, "hour": 1 / (365.25 * 24)}
    try:
        value = float(age)
    except ValueError:
        return None
    factor = factors.get(unit.casefold().rstrip("s"))
    if factor is None or not math.isfinite(value) or value < 0:
        return None
    years = value * factor
    return years if years <= 120 else None


def _age_group(years: float | None) -> str:
    if years is None:
        return UNKNOWN
    for boundary, group in ((18, "0-17"), (45, "18-44"),
                            (65, "45-64"), (75, "65-74")):
        if years < boundary:
            return group
    return "75+"


def _distribution(pairs, denominator: int, categories=()) -> list[dict]:
    """Each bucket carries the unique case IDs supporting its count."""
    groups = {category: set() for category in categories}
    for category, case_id in pairs:
        groups.setdefault(category, set()).add(case_id)
    return [
        {"category": category, "case_count": len(ids),
         "percent_of_cases": round(100 * len(ids) / denominator, 2) if denominator else 0,
         "case_ids": sorted(ids)}
        for category, ids in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def _months(start: str, end: str) -> list[str]:
    year, month = map(int, start[:7].split("-"))
    result = []
    while f"{year:04d}-{month:02d}" <= end[:7]:
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def analyze_csv(csv_path: str | Path, *, country_field: str = "occurcountry",
                top_n: int = 10) -> dict:
    """Read a CSV and return JSON-serializable analyses and source evidence.

    Missing columns essential to case counting raise ValueError. Missing optional
    values and invalid classifications are retained as unknown and flagged.
    Conflicting case fields within the selected version also become unknown.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if country_field not in {"occurcountry", "primarysource_reportercountry"}:
        raise ValueError("Choose occurcountry or primarysource_reportercountry")
    path = Path(csv_path)
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV column names")
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    rows = []
    issues = []

    def issue(code, case_id, detail, records=()):
        issues.append({"code": code, "case_id": case_id,
                       "detail": detail, "source_records": list(records)})

    for record, raw in enumerate(reader, start=1):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"CSV record {record} has an unexpected number of columns")
        row = {key: value.strip() for key, value in raw.items()}
        if not row["safetyreportid"]:
            raise ValueError(f"CSV record {record} has no safetyreportid")
        row["_record"] = record
        version = row.get("safetyreportversion", "")
        if version and (not version.isdigit() or int(version) < 1):
            raise ValueError(f"CSV record {record} has invalid safetyreportversion")
        row["_version"] = int(version) if version else 0
        rows.append(row)
    if not rows:
        raise ValueError("CSV contains no data records")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["safetyreportid"]].append(row)
    cases, selected_records, excluded_records = [], [], []
    term_labels = {}

    for case_id, history in sorted(grouped.items()):
        highest = max(row["_version"] for row in history)
        selected = [row for row in history if row["_version"] == highest]
        records = [row["_record"] for row in selected]
        selected_records.extend(records)
        excluded_records.extend(row["_record"] for row in history if row["_version"] != highest)
        if any(row["_version"] == 0 for row in history):
            issue("missing_version", case_id, "Missing versions rank below numbered versions; if all are missing, all rows are retained.", records)

        def scalar(field, normalize=_text):
            values = {normalize(row.get(field, "")) for row in selected if row.get(field, "")}
            if len(values) > 1:
                issue("conflicting_field", case_id, f"{field}: conflicting selected-version values; retained as unknown.", records)
                return UNKNOWN
            return next(iter(values), UNKNOWN)

        serious_raw = scalar("serious")
        seriousness = {"serious": "serious", "not serious": "non-serious",
                       "non-serious": "non-serious"}.get(serious_raw, UNKNOWN)
        if seriousness == UNKNOWN:
            issue("unknown_seriousness", case_id, f"Unrecognized or missing serious value: {serious_raw}", records)
        expedite_raw = scalar("fulfillexpeditecriteria")
        expedited = expedite_raw if expedite_raw in {"yes", "no"} else UNKNOWN
        if expedited == UNKNOWN:
            issue("unknown_expedited", case_id, "Expedited status unavailable or unrecognized.", records)
        received_raw = scalar("receivedate", str.strip)
        received = _date(received_raw)
        if received is None:
            issue("invalid_receivedate", case_id, f"Cannot parse receivedate: {received_raw}", records)
        age = scalar("patient_patientonsetage", str.strip)
        unit = scalar("patient_patientonsetageunit")
        years = _age_years(age, unit)
        if years is None:
            issue("unknown_age", case_id, f"Age/unit unavailable, invalid or unsupported: {age} / {unit}", records)
        sex_raw = scalar("patient_patientsex")
        sex = sex_raw if sex_raw in {"male", "female", "other", UNKNOWN} else UNKNOWN
        if sex == UNKNOWN:
            issue("unknown_sex", case_id, "Sex unavailable or unrecognized.", records)
        country = scalar(country_field)
        if country == UNKNOWN:
            issue("unknown_country", case_id, f"No unambiguous {country_field} value.", records)
        flags = {field: scalar(field) for field in SERIOUSNESS_FLAGS}
        flags = {key: value if value in {"yes", "no"} else UNKNOWN for key, value in flags.items()}
        if seriousness == "non-serious" and "yes" in flags.values():
            issue("seriousness_disagreement", case_id, "Non-serious classification conflicts with a yes seriousness flag; not silently changed.", records)

        reactions = {}
        for row in selected:
            terms = [term.strip() for term in row["patient_reaction_reactionmeddrapt"].split(",")]
            outcomes = [item.strip().casefold() or UNKNOWN for item in row.get("patient_reaction_reactionoutcome", "").split(",")]
            if len(terms) != len(outcomes):
                issue("outcome_alignment", case_id, "Reaction/outcome list lengths differ; outcomes for this row are unknown rather than guessed.", [row["_record"]])
                outcomes = [UNKNOWN] * len(terms)
            for term, outcome in zip(terms, outcomes):
                if not term:
                    issue("missing_reaction", case_id, "Empty preferred term omitted from reaction analysis.", [row["_record"]])
                    continue
                key = term.casefold()
                term_labels.setdefault(key, term)
                entry = reactions.setdefault(key, {"outcomes": set(), "source_records": set()})
                entry["outcomes"].add(outcome)
                entry["source_records"].add(row["_record"])
        for key, entry in reactions.items():
            if len(entry["outcomes"]) > 1:
                issue("conflicting_outcome", case_id, f"Multiple outcomes for {term_labels[key]}; all retained.", sorted(entry["source_records"]))
        cases.append({
            "case_id": case_id, "report_version": highest or None,
            "source_records": records, "all_source_records": [row["_record"] for row in history],
            "seriousness": seriousness, "expedited": expedited, "receivedate": received,
            "age_years": round(years, 6) if years is not None else None,
            "age_group": _age_group(years), "sex": sex, "country": country,
            "reporter_type": scalar("primarysource_qualification"),
            "seriousness_flags": flags,
            "reactions": [{"preferred_term": term_labels[key],
                           "outcomes": sorted(entry["outcomes"]),
                           "source_records": sorted(entry["source_records"])}
                          for key, entry in sorted(reactions.items())],
        })

    n = len(cases)
    serious_cases = [case for case in cases if case["seriousness"] == "serious"]
    alert_cases = [case for case in cases if case["expedited"] == "yes"]

    def breakdown(field, population=None, categories=()):
        population = cases if population is None else population
        return _distribution(((case[field] or UNKNOWN, case["case_id"]) for case in population), len(population), categories)

    def reaction_counts(population):
        return _distribution(((reaction["preferred_term"], case["case_id"])
                              for case in population for reaction in case["reactions"]), len(population))

    def outcome_counts(population):
        return _distribution(((outcome, case["case_id"]) for case in population
                              for reaction in case["reactions"] for outcome in reaction["outcomes"]), len(population))

    dates = sorted(case["receivedate"] for case in cases if case["receivedate"])
    months = _months(dates[0], dates[-1]) if dates else []
    monthly = []
    for month in months:
        population = [case for case in cases if case["receivedate"] and case["receivedate"].startswith(month)]
        ids = sorted(case["case_id"] for case in population)
        previous = monthly[-1]["case_count"] if monthly else None
        monthly.append({
            "month": month, "case_count": len(ids), "case_ids": ids,
            "change_from_previous_month": len(ids) - previous if previous is not None else None,
            "percent_change": round(100 * (len(ids) - previous) / previous, 2) if previous else None,
            "seriousness": breakdown("seriousness", population, ("serious", "non-serious", UNKNOWN)),
            "country": breakdown("country", population), "outcomes": outcome_counts(population),
        })
    reactions = reaction_counts(cases)
    reaction_details = []
    for reaction in reactions[:top_n]:
        ids = set(reaction["case_ids"])
        population = [case for case in cases if case["case_id"] in ids]
        reaction_details.append({
            "preferred_term": reaction["category"], "age_group": breakdown("age_group", population, AGE_GROUPS),
            "sex": breakdown("sex", population),
            "monthly": [{"month": month, **_distribution(
                ((month, case["case_id"]) for case in population if case["receivedate"] and case["receivedate"].startswith(month)),
                len(population), (month,))[0]} for month in months],
        })
    summary = {
        "input_rows": len(rows), "total_cases": n,
        "selected_version_rows": len(selected_records), "excluded_older_version_rows": len(excluded_records),
        "duplicate_case_id_rows": len(rows) - n,
        "case_reaction_pairs": sum(len(case["reactions"]) for case in cases),
        "serious_cases": len(serious_cases),
        "non_serious_cases": sum(case["seriousness"] == "non-serious" for case in cases),
        "unknown_seriousness_cases": sum(case["seriousness"] == UNKNOWN for case in cases),
        "expedited_cases": len(alert_cases),
        "non_expedited_cases": sum(case["expedited"] == "no" for case in cases),
        "unknown_expedited_cases": sum(case["expedited"] == UNKNOWN for case in cases),
        "unknown_date_cases": sum(case["receivedate"] is None for case in cases),
        "cases_without_reactions": sum(not case["reactions"] for case in cases),
    }
    checks = {
        "seriousness_reconciles": sum(summary[key] for key in ("serious_cases", "non_serious_cases", "unknown_seriousness_cases")) == n,
        "expedited_reconciles": sum(summary[key] for key in ("expedited_cases", "non_expedited_cases", "unknown_expedited_cases")) == n,
        "monthly_reconciles": sum(item["case_count"] for item in monthly) + summary["unknown_date_cases"] == n,
        "reaction_counts_reconcile": sum(item["case_count"] for item in reactions) == summary["case_reaction_pairs"],
        "selected_and_excluded_rows_reconcile": len(selected_records) + len(excluded_records) == len(rows),
    }
    if not all(checks.values()):
        raise AssertionError(f"Analysis reconciliation failed: {checks}")
    return {
        "schema_version": "1.0", "review_status": "draft_requires_human_review",
        "provenance": {"source_file": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                       "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "source_record_numbering": "1-based CSV data records, excluding header; not physical line numbers"},
        "reporting_period": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None,
                             "derived_from": "receivedate of selected case versions",
                             "scope": "Observed data range; no external reporting window or new-case baseline supplied."},
        "summary": summary,
        "case_ids": sorted(grouped),
        "demographics": {"age_groups": breakdown("age_group", categories=AGE_GROUPS),
                         "sex": breakdown("sex", categories=("female", "male", "other", UNKNOWN)),
                         "country": breakdown("country"), "reporter_type": breakdown("reporter_type")},
        "seriousness": breakdown("seriousness", categories=("serious", "non-serious", UNKNOWN)),
        "seriousness_criteria": {field: _distribution(((case["seriousness_flags"][field], case["case_id"]) for case in cases), n, ("yes", "no", UNKNOWN)) for field in SERIOUSNESS_FLAGS},
        "reactions": {"all": reactions, "top": reactions[:top_n],
                      "serious": reaction_counts(serious_cases),
                      "top_serious": reaction_counts(serious_cases)[:top_n],
                      "top_reaction_breakdowns": reaction_details},
        "outcomes": outcome_counts(cases),
        "expedited": {"classification": "fulfillexpeditecriteria=yes; not inferred from seriousness or expectedness",
                      "status": breakdown("expedited", categories=("yes", "no", UNKNOWN)),
                      "case_ids": sorted(case["case_id"] for case in alert_cases),
                      "seriousness": breakdown("seriousness", alert_cases),
                      "reactions": reaction_counts(alert_cases), "outcomes": outcome_counts(alert_cases)},
        "trends": {"monthly": monthly,
                   "peak_months": [item["month"] for item in monthly if item["case_count"] == max(row["case_count"] for row in monthly)] if monthly else [],
                   "interpretation": "Descriptive counts only, not incidence, causality or a confirmed safety signal. Boundary months may be partial; a missing month means zero received cases in this file, not proven complete surveillance."},
        "history_of_actions": {"status": "not_provided", "statement": "No history-of-actions information was supplied for this exercise; this does not establish that no actions occurred."},
        "data_quality": {"missing_optional_columns": sorted((OPTIONAL_COLUMNS | set(SERIOUSNESS_FLAGS)) - set(columns)),
                         "issues": issues, "checks": checks, "excluded_older_source_records": sorted(excluded_records)},
        "methods": {
            "case_selection": "Highest numeric safetyreportversion per safetyreportid; retain all rows tied at that version; no fallback to older field values.",
            "reaction_counting": "Split comma-separated preferred terms; count each case once per case-insensitive term. Different terms and outcomes can overlap across cases.",
            "outcomes": "Align comma-separated outcomes with terms only when lengths match; count unique cases per outcome. Outcome buckets are non-exclusive and need not sum to total cases.",
            "country_field": country_field,
            "age_groups": "Numeric onset age converted from named years/months/weeks/days/hours; unknown or unsupported units and ages outside 0-120 years remain unknown.",
            "case_conflicts": "Conflicting nonempty scalar values within the selected version become unknown with a review flag.",
            "seriousness": "Use serious field without guessing or swapping counts; individual seriousness criteria are non-exclusive.",
            "limitations": ["No product label/CCDS: expectedness is out of scope.",
                            "No supplied SOC mapping: analyze preferred terms only.",
                            "No prior-period baseline: cannot distinguish newly created cases from follow-up reports.",
                            "Drug/indication list alignment is not assumed; no product-specific causality is inferred.",
                            "No submission-timeliness assessment or invented clinical narratives."]},
        "case_listing": cases,
    }


def write_outputs(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """Write analysis and a case-level listing without touching existing reports."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    analysis_path = directory / "analysis_results.json"
    listing_path = directory / "case_listing.csv"
    analysis_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["case_id", "report_version", "receivedate", "seriousness", "expedited",
              "age_years", "age_group", "sex", "country", "reporter_type", "reactions", "source_records"]
    with listing_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in result["case_listing"]:
            row = {key: case[key] for key in fields}
            for key in ("reactions", "source_records"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    return analysis_path, listing_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to the local source CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--country-field", choices=("occurcountry", "primarysource_reportercountry"), default="occurcountry")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    try:
        result = analyze_csv(args.input, country_field=args.country_field, top_n=args.top_n)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    paths = write_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], indent=2))
    print(f"Review flags: {len(result['data_quality']['issues'])}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
