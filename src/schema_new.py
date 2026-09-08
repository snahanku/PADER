"""Data contracts for CSV analysis and evidence-backed report generation.

Use AnalysisResults.model_validate(analyze_csv(path)) at the pipeline boundary.
Use GeneratedSection as the AI response schema, then PADERReport.validate_against
to check evidence pointers against the analysis. These checks establish structure
and reference integrity; they do not prove that prose is supported by its evidence.

The PDF models at the end are temporary compatibility models for extractor_new.py.
The new CSV workflow must use AnalysisResults and PADERReport instead.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Annotated, List, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


Count = Annotated[int, Field(strict=True, ge=0)]
RecordNumber = Annotated[int, Field(strict=True, ge=1)]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
CaseID = Text  # Keep identifiers as strings, including leading zeros.
Percent = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Month = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
Seriousness = Literal["serious", "non-serious", "unknown"]
Flag = Literal["yes", "no", "unknown"]
AgeGroup = Literal["0-17", "18-44", "45-64", "65-74", "75+", "unknown"]


def _valid_date(value: str) -> str:
    date.fromisoformat(value)
    return value


ISODate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(_valid_date)]


def _unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")


class CSVModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AnalysisProvenance(CSVModel):
    source_file: Text
    sha256: Sha256
    generated_at_utc: datetime
    source_record_numbering: Text


class ReportingPeriod(CSVModel):
    start: ISODate | None
    end: ISODate | None
    derived_from: Text
    scope: Text

    @model_validator(mode="after")
    def check_dates(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Reporting start and end must both be known or both be null")
        if self.start is not None and self.start > self.end:
            raise ValueError("Reporting start cannot be after reporting end")
        return self


class AnalysisSummary(CSVModel):
    input_rows: Count
    total_cases: Count
    selected_version_rows: Count
    excluded_older_version_rows: Count
    duplicate_case_id_rows: Count
    case_reaction_pairs: Count
    serious_cases: Count
    non_serious_cases: Count
    unknown_seriousness_cases: Count
    expedited_cases: Count
    non_expedited_cases: Count
    unknown_expedited_cases: Count
    unknown_date_cases: Count
    cases_without_reactions: Count

    @model_validator(mode="after")
    def reconcile(self):
        if self.serious_cases + self.non_serious_cases + self.unknown_seriousness_cases != self.total_cases:
            raise ValueError("Seriousness counts must sum to total_cases")
        if self.expedited_cases + self.non_expedited_cases + self.unknown_expedited_cases != self.total_cases:
            raise ValueError("Expedited counts must sum to total_cases")
        if self.selected_version_rows + self.excluded_older_version_rows != self.input_rows:
            raise ValueError("Selected and excluded rows must sum to input_rows")
        if self.input_rows - self.total_cases != self.duplicate_case_id_rows:
            raise ValueError("duplicate_case_id_rows must equal input_rows minus total_cases")
        if self.selected_version_rows < self.total_cases:
            raise ValueError("Every case must have at least one selected row")
        if max(self.unknown_date_cases, self.cases_without_reactions) > self.total_cases:
            raise ValueError("Unknown/missing case counts cannot exceed total_cases")
        return self


class EvidenceCount(CSVModel):
    case_count: Count
    case_ids: list[CaseID]

    @model_validator(mode="after")
    def check_support(self):
        _unique(self.case_ids, "case_ids")
        if self.case_count != len(self.case_ids):
            raise ValueError("case_count must equal the number of supporting case IDs")
        return self


class CaseDistribution(EvidenceCount):
    category: Text
    percent_of_cases: Percent


class Demographics(CSVModel):
    age_groups: list[CaseDistribution]
    sex: list[CaseDistribution]
    country: list[CaseDistribution]
    reporter_type: list[CaseDistribution]


class ReactionMonth(CaseDistribution):
    month: Month


class ReactionBreakdown(CSVModel):
    preferred_term: Text
    age_group: list[CaseDistribution]
    sex: list[CaseDistribution]
    monthly: list[ReactionMonth]


class ReactionAnalysis(CSVModel):
    all: list[CaseDistribution]
    top: list[CaseDistribution]
    serious: list[CaseDistribution]
    top_serious: list[CaseDistribution]
    top_reaction_breakdowns: list[ReactionBreakdown]


class ExpeditedAnalysis(CSVModel):
    classification: Text
    status: list[CaseDistribution]
    case_ids: list[CaseID]
    seriousness: list[CaseDistribution]
    reactions: list[CaseDistribution]
    outcomes: list[CaseDistribution]


class MonthlyAnalysis(EvidenceCount):
    month: Month
    change_from_previous_month: Annotated[int, Field(strict=True)] | None
    percent_change: Annotated[float, Field(allow_inf_nan=False)] | None
    seriousness: list[CaseDistribution]
    country: list[CaseDistribution]
    outcomes: list[CaseDistribution]


class TrendAnalysis(CSVModel):
    monthly: list[MonthlyAnalysis]
    peak_months: list[Month]
    interpretation: Text


class HistoryOfActions(CSVModel):
    status: Literal["not_provided"]
    statement: Text


class DataQualityIssue(CSVModel):
    code: Text
    case_id: CaseID
    detail: Text
    source_records: list[RecordNumber]


class ReconciliationChecks(CSVModel):
    seriousness_reconciles: bool
    expedited_reconciles: bool
    monthly_reconciles: bool
    reaction_counts_reconcile: bool
    selected_and_excluded_rows_reconcile: bool


class DataQuality(CSVModel):
    missing_optional_columns: list[Text]
    issues: list[DataQualityIssue]
    checks: ReconciliationChecks
    excluded_older_source_records: list[RecordNumber]


class AnalysisMethods(CSVModel):
    case_selection: Text
    reaction_counting: Text
    outcomes: Text
    country_field: Literal["occurcountry", "primarysource_reportercountry"]
    age_groups: Text
    case_conflicts: Text
    seriousness: Text
    limitations: list[Text]


class CaseReaction(CSVModel):
    preferred_term: Text
    outcomes: list[Text] = Field(min_length=1)
    source_records: list[RecordNumber] = Field(min_length=1)


class CaseRecord(CSVModel):
    case_id: CaseID
    report_version: RecordNumber | None
    source_records: list[RecordNumber] = Field(min_length=1)
    all_source_records: list[RecordNumber] = Field(min_length=1)
    seriousness: Seriousness
    expedited: Flag
    receivedate: ISODate | None
    age_years: Annotated[float, Field(ge=0, le=120, allow_inf_nan=False)] | None
    age_group: AgeGroup
    sex: Literal["male", "female", "other", "unknown"]
    country: Text
    reporter_type: Text
    seriousness_flags: dict[Text, Flag]
    reactions: list[CaseReaction]

    @model_validator(mode="after")
    def check_source_records(self):
        _unique(self.source_records, "Selected source records")
        _unique(self.all_source_records, "All source records")
        if not set(self.source_records) <= set(self.all_source_records):
            raise ValueError("Selected records must belong to this case's history")
        _unique([term.preferred_term.casefold() for term in self.reactions], "Case preferred terms")
        for term in self.reactions:
            _unique(term.source_records, "Reaction source records")
            _unique(term.outcomes, "Reaction outcomes")
            if not set(term.source_records) <= set(self.source_records):
                raise ValueError("Reaction evidence must belong to the selected case version")
        return self


class AnalysisResults(CSVModel):
    """Typed form of analyzer.analyze_csv(); no AI-authored values belong here."""

    schema_version: Literal["1.0"]
    review_status: Literal["draft_requires_human_review"]
    provenance: AnalysisProvenance
    reporting_period: ReportingPeriod
    summary: AnalysisSummary
    case_ids: list[CaseID]
    demographics: Demographics
    seriousness: list[CaseDistribution]
    seriousness_criteria: dict[Text, list[CaseDistribution]]
    reactions: ReactionAnalysis
    outcomes: list[CaseDistribution]
    expedited: ExpeditedAnalysis
    trends: TrendAnalysis
    history_of_actions: HistoryOfActions
    data_quality: DataQuality
    methods: AnalysisMethods
    case_listing: list[CaseRecord]

    def content_sha256(self) -> str:
        """Bind report evidence to this analysis, including methods and provenance."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def check_integrity(self):
        _unique(self.case_ids, "Analysis case IDs")
        _unique([case.case_id for case in self.case_listing], "Case listing IDs")
        known = set(self.case_ids)
        cases = {case.case_id: case for case in self.case_listing}
        if set(cases) != known or len(known) != self.summary.total_cases:
            raise ValueError("Case listing, case_ids and total_cases must agree")
        # Walk every aggregate, including nested monthly/term breakdowns.
        def check_references(value):
            if isinstance(value, EvidenceCount) and not set(value.case_ids) <= known:
                raise ValueError("Aggregate references a case ID absent from the case listing")
            if isinstance(value, BaseModel):
                for name in type(value).model_fields:
                    check_references(getattr(value, name))
            elif isinstance(value, dict):
                for child in value.values():
                    check_references(child)
            elif isinstance(value, list):
                for child in value:
                    check_references(child)
        for name in type(self).model_fields:
            check_references(getattr(self, name))
        expected = {
            "serious_cases": sum(case.seriousness == "serious" for case in cases.values()),
            "non_serious_cases": sum(case.seriousness == "non-serious" for case in cases.values()),
            "unknown_seriousness_cases": sum(case.seriousness == "unknown" for case in cases.values()),
            "expedited_cases": sum(case.expedited == "yes" for case in cases.values()),
            "non_expedited_cases": sum(case.expedited == "no" for case in cases.values()),
            "unknown_expedited_cases": sum(case.expedited == "unknown" for case in cases.values()),
            "unknown_date_cases": sum(case.receivedate is None for case in cases.values()),
            "cases_without_reactions": sum(not case.reactions for case in cases.values()),
            "case_reaction_pairs": sum(len(case.reactions) for case in cases.values()),
        }
        if any(getattr(self.summary, key) != value for key, value in expected.items()):
            raise ValueError("Summary values must agree with the case listing")
        selected = [record for case in cases.values() for record in case.source_records]
        history = [record for case in cases.values() for record in case.all_source_records]
        _unique(selected, "Selected records across cases")
        _unique(history, "Source records across cases")
        excluded = self.data_quality.excluded_older_source_records
        _unique(excluded, "Excluded source records")
        if set(history) != set(range(1, self.summary.input_rows + 1)):
            raise ValueError("Source history must account for every input record")
        if set(excluded) != set(history) - set(selected):
            raise ValueError("Excluded records must match unselected case history")
        if len(selected) != self.summary.selected_version_rows or len(excluded) != self.summary.excluded_older_version_rows:
            raise ValueError("Source record counts do not match summary")
        for issue in self.data_quality.issues:
            if issue.case_id not in known or not set(issue.source_records) <= set(cases[issue.case_id].all_source_records):
                raise ValueError("Data-quality issue references records outside its case")
        dates = sorted(case.receivedate for case in cases.values() if case.receivedate)
        if (self.reporting_period.start, self.reporting_period.end) != ((dates[0], dates[-1]) if dates else (None, None)):
            raise ValueError("Reporting period must match observed case receivedates")
        _unique(self.expedited.case_ids, "Expedited case IDs")
        if set(self.expedited.case_ids) != {case.case_id for case in cases.values() if case.expedited == "yes"}:
            raise ValueError("Expedited evidence must match expedited cases")
        _unique([month.month for month in self.trends.monthly], "Trend months")
        monthly_ids = []
        for month in self.trends.monthly:
            monthly_ids.extend(month.case_ids)
            if any(not cases[case_id].receivedate or not cases[case_id].receivedate.startswith(month.month) for case_id in month.case_ids):
                raise ValueError("Monthly evidence must match case receivedates")
        _unique(monthly_ids, "Cases across months")
        if set(monthly_ids) != {case.case_id for case in cases.values() if case.receivedate}:
            raise ValueError("Monthly cases must cover all cases with known dates")

        def check_buckets(buckets, pairs, population):
            expected_groups = {}
            for category, case_id in pairs:
                expected_groups.setdefault(category, set()).add(case_id)
            _unique([bucket.category for bucket in buckets], "Distribution categories")
            for bucket in buckets:
                if set(bucket.case_ids) != expected_groups.get(bucket.category, set()):
                    raise ValueError(f"Incorrect supporting cases for category: {bucket.category}")
                percent = round(100 * bucket.case_count / len(population), 2) if population else 0
                if abs(bucket.percent_of_cases - percent) > 0.000001:
                    raise ValueError(f"Incorrect percentage for category: {bucket.category}")
            if not set(expected_groups) <= {bucket.category for bucket in buckets}:
                raise ValueError("Distribution omits populated categories")

        def fields(buckets, population, field):
            check_buckets(buckets, ((getattr(case, field), case.case_id) for case in population), population)

        def reactions(buckets, population):
            check_buckets(buckets, ((term.preferred_term, case.case_id) for case in population for term in case.reactions), population)

        def outcomes(buckets, population):
            check_buckets(buckets, ((outcome, case.case_id) for case in population for term in case.reactions for outcome in term.outcomes), population)

        population = list(cases.values())
        serious = [case for case in population if case.seriousness == "serious"]
        expedited = [case for case in population if case.expedited == "yes"]
        for name, field in (("age_groups", "age_group"), ("sex", "sex"),
                            ("country", "country"), ("reporter_type", "reporter_type")):
            fields(getattr(self.demographics, name), population, field)
        fields(self.seriousness, population, "seriousness")
        fields(self.expedited.status, population, "expedited")
        fields(self.expedited.seriousness, expedited, "seriousness")
        reactions(self.reactions.all, population)
        reactions(self.reactions.serious, serious)
        reactions(self.expedited.reactions, expedited)
        outcomes(self.outcomes, population)
        outcomes(self.expedited.outcomes, expedited)
        flag_names = {"seriousnessdeath", "seriousnesslifethreatening", "seriousnesshospitalization",
                      "seriousnessdisabling", "seriousnesscongenitalanomali", "seriousnessother"}
        if set(self.seriousness_criteria) != flag_names or any(set(case.seriousness_flags) != flag_names for case in population):
            raise ValueError("All six independent seriousness criteria must be represented")
        for flag, buckets in self.seriousness_criteria.items():
            check_buckets(buckets, ((case.seriousness_flags[flag], case.case_id) for case in population), population)
        for full, top in ((self.reactions.all, self.reactions.top),
                          (self.reactions.serious, self.reactions.top_serious)):
            if full != sorted(full, key=lambda bucket: (-bucket.case_count, bucket.category)) or top != full[:len(top)]:
                raise ValueError("Top reactions must be the leading entries of the ranked distribution")
        month_names = [month.month for month in self.trends.monthly]
        if month_names != sorted(month_names):
            raise ValueError("Trend months must be chronological")
        previous = None
        for month in self.trends.monthly:
            members = [cases[case_id] for case_id in month.case_ids]
            fields(month.seriousness, members, "seriousness")
            fields(month.country, members, "country")
            outcomes(month.outcomes, members)
            expected_change = month.case_count - previous if previous is not None else None
            expected_percent = round(100 * expected_change / previous, 2) if previous else None
            if month.change_from_previous_month != expected_change or month.percent_change != expected_percent:
                raise ValueError("Monthly changes must reconcile with monthly case counts")
            previous = month.case_count
        peak = max((month.case_count for month in self.trends.monthly), default=0)
        if self.trends.peak_months != [month.month for month in self.trends.monthly if month.case_count == peak]:
            raise ValueError("Peak months must match the monthly case counts")
        if [entry.preferred_term for entry in self.reactions.top_reaction_breakdowns] != [entry.category for entry in self.reactions.top]:
            raise ValueError("Reaction breakdowns must match top reactions")
        for entry, bucket in zip(self.reactions.top_reaction_breakdowns, self.reactions.top):
            members = [cases[case_id] for case_id in bucket.case_ids]
            fields(entry.age_group, members, "age_group")
            fields(entry.sex, members, "sex")
            if [month.month for month in entry.monthly] != month_names:
                raise ValueError("Reaction months must match the overall timeline")
            for month in entry.monthly:
                if month.category != month.month:
                    raise ValueError("Reaction month category must match month")
                check_buckets([month], ((month.month, case.case_id) for case in members if case.receivedate and case.receivedate.startswith(month.month)), members)
        if sum(bucket.case_count for bucket in self.reactions.all) != self.summary.case_reaction_pairs:
            raise ValueError("Reaction totals must reconcile with case-reaction pairs")
        if not all(self.data_quality.checks.model_dump().values()):
            raise ValueError("Analysis contains a failed reconciliation check")
        return self


SectionID = Literal[
    "reporting_period", "narrative_summary", "case_summary", "reaction_analysis",
    "serious_cases", "trends", "history_of_actions", "case_listing",
]


class EvidenceReference(CSVModel):
    """JSON Pointer into analysis_results.json, e.g. /summary/total_cases."""

    analysis_pointer: Annotated[str, Field(pattern=r"^/", min_length=2)]


class SupportedStatement(CSVModel):
    text: Text
    evidence: list[EvidenceReference] = Field(min_length=1)


class GeneratedSection(CSVModel):
    """AI response only; review decisions are kept out of this schema."""

    section_id: SectionID
    title: Text
    statements: list[SupportedStatement] = Field(min_length=1)


class ReportMetadata(CSVModel):
    product: Text
    report_type: Literal["PADER"] = "PADER"
    application_number: Text | None = None
    applicant_sponsor: Text | None = None
    reporting_period: ReportingPeriod


class ReportReview(CSVModel):
    status: Literal["draft", "flagged", "approved"] = "draft"
    reviewer: Text | None = None
    reviewed_at: datetime | None = None
    notes: str = ""

    @model_validator(mode="after")
    def require_review_record(self):
        if self.status != "draft":
            if self.reviewer is None or self.reviewed_at is None:
                raise ValueError("Flagging or approving requires reviewer identity and timestamp")
            if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
                raise ValueError("Review timestamp must include a time zone")
        return self


class PADERReport(CSVModel):
    """Report linked to its analysis; human review is distinct from AI output.

    Call validate_against(analysis) before saving. This checks reference integrity,
    not semantic entailment of generated text; a reviewer must assess the prose.
    """

    metadata: ReportMetadata
    analysis_sha256: Sha256
    sections: list[GeneratedSection] = Field(min_length=1)
    review: ReportReview = Field(default_factory=ReportReview)

    @model_validator(mode="after")
    def unique_sections(self):
        _unique([section.section_id for section in self.sections], "Report section IDs")
        if self.review.status == "approved" and {section.section_id for section in self.sections} != {
            "reporting_period", "narrative_summary", "case_summary", "reaction_analysis",
            "serious_cases", "trends", "history_of_actions", "case_listing",
        }:
            raise ValueError("An approved report must include all eight required sections")
        return self

    def validate_against(self, analysis: AnalysisResults) -> PADERReport:
        if self.analysis_sha256 != analysis.content_sha256():
            raise ValueError("Report analysis hash differs from the referenced analysis")
        if self.metadata.reporting_period != analysis.reporting_period:
            raise ValueError("Report reporting period differs from the analysis")
        document = analysis.model_dump(mode="json")
        for section in self.sections:
            for statement in section.statements:
                for evidence in statement.evidence:
                    value = document
                    try:
                        for token in evidence.analysis_pointer[1:].split("/"):
                            # Decode JSON Pointer escapes, rejecting invalid forms.
                            if any(not part or part[0] not in "01" for part in token.split("~")[1:]):
                                raise ValueError("Invalid JSON Pointer escape")
                            key = token.replace("~1", "/").replace("~0", "~")
                            if isinstance(value, list):
                                if not key.isascii() or not key.isdigit() or (len(key) > 1 and key[0] == "0"):
                                    raise ValueError("Invalid list index")
                                value = value[int(key)]
                            elif isinstance(value, dict):
                                value = value[key]
                            else:
                                raise ValueError("Cannot traverse scalar evidence")
                    except (KeyError, IndexError, ValueError) as error:
                        raise ValueError(f"Unresolved evidence pointer: {evidence.analysis_pointer}") from error
        return self


# Legacy PDF-only models. Kept until extractor_new.py and main.py are migrated.
# Do not use their unlabelled fields for CSV expectedness classifications.


class PADERMetadata(BaseModel):
  pader_control_number: str = Field(
      default="UNKNOWN", description="PADER Control Number"
  )
  application_number: str = Field(
      default="UNKNOWN", description="Application Number / NDA / NDA-B"
  )
  applicant_sponsor: str = Field(
      default="UNKNOWN", description="Applicant or Sponsor name"
  )
  reporting_interval_start: str = Field(
      default="YYYY-MM-DD", description="Reporting period start date"
  )
  reporting_interval_end: str = Field(
      default="YYYY-MM-DD", description="Reporting period end date"
  )


class ReactionTotals(BaseModel):
  total_serious_interval: int = Field(
      default=0, description="Total serious cases in interval"
  )
  total_nonserious_interval: int = Field(
      default=0, description="Total non-serious cases in interval"
  )
  total_reactions: int = Field(
      default=0, description="Grand total interval reactions"
  )


class AlertTotals(BaseModel):
  serious_unlabelled_non_fatal: int = Field(
      default=0, description="Count of non-fatal 15-day alerts"
  )
  serious_unlabelled_fatal: int = Field(
      default=0, description="Count of fatal 15-day alerts"
  )
  total_15_day_alerts: int = Field(
      default=0, description="Grand total of 15-day alerts"
  )


class UnlabelledTermItem(BaseModel):
  preferred_term: str = Field(..., description="MedDRA Preferred Term")
  case_count: int = Field(
      ..., description="Number of reported cases for this term"
  )


class UnlabelledTermDetail(BaseModel):
  preferred_term: str
  case_count: int
  clinical_narrative: str


class PADERMetricsOnly(BaseModel):
  drug_name: str
  metadata: PADERMetadata
  reaction_totals: ReactionTotals
  alert_totals: AlertTotals


class PADERTermsOnly(BaseModel):
  unlabelled_terms: List[UnlabelledTermItem]


class PADERExtraction(BaseModel):
  drug_name: str
  metadata: PADERMetadata
  reaction_totals: ReactionTotals
  alert_totals: AlertTotals
  unlabelled_terms: List[UnlabelledTermDetail]
