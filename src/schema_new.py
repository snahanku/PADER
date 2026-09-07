import logging
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger("pader_schema")


class ReportMetadata(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    pader_control_number: str = Field(
        ..., description="Full PADER Control Number, e.g., PADER-FDA-Y0AHP"
    )
    application_number: str = Field(
        ..., description="Application/NDA/ANDA number, e.g., B-1"
    )
    applicant_sponsor: str = Field(
        ..., description="Applicant/Sponsor organization name"
    )
    # Kept as str (not `date`) on purpose: the raw PDF title page contains
    # literal placeholders like "<START_DATE>". Typing this as `date` would
    # make pydantic reject the LLM's output before main.py's regex-based
    # placeholder recovery ever gets a chance to run. Recovery happens in
    # main.py's post_process_data(); this validator only guards ordering
    # once real dates are in place.
    reporting_interval_start: str = Field(
        ..., description="Reporting interval start date in YYYY-MM-DD format"
    )
    reporting_interval_end: str = Field(
        ..., description="Reporting interval end date in YYYY-MM-DD format"
    )

    @model_validator(mode="after")
    def check_interval_order(self) -> "ReportMetadata":
        s, e = self.reporting_interval_start, self.reporting_interval_end
        if _looks_like_iso_date(s) and _looks_like_iso_date(e) and e < s:
            logger.warning(
                "reporting_interval_end (%s) precedes reporting_interval_start (%s); swapping.",
                e, s,
            )
            self.reporting_interval_start, self.reporting_interval_end = e, s
        return self


def _looks_like_iso_date(value: str) -> bool:
    import re
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or ""))


class IntervalReactionTotals(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    total_reactions: int = Field(..., description="Total interval reactions count")
    total_serious_interval: int = Field(
        ..., description="Total serious interval reactions count"
    )
    total_nonserious_interval: int = Field(
        ..., description="Total non-serious interval reactions count"
    )

    @model_validator(mode="after")
    def reconcile_totals(self) -> "IntervalReactionTotals":
        expected = self.total_serious_interval + self.total_nonserious_interval
        if expected != self.total_reactions:
            logger.warning(
                "total_reactions (%s) != serious+nonserious (%s); recomputing total_reactions.",
                self.total_reactions, expected,
            )
            self.total_reactions = expected
        return self


class AlertTotals15Day(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # --- Serious, Unlabelled — Non-Fatal breakdown by source ---
    serious_unlabelled_non_fatal_solicited_study: int = Field(
        0, description="Non-fatal serious unlabelled count, Solicited (Study) source"
    )
    serious_unlabelled_non_fatal_solicited_other: int = Field(
        0, description="Non-fatal serious unlabelled count, Solicited (Other) source"
    )
    serious_unlabelled_non_fatal_spontaneous: int = Field(
        0, description="Non-fatal serious unlabelled count, Spontaneous source"
    )
    serious_unlabelled_non_fatal_total: int = Field(
        0, description="Total non-fatal serious unlabelled alerts across all sources"
    )

    # --- Serious, Unlabelled — Fatal breakdown by source ---
    serious_unlabelled_fatal_solicited_study: int = Field(
        0, description="Fatal serious unlabelled count, Solicited (Study) source"
    )
    serious_unlabelled_fatal_solicited_other: int = Field(
        0, description="Fatal serious unlabelled count, Solicited (Other) source"
    )
    serious_unlabelled_fatal_spontaneous: int = Field(
        0, description="Fatal serious unlabelled count, Spontaneous source"
    )
    serious_unlabelled_fatal_total: int = Field(
        0, description="Total fatal serious unlabelled alerts across all sources"
    )

    total_15_day_alerts: int = Field(
        0, description="Grand total of 15-day alerts (non-fatal total + fatal total)"
    )

    @model_validator(mode="after")
    def reconcile_alert_totals(self) -> "AlertTotals15Day":
        expected_non_fatal = (
            self.serious_unlabelled_non_fatal_solicited_study
            + self.serious_unlabelled_non_fatal_solicited_other
            + self.serious_unlabelled_non_fatal_spontaneous
        )
        if expected_non_fatal != self.serious_unlabelled_non_fatal_total:
            logger.warning(
                "non_fatal_total (%s) != sum of sub-categories (%s); recomputing.",
                self.serious_unlabelled_non_fatal_total, expected_non_fatal,
            )
            self.serious_unlabelled_non_fatal_total = expected_non_fatal

        expected_fatal = (
            self.serious_unlabelled_fatal_solicited_study
            + self.serious_unlabelled_fatal_solicited_other
            + self.serious_unlabelled_fatal_spontaneous
        )
        if expected_fatal != self.serious_unlabelled_fatal_total:
            logger.warning(
                "fatal_total (%s) != sum of sub-categories (%s); recomputing.",
                self.serious_unlabelled_fatal_total, expected_fatal,
            )
            self.serious_unlabelled_fatal_total = expected_fatal

        expected_grand_total = (
            self.serious_unlabelled_non_fatal_total + self.serious_unlabelled_fatal_total
        )
        if expected_grand_total != self.total_15_day_alerts:
            logger.warning(
                "total_15_day_alerts (%s) != non_fatal_total + fatal_total (%s); recomputing.",
                self.total_15_day_alerts, expected_grand_total,
            )
            self.total_15_day_alerts = expected_grand_total

        return self


class UnlabelledTermSummary(BaseModel):
    preferred_term: str = Field(
        ..., description="Name of the preferred term (e.g., Acute kidney injury)"
    )
    case_count: int = Field(..., description="Number of cases for this term")
    clinical_narrative: str = Field(
        ..., description="Concise clinical narrative summary in 2-3 sentences"
    )

    @model_validator(mode="after")
    def check_narrative_matches_count(self) -> "UnlabelledTermSummary":
        # Logged, not raised: a mismatch here is a strong signal the model
        # conflated two mentions of the same term (this is exactly what
        # happened with "Fatigue" in the first extraction run) — but a hard
        # crash would make the whole pipeline unusable on a small/imperfect
        # model. Surface it for human review instead.
        if str(self.case_count) not in self.clinical_narrative:
            logger.warning(
                "Narrative for '%s' does not mention its own case_count (%s) — "
                "possible conflation with another section; flagging for review.",
                self.preferred_term, self.case_count,
            )
        return self


class PADERMetricsOnly(BaseModel):
    drug_name: str
    metadata: ReportMetadata
    reaction_totals: IntervalReactionTotals
    alert_totals: AlertTotals15Day


class PADERTermsOnly(BaseModel):
    # Populated from a regex hint in extractor_new.py (the document states
    # this explicitly, e.g. "the following 5 unlabelled Preferred Terms were
    # identified") rather than trusted blindly from the LLM. Defaults to 0
    # (= "unknown / not checked") if the document text doesn't state a count.
    expected_unlabelled_term_count: int = Field(
        0,
        description=(
            "Number of unlabelled Preferred Terms the source document "
            "explicitly states were identified. 0 means not detected / not "
            "checked."
        ),
    )
    unlabelled_terms: List[UnlabelledTermSummary] = Field(default_factory=list)

    @field_validator("unlabelled_terms")
    @classmethod
    def dedupe_terms(
        cls, v: List[UnlabelledTermSummary]
    ) -> List[UnlabelledTermSummary]:
        seen = set()
        unique = []
        for term in v:
            key = term.preferred_term.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(term)
        return unique

    @model_validator(mode="after")
    def check_term_completeness(self) -> "PADERTermsOnly":
        if (
            self.expected_unlabelled_term_count
            and len(self.unlabelled_terms) != self.expected_unlabelled_term_count
        ):
            logger.warning(
                "Document states %s unlabelled terms but extraction found %s — "
                "likely an omitted term; flagging for review.",
                self.expected_unlabelled_term_count, len(self.unlabelled_terms),
            )
        return self


class PADERExtraction(PADERMetricsOnly, PADERTermsOnly):
    """Full PADER extraction combining metrics and unlabelled term summaries."""
    pass