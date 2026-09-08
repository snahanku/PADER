from typing import List, Optional
from pydantic import BaseModel, Field


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