import re
from typing import List
import pdfplumber
from src.schema_new import (
    AlertTotals,
    PADERExtraction,
    PADERMetricsOnly,
    PADERTermsOnly,
    ReactionTotals,
    UnlabelledTermDetail,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def get_llm():
  return ChatOllama(
      model="qwen2.5:3b",
      temperature=0.0,
      options={"num_ctx": 8192, "num_predict": 4096},
  )


def extract_tables_with_pdfplumber(pdf_path: str) -> tuple[ReactionTotals, AlertTotals]:
  """
  Deterministic extraction of table metrics using pdfplumber.
  Eliminates LLM table column/row inversion errors completely.
  """
  rx_totals = ReactionTotals()
  al_totals = AlertTotals()

  try:
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages[:3]:  # Tables are typically in the first 3 pages
        tables = page.extract_tables()
        for table in tables:
          for row in table:
            row_str = " ".join([str(cell) for cell in row if cell]).lower()

            # Parse Reaction Totals Table
            if "serious" in row_str or "interval" in row_str:
              numbers = [int(n) for n in re.findall(r"\b\d+\b", row_str)]
              if len(numbers) >= 2:
                # Higher number is total serious in PADERs
                rx_totals.total_serious_interval = max(numbers)
                rx_totals.total_nonserious_interval = min(numbers)
                rx_totals.total_reactions = sum(numbers)

            # Parse 15-Day Alert Table
            if "15-day" in row_str or "alert" in row_str or "unlabelled" in row_str:
              numbers = [int(n) for n in re.findall(r"\b\d+\b", row_str)]
              if len(numbers) >= 3:
                al_totals.serious_unlabelled_non_fatal = numbers[0]
                al_totals.serious_unlabelled_fatal = numbers[1]
                al_totals.total_15_day_alerts = numbers[2]
              elif len(numbers) == 2:
                al_totals.serious_unlabelled_non_fatal = numbers[0]
                al_totals.serious_unlabelled_fatal = numbers[1]
                al_totals.total_15_day_alerts = sum(numbers)

  except Exception as e:
    print(f"[WARNING] pdfplumber table extraction fallback triggered: {e}")

  return rx_totals, al_totals


def slice_terms_section(text: str) -> str:
  """Locates and extracts the 'Case Presentation' / 'Unlabelled Adverse Events' section."""
  keywords = ["Case Presentation", "Unlabelled Adverse Events", "SUMMARY OF UNLABELLED"]
  start_pos = -1
  for kw in keywords:
    pos = text.find(kw)
    if pos != -1:
      start_pos = pos
      break
  return text[start_pos:] if start_pos != -1 else text


def generate_single_narrative(llm, term: str, count: int, section_text: str) -> str:
  """Generates a dedicated clinical narrative for a specific preferred term to prevent context drift."""
  prompt = ChatPromptTemplate.from_messages([
      ("system", (
          "You are a clinical pharmacovigilance specialist.\n"
          "Write a factual, objective 3-5 sentence clinical summary for the preferred term '{term}' "
          "based strictly on the details in the provided case text. Include demographics, comorbidities, "
          "concomitant medications, clinical outcomes, and causality where available."
      )),
      ("human", "CASE TEXT:\n{text}\n\nClinical Summary for {term}:")
  ])
  chain = prompt | llm
  res = chain.invoke({"term": term, "text": section_text[:3000]})
  return res.content.strip()


def extract_pader_data(document_text: str, pdf_path: str) -> PADERExtraction:
  llm = get_llm()

  # 1. Deterministic Table Parsing
  rx_totals, al_totals = extract_tables_with_pdfplumber(pdf_path)

  # 2. Extract Document Metadata via LLM (Pass 1)
  metrics_parser = PydanticOutputParser(pydantic_object=PADERMetricsOnly)
  metrics_llm = llm.bind(format=PADERMetricsOnly.model_json_schema())
  metrics_prompt = ChatPromptTemplate.from_messages([
      ("system", (
          "You are a pharmacovigilance assistant.\n"
          "Extract drug_name, pader_control_number, application_number, applicant_sponsor, "
          "reporting_interval_start (YYYY-MM-DD), and reporting_interval_end (YYYY-MM-DD)."
      )),
      ("human", "DOCUMENT SNIPPET:\n{snippet}\n\nExtract metadata JSON:")
  ])
  metrics_chain = metrics_prompt | metrics_llm | metrics_parser
  metrics_res = metrics_chain.invoke({"snippet": document_text[:3000]})

  # Combine deterministic tables if pdfplumber found valid counts
  if rx_totals.total_reactions > 0:
    metrics_res.reaction_totals = rx_totals
  if al_totals.total_15_day_alerts > 0:
    metrics_res.alert_totals = al_totals

  # 3. Lightweight Term & Count Extraction (Pass 2 - Step A)
  terms_parser = PydanticOutputParser(pydantic_object=PADERTermsOnly)
  terms_llm = llm.bind(format=PADERTermsOnly.model_json_schema())
  terms_prompt = ChatPromptTemplate.from_messages([
      ("system", (
          "You are an expert clinical pharmacovigilance specialist.\n"
          "Scan the text section and extract ALL unlabelled preferred terms and their exact case counts.\n"
          "Do not skip any term. Capture every term present (e.g. Acute kidney injury, Drug ineffective, "
          "Hypotension, Fatigue, Drug interaction)."
      )),
      ("human", "CASE PRESENTATION SECTION:\n{terms_text}\n\nExtract JSON list of terms and counts:")
  ])
  terms_chain = terms_prompt | terms_llm | terms_parser
  sliced_terms_text = slice_terms_section(document_text)
  raw_terms = terms_chain.invoke({"terms_text": sliced_terms_text})

  # 4. Generate Clinical Narratives Individually (Pass 2 - Step B)
  detailed_terms: List[UnlabelledTermDetail] = []
  for item in raw_terms.unlabelled_terms:
    narrative = generate_single_narrative(llm, item.preferred_term, item.case_count, sliced_terms_text)
    detailed_terms.append(
        UnlabelledTermDetail(
            preferred_term=item.preferred_term,
            case_count=item.case_count,
            clinical_narrative=narrative,
        )
    )

  return PADERExtraction(
      drug_name=metrics_res.drug_name,
      metadata=metrics_res.metadata,
      reaction_totals=metrics_res.reaction_totals,
      alert_totals=metrics_res.alert_totals,
      unlabelled_terms=detailed_terms,
  )