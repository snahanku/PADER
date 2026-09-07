import logging
import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.schema_new import PADERExtraction, PADERMetricsOnly, PADERTermsOnly

logger = logging.getLogger("pader_extractor")


def get_llm():
    return ChatOllama(
        model="qwen2.5:3b",
        temperature=0.0,
        options={
            "num_ctx": 8192,
            "num_predict": 4096,
        },
    )


# ---------------------------------------------------------------------------
# SECTION SLICING
#
# Root cause of the original 0/67 fatal-alert miss: the old
# slice_metrics_section() used a hardcoded text[:3500] cutoff. The 15-day
# alert / ICSR tabulation tables sit at the very END of the document (after
# Introduction, Reaction Totals, Case Presentation, and Safety Section), so
# that cutoff never reached them — the model wasn't wrong, it was extracting
# from a blank context.
#
# Root cause of the dropped "Drug interaction" term and the conflated
# "Fatigue" narrative: the old slice_terms_section() had a start anchor but
# no end anchor, so it fed the model Case Presentation + Safety Section +
# ICSR tables all at once. The Safety Section separately mentions "fatigue"
# as a generic labeled ADR, and the model merged that mention with the
# distinct Case Presentation narrative for Fatigue.
# ---------------------------------------------------------------------------

def slice_metrics_section(text: str) -> str:
    """Returns the header/reaction-totals block plus the 15-day alert / ICSR
    tabulation tables, skipping the narrative sections in between (which
    aren't needed for metrics and would otherwise just add noise/length)."""
    parts = []

    case_pres_pos = text.find("Case Presentation")
    header_end = case_pres_pos if case_pres_pos != -1 else min(len(text), 4000)
    parts.append(text[:header_end])

    alert_start_keywords = [
        "Summary of ICSRs Case Tabulation",
        "Summary of 15-Day Alert Reports",
        "Summary of All ICSR Cases",
    ]
    alert_start = -1
    for kw in alert_start_keywords:
        pos = text.find(kw)
        if pos != -1:
            alert_start = pos
            break

    if alert_start != -1:
        parts.append(text[alert_start:])
    else:
        logger.warning(
            "Could not locate 15-day alert / ICSR tabulation section by "
            "keyword — alert totals may be incomplete."
        )

    combined = "\n\n---\n\n".join(p for p in parts if p.strip())
    return combined if combined.strip() else text[:6000]


def slice_terms_section(text: str) -> str:
    """Locates the 'Case Presentation' (unlabelled terms) section and bounds
    its END at the next major heading, so the model never sees the Safety
    Section's separate, generic ADR list in the same context window."""
    start_keywords = [
        "Case Presentation",
        "Unlabelled Adverse Events",
        "SUMMARY OF UNLABELLED",
    ]
    start_pos = -1
    for kw in start_keywords:
        pos = text.find(kw)
        if pos != -1:
            start_pos = pos
            break

    if start_pos == -1:
        return text

    remainder = text[start_pos:]

    end_keywords = ["Safety Section", "Summary of ICSRs Case Tabulation"]
    end_pos = -1
    for kw in end_keywords:
        pos = remainder.find(kw)
        if pos != -1 and (end_pos == -1 or pos < end_pos):
            end_pos = pos

    return remainder[:end_pos] if end_pos != -1 else remainder


def detect_expected_term_count(text: str) -> int:
    """The source document states its own ground truth, e.g. 'the following
    5 unlabelled Preferred Terms were identified'. Pulling this via regex is
    more reliable than trusting the LLM to count and self-report it."""
    match = re.search(r"following\s+(\d+)\s+unlabell?ed", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# ---------------------------------------------------------------------------
# RETRY WRAPPER
#
# Once the schema's validators can flag problems, a single bad LLM pass
# shouldn't silently propagate. This gives the model one chance to see its
# own error and correct itself before we fall back to best-effort output.
# ---------------------------------------------------------------------------

def invoke_with_retry(chain, base_inputs: dict, max_retries: int = 1):
    inputs = {**base_inputs, "retry_note": ""}
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:  # covers OutputParserException / ValidationError
            last_error = e
            logger.warning("Extraction attempt %s failed: %s", attempt + 1, e)
            inputs = {
                **base_inputs,
                "retry_note": (
                    f"\n\nYOUR PREVIOUS ATTEMPT FAILED VALIDATION WITH THIS ERROR:\n"
                    f"{e}\nCarefully re-read the document and correct the JSON output."
                ),
            }
    raise last_error


def extract_pader_data(document_text: str) -> PADERExtraction:
    llm = get_llm()

    # =========================================================================
    # PASS 1: METRICS & TABLES
    # =========================================================================
    metrics_parser = PydanticOutputParser(pydantic_object=PADERMetricsOnly)
    metrics_llm = llm.bind(format=PADERMetricsOnly.model_json_schema())

    metrics_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert pharmacovigilance data extraction assistant.\n"
            "Extract report metadata and numerical tables strictly from the"
            " provided PADER document snippet.\n\n"
            "GENERAL METADATA RULES:\n"
            "1. Extract the exact drug name, control number, application number,"
            " and sponsor name.\n"
            "2. Extract reporting interval start and end dates strictly in"
            " YYYY-MM-DD format, from the actual reported dates in the"
            " Introduction text (not title-page placeholders like"
            " '<START_DATE>').\n\n"
            "GENERAL TABLE RULES:\n"
            "1. REACTION TOTALS (from the 'Summary Tabulation of Adverse"
            " Events' table):\n"
            "   - total_serious_interval, total_nonserious_interval,"
            " total_reactions (grand total row).\n"
            "2. 15-DAY ALERT / ICSR TABULATION (from the tables titled"
            " 'Summary of 15-Day Alert Reports' / 'Summary of ICSRs Case"
            " Tabulation' / 'Summary of All ICSR Cases'):\n"
            "   These tables break counts down by source: Solicited (Study),"
            " Solicited (Other), and Spontaneous. Extract EACH source"
            " sub-count separately for both the 'Serious, Unlabelled —"
            " Non-Fatal' row and the 'Serious, Unlabelled — Fatal' row, as"
            " well as each row's total and the grand total_15_day_alerts.\n"
            "   Do NOT output zero for any of these fields unless the table"
            " itself shows zero — search the ENTIRE snippet for this table"
            " before answering.",
        ),
        (
            "human",
            "DOCUMENT SNIPPET:\n{metrics_text}\n\n{retry_note}\n\nExtract JSON metrics:",
        ),
    ])

    metrics_chain = metrics_prompt | metrics_llm | metrics_parser
    sliced_metrics = slice_metrics_section(document_text)
    metrics_res = invoke_with_retry(
        metrics_chain, {"metrics_text": sliced_metrics}
    )

    # =========================================================================
    # PASS 2: UNLABELLED TERMS & NARRATIVES
    # =========================================================================
    terms_parser = PydanticOutputParser(pydantic_object=PADERTermsOnly)
    terms_llm = llm.bind(format=PADERTermsOnly.model_json_schema())

    sliced_terms = slice_terms_section(document_text)
    expected_count = detect_expected_term_count(document_text)

    terms_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert clinical pharmacovigilance specialist.\n"
            "Scan the provided 'Case Presentation' section ONLY (ignore any"
            " other section) and extract every distinct unlabelled preferred"
            " term listed there.\n\n"
            "GENERAL EXTRACTION RULES:\n"
            "1. COMPLETE EXTRACTION: The document states exactly how many"
            " unlabelled terms are covered in this section"
            f" ({expected_count if expected_count else 'a stated number'})."
            " You MUST return exactly that many distinct terms — do not stop"
            " early or omit any.\n"
            "2. CASE COUNTS: Extract the exact numerical case count stated"
            " for each preferred term, and restate that same number inside"
            " the narrative you write for it.\n"
            "3. CLINICAL NARRATIVES: Write a 2-3 sentence narrative per term"
            " based only on the details given for that specific term"
            " (demographics, comorbidities, concomitant medications,"
            " outcomes, causality). Do not borrow details from a different"
            " term or a different section.\n"
            "4. NO DUPLICATES: Each preferred term must appear only once in"
            " the output list.\n"
            f"5. Set expected_unlabelled_term_count to {expected_count}.",
        ),
        (
            "human",
            "CASE PRESENTATION SECTION:\n{terms_text}\n\n{retry_note}\n\n"
            "Extract ALL unlabelled terms, counts, and narratives into JSON:",
        ),
    ])

    terms_chain = terms_prompt | terms_llm | terms_parser
    terms_res = invoke_with_retry(terms_chain, {"terms_text": sliced_terms})

    # Trust the regex-detected count over whatever the LLM echoed back, since
    # it's read directly from the document's own explicit statement.
    if expected_count and terms_res.expected_unlabelled_term_count != expected_count:
        terms_res.expected_unlabelled_term_count = expected_count

    return PADERExtraction(
        drug_name=metrics_res.drug_name,
        metadata=metrics_res.metadata,
        reaction_totals=metrics_res.reaction_totals,
        alert_totals=metrics_res.alert_totals,
        expected_unlabelled_term_count=terms_res.expected_unlabelled_term_count,
        unlabelled_terms=terms_res.unlabelled_terms,
    )