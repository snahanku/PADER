# Version 1: proposed baseline design

**Author:** Snahanku Karar  
**Status:** Design-only submission. This is a proposed baseline, not a claim that
a separate version 1 implementation has been built or evaluated.

## Objective

Convert a text-based PADER PDF into structured report metadata, reaction and
15-day alert totals, and unlabelled preferred terms with short clinical
narratives. Produce a readable Markdown report plus JSON and CSV outputs.

## Proposed approach

1. Read the PDF with `pdfplumber` and clean repeated page markers and whitespace.
2. Pass the cleaned document to a local instruction model through Ollama. For
   this baseline, use one extraction request with an explicit JSON schema and
   instructions to use only information supported by the source document.
3. Parse the response with Pydantic. On a parsing or validation failure, retry
   once with the error included; if it still fails, return a visible failure.
4. Check totals, duplicate preferred terms, missing fields, and reporting dates.
   Flag discrepancies for review rather than inventing unsupported values.
5. Render `output/report_output.md` and save structured JSON and a term-level CSV
   in the same output directory.

## Reuse and implementation boundary

The submitted implementation is in `main.py` and `src/`. Its PDF loader, schemas,
and report rendering provide reusable components for this proposed baseline.
The current extractor already uses a more targeted two-pass approach: one
request for metadata and numerical tables and another for case narratives.
Its prompts are inline in `src/extractor_new.py`. This document does not replace
or duplicate that implementation.

## Evaluation plan and limitations

Compare extracted fields against a manually reviewed reference for the sample
PDF. Check each numerical table cell, reporting dates, term coverage, duplicate
terms, and whether each narrative is supported by the corresponding source
section. Record schema failures and retry outcomes. No baseline evaluation
results are claimed here.

A single request may miss tables near the end of a long document or mix details
across sections. Scanned PDFs would require an OCR step. These limitations
motivate section-specific extraction and human review of flagged results.
