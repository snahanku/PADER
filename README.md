# GenAR AI Engineering Challenge

Submitted by **Snahanku Karar**.

Use `snahanku-karar-genar-challenge/` as the folder name when packaging this repository for submission.

PADER data extraction project.

- Entry point: `main.py`
- Implementation: `src/`
- Generated PADER-style report: [report_output.md](output/report_output.md)
- Inline prompts: `src/extractor_new.py`
- Sample input PDF: `data/`
- Structured JSON and CSV outputs: `output/`

## Submission layout

```text
snahanku-karar-genar-challenge/
|-- README.md
|-- main.py
|-- requirements.txt
|-- .env.example
|-- .gitattributes
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- extractor_new.py       # Inline metrics and narrative prompts
|   |-- pdf_loader.py
|   |-- schema_new.py
|   `-- models/
|       `-- qwen2.5-3b-instruct-q4_k_m.gguf  # Git LFS
|-- data/
|   `-- PADER-FDA-Y0AHP_PADER_Full_sample_data_B-1_CLIENT_DEV_01_FDA_v1_20260810.pdf
|-- output/
|   |-- report_output.md
|   |-- extracted_pader_summary.json
|   `-- extracted_unlabelled_terms.csv
`-- version1/
    `-- README.md             # Proposed baseline design document
```

The report is kept in `output/`. Prompts remain inline in
`extract_pader_data()` in `src/extractor_new.py`: `metrics_prompt` and
`terms_prompt`. A separate `prompts/` directory is unnecessary.

The [version 1 design document](version1/README.md) provides the design-document
option from the submission guide. It describes a proposed baseline, not a
separately implemented or tested historical version.

The architecture diagram is deferred as requested. The `.git/` directory is
optional for submission; exclude local `.env` credentials and generated caches.
The model is stored through Git LFS, so use `git lfs pull` after cloning to obtain
the actual model file. The current extractor calls Ollama with `qwen2.5:3b`.
