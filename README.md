# Docling Preprocessor Factory

_Note: This README was developed with the help of AI._

Related blog post [Building a Reproducible AI-Generated Project with ChatGPT, Codex, and Docling in VS Code](https://suedbroecker.net/2026/02/22/building-a-reproducible-ai-generated-project-with-chatgpt-codex-and-docling-in-vs-code/) on [suedbroecker.net](https://www.suedbroecker.com).

This repository provides a local-first preprocessing pipeline for multi-format business documents. It regenerates a representative sample corpus, extracts unit-wise content from each supported file, performs best-effort OCR where possible, and writes normalized JSONL output for downstream chunking and AI workflows.

## What The Code Does

- `code/scripts/run_local.sh` loads `.env` if present, validates Python `3.12`, creates or reuses `code/.venv`, installs pinned dependencies, runs the preprocessor, and then runs the verifier.
- `code/preprocess_app.py` regenerates sample files in `code/examples/` on every run and processes all supported files recursively.
- `code/verify_output.py` fails the run if `code/output/preprocessed.jsonl` is missing, empty, structurally invalid, or missing coverage for any supported example file.
- `code/Docling_multi_format_preprocessing_pipeline.py` is an alternate standalone Docling-centric module and is not invoked by the default shell workflow.

## Supported Formats

| Format | Output unit | Implementation detail |
| --- | --- | --- |
| `.pdf` | page | Uses Docling first for markdown extraction and falls back to PyMuPDF text extraction when needed. OCR is attempted page-by-page when Tesseract is available. |
| `.docx` | document | Extracts paragraph text with `python-docx` and attempts OCR on embedded images. |
| `.pptx` | slide | Extracts slide text and attempts OCR on slide images. |
| `.ppt` | slide | Requires LibreOffice to convert to `.pptx` before processing. If `soffice` is unavailable, the sample `.ppt` is skipped. |
| `.xlsx` | sheet | Extracts sheet data as markdown tables and attempts OCR on embedded worksheet images. |
| `.xls` | sheet | Extracts sheet data as markdown tables. No image OCR path is implemented for legacy `.xls`. |

## Main Files

| Path | Purpose |
| --- | --- |
| `code/scripts/run_local.sh` | End-to-end local bootstrap and verification entrypoint. |
| `code/preprocess_app.py` | Sample generation plus multi-format preprocessing logic. |
| `code/verify_output.py` | JSONL structure and source-coverage validation. |
| `code/examples/` | Generated sample documents processed by the application. |
| `code/output/preprocessed.jsonl` | Normalized output written by the preprocessor. |
| `requirements.txt` | Pinned top-level Python dependency set. |
| `code/scripts/requirements.txt` | Runtime requirements file rewritten by `run_local.sh` from the same pinned versions. |

## Quick Start

```bash
cd /path/to/docling_preprocessor_factory_public
cp .env_template .env
# Optional: edit .env if python3.12, tesseract, or soffice are not on PATH
bash code/scripts/run_local.sh
```

`.env` is optional. The script already computes project-relative defaults for the project, code, examples, output, and virtualenv directories. In practice, the most common overrides are:

- `DOC_PREPROCESS_PYTHON_BIN`
- `DOC_PREPROCESS_LIBREOFFICE_BIN`
- `DOC_PREPROCESS_TESSERACT_BIN`
- `DOC_PREPROCESS_EXAMPLES_DIR`
- `DOC_PREPROCESS_OUTPUT_DIR`

The first run needs network access because the helper script installs Python dependencies into `code/.venv`.

## Output Schema

Each JSONL record written to `code/output/preprocessed.jsonl` contains these required fields:

| Field | Meaning |
| --- | --- |
| `source_file_path` | Absolute path to the processed source document under `code/examples/`. |
| `input_format` | File format such as `pdf`, `docx`, `pptx`, `ppt`, `xlsx`, or `xls`. |
| `unit_number` | One-based page, slide, sheet, or document index. |
| `unit_type` | Logical unit type: `page`, `slide`, `sheet`, or `document`. |
| `text_markdown` | Extracted textual content rendered as markdown. |
| `ocr_image_text` | OCR text extracted from images when available; otherwise `null`. |
| `metadata` | Format-specific metadata such as page count, slide count, sheet name, and OCR/runtime flags. |

The verifier also checks that every supported file under `code/examples/` is represented in the JSONL output.

## Configuration

Important environment variables loaded by the default workflow:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOC_PREPROCESS_REQUIRED_PYTHON` | `3.12` | Exact major/minor Python version enforced by both the shell script and Python entrypoints. |
| `DOC_PREPROCESS_PYTHON_BIN` | `python3.12` | Python executable used to create and run the virtual environment. |
| `DOC_PREPROCESS_EXAMPLES_DIR` | `code/examples` | Directory where the sample corpus is generated and then scanned. |
| `DOC_PREPROCESS_OUTPUT_DIR` | `code/output` | Directory for generated artifacts. |
| `DOC_PREPROCESS_OUTPUT_JSONL` | `code/output/preprocessed.jsonl` | Final JSONL file written by the preprocessor. |
| `DOC_PREPROCESS_LIBREOFFICE_BIN` | auto-detected `soffice` or empty | Optional path for `.ppt` conversion support. |
| `DOC_PREPROCESS_TESSERACT_BIN` | `tesseract` | OCR executable used by `pytesseract`. |
| `DOC_PREPROCESS_SUPPORTED_EXTENSIONS` | `.pdf,.docx,.xlsx,.xls,.pptx,.ppt` | Recursive file extension allowlist. |
| `DOC_PREPROCESS_REQUIRED_FIELDS` | `source_file_path,input_format,unit_number,unit_type,text_markdown,ocr_image_text,metadata` | Required JSONL keys enforced by the verifier. |

## Open-Source Dependencies

This repository is Apache-2.0 licensed and uses open-source dependencies only.

The project pins exact runtime versions in `requirements.txt`. The same pinned set is written to `code/scripts/requirements.txt` by `run_local.sh` before installation. The table below reflects the locally verified installed versions on `2026-04-08`.

Current verified Python version:

- `Python 3.12.12`

Current verified main runtime libraries:

| Library | Installed version | License | Used for |
| --- | --- | --- | --- |
| `docling` | `2.20.0` | MIT | Primary PDF-to-markdown extraction path. |
| `reportlab` | `4.2.5` | BSD-style | Sample PDF generation. |
| `python-docx` | `1.1.2` | MIT | DOCX generation and text extraction. |
| `openpyxl` | `3.1.5` | MIT | XLSX generation and worksheet extraction. |
| `xlwt` | `1.3.0` | BSD | Legacy XLS generation. |
| `xlrd` | `2.0.1` | BSD | Legacy XLS reading. |
| `python-pptx` | `1.0.2` | MIT | PPTX generation and slide extraction. |
| `Pillow` | `10.4.0` | HPND | OCR sample image generation and image handling. |
| `PyMuPDF` | `1.24.11` | AGPL-3.0 | PDF fallback text extraction and page rasterization for OCR. |
| `pytesseract` | `0.3.13` | Apache-2.0 | Python wrapper for OCR execution. |

`PyMuPDF` is open source, but it is licensed under AGPL-3.0. Review that license carefully if you plan to redistribute or network-expose a derived application.

## More Detail

- [Quickstart](./QUICKSTART.md)
- [Documentation](./documentation/DOCUMENTATION.md)
