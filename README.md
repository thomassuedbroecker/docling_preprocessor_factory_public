# Docling Preprocessor Factory

_Note: This README was developed with the help of AI._

Related blog post [Building a Reproducible AI-Generated Project with ChatGPT, Codex, and Docling in VS Code](https://suedbroecker.net/2026/02/22/building-a-reproducible-ai-generated-project-with-chatgpt-codex-and-docling-in-vs-code/) on [suedbroecker.net](https://www.suedbroecker.com).

This repository provides a local-first preprocessing pipeline for multi-format business documents. It regenerates a representative sample corpus, extracts unit-wise content from each supported file, performs best-effort OCR where possible, and writes normalized JSONL output for downstream chunking and AI workflows.

## What The Code Does

- `code/scripts/run_local.sh` loads `.env` if present, validates Python `3.12`, creates or reuses `code/.venv`, installs pinned dependencies, runs the preprocessor, and then runs both verification steps.
- `code/preprocess_app.py` regenerates sample files in `code/examples/` on every run and processes all supported files recursively.
- `code/verify_output.py` fails the run if `code/output/preprocessed.jsonl` is missing, empty, structurally invalid, or missing coverage for any supported example file.
- `code/verify_project_consistency.py` fails the run if the dependency documentation, requirements files, and standalone Docling notes drift out of sync.
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
| `code/verify_project_consistency.py` | Verifies that dependency docs, requirements files, and Docling-only notes stay in sync. |
| `code/dependency_profiles.py` | Canonical dependency profiles used by the bootstrap script and project consistency verifier. |
| `code/docling_config_examples.py` | Extracted Docling-only converter configurations for the example pipeline, including OCR variants that need extra runtime support. |
| `code/examples/` | Generated sample documents processed by the application. |
| `code/output/preprocessed.jsonl` | Normalized output written by the preprocessor. |
| `requirements.txt` | Pinned top-level Python dependency set. |
| `requirements-docling-only.txt` | Trimmed Python dependency set for the standalone Docling-only modules. |
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
| `DOC_PREPROCESS_DEPENDENCY_PROFILES_FILE` | `code/dependency_profiles.py` | Canonical dependency profile module used to write the bootstrap requirements file. |
| `DOC_PREPROCESS_LIBREOFFICE_BIN` | auto-detected `soffice` or empty | Optional path for `.ppt` conversion support. |
| `DOC_PREPROCESS_TESSERACT_BIN` | `tesseract` | OCR executable used by `pytesseract`. |
| `DOC_PREPROCESS_SUPPORTED_EXTENSIONS` | `.pdf,.docx,.xlsx,.xls,.pptx,.ppt` | Recursive file extension allowlist. |
| `DOC_PREPROCESS_REQUIRED_FIELDS` | `source_file_path,input_format,unit_number,unit_type,text_markdown,ocr_image_text,metadata` | Required JSONL keys enforced by the verifier. |
| `DOC_PREPROCESS_VERIFY_PROJECT_FILE` | `code/verify_project_consistency.py` | Consistency verifier that checks dependency docs and requirements files. |

## Docling Configuration For The Example

The Docling-only setup used by the standalone example has been extracted to `code/docling_config_examples.py`.

The relevant configuration patterns are:

- `build_example_converter()` for the current example setup using `DocumentConverter` with explicit `PdfFormatOption`, `WordFormatOption`, and `PowerpointFormatOption`
- `build_example_converter_without_ocr()` when you only want native PDF text extraction
- `build_example_converter_with_tesseract()` when you want Docling OCR through the system `tesseract` binary
- `build_example_converter_with_rapidocr()` when you want Docling OCR through `rapidocr_onnxruntime`

Minimal example:

```python
from docling_config_examples import build_example_converter, export_pdf_pages_as_markdown

converter = build_example_converter()
pages = export_pdf_pages_as_markdown("code/examples/sample.pdf", converter)
```

When an additional OCR runtime is needed, the relevant Docling configuration is explicit in code:

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

pdf_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=TesseractCliOcrOptions(
        lang=["eng"],
        tesseract_cmd="tesseract",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
    }
)
```

Additional OCR runtime requirements for these variants:

- `build_example_converter()` uses Docling's EasyOCR-based PDF pipeline in the tested environment
- `build_example_converter_with_tesseract()` needs the `tesseract` executable installed and on `PATH`
- `build_example_converter_with_rapidocr()` needs `pip install rapidocr_onnxruntime`

## Test Commands

This repository currently uses executable end-to-end checks instead of a separate `pytest` or `unittest` suite.

Default end-to-end command:

```bash
bash code/scripts/run_local.sh
```

Direct entrypoint checks used for verification:

```bash
code/.venv/bin/python code/preprocess_app.py \
  --input-dir code/examples \
  --output-dir code/output \
  --output-jsonl code/output/preprocessed.jsonl \
  --tesseract-bin tesseract \
  --supported-extensions .pdf,.docx,.xlsx,.xls,.pptx,.ppt \
  --required-python 3.12 \
  --sample-image-name sample_ocr_image.png \
  --sample-pdf-name sample.pdf \
  --sample-docx-name sample.docx \
  --sample-xlsx-name sample.xlsx \
  --sample-xls-name sample.xls \
  --sample-pptx-name sample.pptx \
  --sample-ppt-name sample.ppt

code/.venv/bin/python code/verify_output.py \
  --examples-dir code/examples \
  --output-jsonl code/output/preprocessed.jsonl \
  --supported-extensions .pdf,.docx,.xlsx,.xls,.pptx,.ppt \
  --required-fields source_file_path,input_format,unit_number,unit_type,text_markdown,ocr_image_text,metadata \
  --required-python 3.12

code/.venv/bin/python code/verify_project_consistency.py \
  --project-dir . \
  --required-python 3.12
```

## Tested

Current verified state as of `2026-04-08`:

| State | Meaning |
| --- | --- |
| 🟢 | Executed and passed |
| 🟡 | Executed and passed with a known non-blocking warning or environment limitation |
| 🔴 | Failed or currently blocking |

| State | Check | Result | Details |
| --- | --- | --- | --- |
| 🟢 | `bash code/scripts/run_local.sh` | Passed on `2026-04-08` | Created `code/.venv`, installed dependencies, regenerated the sample corpus, wrote `code/output/preprocessed.jsonl`, and finished with both verification steps succeeding. |
| 🟢 | `code/.venv/bin/python code/preprocess_app.py ...` | Passed on `2026-04-08` | Rewrote `code/output/preprocessed.jsonl` and generated `9` JSONL records from `5` supported example documents in the current environment. |
| 🟢 | `code/.venv/bin/python code/verify_output.py ...` | Passed on `2026-04-08` | Validated JSONL parseability, required fields, and source coverage for the generated sample set. |
| 🟢 | `code/.venv/bin/python code/verify_project_consistency.py ...` | Passed on `2026-04-08` | Validated dependency profiles against `requirements.txt`, `requirements-docling-only.txt`, `code/scripts/requirements.txt`, and the documentation blocks in the repository. |
| 🟢 | Runtime toolchain versions | Verified on `2026-04-08` | `Python 3.12.12` in `code/.venv` and `tesseract 5.5.1` at `/opt/homebrew/bin/tesseract`. |
| 🟡 | LibreOffice availability | Checked on `2026-04-08` | `soffice` was not installed, so legacy `.ppt` sample generation and `.ppt` processing were skipped by design. |
| 🟡 | Known runtime warning | Present on `2026-04-08` | `torch.utils.data.dataloader` emitted a non-blocking `pin_memory` warning during preprocessing; the run still passed. |
| 🟢 | Current failing checks | None on `2026-04-08` | No blocking failures remained after the successful end-to-end run. |

What is covered by the available checks:

- creation and reuse of `code/.venv` plus dependency installation through the default bootstrap script
- sample document generation under `code/examples/` for PDF, DOCX, PPTX, XLSX, and XLS
- preprocessing and JSONL write paths for page-wise, slide-wise, sheet-wise, and document-level extraction
- verifier checks for JSONL existence, non-empty output, required fields, and source-file coverage
- dependency profile consistency across code, requirements files, and documentation
- OCR execution in the current environment with local `tesseract 5.5.1`
- graceful skip behavior for legacy `.ppt` support when LibreOffice is unavailable

## Open-Source Dependencies

This repository is Apache-2.0 licensed and uses open-source dependencies only.

The project maintains two Python dependency profiles:

- `requirements.txt` for the repository's default end-to-end workflow
- `requirements-docling-only.txt` for the standalone Docling-only modules

Why additional libraries are present in the default workflow:

<!-- BEGIN:DEPENDENCY_SCOPE_EXPLANATION -->
- These libs are not required by Docling itself.
- They are required by this repository's default implementation choices.
- If you reduce the project to Docling-only conversion, most of them can be removed.
<!-- END:DEPENDENCY_SCOPE_EXPLANATION -->

The default shell workflow still writes the canonical default profile into `code/scripts/requirements.txt` before installation. The tables below reflect the locally verified versions on `2026-04-08`.

Current verified Python version:

- `Python 3.12.12`

Current verified default workflow runtime libraries:

<!-- BEGIN:DEFAULT_WORKFLOW_DEPENDENCIES -->
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
<!-- END:DEFAULT_WORKFLOW_DEPENDENCIES -->

Trimmed Docling-only dependency profile:

<!-- BEGIN:DOCLING_ONLY_DEPENDENCIES -->
| Library | Installed version | License | Used for |
| --- | --- | --- | --- |
| `docling` | `2.20.0` | MIT | Standalone Docling-only conversion modules. |
<!-- END:DOCLING_ONLY_DEPENDENCIES -->

Optional runtime notes for Docling-only variants:

<!-- BEGIN:OPTIONAL_RUNTIME_NOTES -->
- `tesseract` is still needed when you select the Tesseract OCR path.
- `rapidocr_onnxruntime` is only needed when you select the RapidOCR path.
- `soffice` is only needed by the default workflow for legacy `.ppt` conversion.
<!-- END:OPTIONAL_RUNTIME_NOTES -->

`PyMuPDF` is open source, but it is licensed under AGPL-3.0. Review that license carefully if you plan to redistribute or network-expose a derived application.

## More Detail

- [Quickstart](./QUICKSTART.md)
- [Documentation](./documentation/DOCUMENTATION.md)
