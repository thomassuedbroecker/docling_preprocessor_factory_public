# Quickstart

_Note: This quickstart was developed with the help of AI._

## Prerequisites
- Python 3.12
- Internet access on the first run so dependencies can be installed into `code/.venv`
- Optional: LibreOffice (`soffice`) for `.ppt` generation and `.ppt` processing
- Optional: Tesseract for OCR text extraction from embedded images

## Illustrative Snippet (Not Executable As-Is)
```bash
# Demonstration only: this inline snippet is illustrative and will not run as-is.
python preprocess.py --input-dir examples --output-dir output
```

## Execute The Real Application
`code/scripts/run_local.sh` automatically loads `.env` from the project root when the file exists. The script already computes project-relative defaults, so `.env` is only needed when you want to override them.

```bash
# Optional if you want to override defaults:
cp .env_template .env
# Typical overrides:
# - DOC_PREPROCESS_PYTHON_BIN
# - DOC_PREPROCESS_LIBREOFFICE_BIN
# - DOC_PREPROCESS_TESSERACT_BIN
# - DOC_PREPROCESS_EXAMPLES_DIR
# - DOC_PREPROCESS_OUTPUT_DIR

bash code/scripts/run_local.sh
```

What the helper script does:

- validates that the selected Python resolves to version `3.12`
- creates or reuses `code/.venv`
- rewrites `code/scripts/requirements.txt` with the pinned dependency set
- installs dependencies
- regenerates sample input files in `code/examples/`
- writes `code/output/preprocessed.jsonl`
- runs `code/verify_output.py` to validate structure and source coverage

## Input, Output, and Verification
- The application loads input documents from `code/examples/`.
- The automation writes output artifacts to `code/output/preprocessed.jsonl`.
- `run_local.sh` verifies that:
  - the JSONL file exists and is not empty,
  - each JSONL record contains required fields,
  - every supported file in `code/examples/` is represented in the output.
- If verification fails, the script exits non-zero.
- If verification succeeds, it prints a clear `SUCCESS` message.
