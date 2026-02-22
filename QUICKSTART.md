# Quickstart

_Note: This quickstart was developed with the help of AI._

## Prerequisites
- Python 3.12
- Optional: LibreOffice (required for `.ppt` generation/conversion)

## Illustrative Snippet (Not Executable As-Is)
```bash
# Demonstration only: this inline snippet is illustrative and will not run as-is.
python preprocess.py --input-dir examples --output-dir output
```

## Execute The Real Application
`code/scripts/run_local.sh` automatically loads `.env` from the project root.

```bash
cp .env_template .env
# Edit .env and set at least:
# - DOC_PREPROCESS_PROJECT_DIR
# - DOC_PREPROCESS_PYTHON_BIN
# - DOC_PREPROCESS_REQUIRED_PYTHON
# Optional:
# - DOC_PREPROCESS_LIBREOFFICE_BIN
# - DOC_PREPROCESS_EXAMPLES_DIR
# - DOC_PREPROCESS_OUTPUT_DIR

bash code/scripts/run_local.sh
```

## Input, Output, and Verification
- The application loads input documents from `code/examples/`.
- The automation writes output artifacts to `code/output/preprocessed.jsonl`.
- `run_local.sh` verifies that:
  - the JSONL file exists and is not empty,
  - each JSONL record contains required fields,
  - every supported file in `code/examples/` is represented in the output.
- If verification fails, the script exits non-zero.
- If verification succeeds, it prints a clear `SUCCESS` message.
