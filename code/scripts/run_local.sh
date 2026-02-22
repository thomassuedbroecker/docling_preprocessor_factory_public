#!/usr/bin/env bash
# This code was developed with the help of AI.
set -euo pipefail

trap 'echo "ERROR: A command failed. Check logs above for details." >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${DEFAULT_CODE_DIR}/.." && pwd)"

DOC_PREPROCESS_ENV_FILE="${DOC_PREPROCESS_ENV_FILE:-${DEFAULT_PROJECT_DIR}/.env}"
if [[ -f "${DOC_PREPROCESS_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOC_PREPROCESS_ENV_FILE}"
  set +a
fi

export DOC_PREPROCESS_SCRIPT_DIR="${DOC_PREPROCESS_SCRIPT_DIR:-${SCRIPT_DIR}}"
export DOC_PREPROCESS_CODE_DIR="${DOC_PREPROCESS_CODE_DIR:-${DEFAULT_CODE_DIR}}"
export DOC_PREPROCESS_PROJECT_DIR="${DOC_PREPROCESS_PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"

export DOC_PREPROCESS_REQUIRED_PYTHON="${DOC_PREPROCESS_REQUIRED_PYTHON:-3.12}"
export DOC_PREPROCESS_PYTHON_BIN="${DOC_PREPROCESS_PYTHON_BIN:-python3.12}"

export DOC_PREPROCESS_EXAMPLES_DIR="${DOC_PREPROCESS_EXAMPLES_DIR:-${DOC_PREPROCESS_CODE_DIR}/examples}"
export DOC_PREPROCESS_OUTPUT_DIR="${DOC_PREPROCESS_OUTPUT_DIR:-${DOC_PREPROCESS_CODE_DIR}/output}"
export DOC_PREPROCESS_VENV_DIR="${DOC_PREPROCESS_VENV_DIR:-${DOC_PREPROCESS_CODE_DIR}/.venv}"
export DOC_PREPROCESS_REQUIREMENTS_FILE="${DOC_PREPROCESS_REQUIREMENTS_FILE:-${DOC_PREPROCESS_SCRIPT_DIR}/requirements.txt}"
export DOC_PREPROCESS_APP_FILE="${DOC_PREPROCESS_APP_FILE:-${DOC_PREPROCESS_CODE_DIR}/preprocess_app.py}"
export DOC_PREPROCESS_VERIFY_FILE="${DOC_PREPROCESS_VERIFY_FILE:-${DOC_PREPROCESS_CODE_DIR}/verify_output.py}"
export DOC_PREPROCESS_OUTPUT_JSONL="${DOC_PREPROCESS_OUTPUT_JSONL:-${DOC_PREPROCESS_OUTPUT_DIR}/preprocessed.jsonl}"

export DOC_PREPROCESS_TESSERACT_BIN="${DOC_PREPROCESS_TESSERACT_BIN:-tesseract}"
export DOC_PREPROCESS_SUPPORTED_EXTENSIONS="${DOC_PREPROCESS_SUPPORTED_EXTENSIONS:-.pdf,.docx,.xlsx,.xls,.pptx,.ppt}"
export DOC_PREPROCESS_REQUIRED_FIELDS="${DOC_PREPROCESS_REQUIRED_FIELDS:-source_file_path,input_format,unit_number,unit_type,text_markdown,ocr_image_text,metadata}"

export DOC_PREPROCESS_SAMPLE_IMAGE_NAME="${DOC_PREPROCESS_SAMPLE_IMAGE_NAME:-sample_ocr_image.png}"
export DOC_PREPROCESS_SAMPLE_PDF_NAME="${DOC_PREPROCESS_SAMPLE_PDF_NAME:-sample.pdf}"
export DOC_PREPROCESS_SAMPLE_DOCX_NAME="${DOC_PREPROCESS_SAMPLE_DOCX_NAME:-sample.docx}"
export DOC_PREPROCESS_SAMPLE_XLSX_NAME="${DOC_PREPROCESS_SAMPLE_XLSX_NAME:-sample.xlsx}"
export DOC_PREPROCESS_SAMPLE_XLS_NAME="${DOC_PREPROCESS_SAMPLE_XLS_NAME:-sample.xls}"
export DOC_PREPROCESS_SAMPLE_PPTX_NAME="${DOC_PREPROCESS_SAMPLE_PPTX_NAME:-sample.pptx}"
export DOC_PREPROCESS_SAMPLE_PPT_NAME="${DOC_PREPROCESS_SAMPLE_PPT_NAME:-sample.ppt}"

if [[ -z "${DOC_PREPROCESS_LIBREOFFICE_BIN:-}" ]]; then
  if command -v soffice >/dev/null 2>&1; then
    export DOC_PREPROCESS_LIBREOFFICE_BIN
    DOC_PREPROCESS_LIBREOFFICE_BIN="$(command -v soffice)"
  else
    export DOC_PREPROCESS_LIBREOFFICE_BIN=""
    echo "INFO: LibreOffice was not found. Legacy .ppt generation/conversion will be skipped." >&2
    echo "INFO: Set DOC_PREPROCESS_LIBREOFFICE_BIN to a valid soffice path to enable .ppt support." >&2
  fi
fi

if ! command -v "${DOC_PREPROCESS_TESSERACT_BIN}" >/dev/null 2>&1; then
  echo "INFO: Tesseract was not found via DOC_PREPROCESS_TESSERACT_BIN='${DOC_PREPROCESS_TESSERACT_BIN}'." >&2
  echo "INFO: OCR image text will run in best-effort mode and may be empty." >&2
fi

validate_python_version() {
  local python_bin="$1"
  local required_version="$2"
  local current_version

  current_version="$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${current_version}" != "${required_version}" ]]; then
    echo "ERROR: Python ${required_version} is required, but ${python_bin} resolved to ${current_version}." >&2
    exit 1
  fi
}

if ! command -v "${DOC_PREPROCESS_PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: DOC_PREPROCESS_PYTHON_BIN='${DOC_PREPROCESS_PYTHON_BIN}' was not found." >&2
  echo "ERROR: Install Python ${DOC_PREPROCESS_REQUIRED_PYTHON} and set DOC_PREPROCESS_PYTHON_BIN accordingly." >&2
  exit 1
fi

validate_python_version "${DOC_PREPROCESS_PYTHON_BIN}" "${DOC_PREPROCESS_REQUIRED_PYTHON}"

if [[ ! -f "${DOC_PREPROCESS_APP_FILE}" ]]; then
  echo "ERROR: Python application file not found: ${DOC_PREPROCESS_APP_FILE}" >&2
  exit 1
fi

if [[ ! -f "${DOC_PREPROCESS_VERIFY_FILE}" ]]; then
  echo "ERROR: Verification script file not found: ${DOC_PREPROCESS_VERIFY_FILE}" >&2
  exit 1
fi

mkdir -p "${DOC_PREPROCESS_EXAMPLES_DIR}" "${DOC_PREPROCESS_OUTPUT_DIR}"

cat > "${DOC_PREPROCESS_REQUIREMENTS_FILE}" <<'REQ'
docling==2.20.0
reportlab==4.2.5
python-docx==1.1.2
openpyxl==3.1.5
xlwt==1.3.0
xlrd==2.0.1
python-pptx==1.0.2
Pillow==10.4.0
PyMuPDF==1.24.11
pytesseract==0.3.13
REQ

if [[ ! -d "${DOC_PREPROCESS_VENV_DIR}" ]]; then
  "${DOC_PREPROCESS_PYTHON_BIN}" -m venv "${DOC_PREPROCESS_VENV_DIR}"
fi

VENV_PY="${DOC_PREPROCESS_VENV_DIR}/bin/python"
VENV_PIP="${DOC_PREPROCESS_VENV_DIR}/bin/pip"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: Virtual environment Python is missing at ${VENV_PY}. Remove ${DOC_PREPROCESS_VENV_DIR} and rerun." >&2
  exit 1
fi

validate_python_version "${VENV_PY}" "${DOC_PREPROCESS_REQUIRED_PYTHON}"

if ! "${VENV_PY}" -m pip install --upgrade pip setuptools wheel; then
  echo "WARNING: Could not upgrade pip/setuptools/wheel. Continuing with existing versions." >&2
fi

if ! "${VENV_PIP}" install -r "${DOC_PREPROCESS_REQUIREMENTS_FILE}"; then
  echo "ERROR: Failed to install dependencies from ${DOC_PREPROCESS_REQUIREMENTS_FILE}." >&2
  echo "ERROR: Check internet connectivity or configure a local Python package mirror, then rerun." >&2
  exit 1
fi

APP_CMD=(
  "${VENV_PY}" "${DOC_PREPROCESS_APP_FILE}"
  "--input-dir" "${DOC_PREPROCESS_EXAMPLES_DIR}"
  "--output-dir" "${DOC_PREPROCESS_OUTPUT_DIR}"
  "--output-jsonl" "${DOC_PREPROCESS_OUTPUT_JSONL}"
  "--tesseract-bin" "${DOC_PREPROCESS_TESSERACT_BIN}"
  "--supported-extensions" "${DOC_PREPROCESS_SUPPORTED_EXTENSIONS}"
  "--required-python" "${DOC_PREPROCESS_REQUIRED_PYTHON}"
  "--sample-image-name" "${DOC_PREPROCESS_SAMPLE_IMAGE_NAME}"
  "--sample-pdf-name" "${DOC_PREPROCESS_SAMPLE_PDF_NAME}"
  "--sample-docx-name" "${DOC_PREPROCESS_SAMPLE_DOCX_NAME}"
  "--sample-xlsx-name" "${DOC_PREPROCESS_SAMPLE_XLSX_NAME}"
  "--sample-xls-name" "${DOC_PREPROCESS_SAMPLE_XLS_NAME}"
  "--sample-pptx-name" "${DOC_PREPROCESS_SAMPLE_PPTX_NAME}"
  "--sample-ppt-name" "${DOC_PREPROCESS_SAMPLE_PPT_NAME}"
)

if [[ -n "${DOC_PREPROCESS_LIBREOFFICE_BIN}" ]]; then
  APP_CMD+=("--libreoffice-bin" "${DOC_PREPROCESS_LIBREOFFICE_BIN}")
fi

"${APP_CMD[@]}"

if [[ ! -f "${DOC_PREPROCESS_OUTPUT_JSONL}" ]]; then
  echo "ERROR: Verification failed. Expected output file not found: ${DOC_PREPROCESS_OUTPUT_JSONL}" >&2
  exit 1
fi

if [[ ! -s "${DOC_PREPROCESS_OUTPUT_JSONL}" ]]; then
  echo "ERROR: Verification failed. Output file is empty: ${DOC_PREPROCESS_OUTPUT_JSONL}" >&2
  exit 1
fi

VERIFY_CMD=(
  "${VENV_PY}" "${DOC_PREPROCESS_VERIFY_FILE}"
  "--examples-dir" "${DOC_PREPROCESS_EXAMPLES_DIR}"
  "--output-jsonl" "${DOC_PREPROCESS_OUTPUT_JSONL}"
  "--supported-extensions" "${DOC_PREPROCESS_SUPPORTED_EXTENSIONS}"
  "--required-fields" "${DOC_PREPROCESS_REQUIRED_FIELDS}"
  "--required-python" "${DOC_PREPROCESS_REQUIRED_PYTHON}"
)

"${VERIFY_CMD[@]}"

echo "SUCCESS: End-to-end run completed and verification passed. Output file: ${DOC_PREPROCESS_OUTPUT_JSONL}"
