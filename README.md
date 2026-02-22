# Docling Preprocessor Factory

_Note: This readme was developed with the help of AI._

## Repository Objective
This repository provides a reusable, local-first preprocessing pipeline for multi-format business documents (PDF, DOCX, XLS/XLSX, PPT/PPTX).  
Its objective is to generate sample input files, extract content unit-wise (page, slide, sheet, document), perform best-effort OCR, and produce standardized JSONL output suitable for downstream chunking and AI workflows.

## Scope
- Run locally on macOS via Bash automation in `code/scripts/run_local.sh`
- Use Python 3.12 with class-based implementation for reusability
- Read environment-specific configuration from environment variables (via `.env`)
- Validate execution and fail fast when output verification does not pass

## [Quickstart](./QUICKSTART.md)

## [Documentation](./documentation/DOCUMENTATION.md)
