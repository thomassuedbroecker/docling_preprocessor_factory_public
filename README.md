# Docling Preprocessor Factory

_Note: This readme was developed with the help of AI._

Related blog post [Building a Reproducible AI-Generated Project with ChatGPT, Codex, and Docling in VS Code](https://suedbroecker.net/2026/02/22/building-a-reproducible-ai-generated-project-with-chatgpt-codex-and-docling-in-vs-code/) on [suedbroecker.net](https://www.suedbroecker.com).

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
