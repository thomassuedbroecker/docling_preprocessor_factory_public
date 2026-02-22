"""
Docling Multi-Format Preprocessing Pipeline
===========================================

Purpose
-------
This module implements a structured, class-based preprocessing pipeline
using Docling to extract content from:

- PDF  → page-wise
- DOCX → document-level (Word does not expose stable physical pages)
- PPTX → slide-wise
- XLSX → sheet-wise (if supported by installed Docling version)

The extracted content is exported as Markdown per logical unit
(page / slide / sheet / document) and serialized as JSONL.
This prepares the data for downstream chunking and embedding workflows.

Design Principles
-----------------
- Clear separation of concerns
- Format detection isolated from conversion
- Extraction logic separated from orchestration
- JSONL output suitable for large-scale pipelines
- Easy extension for additional formats or metadata enrichment

Dependencies
------------
pip install docling

Notes
-----
- Page numbering in Docling is typically 1-based.
- DOCX does not reliably expose physical page boundaries.
- XLSX support depends on the installed Docling version.
"""

from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, Optional

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption,
)
from docling.datamodel.base_models import InputFormat


# ==========================================================
# Data Model
# ==========================================================

@dataclass(frozen=True)
class ExtractionUnit:
    """
    Represents one logical extraction unit.

    Depending on format:
        PDF   → page
        PPTX  → slide
        XLSX  → sheet
        DOCX  → full document (single unit)
    """
    source_path: str
    input_format: str
    unit_number: int          # 1-based index
    unit_type: str            # "page", "slide", "sheet", "document"
    text_markdown: str
    metadata: Dict[str, object]


# ==========================================================
# File Type Detection
# ==========================================================

class FileTypeDetector:
    """
    Determines the Docling InputFormat based on file extension.
    """

    EXTENSION_MAPPING: Dict[str, InputFormat] = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".xlsx": getattr(InputFormat, "XLSX", None),
        ".xls": getattr(InputFormat, "XLSX", None),
    }

    def detect(self, file_path: Path) -> InputFormat:
        ext = file_path.suffix.lower()
        detected_format = self.EXTENSION_MAPPING.get(ext)

        if detected_format is None:
            mime, _ = mimetypes.guess_type(str(file_path))
            raise ValueError(
                f"Unsupported file type: {file_path.name} "
                f"(extension={ext}, mimetype={mime})"
            )

        return detected_format


# ==========================================================
# Converter Factory
# ==========================================================

class DoclingConverterFactory:
    """
    Responsible for constructing a properly configured
    Docling DocumentConverter instance.
    """

    @staticmethod
    def build() -> DocumentConverter:
        format_options = {
            InputFormat.PDF: PdfFormatOption(),
            InputFormat.DOCX: WordFormatOption(),
            InputFormat.PPTX: PowerpointFormatOption(),
        }

        return DocumentConverter(format_options=format_options)


# ==========================================================
# Extraction Logic
# ==========================================================

class UnitExtractor:
    """
    Extracts logical units from a Docling conversion result.
    """

    def extract(self, file_path: Path, input_format: InputFormat) -> Iterator[ExtractionUnit]:
        converter = DoclingConverterFactory.build()
        result = converter.convert(str(file_path))
        document = result.document

        unit_type = self._resolve_unit_type(input_format)

        if input_format == InputFormat.PDF:
            total_pages = document.num_pages()
            for page_no in range(1, total_pages + 1):
                yield ExtractionUnit(
                    source_path=str(file_path),
                    input_format=input_format.name,
                    unit_number=page_no,
                    unit_type="page",
                    text_markdown=document.export_to_markdown(page_no=page_no),
                    metadata={
                        "total_pages": total_pages,
                        "page_number": page_no,
                    },
                )

        elif input_format == InputFormat.PPTX:
            total_slides = document.num_pages()
            for slide_no in range(1, total_slides + 1):
                yield ExtractionUnit(
                    source_path=str(file_path),
                    input_format=input_format.name,
                    unit_number=slide_no,
                    unit_type="slide",
                    text_markdown=document.export_to_markdown(page_no=slide_no),
                    metadata={
                        "total_slides": total_slides,
                        "slide_number": slide_no,
                    },
                )

        elif input_format.name == "XLSX":
            # XLSX may not expose page count depending on version.
            # Fallback: export entire document as one logical unit.
            yield ExtractionUnit(
                source_path=str(file_path),
                input_format=input_format.name,
                unit_number=1,
                unit_type="sheet",
                text_markdown=document.export_to_markdown(),
                metadata={
                    "note": "Sheet-wise extraction depends on Docling version.",
                },
            )

        else:
            # DOCX fallback (single document-level unit)
            yield ExtractionUnit(
                source_path=str(file_path),
                input_format=input_format.name,
                unit_number=1,
                unit_type="document",
                text_markdown=document.export_to_markdown(),
                metadata={
                    "note": "DOCX does not guarantee physical page segmentation.",
                },
            )

    @staticmethod
    def _resolve_unit_type(input_format: InputFormat) -> str:
        if input_format == InputFormat.PDF:
            return "page"
        if input_format == InputFormat.PPTX:
            return "slide"
        if input_format.name == "XLSX":
            return "sheet"
        return "document"


# ==========================================================
# JSONL Writer
# ==========================================================

class JSONLWriter:
    """
    Writes extraction units into JSONL format.
    """

    @staticmethod
    def write(units: Iterator[ExtractionUnit], output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            for unit in units:
                f.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")


# ==========================================================
# Orchestrator
# ==========================================================

class PreprocessingPipeline:
    """
    High-level orchestration class coordinating:
        - Format detection
        - Unit extraction
        - JSONL serialization
    """

    def __init__(self):
        self.detector = FileTypeDetector()
        self.extractor = UnitExtractor()
        self.writer = JSONLWriter()

    def process(self, input_path: Path, output_path: Path) -> None:
        input_format = self.detector.detect(input_path)
        units = self.extractor.extract(input_path, input_format)
        self.writer.write(units, output_path)


# ==========================================================
# Example Usage
# ==========================================================

class PipelineCLI:
    """
    Environment-driven CLI wrapper for local invocation.

    Required environment variables:
        DOC_PREPROCESS_PIPELINE_INPUT_FILE
        DOC_PREPROCESS_PIPELINE_OUTPUT_FILE
    """

    ENV_INPUT_FILE = "DOC_PREPROCESS_PIPELINE_INPUT_FILE"
    ENV_OUTPUT_FILE = "DOC_PREPROCESS_PIPELINE_OUTPUT_FILE"

    @classmethod
    def run(cls) -> None:
        input_file_raw = os.environ.get(cls.ENV_INPUT_FILE, "").strip()
        output_file_raw = os.environ.get(cls.ENV_OUTPUT_FILE, "").strip()

        if not input_file_raw:
            raise RuntimeError(
                f"Missing environment variable '{cls.ENV_INPUT_FILE}'."
            )
        if not output_file_raw:
            raise RuntimeError(
                f"Missing environment variable '{cls.ENV_OUTPUT_FILE}'."
            )

        pipeline = PreprocessingPipeline()
        input_file = Path(input_file_raw)
        output_file = Path(output_file_raw)
        pipeline.process(input_file, output_file)
        print(f"Preprocessing complete. Output written to: {output_file}")


if __name__ == "__main__":
    PipelineCLI.run()
