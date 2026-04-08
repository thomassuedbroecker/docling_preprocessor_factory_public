"""
Focused Docling configuration examples extracted from this repository.

The default application in this project mixes Docling with other format-specific
libraries. This module shows only the Docling-specific converter setup that is
relevant for the example pipeline and the OCR variants that need extra runtime
support.

For the Docling-only examples in this file, the minimal Python dependency file
is `requirements-docling-only.txt`.
"""

from __future__ import annotations

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)


def _base_format_options(pdf_format_option: PdfFormatOption) -> dict[InputFormat, object]:
    return {
        InputFormat.PDF: pdf_format_option,
        InputFormat.DOCX: WordFormatOption(),
        InputFormat.PPTX: PowerpointFormatOption(),
    }


def build_example_converter() -> DocumentConverter:
    """
    Shows the explicit Docling configuration used for the example pipeline.

    In the tested `docling==2.20.0` environment this uses EasyOCR through
    Docling's PDF pipeline.
    """
    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=EasyOcrOptions(),
    )
    return DocumentConverter(
        format_options=_base_format_options(
            PdfFormatOption(pipeline_options=pdf_options)
        )
    )


def build_example_converter_without_ocr() -> DocumentConverter:
    """
    Minimal Docling setup when only native PDF text extraction is needed.
    """
    pdf_options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options=_base_format_options(
            PdfFormatOption(pipeline_options=pdf_options)
        )
    )


def build_example_converter_with_tesseract(
    tesseract_cmd: str = "tesseract",
) -> DocumentConverter:
    """
    OCR variant that requires the system `tesseract` binary.
    """
    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=TesseractCliOcrOptions(
            lang=["eng"],
            tesseract_cmd=tesseract_cmd,
        ),
    )
    return DocumentConverter(
        format_options=_base_format_options(
            PdfFormatOption(pipeline_options=pdf_options)
        )
    )


def build_example_converter_with_rapidocr() -> DocumentConverter:
    """
    OCR variant that requires `pip install rapidocr_onnxruntime`.
    """
    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(),
    )
    return DocumentConverter(
        format_options=_base_format_options(
            PdfFormatOption(pipeline_options=pdf_options)
        )
    )


def export_pdf_pages_as_markdown(
    pdf_path: str,
    converter: DocumentConverter,
) -> dict[int, str]:
    """
    Small helper used by the example snippets below.
    """
    result = converter.convert(pdf_path)
    document = result.document
    total_pages = int(document.num_pages())
    return {
        page_number: document.export_to_markdown(page_no=page_number) or ""
        for page_number in range(1, total_pages + 1)
    }
