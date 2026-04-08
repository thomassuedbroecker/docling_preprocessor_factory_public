#!/usr/bin/env python3
# This code was developed with the help of AI.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DependencySpec:
    name: str
    version: str
    license_name: str
    used_for: str


DEPENDENCY_SCOPE_EXPLANATION: tuple[str, ...] = (
    "These libs are not required by Docling itself.",
    "They are required by this repository's default implementation choices.",
    "If you reduce the project to Docling-only conversion, most of them can be removed.",
)

DEFAULT_WORKFLOW_DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("docling", "2.20.0", "MIT", "Primary PDF-to-markdown extraction path."),
    DependencySpec("reportlab", "4.2.5", "BSD-style", "Sample PDF generation."),
    DependencySpec("python-docx", "1.1.2", "MIT", "DOCX generation and text extraction."),
    DependencySpec("openpyxl", "3.1.5", "MIT", "XLSX generation and worksheet extraction."),
    DependencySpec("xlwt", "1.3.0", "BSD", "Legacy XLS generation."),
    DependencySpec("xlrd", "2.0.1", "BSD", "Legacy XLS reading."),
    DependencySpec("python-pptx", "1.0.2", "MIT", "PPTX generation and slide extraction."),
    DependencySpec("Pillow", "10.4.0", "HPND", "OCR sample image generation and image handling."),
    DependencySpec("PyMuPDF", "1.24.11", "AGPL-3.0", "PDF fallback text extraction and page rasterization for OCR."),
    DependencySpec("pytesseract", "0.3.13", "Apache-2.0", "Python wrapper for OCR execution."),
)

DOCLING_ONLY_DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("docling", "2.20.0", "MIT", "Standalone Docling-only conversion modules."),
)

OPTIONAL_RUNTIME_NOTES: tuple[str, ...] = (
    "`tesseract` is still needed when you select the Tesseract OCR path.",
    "`rapidocr_onnxruntime` is only needed when you select the RapidOCR path.",
    "`soffice` is only needed by the default workflow for legacy `.ppt` conversion.",
)

PROFILE_ALIASES = {
    "default": DEFAULT_WORKFLOW_DEPENDENCIES,
    "default-workflow": DEFAULT_WORKFLOW_DEPENDENCIES,
    "docling-only": DOCLING_ONLY_DEPENDENCIES,
    "docling_only": DOCLING_ONLY_DEPENDENCIES,
}


def get_profile(profile_name: str) -> tuple[DependencySpec, ...]:
    try:
        return PROFILE_ALIASES[profile_name]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILE_ALIASES))
        raise ValueError(f"Unsupported dependency profile '{profile_name}'. Expected one of: {supported}") from exc


def render_requirements(dependencies: Sequence[DependencySpec]) -> str:
    return "\n".join(f"{dependency.name}=={dependency.version}" for dependency in dependencies) + "\n"


def render_markdown_table(dependencies: Sequence[DependencySpec]) -> str:
    lines = [
        "| Library | Installed version | License | Used for |",
        "| --- | --- | --- | --- |",
    ]
    for dependency in dependencies:
        lines.append(
            f"| `{dependency.name}` | `{dependency.version}` | {dependency.license_name} | {dependency.used_for} |"
        )
    return "\n".join(lines)


def render_markdown_bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


class CLI:
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Render canonical dependency profiles for this project.")
        subparsers = parser.add_subparsers(dest="command", required=True)

        write_requirements = subparsers.add_parser(
            "write-requirements",
            help="Write a canonical requirements file for the selected profile.",
        )
        write_requirements.add_argument("--profile", required=True, type=str)
        write_requirements.add_argument("--output", required=True, type=str)

        print_markdown = subparsers.add_parser(
            "print-markdown-table",
            help="Print the markdown dependency table for the selected profile.",
        )
        print_markdown.add_argument("--profile", required=True, type=str)

        subparsers.add_parser(
            "print-scope-bullets",
            help="Print the canonical explanation bullets for dependency scope.",
        )

        print_optional_runtimes = subparsers.add_parser(
            "print-optional-runtime-bullets",
            help="Print optional runtime notes for OCR and conversion variants.",
        )
        print_optional_runtimes.set_defaults(command="print-optional-runtime-bullets")

        return parser

    @staticmethod
    def run() -> int:
        parser = CLI.build_parser()
        args = parser.parse_args()

        if args.command == "write-requirements":
            output_path = Path(args.output).resolve()
            output_path.write_text(
                render_requirements(get_profile(args.profile)),
                encoding="utf-8",
            )
            return 0

        if args.command == "print-markdown-table":
            print(render_markdown_table(get_profile(args.profile)))
            return 0

        if args.command == "print-scope-bullets":
            print(render_markdown_bullets(DEPENDENCY_SCOPE_EXPLANATION))
            return 0

        if args.command == "print-optional-runtime-bullets":
            print(render_markdown_bullets(OPTIONAL_RUNTIME_NOTES))
            return 0

        raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(CLI.run())
