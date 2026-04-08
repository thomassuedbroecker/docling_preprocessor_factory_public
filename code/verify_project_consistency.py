#!/usr/bin/env python3
# This code was developed with the help of AI.

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dependency_profiles import (
    DEFAULT_WORKFLOW_DEPENDENCIES,
    DEPENDENCY_SCOPE_EXPLANATION,
    DOCLING_ONLY_DEPENDENCIES,
    OPTIONAL_RUNTIME_NOTES,
    render_markdown_bullets,
    render_markdown_table,
    render_requirements,
)


class Logger:
    @staticmethod
    def error(message: str) -> None:
        print(f"ERROR: {message}", file=sys.stderr)


class PythonVersionEnforcer:
    def __init__(self, required_version: str):
        self.required_version = required_version

    def validate(self) -> None:
        parts = self.required_version.strip().split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise RuntimeError(
                f"Invalid required Python version format: '{self.required_version}'. Use '<major>.<minor>' like '3.12'."
            )

        required_major, required_minor = int(parts[0]), int(parts[1])
        current = (sys.version_info.major, sys.version_info.minor)
        if current != (required_major, required_minor):
            raise RuntimeError(
                f"Python {required_major}.{required_minor} is required, but current runtime is {current[0]}.{current[1]}."
            )


@dataclass(frozen=True)
class VerificationConfig:
    project_dir: Path
    required_python: str


class ConfigLoader:
    ENV_PROJECT_DIR = "DOC_PREPROCESS_PROJECT_DIR"
    ENV_REQUIRED_PYTHON = "DOC_PREPROCESS_REQUIRED_PYTHON"

    def __init__(self, args: argparse.Namespace, env: dict[str, str]):
        self.args = args
        self.env = env

    def load(self) -> VerificationConfig:
        return VerificationConfig(
            project_dir=self._path_value(self.args.project_dir, self.ENV_PROJECT_DIR),
            required_python=self._text_value(self.args.required_python, self.ENV_REQUIRED_PYTHON),
        )

    def _path_value(self, arg_value: str | None, env_name: str) -> Path:
        return Path(self._text_value(arg_value, env_name)).resolve()

    def _text_value(self, arg_value: str | None, env_name: str) -> str:
        if arg_value is not None and str(arg_value).strip() != "":
            return str(arg_value).strip()

        env_value = str(self.env.get(env_name, "")).strip()
        if env_value == "":
            raise RuntimeError(f"Missing configuration. Set environment variable '{env_name}' or pass CLI option.")
        return env_value


class CLIParserFactory:
    @staticmethod
    def build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Verify project dependency docs and requirements stay in sync.")
        parser.add_argument("--project-dir", default=None, type=str)
        parser.add_argument("--required-python", default=None, type=str)
        return parser


class ProjectConsistencyVerifier:
    def __init__(self, config: VerificationConfig):
        self.config = config

    def verify(self) -> None:
        if not self.config.project_dir.is_dir():
            raise RuntimeError(f"Project directory does not exist: {self.config.project_dir}")

        self._verify_requirements_files()
        self._verify_readme()
        self._verify_quickstart()
        self._verify_architecture_doc()
        self._verify_docling_module_notes()

    def _verify_requirements_files(self) -> None:
        default_requirements = render_requirements(DEFAULT_WORKFLOW_DEPENDENCIES)
        docling_only_requirements = render_requirements(DOCLING_ONLY_DEPENDENCIES)

        self._assert_file_text(
            self.config.project_dir / "requirements.txt",
            default_requirements,
            "Top-level default workflow requirements are out of sync.",
        )
        self._assert_file_text(
            self.config.project_dir / "code/scripts/requirements.txt",
            default_requirements,
            "Bootstrap requirements are out of sync.",
        )
        self._assert_file_text(
            self.config.project_dir / "requirements-docling-only.txt",
            docling_only_requirements,
            "Docling-only requirements are out of sync.",
        )

    def _verify_readme(self) -> None:
        readme_path = self.config.project_dir / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")

        self._assert_block(
            readme_text,
            "README dependency scope explanation",
            "DEPENDENCY_SCOPE_EXPLANATION",
            render_markdown_bullets(DEPENDENCY_SCOPE_EXPLANATION),
        )
        self._assert_block(
            readme_text,
            "README default dependency table",
            "DEFAULT_WORKFLOW_DEPENDENCIES",
            render_markdown_table(DEFAULT_WORKFLOW_DEPENDENCIES),
        )
        self._assert_block(
            readme_text,
            "README Docling-only dependency table",
            "DOCLING_ONLY_DEPENDENCIES",
            render_markdown_table(DOCLING_ONLY_DEPENDENCIES),
        )
        self._assert_block(
            readme_text,
            "README optional runtime notes",
            "OPTIONAL_RUNTIME_NOTES",
            render_markdown_bullets(OPTIONAL_RUNTIME_NOTES),
        )
        self._assert_contains(readme_text, "`requirements-docling-only.txt`", "README must mention the trimmed requirements file.")
        self._assert_contains(
            readme_text,
            "`code/verify_project_consistency.py`",
            "README must mention the project consistency verifier.",
        )

    def _verify_quickstart(self) -> None:
        quickstart_path = self.config.project_dir / "QUICKSTART.md"
        quickstart_text = quickstart_path.read_text(encoding="utf-8")

        self._assert_block(
            quickstart_text,
            "QUICKSTART dependency scope explanation",
            "DEPENDENCY_SCOPE_EXPLANATION",
            render_markdown_bullets(DEPENDENCY_SCOPE_EXPLANATION),
        )
        self._assert_contains(
            quickstart_text,
            "`requirements-docling-only.txt`",
            "QUICKSTART must mention the Docling-only requirements file.",
        )
        self._assert_contains(
            quickstart_text,
            "`code/verify_project_consistency.py`",
            "QUICKSTART must mention the project consistency verifier.",
        )

    def _verify_architecture_doc(self) -> None:
        architecture_path = self.config.project_dir / "documentation/DOCUMENTATION.md"
        architecture_text = architecture_path.read_text(encoding="utf-8")

        self._assert_block(
            architecture_text,
            "Architecture dependency scope explanation",
            "DEPENDENCY_SCOPE_EXPLANATION",
            render_markdown_bullets(DEPENDENCY_SCOPE_EXPLANATION),
        )
        self._assert_block(
            architecture_text,
            "Architecture default dependency table",
            "DEFAULT_WORKFLOW_DEPENDENCIES",
            render_markdown_table(DEFAULT_WORKFLOW_DEPENDENCIES),
        )
        self._assert_block(
            architecture_text,
            "Architecture Docling-only dependency table",
            "DOCLING_ONLY_DEPENDENCIES",
            render_markdown_table(DOCLING_ONLY_DEPENDENCIES),
        )
        self._assert_contains(
            architecture_text,
            "`code/verify_project_consistency.py`",
            "Architecture doc must mention the project consistency verifier.",
        )

    def _verify_docling_module_notes(self) -> None:
        standalone_text = (self.config.project_dir / "code/Docling_multi_format_preprocessing_pipeline.py").read_text(
            encoding="utf-8"
        )
        examples_text = (self.config.project_dir / "code/docling_config_examples.py").read_text(encoding="utf-8")

        expected_note = "requirements-docling-only.txt"
        if expected_note not in standalone_text:
            raise RuntimeError("Standalone Docling module must mention requirements-docling-only.txt.")
        if expected_note not in examples_text:
            raise RuntimeError("Docling configuration examples must mention requirements-docling-only.txt.")

    @staticmethod
    def _assert_file_text(path: Path, expected_text: str, message: str) -> None:
        if not path.is_file():
            raise RuntimeError(f"Required file does not exist: {path}")
        actual_text = path.read_text(encoding="utf-8")
        if actual_text != expected_text:
            raise RuntimeError(f"{message} Expected exact contents in {path}.")

    @staticmethod
    def _assert_contains(text: str, needle: str, message: str) -> None:
        if needle not in text:
            raise RuntimeError(message)

    @staticmethod
    def _assert_block(text: str, label: str, block_name: str, expected_content: str) -> None:
        start_marker = f"<!-- BEGIN:{block_name} -->"
        end_marker = f"<!-- END:{block_name} -->"
        try:
            start_index = text.index(start_marker) + len(start_marker)
            end_index = text.index(end_marker, start_index)
        except ValueError as exc:
            raise RuntimeError(f"{label} markers are missing.") from exc

        actual_content = text[start_index:end_index].strip()
        if actual_content != expected_content.strip():
            raise RuntimeError(f"{label} is out of sync.")


class VerificationApplication:
    def __init__(self, args: argparse.Namespace, env: dict[str, str]):
        self.config = ConfigLoader(args=args, env=env).load()

    def run(self) -> int:
        PythonVersionEnforcer(self.config.required_python).validate()
        ProjectConsistencyVerifier(self.config).verify()
        return 0


class ApplicationEntryPoint:
    @staticmethod
    def run() -> int:
        parser = CLIParserFactory.build()
        args = parser.parse_args()
        app = VerificationApplication(args=args, env=dict(os.environ))
        return app.run()


if __name__ == "__main__":
    try:
        raise SystemExit(ApplicationEntryPoint.run())
    except Exception as exc:
        Logger.error(f"Project consistency verification failed. {exc}")
        raise SystemExit(1)
