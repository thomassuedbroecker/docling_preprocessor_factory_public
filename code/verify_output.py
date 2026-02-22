#!/usr/bin/env python3
# This code was developed with the help of AI.

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


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
    examples_dir: Path
    output_jsonl: Path
    supported_extensions: Set[str]
    required_fields: Set[str]
    required_python: str


class VerificationConfigLoader:
    ENV_EXAMPLES_DIR = "DOC_PREPROCESS_EXAMPLES_DIR"
    ENV_OUTPUT_JSONL = "DOC_PREPROCESS_OUTPUT_JSONL"
    ENV_SUPPORTED_EXTENSIONS = "DOC_PREPROCESS_SUPPORTED_EXTENSIONS"
    ENV_REQUIRED_FIELDS = "DOC_PREPROCESS_REQUIRED_FIELDS"
    ENV_REQUIRED_PYTHON = "DOC_PREPROCESS_REQUIRED_PYTHON"

    def __init__(self, args: argparse.Namespace, env: Dict[str, str]):
        self.args = args
        self.env = env

    def load(self) -> VerificationConfig:
        examples_dir = self._path_value(self.args.examples_dir, self.ENV_EXAMPLES_DIR)
        output_jsonl = self._path_value(self.args.output_jsonl, self.ENV_OUTPUT_JSONL)

        supported_extensions = self._csv_to_set(
            self._text_value(self.args.supported_extensions, self.ENV_SUPPORTED_EXTENSIONS), normalize_extensions=True
        )
        required_fields = self._csv_to_set(
            self._text_value(self.args.required_fields, self.ENV_REQUIRED_FIELDS), normalize_extensions=False
        )

        required_python = self._text_value(self.args.required_python, self.ENV_REQUIRED_PYTHON)

        return VerificationConfig(
            examples_dir=examples_dir,
            output_jsonl=output_jsonl,
            supported_extensions=supported_extensions,
            required_fields=required_fields,
            required_python=required_python,
        )

    def _path_value(self, arg_value: str | None, env_name: str) -> Path:
        value = self._text_value(arg_value, env_name)
        return Path(value).resolve()

    def _text_value(self, arg_value: str | None, env_name: str) -> str:
        if arg_value is not None and str(arg_value).strip() != "":
            return str(arg_value).strip()

        env_value = str(self.env.get(env_name, "")).strip()
        if env_value == "":
            raise RuntimeError(f"Missing configuration. Set environment variable '{env_name}' or pass CLI option.")
        return env_value

    @staticmethod
    def _csv_to_set(raw_value: str, normalize_extensions: bool) -> Set[str]:
        parsed: Set[str] = set()
        for token in raw_value.split(","):
            item = token.strip()
            if not item:
                continue
            if normalize_extensions:
                normalized = item.lower()
                parsed.add(normalized if normalized.startswith(".") else f".{normalized}")
            else:
                parsed.add(item)

        if not parsed:
            raise RuntimeError("Configured CSV list is empty.")

        return parsed


class CLIParserFactory:
    @staticmethod
    def build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Verify preprocessing JSONL output.")
        parser.add_argument("--examples-dir", default=None, type=str)
        parser.add_argument("--output-jsonl", default=None, type=str)
        parser.add_argument("--supported-extensions", default=None, type=str)
        parser.add_argument("--required-fields", default=None, type=str)
        parser.add_argument("--required-python", default=None, type=str)
        return parser


class OutputVerifier:
    def __init__(self, config: VerificationConfig):
        self.config = config

    def verify(self) -> None:
        if not self.config.examples_dir.is_dir():
            raise RuntimeError(f"Examples directory does not exist: {self.config.examples_dir}")
        if not self.config.output_jsonl.is_file():
            raise RuntimeError(f"Output file does not exist: {self.config.output_jsonl}")

        docs = sorted(
            path.resolve()
            for path in self.config.examples_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.config.supported_extensions
        )
        if not docs:
            raise RuntimeError(
                f"No supported files found in examples directory: {self.config.examples_dir}"
            )

        records = self._load_and_validate_records()
        record_sources = self._validate_record_sources(records)

        missing_docs = [str(path) for path in docs if path not in record_sources]
        if missing_docs:
            raise RuntimeError("Some example files were not processed:\n" + "\n".join(missing_docs))

    def _load_and_validate_records(self) -> List[dict]:
        records: List[dict] = []
        with self.config.output_jsonl.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON on line {line_no} in {self.config.output_jsonl}: {exc}"
                    ) from exc

                missing = sorted(self.config.required_fields - set(record.keys()))
                if missing:
                    raise RuntimeError(
                        f"Missing required fields {missing} on line {line_no} in {self.config.output_jsonl}"
                    )

                records.append(record)

        if not records:
            raise RuntimeError(f"No records found in {self.config.output_jsonl}")
        return records

    def _validate_record_sources(self, records: List[dict]) -> Set[Path]:
        valid_sources: Set[Path] = set()
        for idx, record in enumerate(records, start=1):
            source = Path(str(record.get("source_file_path", ""))).resolve()
            if not source.exists():
                raise RuntimeError(f"Record {idx} points to missing source file: {source}")
            try:
                source.relative_to(self.config.examples_dir)
            except ValueError as exc:
                raise RuntimeError(f"Record {idx} source is not in examples/: {source}") from exc
            valid_sources.add(source)

        return valid_sources


class VerificationApplication:
    def __init__(self, args: argparse.Namespace, env: Dict[str, str]):
        self.config = VerificationConfigLoader(args=args, env=env).load()

    def run(self) -> int:
        PythonVersionEnforcer(self.config.required_python).validate()
        verifier = OutputVerifier(self.config)
        verifier.verify()
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
        Logger.error(f"Verification failed. {exc}")
        raise SystemExit(1)
