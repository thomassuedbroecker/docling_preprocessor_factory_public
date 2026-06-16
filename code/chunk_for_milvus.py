#!/usr/bin/env python3
# This code was developed with the help of AI.

"""Generate Milvus-ready chunk records from preprocessed JSONL output.

This module reads the unit-wise JSONL produced by ``preprocess_app.py`` and
emits a separate chunk JSONL where each line is one size-bounded text chunk
plus a configurable ``metadata`` object. The metadata object is shaped by an
external metadata-definition file (see ``--metadata-definition-file``) that
controls two things:

- ``field_allowlist``: which keys from each source record's ``metadata`` are
  carried into the chunk. An empty/missing allowlist keeps every key.
- ``static_fields``: fixed key/value pairs injected into every chunk's
  metadata (for example a Milvus collection name or a source-system tag).

No live database connection is made here. The output is a portable JSONL that
a downstream embedding/insert step can load into Milvus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class Logger:
    @staticmethod
    def info(message: str) -> None:
        print(f"INFO: {message}", file=sys.stderr)

    @staticmethod
    def warning(message: str) -> None:
        print(f"WARNING: {message}", file=sys.stderr)

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
class ChunkConfig:
    input_jsonl: Path
    chunk_output_jsonl: Path
    metadata_definition_file: Path
    chunk_size: int
    chunk_overlap: int
    include_ocr: bool
    required_python: str


class ChunkConfigLoader:
    ENV_INPUT_JSONL = "DOC_PREPROCESS_OUTPUT_JSONL"
    ENV_CHUNK_OUTPUT_JSONL = "DOC_PREPROCESS_CHUNK_OUTPUT_JSONL"
    ENV_METADATA_DEFINITION_FILE = "DOC_PREPROCESS_METADATA_DEFINITION_FILE"
    ENV_CHUNK_SIZE = "DOC_PREPROCESS_CHUNK_SIZE"
    ENV_CHUNK_OVERLAP = "DOC_PREPROCESS_CHUNK_OVERLAP"
    ENV_CHUNK_INCLUDE_OCR = "DOC_PREPROCESS_CHUNK_INCLUDE_OCR"
    ENV_REQUIRED_PYTHON = "DOC_PREPROCESS_REQUIRED_PYTHON"

    def __init__(self, env: Dict[str, str], args: argparse.Namespace):
        self.env = env
        self.args = args

    def load(self) -> ChunkConfig:
        return ChunkConfig(
            input_jsonl=self._path_value(self.args.input_jsonl, self.ENV_INPUT_JSONL),
            chunk_output_jsonl=self._path_value(self.args.chunk_output_jsonl, self.ENV_CHUNK_OUTPUT_JSONL),
            metadata_definition_file=self._path_value(
                self.args.metadata_definition_file, self.ENV_METADATA_DEFINITION_FILE
            ),
            chunk_size=self._int_value(self.args.chunk_size, self.ENV_CHUNK_SIZE, minimum=1),
            chunk_overlap=self._int_value(self.args.chunk_overlap, self.ENV_CHUNK_OVERLAP, minimum=0),
            include_ocr=self._bool_value(self.args.include_ocr, self.ENV_CHUNK_INCLUDE_OCR),
            required_python=self._text_value(self.args.required_python, self.ENV_REQUIRED_PYTHON),
        )

    def _path_value(self, arg_value: Optional[str], env_name: str) -> Path:
        return Path(self._text_value(arg_value, env_name)).resolve()

    def _text_value(self, arg_value: Optional[str], env_name: str) -> str:
        if arg_value is not None and str(arg_value).strip() != "":
            return str(arg_value).strip()
        env_value = str(self.env.get(env_name, "")).strip()
        if env_value == "":
            raise RuntimeError(f"Missing configuration. Set environment variable '{env_name}' or pass CLI option.")
        return env_value

    def _int_value(self, arg_value: Optional[str], env_name: str, minimum: int) -> int:
        raw = self._text_value(arg_value, env_name)
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"Configuration '{env_name}' must be an integer, got '{raw}'.") from exc
        if value < minimum:
            raise RuntimeError(f"Configuration '{env_name}' must be >= {minimum}, got {value}.")
        return value

    def _bool_value(self, arg_value: Optional[str], env_name: str) -> bool:
        if arg_value is not None and str(arg_value).strip() != "":
            raw = str(arg_value).strip()
        else:
            raw = str(self.env.get(env_name, "")).strip()
        if raw == "":
            return True
        normalized = raw.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(f"Configuration '{env_name}' must be a boolean (true/false), got '{raw}'.")


class CLIParserFactory:
    @staticmethod
    def build() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Generate Milvus-ready chunk records from preprocessed JSONL.")
        parser.add_argument("--input-jsonl", default=None, type=str)
        parser.add_argument("--chunk-output-jsonl", default=None, type=str)
        parser.add_argument("--metadata-definition-file", default=None, type=str)
        parser.add_argument("--chunk-size", default=None, type=str)
        parser.add_argument("--chunk-overlap", default=None, type=str)
        parser.add_argument("--include-ocr", default=None, type=str)
        parser.add_argument("--required-python", default=None, type=str)
        return parser


class MetadataDefinition:
    """Shapes the per-chunk metadata object from a source record's metadata."""

    def __init__(self, field_allowlist: List[str], static_fields: Dict[str, Any]):
        self.field_allowlist = field_allowlist
        self.static_fields = static_fields

    def build(self, source_metadata: Dict[str, Any]) -> Dict[str, Any]:
        if self.field_allowlist:
            selected = {key: source_metadata[key] for key in self.field_allowlist if key in source_metadata}
        else:
            selected = dict(source_metadata)
        # Static fields are always present and win on key collisions.
        selected.update(self.static_fields)
        return selected


class MetadataDefinitionLoader:
    @staticmethod
    def load(path: Path) -> MetadataDefinition:
        if not path.is_file():
            raise RuntimeError(f"Metadata definition file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in metadata definition file {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Metadata definition file must contain a JSON object.")

        allowlist = data.get("field_allowlist", [])
        if not isinstance(allowlist, list) or not all(isinstance(item, str) for item in allowlist):
            raise RuntimeError("'field_allowlist' must be a list of strings in the metadata definition file.")

        static_fields = data.get("static_fields", {})
        if not isinstance(static_fields, dict) or not all(isinstance(key, str) for key in static_fields):
            raise RuntimeError("'static_fields' must be a JSON object with string keys in the metadata definition file.")

        return MetadataDefinition(field_allowlist=allowlist, static_fields=static_fields)


class TextChunker:
    """Splits text into size-bounded, overlapping chunks on word/line boundaries."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_size < 1:
            raise RuntimeError("chunk-size must be a positive integer.")
        if chunk_overlap < 0:
            raise RuntimeError("chunk-overlap must be zero or a positive integer.")
        if chunk_overlap >= chunk_size:
            raise RuntimeError("chunk-overlap must be smaller than chunk-size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: List[str] = []
        start = 0
        length = len(normalized)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                window = normalized[start:end]
                boundary = window.rfind("\n")
                if boundary == -1:
                    boundary = window.rfind(" ")
                # Honor the boundary only when it keeps the chunk at least half-size,
                # so we avoid emitting many tiny chunks.
                if boundary != -1 and boundary >= self.chunk_size // 2:
                    end = start + boundary + 1

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= length:
                break

            next_start = end - self.chunk_overlap
            # Guarantee forward progress even in degenerate boundary cases.
            start = next_start if next_start > start else end

        return chunks


class ChunkRecordFactory:
    REQUIRED_FIELDS = (
        "chunk_id",
        "text",
        "source_file_path",
        "input_format",
        "unit_number",
        "unit_type",
        "chunk_index",
        "chunk_count",
        "metadata",
    )

    @staticmethod
    def build(
        source_record: Dict[str, Any],
        chunk_text: str,
        chunk_index: int,
        chunk_count: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_file_path = str(source_record.get("source_file_path", ""))
        unit_number = source_record.get("unit_number", 0)
        digest = hashlib.sha1(f"{source_file_path}:{unit_number}".encode("utf-8")).hexdigest()[:10]
        return {
            "chunk_id": f"{digest}-u{unit_number}-c{chunk_index}",
            "text": chunk_text,
            "source_file_path": source_file_path,
            "input_format": source_record.get("input_format", ""),
            "unit_number": unit_number,
            "unit_type": source_record.get("unit_type", ""),
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "metadata": metadata,
        }


class ChunkEmitter:
    def __init__(self, config: ChunkConfig, metadata_definition: MetadataDefinition, chunker: TextChunker):
        self.config = config
        self.metadata_definition = metadata_definition
        self.chunker = chunker

    def run(self) -> int:
        records = self._load_source_records()

        self.config.chunk_output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        units_without_text = 0
        with self.config.chunk_output_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                chunk_records = self._chunk_record(record)
                if not chunk_records:
                    units_without_text += 1
                    continue
                for chunk_record in chunk_records:
                    handle.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                    total_chunks += 1

        if total_chunks == 0:
            raise RuntimeError(
                f"No chunks were produced from {self.config.input_jsonl}. "
                "Verify the preprocessed output contains extractable text."
            )

        self._verify_output()

        if units_without_text:
            Logger.warning(f"{units_without_text} source unit(s) produced no chunk because no text was available.")
        Logger.info(
            f"Wrote {total_chunks} chunk record(s) from {len(records)} source unit(s) to {self.config.chunk_output_jsonl}"
        )
        return 0

    def _load_source_records(self) -> List[Dict[str, Any]]:
        if not self.config.input_jsonl.is_file():
            raise RuntimeError(f"Input JSONL does not exist: {self.config.input_jsonl}")

        records: List[Dict[str, Any]] = []
        with self.config.input_jsonl.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON on line {line_no} in {self.config.input_jsonl}: {exc}") from exc
                records.append(record)

        if not records:
            raise RuntimeError(f"No records found in {self.config.input_jsonl}")
        return records

    def _unit_text(self, record: Dict[str, Any]) -> str:
        parts: List[str] = []
        text_markdown = str(record.get("text_markdown") or "").strip()
        if text_markdown:
            parts.append(text_markdown)
        if self.config.include_ocr:
            ocr_text = record.get("ocr_image_text")
            if ocr_text:
                ocr_text = str(ocr_text).strip()
                if ocr_text:
                    parts.append(f"[OCR]\n{ocr_text}")
        return "\n\n".join(parts)

    def _chunk_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        unit_text = self._unit_text(record)
        chunk_texts = self.chunker.split(unit_text)
        if not chunk_texts:
            return []

        source_metadata = record.get("metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        metadata = self.metadata_definition.build(source_metadata)

        chunk_count = len(chunk_texts)
        return [
            ChunkRecordFactory.build(
                source_record=record,
                chunk_text=chunk_text,
                chunk_index=index,
                chunk_count=chunk_count,
                metadata=metadata,
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]

    def _verify_output(self) -> None:
        with self.config.chunk_output_jsonl.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                missing = sorted(set(ChunkRecordFactory.REQUIRED_FIELDS) - set(record.keys()))
                if missing:
                    raise RuntimeError(
                        f"Chunk record on line {line_no} is missing required fields {missing} "
                        f"in {self.config.chunk_output_jsonl}"
                    )
                if not str(record.get("text") or "").strip():
                    raise RuntimeError(f"Chunk record on line {line_no} has empty text.")
                if not isinstance(record.get("metadata"), dict):
                    raise RuntimeError(f"Chunk record on line {line_no} has a non-object metadata field.")
                if record["chunk_index"] >= record["chunk_count"]:
                    raise RuntimeError(f"Chunk record on line {line_no} has chunk_index >= chunk_count.")


class ChunkApplication:
    def __init__(self, args: argparse.Namespace, env: Dict[str, str]):
        self.config = ChunkConfigLoader(env=env, args=args).load()

    def run(self) -> int:
        PythonVersionEnforcer(self.config.required_python).validate()
        metadata_definition = MetadataDefinitionLoader.load(self.config.metadata_definition_file)
        chunker = TextChunker(self.config.chunk_size, self.config.chunk_overlap)
        emitter = ChunkEmitter(config=self.config, metadata_definition=metadata_definition, chunker=chunker)
        return emitter.run()


class ApplicationEntryPoint:
    @staticmethod
    def run() -> int:
        parser = CLIParserFactory.build()
        args = parser.parse_args()
        app = ChunkApplication(args=args, env=dict(os.environ))
        return app.run()


if __name__ == "__main__":
    try:
        raise SystemExit(ApplicationEntryPoint.run())
    except Exception as exc:
        Logger.error(str(exc))
        raise SystemExit(1)
