"""Read JSONL and safe YAML, with atomic file replacement; implemented in ."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO

import yaml


@contextmanager
def _atomic_writer(path: str | Path) -> Iterator[TextIO]:
    """Stage a UTF-8 file beside its destination and replace it on success."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_jsonl(path: str | Path) -> list[Any]:
    """Read nonblank JSONL records in order, raising on malformed JSON."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Atomically write JSON records, preserving the old file on failure."""
    with _atomic_writer(path) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")


def read_yaml(path: str | Path) -> Any:
    """Read YAML without constructing arbitrary Python objects."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(path: str | Path, value: Any) -> None:
    """Atomically write safe YAML while preserving mapping order."""
    with _atomic_writer(path) as stream:
        yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
