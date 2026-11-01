"""exercise serialization and atomic replacement, including failures."""

import json

import pytest
import yaml

from cgcr.utils.io import read_jsonl, read_yaml, write_jsonl, write_yaml


def test_jsonl_round_trip_and_no_temporary_files(tmp_path):
    records = [{"id": 1, "text": "café اردو"}, {"id": 2, "value": None}]
    target = tmp_path / "nested" / "records.jsonl"
    write_jsonl(target, iter(records))
    assert read_jsonl(target) == records
    assert list(target.parent.iterdir()) == [target]
    assert "café" in target.read_text(encoding="utf-8")
    write_jsonl(target, [{"replacement": True}])
    assert read_jsonl(target) == [{"replacement": True}]
    assert list(target.parent.iterdir()) == [target]


def test_empty_jsonl_and_blank_lines(tmp_path):
    target = tmp_path / "records.jsonl"
    write_jsonl(target, [])
    assert read_jsonl(target) == []
    target.write_text('\n{"id": 1}\n  \n', encoding="utf-8")
    assert read_jsonl(target) == [{"id": 1}]


def test_failed_atomic_jsonl_write_preserves_existing_file(tmp_path):
    target = tmp_path / "records.jsonl"
    write_jsonl(target, [{"old": True}])
    original = target.read_bytes()

    def interrupted_records():
        yield {"new": True}
        raise RuntimeError("interrupted input")

    with pytest.raises(RuntimeError, match="interrupted input"):
        write_jsonl(target, interrupted_records())
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_failed_json_encoding_leaves_no_target_or_temp_file(tmp_path):
    target = tmp_path / "records.jsonl"
    with pytest.raises(TypeError):
        write_jsonl(target, [{"unsupported": object()}])
    assert list(tmp_path.iterdir()) == []


def test_malformed_jsonl_raises(tmp_path):
    target = tmp_path / "records.jsonl"
    target.write_text("{broken}\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(target)


def test_yaml_round_trip_and_no_temporary_files(tmp_path):
    value = {"seed": 42, "nested": {"names": ["café", "اردو"], "enabled": True}}
    target = tmp_path / "nested" / "config.yaml"
    write_yaml(target, value)
    assert read_yaml(target) == value
    assert list(target.parent.iterdir()) == [target]


def test_yaml_does_not_construct_python_objects(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("!!python/tuple [1, 2]\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        read_yaml(target)
