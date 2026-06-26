"""Tests for structured JSON logging."""

from __future__ import annotations

import io
import json
import logging

from options_engine.core.logging import bind_context, configure_logging, get_logger


def _capture(level: str = "INFO") -> io.StringIO:
    buffer = io.StringIO()
    configure_logging(level=level, stream=buffer, force=True)
    return buffer


def _last_record(buffer: io.StringIO) -> dict[str, object]:
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert lines, "no log output captured"
    return json.loads(lines[-1])


def test_emits_valid_json_with_core_fields() -> None:
    buffer = _capture()
    get_logger("test").info("hello")
    record = _last_record(buffer)
    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["logger"] == "test"
    assert "timestamp" in record
    assert "location" in record


def test_structured_extra_is_merged() -> None:
    buffer = _capture()
    get_logger("test").info("entry", extra={"win_prob": 0.8, "credit": 1.25})
    record = _last_record(buffer)
    assert record["win_prob"] == 0.8
    assert record["credit"] == 1.25


def test_bound_context_is_attached_and_restored() -> None:
    buffer = _capture()
    log = get_logger("test")
    with bind_context(correlation_id="abc", symbol="SPX"):
        log.info("inside")
    log.info("outside")
    lines = [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]
    inside, outside = lines[-2], lines[-1]
    assert inside["correlation_id"] == "abc"
    assert inside["symbol"] == "SPX"
    assert "correlation_id" not in outside


def test_nested_context_merges() -> None:
    buffer = _capture()
    log = get_logger("test")
    with bind_context(a="1"), bind_context(b="2"):
        log.info("nested")
    record = _last_record(buffer)
    assert record["a"] == "1"
    assert record["b"] == "2"


def test_explicit_extra_overrides_context() -> None:
    buffer = _capture()
    log = get_logger("test")
    with bind_context(symbol="SPX"):
        log.info("override", extra={"symbol": "NDX"})
    assert _last_record(buffer)["symbol"] == "NDX"


def test_exception_is_rendered() -> None:
    buffer = _capture()
    log = get_logger("test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    record = _last_record(buffer)
    assert "exception" in record
    assert "ValueError" in str(record["exception"])


def test_unserializable_extra_does_not_crash() -> None:
    buffer = _capture()

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    get_logger("test").info("odd", extra={"obj": Weird()})
    record = _last_record(buffer)
    assert record["obj"] == "<weird>"


def test_configure_is_idempotent() -> None:
    buffer = _capture()
    configure_logging(level="DEBUG", stream=buffer, force=True)
    # Only one handler should be attached after repeated configuration with force.
    assert len(logging.getLogger().handlers) == 1
