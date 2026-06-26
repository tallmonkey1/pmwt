"""Structured JSON logging for observability (SPEC §11).

Every meaningful decision and failure in the engine must be observable and reconstructable
after the fact. To that end this module configures the standard library ``logging`` to
emit one JSON object per line, including any structured ``extra`` fields supplied at the
call site, plus an optional correlation id that ties together all log records belonging to
a single decision or order.

Why the standard library instead of a third-party logger? Dependency minimization
(SPEC §13.15): structured logging is achievable with zero extra dependencies, and the
standard library is battle-tested and ubiquitous.

Usage::

    from options_engine.core.logging import configure_logging, get_logger, bind_context

    configure_logging(level="INFO")
    log = get_logger(__name__)
    with bind_context(correlation_id="trade-123", symbol="SPX"):
        log.info("entry_signal", extra={"win_prob": 0.78, "credit": 1.25})
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

__all__ = ["JsonFormatter", "bind_context", "configure_logging", "get_logger"]

# Context variable holding ambient structured fields merged into every log record emitted
# within the active context (e.g. correlation id, symbol, mode). Using contextvars makes
# this safe across threads and asyncio tasks.
# Default is an *empty, immutable* mapping shared by all contexts; binds always create a
# new dict (see ``bind_context``) so the default is never mutated. Using a module-level
# constant avoids the mutable-default pitfall while keeping ``.get()`` allocation-free.
_EMPTY_CONTEXT: Mapping[str, Any] = MappingProxyType({})
_LOG_CONTEXT: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "options_engine_log_context", default=_EMPTY_CONTEXT
)

# Attributes present on a stdlib LogRecord that we render explicitly or deliberately omit;
# anything *not* in this set that appears on the record is treated as a structured extra.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Render :class:`logging.LogRecord` objects as single-line JSON.

    The output always includes ISO-8601 UTC timestamp, level, logger name, source
    location, and message. Structured ``extra`` fields and the ambient log context are
    merged into the top-level object. Exceptions are rendered with their traceback.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = _dt.datetime.fromtimestamp(record.created, tz=_dt.UTC)
        payload: dict[str, Any] = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "location": f"{record.module}:{record.funcName}:{record.lineno}",
        }

        # Merge ambient context first so explicit per-call extras can override it.
        context = _LOG_CONTEXT.get()
        if context:
            payload.update(context)

        # Merge any structured fields passed via ``extra=...``.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _json_default(obj: Any) -> str:
    """Fallback serializer so logging never crashes on an exotic ``extra`` value."""
    try:
        return str(obj)
    except Exception:  # pragma: no cover - extremely defensive
        return "<unserializable>"


def configure_logging(
    *,
    level: str | int = "INFO",
    stream: Any = None,
    force: bool = True,
) -> None:
    """Configure the root logger to emit structured JSON.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``) or numeric level.
    stream:
        Output stream; defaults to ``sys.stderr`` so that JSON logs do not pollute any
        stdout data channel.
    force:
        If True (default), removes existing handlers first so repeated configuration in
        tests/notebooks is idempotent.
    """
    target_stream = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(target_stream)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    if force:
        for existing in list(root.handlers):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Thin wrapper kept for a single import surface."""
    return logging.getLogger(name)


@contextmanager
def bind_context(**fields: Any) -> Iterator[None]:
    """Bind structured fields into all log records emitted within the block.

    Nested binds merge; on exit the previous context is restored. This is how correlation
    ids and per-decision metadata are attached without threading them through every call.
    """
    current = _LOG_CONTEXT.get()
    merged = {**current, **fields}
    token = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)
