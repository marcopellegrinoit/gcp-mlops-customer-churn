"""Log configuration that survives the trip into Cloud Logging with its severity intact.

Python's `logging.basicConfig` installs a single handler on `sys.stderr`. Every GCP log
agent reads an unstructured stderr line as `ERROR` severity, so a plain `logging.info`
call — an HPO trial score, a promotion decision — arrives in Logs Explorer flagged red.
The `INFO` in a `%(levelname)s`-formatted line is just message text; it is never read as
the entry's severity.

Two mechanisms are combined here so the result is correct on both runtimes this repo
deploys to:

* **Structured JSON payloads.** Cloud Run parses a JSON object on stdout/stderr and lifts
  the `severity` field onto the entry itself, giving exact severities (`WARNING` stays
  `WARNING` instead of collapsing into `ERROR`).
* **Severity-based stream routing.** Vertex AI Pipelines does not reliably parse that
  payload, and falls back to classifying by stream. Records below `WARNING` are therefore
  written to stdout and the rest to stderr, so the stream-based reading degrades to an
  accurate `INFO`/`ERROR` split rather than marking everything as an error.

Anything passed via `logging.info(..., extra={...})` is merged into the JSON payload, so
run context (trial number, model version) becomes a queryable field rather than substring
-matchable message text.
"""

import json
import logging
import sys

# Descending so the first match wins. Cloud Logging accepts DEBUG/INFO/WARNING/ERROR/
# CRITICAL verbatim, but a custom or intermediate level number (logging.addLevelName, or
# WARNING+5) has no name it recognises — each one is floored to the severity below it.
_SEVERITY_LADDER: tuple[tuple[int, str], ...] = (
    (logging.CRITICAL, "CRITICAL"),
    (logging.ERROR, "ERROR"),
    (logging.WARNING, "WARNING"),
    (logging.INFO, "INFO"),
    (logging.DEBUG, "DEBUG"),
)

# Records at or above this level go to stderr; everything quieter goes to stdout.
_STDERR_THRESHOLD = logging.WARNING

# Attributes every LogRecord carries. Anything outside this set arrived via `extra=` and is
# merged into the payload as a queryable jsonPayload field. Built from a live record rather
# than hardcoded so a new stdlib attribute cannot silently start leaking into the output.
_STANDARD_RECORD_ATTRS = frozenset(
    set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName"}
)


def severity_of(levelno: int) -> str:
    """Return the Cloud Logging severity for a Python level number."""
    for threshold, severity in _SEVERITY_LADDER:
        if levelno >= threshold:
            return severity
    return "DEBUG"


class StructuredFormatter(logging.Formatter):
    """Render a record as the single-line JSON object Cloud Logging parses natively."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a compact JSON string."""
        message = record.getMessage()
        if record.exc_info:
            # Appended rather than sent as its own field: Logs Explorer shows `message` as
            # the entry summary, so an exception in a sibling field stays collapsed.
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        payload = {
            "severity": severity_of(record.levelno),
            "message": message,
            "logger": record.name,
            # Special-cased by Cloud Logging into the entry's sourceLocation, which is how
            # Logs Explorer renders a file:line link back to the emitting call.
            "logging.googleapis.com/sourceLocation": {
                "file": record.pathname,
                "line": str(record.lineno),
                "function": record.funcName,
            },
        }
        payload.update(
            {
                key: value
                for key, value in vars(record).items()
                if key not in _STANDARD_RECORD_ATTRS and key not in payload
            }
        )
        # default=str keeps a stray non-serialisable `extra` value from raising inside the
        # handler, where the error would be swallowed and the log line lost.
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the structured stdout/stderr handler pair on the root logger.

    Replaces any handlers already present, so this is safe to call from a container
    entrypoint that a library may have configured first. Call it once, at startup.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for existing in root.handlers[:]:
        root.removeHandler(existing)

    stdout_handler = logging.StreamHandler(sys.stdout)
    # A filter, not setLevel: this handler needs an upper bound, and setLevel only sets a
    # lower one. Without it every WARNING+ record would be emitted on both streams.
    stdout_handler.addFilter(lambda record: record.levelno < _STDERR_THRESHOLD)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(_STDERR_THRESHOLD)

    for handler in (stdout_handler, stderr_handler):
        handler.setFormatter(StructuredFormatter())
        root.addHandler(handler)
