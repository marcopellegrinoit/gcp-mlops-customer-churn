import json
import logging

import pytest
from obs_common.logging import configure_logging, severity_of


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _emit(level, message, capsys, *, level_arg=logging.DEBUG):
    configure_logging(level_arg)
    logging.getLogger("test").log(level, message)
    captured = capsys.readouterr()
    return captured.out, captured.err


@pytest.mark.parametrize(
    ("levelno", "expected"),
    [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRITICAL"),
    ],
)
def test_severity_of_maps_standard_levels(levelno, expected):
    assert severity_of(levelno) == expected


def test_severity_of_floors_intermediate_levels():
    assert severity_of(logging.WARNING + 5) == "WARNING"
    assert severity_of(logging.CRITICAL + 10) == "CRITICAL"
    assert severity_of(1) == "DEBUG"


def test_info_goes_to_stdout_only(capsys):
    out, err = _emit(logging.INFO, "hpo trial 0073", capsys)

    assert err == ""
    assert json.loads(out)["severity"] == "INFO"


def test_error_goes_to_stderr_only(capsys):
    out, err = _emit(logging.ERROR, "gate failed", capsys)

    assert out == ""
    assert json.loads(err)["severity"] == "ERROR"


def test_warning_keeps_its_own_severity(capsys):
    _, err = _emit(logging.WARNING, "drift detected", capsys)

    assert json.loads(err)["severity"] == "WARNING"


def test_payload_carries_message_logger_and_source_location(capsys):
    out, _ = _emit(logging.INFO, "cv_pr_auc_mean=0.2461", capsys)
    payload = json.loads(out)

    assert payload["message"] == "cv_pr_auc_mean=0.2461"
    assert payload["logger"] == "test"
    source = payload["logging.googleapis.com/sourceLocation"]
    assert source["file"].endswith("test_logging.py")
    assert source["function"] == "_emit"


def test_percent_style_args_are_interpolated(capsys):
    configure_logging(logging.INFO)
    logging.getLogger("test").info("HPO trial %04d: mean=%.4f", 73, 0.2461)

    assert json.loads(capsys.readouterr().out)["message"] == "HPO trial 0073: mean=0.2461"


def test_exception_traceback_is_folded_into_message(capsys):
    configure_logging(logging.INFO)
    try:
        raise ValueError("bad params")
    except ValueError:
        logging.getLogger("test").exception("hpo crashed")

    payload = json.loads(capsys.readouterr().err)
    assert payload["severity"] == "ERROR"
    assert payload["message"].startswith("hpo crashed\n")
    assert "ValueError: bad params" in payload["message"]


def test_stack_info_is_folded_into_message(capsys):
    configure_logging(logging.INFO)
    logging.getLogger("test").info("where am i", stack_info=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["message"].startswith("where am i\nStack (most recent call last):")


def test_each_record_is_exactly_one_line(capsys):
    configure_logging(logging.INFO)
    log = logging.getLogger("test")
    log.info("first")
    log.info("second")

    lines = capsys.readouterr().out.strip().split("\n")
    assert [json.loads(line)["message"] for line in lines] == ["first", "second"]


def test_level_argument_filters_quieter_records(capsys):
    out, err = _emit(logging.DEBUG, "noisy", capsys, level_arg=logging.INFO)

    assert (out, err) == ("", "")


def test_configure_logging_replaces_existing_handlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configure_logging()

    handlers = logging.getLogger().handlers
    assert len(handlers) == 2
    assert all(isinstance(h.formatter, type(handlers[0].formatter)) for h in handlers)


def test_extra_fields_are_merged_into_the_payload(capsys):
    configure_logging(logging.INFO)
    logging.getLogger("test").info("trial done", extra={"trial": 73, "cv_pr_auc_mean": 0.2461})

    payload = json.loads(capsys.readouterr().out)
    assert payload["trial"] == 73
    assert payload["cv_pr_auc_mean"] == 0.2461


def test_standard_record_attributes_stay_out_of_the_payload(capsys):
    out, _ = _emit(logging.INFO, "clean", capsys)

    assert set(json.loads(out)) == {
        "severity",
        "message",
        "logger",
        "logging.googleapis.com/sourceLocation",
    }


def test_non_serialisable_extra_does_not_raise(capsys):
    configure_logging(logging.INFO)
    logging.getLogger("test").info("with extra", extra={"blob": object()})

    payload = json.loads(capsys.readouterr().out)
    assert payload["message"] == "with extra"
    assert payload["blob"].startswith("<object object at")
