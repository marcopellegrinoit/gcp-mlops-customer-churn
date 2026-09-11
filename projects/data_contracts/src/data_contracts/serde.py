"""JSON encoding for the artifacts and control-plane files these contracts describe."""

import json

from pydantic import BaseModel


def to_json(model: BaseModel) -> str:
    """Serialise a contract model the way every reader of these files already expects.

    Deliberately ``json.dumps`` over a Python-mode dump rather than ``model_dump_json``.
    Pydantic's JSON serialiser writes non-finite floats as ``null`` by default, and
    ``bin_edges`` is bounded by ``-inf``/``inf`` on purpose — so that a value beyond the
    training range lands in the outermost bucket instead of falling outside every bucket.
    Silently rewriting those two edges to ``null`` would make a rebuilt baseline
    unreadable. ``json.dumps`` emits ``Infinity``, which is what the existing artifacts
    already contain and what ``json.loads`` reads back.

    ``exclude_none`` keeps "field absent" distinguishable from "field present and null".
    Both this package's readers and the orchestrator workflow test for key *presence* to
    detect a legacy artifact or an optional section, so writing explicit nulls would read
    as "present, and false-y" and quietly take the wrong branch.
    """
    return json.dumps(model.model_dump(mode="python", exclude_none=True))
