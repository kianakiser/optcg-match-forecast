"""Tests for the feature-pipeline ingest entry point.

No network. What these protect is the behaviour that makes a scheduled job survivable:
it must not need credentials, it must not re-ingest what it already has, one bad event
must not lose the run, and an empty day must be a success rather than a failure.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from optcg_forecast.features.client import ApiError
from optcg_forecast.features.ingest import IngestStats
from optcg_forecast.features.run import (
    already_ingested,
    discover,
    ingest_event,
    main,
    write_landing,
)

EVENTS = [
    {
        "id": "big1",
        "date": "2026-09-17T18:00:00.000Z",
        "players": 64,
        "organizerId": 2339,
        "name": "ChinoizeCup #111",
    },
    {"id": "small", "date": "2026-09-17T18:00:00.000Z", "players": 10, "name": "Versus Cup"},
    {"id": "nodate", "players": 50, "name": "Mystery"},
    {"id": "big2", "date": "2026-09-16T18:00:00.000Z", "players": 40, "name": "Rumble"},
]

PAYLOAD = {
    "standings": [
        {
            "player": "a",
            "deck": {"id": "OP05-098"},
            "country": "CH",
            "decklist": {"leader": {"set": "OP05", "number": "098"}},
            "placing": 1,
            "record": {"wins": 3, "losses": 0},
            "drop": None,
        },
        {
            "player": "b",
            "deck": {"id": "OP01-003"},
            "country": "DE",
            "decklist": {"leader": {"set": "OP01", "number": "003"}},
            "placing": None,
            "record": {"wins": 0, "losses": 3},
            "drop": 2,
        },
    ],
    "pairings": [
        {"round": 1, "phase": 1, "table": 1, "winner": "a", "player1": "a", "player2": "b"},
    ],
}


class FakeClient:
    """Stands in for LimitlessClient without touching the network."""

    def __init__(self, index, payloads=None, fail_on=()):
        self._index, self._payloads, self._fail_on = index, payloads or {}, set(fail_on)
        self.requests_made = self.cache_hits = 0
        self.fetched: list[str] = []

    def tournaments(self, limit=50, **_):
        self.requests_made += 1
        return self._index

    def event_payload(self, event_id):
        self.fetched.append(event_id)
        if event_id in self._fail_on:
            raise ApiError(f"boom: {event_id}")
        return self._payloads.get(event_id, PAYLOAD)


# --------------------------------------------------------------------------- discover


def test_discover_applies_the_size_filter():
    got = discover(FakeClient(EVENTS), min_players=32, limit=50, already_have=set())
    assert {e["id"] for e in got} == {"big1", "big2"}


def test_discover_skips_events_with_no_date():
    """No date means no out-of-time split, so the row would be untrainable."""
    got = discover(FakeClient(EVENTS), min_players=1, limit=50, already_have=set())
    assert "nodate" not in {e["id"] for e in got}


def test_discover_is_idempotent_against_what_is_already_held():
    got = discover(FakeClient(EVENTS), min_players=32, limit=50, already_have={"big1"})
    assert {e["id"] for e in got} == {"big2"}


def test_lowering_min_players_widens_the_field():
    """This is the lever that reduces single-organiser concentration."""
    wide = discover(FakeClient(EVENTS), min_players=8, limit=50, already_have=set())
    assert {e["id"] for e in wide} == {"big1", "big2", "small"}


# --------------------------------------------------------------------------- ingest


def test_ingest_event_strips_outcome_fields_and_keeps_droppers():
    stats = IngestStats()
    rec = ingest_event(FakeClient(EVENTS), EVENTS[0], stats)
    assert rec["event_date"] == "2026-09-17"
    assert len(rec["entrants"]) == 2, "the dropper must stay in the population"
    assert stats.leaky_fields_stripped == {"placing", "record", "drop"}
    for entrant in rec["entrants"]:
        assert not {"placing", "record", "drop"} & entrant.keys()


def test_ingest_event_carries_the_date_onto_every_row():
    rec = ingest_event(FakeClient(EVENTS), EVENTS[0], IngestStats())
    assert all(m["event_date"] == date(2026, 9, 17) for m in rec["matches"])


# --------------------------------------------------------------------------- landing


def test_write_landing_partitions_by_month(tmp_path: Path):
    rec = ingest_event(FakeClient(EVENTS), EVENTS[0], IngestStats())
    path = write_landing(tmp_path, rec)
    assert path.parent.name == "2026-09"
    assert json.loads(path.read_text())["event_id"] == "big1"
    assert already_ingested(tmp_path) == {"big1"}


# --------------------------------------------------------------------------- run


def test_a_quiet_day_is_a_success_not_a_failure(tmp_path, monkeypatch, caplog):
    """Roughly 40% of days add nothing. That must exit 0."""
    monkeypatch.setattr("optcg_forecast.features.run.LimitlessClient", lambda **kw: FakeClient([]))
    rc = main(
        [
            "--out-dir",
            str(tmp_path / "landing"),
            "--feature-root",
            str(tmp_path / "features"),
            "--dry-run",
        ]
    )
    assert rc == 0


def test_one_bad_event_does_not_lose_the_run(tmp_path, monkeypatch):
    client = FakeClient(EVENTS, fail_on={"big1"})
    monkeypatch.setattr("optcg_forecast.features.run.LimitlessClient", lambda **kw: client)
    rc = main(
        ["--out-dir", str(tmp_path / "landing"), "--feature-root", str(tmp_path / "features")]
    )
    assert rc == 0
    assert already_ingested(tmp_path / "landing") == {"big2"}, "the healthy event must still land"


def test_unreachable_index_fails_loudly(tmp_path, monkeypatch):
    class Dead(FakeClient):
        def tournaments(self, limit=50, **_):
            raise ApiError("index unreachable")

    monkeypatch.setattr("optcg_forecast.features.run.LimitlessClient", lambda **kw: Dead([]))
    assert (
        main(["--out-dir", str(tmp_path / "landing"), "--feature-root", str(tmp_path / "features")])
        == 1
    )


def test_needs_no_credentials(tmp_path, monkeypatch):
    """A scheduled run must work with an empty environment; the endpoints are keyless."""
    for var in ("SOURCE_API_KEY", "HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT", "GCP_PROJECT_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "optcg_forecast.features.run.LimitlessClient", lambda **kw: FakeClient(EVENTS)
    )
    assert (
        main(
            [
                "--out-dir",
                str(tmp_path / "landing"),
                "--feature-root",
                str(tmp_path / "features"),
                "--dry-run",
            ]
        )
        == 0
    )


def test_running_twice_ingests_nothing_the_second_time(tmp_path, monkeypatch):
    client = FakeClient(EVENTS)
    monkeypatch.setattr("optcg_forecast.features.run.LimitlessClient", lambda **kw: client)
    assert (
        main(["--out-dir", str(tmp_path / "landing"), "--feature-root", str(tmp_path / "features")])
        == 0
    )
    first = len(client.fetched)
    assert (
        main(["--out-dir", str(tmp_path / "landing"), "--feature-root", str(tmp_path / "features")])
        == 0
    )
    assert len(client.fetched) == first, "immutable events must not be refetched"


@pytest.mark.parametrize("flag", ["--dry-run"])
def test_dry_run_writes_nothing(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(
        "optcg_forecast.features.run.LimitlessClient", lambda **kw: FakeClient(EVENTS)
    )
    main(
        ["--out-dir", str(tmp_path / "landing"), "--feature-root", str(tmp_path / "features"), flag]
    )
    assert already_ingested(tmp_path / "landing") == set()
