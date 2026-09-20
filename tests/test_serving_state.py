"""Tests for the records a prediction needs.

The one that matters is `test_serving_reproduces_training_features`. Everything else in this
project guards against the model seeing the future; this guards against the opposite failure,
where serving and training compute the same feature differently and the model quietly answers
questions in units it was never trained in.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from optcg_forecast.features.compute import FeatureBuilder, build
from optcg_forecast.features.ingest import Entrant, Match
from optcg_forecast.features.serving_state import (
    Stamp,
    digest_events,
    read_state,
    stamp_for,
    write_state,
)

START = date(2026, 1, 1)


def ent(event, player, leader):
    return Entrant(
        event_id=event,
        event_date=START,
        player=player,
        leader_id=leader,
        country="CH",
        decklist=None,
        did_drop=False,
    )


def mt(event, p1, p2, winner, d, rnd=1):
    return Match(
        event_id=event, event_date=d, round=rnd, table=1, player1=p1, player2=p2, winner=winner
    )


def corpus(n_events=8):
    """A small history with players and leaders that recur, so records actually accumulate."""
    events = []
    for e in range(n_events):
        d = START + timedelta(days=7 * e)
        players = [f"p{i}" for i in range(6)]
        leaders = {p: f"L{i % 3}" for i, p in enumerate(players)}
        entrants = [ent(f"e{e}", p, leaders[p]) for p in players]
        matches = [
            mt(f"e{e}", players[0], players[1], players[e % 2], d),
            mt(f"e{e}", players[2], players[3], players[2 + e % 2], d),
            mt(f"e{e}", players[4], players[5], players[4 + e % 2], d),
        ]
        events.append((d, entrants, matches))
    return events


# ------------------------------------------------------------------ round trip


def test_state_round_trips(tmp_path):
    builder = FeatureBuilder()
    list(build(iter(corpus()), builder=builder))
    stamp = stamp_for([f"e{i}" for i in range(8)], "2026-02-19")

    target = write_state(builder, stamp, tmp_path / "serving_state.json")
    loaded, loaded_stamp = read_state(target)

    assert loaded_stamp == stamp
    assert {k: (v.games, v.wins) for k, v in loaded.leaders.items()} == {
        k: (v.games, v.wins) for k, v in builder.leaders.items() if v.games
    }
    assert {k: (v.games, v.wins) for k, v in loaded.players.items()} == {
        k: (v.games, v.wins) for k, v in builder.players.items() if v.games
    }
    assert {k: (v.games, v.wins) for k, v in loaded.cells.items()} == {
        k: (v.games, v.wins) for k, v in builder.cells.items() if v.games
    }


def test_cell_keys_survive_being_stringified(tmp_path):
    """Cells are keyed by a tuple of leader ids; JSON has no tuple keys."""
    builder = FeatureBuilder()
    list(build(iter(corpus()), builder=builder))
    loaded, _ = read_state(
        write_state(builder, stamp_for(["e0"], "2026-01-01"), tmp_path / "s.json")
    )
    assert loaded.cells, "there must be cells to check"
    for key in loaded.cells:
        assert isinstance(key, tuple) and len(key) == 2
        assert key == tuple(sorted(key)), "the order-independent pairing key must survive"


# --------------------------------------------- the training/serving skew guard


def test_serving_reproduces_training_features(tmp_path):
    """Features built from loaded state must equal what training emitted for the same match.

    Build state over everything except the last event, persist it, load it back, and ask it for
    the features of the last event's matches. Those must be identical to the rows the pipeline
    itself produced for that event, because the pipeline emitted them from exactly this state.
    """
    events = corpus()
    history, (last_date, last_entrants, last_matches) = events[:-1], events[-1]

    # What training produced for the final event, from state built over the history.
    trainer = FeatureBuilder()
    list(build(iter(history), builder=trainer))
    leader_of = {e.player: e.leader_id for e in last_entrants if e.leader_id}
    expected = [
        trainer.features_for(m, leader_of[m.player1], leader_of[m.player2]) for m in last_matches
    ]

    # What serving produces, from the same state after a trip through disk.
    saved = write_state(
        trainer, stamp_for([f"e{i}" for i in range(7)], str(last_date)), tmp_path / "s.json"
    )
    served, _ = read_state(saved)
    actual = [
        served.features_for(m, leader_of[m.player1], leader_of[m.player2]) for m in last_matches
    ]

    for want, got in zip(expected, actual, strict=True):
        assert got.leader_strength_diff == pytest.approx(want.leader_strength_diff)
        assert got.player_strength_diff == pytest.approx(want.player_strength_diff)
        assert got.cell_rate == pytest.approx(want.cell_rate)
        assert got.experience_diff == pytest.approx(want.experience_diff)
        assert (got.p1_leader_games, got.p2_leader_games) == (
            want.p1_leader_games,
            want.p2_leader_games,
        )
        assert got.cell_games == want.cell_games
        assert got.coverage == want.coverage


def test_an_unknown_handle_is_cold_not_average(tmp_path):
    """A player nobody has seen must read as no evidence, not as a middling one."""
    builder = FeatureBuilder()
    list(build(iter(corpus()), builder=builder))
    served, _ = read_state(
        write_state(builder, stamp_for(["e0"], "2026-01-01"), tmp_path / "s.json")
    )
    row = served.features_for(
        mt("query", "p0", "someone-who-has-never-played", "p0", START), "L0", "L1"
    )
    assert served.players["someone-who-has-never-played"].games == 0
    assert row.experience_diff > 0, "the known player must show more experience"


def test_loaded_state_will_not_silently_continue_the_walk(tmp_path):
    """Resumed state does not know which date it stopped at, so it must not accept more."""
    builder = FeatureBuilder()
    list(build(iter(corpus()), builder=builder))
    served, _ = read_state(
        write_state(builder, stamp_for(["e0"], "2026-01-01"), tmp_path / "s.json")
    )
    assert served._last_date is None


# ------------------------------------------------------------------- the stamp


def test_digest_ignores_order_and_duplicates():
    assert digest_events(["b", "a"]) == digest_events(["a", "b", "a"])


def test_a_different_corpus_is_a_different_stamp():
    a = stamp_for(["e1", "e2"], "2026-01-01")
    b = stamp_for(["e1", "e2", "e3"], "2026-01-08")
    assert not a.matches(b)
    assert a.matches(stamp_for(["e2", "e1"], "2026-01-01"))


def test_the_stamp_ignores_the_date_but_not_the_events():
    """Re-materialising the same events must not invalidate a model over a cosmetic field."""
    a = stamp_for(["e1", "e2"], "2026-01-01")
    b = Stamp(a.feature_set_version, "2026-09-01", a.events, a.event_digest)
    assert a.matches(b)


def test_state_file_is_json_a_human_can_open(tmp_path):
    builder = FeatureBuilder()
    list(build(iter(corpus()), builder=builder))
    target = write_state(builder, stamp_for(["e0"], "2026-01-01"), tmp_path / "serving_state.json")
    payload = json.loads(target.read_text())
    assert set(payload) == {"stamp", "leaders", "players", "cells"}
    assert all(len(v) == 2 for v in payload["players"].values()), "[games, wins] per entry"
