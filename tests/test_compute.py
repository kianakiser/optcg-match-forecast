"""Tests for feature computation.

The important ones here are the leakage tests. Everything else is arithmetic; leakage is the
thing that would quietly ruin the project, so it is tested by construction rather than by
inspection.
"""

from __future__ import annotations

from datetime import date

import pytest

from optcg_forecast.features.compute import (
    MIN_GAMES_FOR_CELL,
    FeatureBuilder,
    build,
)
from optcg_forecast.features.ingest import Entrant, Match

D1, D2, D3 = date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)


def ent(player, leader):
    return Entrant(
        event_id="e",
        event_date=D1,
        player=player,
        leader_id=leader,
        country="CH",
        decklist={"leader": {}},
        did_drop=False,
    )


def match(event, p1, p2, winner, rnd=1, d=D1):
    return Match(
        event_id=event, event_date=d, round=rnd, table=1, player1=p1, player2=p2, winner=winner
    )


# --------------------------------------------------------------------------- leakage


def test_a_match_cannot_see_its_own_result():
    """The first row ever emitted must have zero evidence behind it."""
    b = FeatureBuilder()
    rows = b.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"})
    assert rows[0].p1_leader_games == 0
    assert rows[0].cell_games == 0
    assert rows[0].coverage == "cold"
    assert rows[0].leader_strength_diff == 0.0


def test_matches_in_one_event_cannot_see_each_other():
    """All rows from one event share a snapshot taken before the event began.

    Otherwise round 1 would inform round 4 features - the same in-event leakage the `table`
    field was rejected for.
    """
    b = FeatureBuilder()
    rows = b.process_event(
        D1,
        [match("e1", "a", "b", "a", rnd=1), match("e1", "c", "d", "c", rnd=2)],
        {"a": "L1", "b": "L2", "c": "L1", "d": "L2"},
    )
    assert rows[0].p1_leader_games == rows[1].p1_leader_games == 0
    assert rows[0].cell_games == rows[1].cell_games == 0


def test_state_advances_between_events():
    b = FeatureBuilder()
    b.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"})
    rows = b.process_event(D2, [match("e2", "x", "y", "x")], {"x": "L1", "y": "L2"})
    assert rows[0].p1_leader_games == 1, "the earlier event must now be visible"
    assert rows[0].cell_games == 1
    assert rows[0].leader_strength_diff > 0, "L1 beat L2, so L1 should look stronger"


def test_out_of_order_events_are_refused():
    """Accepting them silently would reintroduce the leakage this module exists to prevent."""
    b = FeatureBuilder()
    b.process_event(D2, [match("e2", "a", "b", "a")], {"a": "L1", "b": "L2"})
    with pytest.raises(ValueError, match="date order"):
        b.process_event(D1, [match("e1", "c", "d", "c")], {"c": "L1", "d": "L2"})


def test_no_future_information_in_a_full_replay():
    """Strongest form: replay a history and assert each row only ever saw earlier games."""
    events = []
    for i, d in enumerate([D1, D2, D3]):
        e = f"e{i}"
        events.append(
            (
                d,
                [ent("p1", "L1"), ent("p2", "L2")],
                [match(e, "p1", "p2", "p1", d=d)],
            )
        )
    rows = list(build(events))
    assert [r.p1_leader_games for r in rows] == [0, 1, 2], "each row sees only prior events"
    assert [r.cell_games for r in rows] == [0, 1, 2]


# --------------------------------------------------------------------------- shrinkage


def test_a_perfect_record_is_not_certainty():
    """3-0 is not a 100% deck; thin evidence must be pulled toward the middle."""
    b = FeatureBuilder()
    for i in range(3):
        b.process_event(
            date(2026, 1, i + 1), [match(f"e{i}", "a", "b", "a")], {"a": "L1", "b": "L2"}
        )
    rate = b.leaders["L1"].shrunk_rate
    assert 0.5 < rate < 0.65, f"3-0 should be modest, got {rate:.3f}"


def test_more_evidence_moves_the_estimate_further():
    b = FeatureBuilder()
    for i in range(60):
        b.process_event(
            date(2026, 1, 1)
            if i == 0
            else date(2026, 1, 1).replace(day=1)
            if False
            else date(2026, 1, 1) + __import__("datetime").timedelta(days=i),
            [match(f"e{i}", "a", "b", "a")],
            {"a": "L1", "b": "L2"},
        )
    assert b.leaders["L1"].shrunk_rate > 0.75, "60-0 should be convincing"


# --------------------------------------------------------------------------- symmetry


def test_the_pairing_key_is_symmetric():
    """A-vs-B and B-vs-A must pool their evidence, not split it."""
    b = FeatureBuilder()
    b.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"})
    b.process_event(D2, [match("e2", "c", "d", "d")], {"c": "L2", "d": "L1"})
    rows = b.process_event(D3, [match("e3", "x", "y", "x")], {"x": "L1", "y": "L2"})
    assert rows[0].cell_games == 2, "both orderings must land in the same cell"


def test_features_are_differences_so_seat_order_flips_sign():
    b = FeatureBuilder()
    b.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"})
    fwd = b.process_event(D2, [match("e2", "x", "y", "x")], {"x": "L1", "y": "L2"})[0]
    b2 = FeatureBuilder()
    b2.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"})
    rev = b2.process_event(D2, [match("e2", "x", "y", "x")], {"x": "L2", "y": "L1"})[0]
    assert fwd.leader_strength_diff == pytest.approx(-rev.leader_strength_diff)


# --------------------------------------------------------------------------- coverage


def test_coverage_is_cold_with_no_history():
    b = FeatureBuilder()
    rows = b.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"})
    assert rows[0].coverage == "cold"


def test_coverage_becomes_solid_once_the_cell_is_populated():
    b = FeatureBuilder()
    import datetime

    for i in range(MIN_GAMES_FOR_CELL):
        b.process_event(
            D1 + datetime.timedelta(days=i),
            [match(f"e{i}", "a", "b", "a")],
            {"a": "L1", "b": "L2"},
        )
    rows = b.process_event(D3, [match("z", "x", "y", "x")], {"x": "L1", "y": "L2"})
    assert rows[0].cell_games >= MIN_GAMES_FOR_CELL
    assert rows[0].coverage == "solid"


def test_rows_without_a_resolvable_leader_are_dropped():
    b = FeatureBuilder()
    rows = b.process_event(D1, [match("e1", "a", "b", "a")], {"a": "L1"})  # b has no leader
    assert rows == []
