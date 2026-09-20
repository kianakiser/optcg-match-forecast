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


# ----------------------------------------------------------------- same-day events


def test_two_events_on_one_date_cannot_see_each_other():
    """The leak this fixes: 21 dates in the real corpus carry more than one event.

    Processing them one at a time let the second train on the first, and the order came from the
    event id, which is not chronological. Both events must see the same pre-date state.
    """
    b = FeatureBuilder()
    first = ([match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"}, None)
    second = ([match("e2", "c", "d", "c")], {"c": "L1", "d": "L2"}, None)

    rows = b.process_day(D1, [first, second])

    assert len(rows) == 2
    r1, r2 = rows
    assert r1.p1_leader_games == r2.p1_leader_games == 0
    assert r1.cell_games == r2.cell_games == 0
    assert r1.leader_strength_diff == r2.leader_strength_diff == 0.0


def test_the_whole_day_is_learned_before_the_next_one():
    """Emitting from a shared snapshot must not mean the day is forgotten."""
    b = FeatureBuilder()
    b.process_day(
        D1,
        [
            ([match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"}, None),
            ([match("e2", "c", "d", "c")], {"c": "L1", "d": "L2"}, None),
        ],
    )
    rows = b.process_day(D2, [([match("e3", "x", "y", "x")], {"x": "L1", "y": "L2"}, None)])

    assert rows[0].cell_games == 2, "both of the previous day's matches must be known"
    assert rows[0].leader_strength_diff > 0, "L1 won both, so it must look stronger"


def test_order_within_a_day_does_not_change_anything():
    """If the day's order mattered, the arbitrary event-id ordering would be a hidden input."""
    ev_a = ([match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"}, None)
    ev_b = ([match("e2", "c", "d", "c")], {"c": "L3", "d": "L4"}, None)

    forwards = FeatureBuilder()
    forwards.process_day(D1, [ev_a, ev_b])
    backwards = FeatureBuilder()
    backwards.process_day(D1, [ev_b, ev_a])

    after_f = forwards.process_day(
        D2, [([match("e3", "x", "y", "x")], {"x": "L1", "y": "L2"}, None)]
    )
    after_b = backwards.process_day(
        D2, [([match("e3", "x", "y", "x")], {"x": "L1", "y": "L2"}, None)]
    )
    assert after_f[0].leader_strength_diff == after_b[0].leader_strength_diff
    assert after_f[0].cell_rate == after_b[0].cell_rate


def test_a_date_cannot_be_processed_twice():
    """Splitting a date across two calls is the leak wearing a different hat."""
    b = FeatureBuilder()
    b.process_day(D1, [([match("e1", "a", "b", "a")], {"a": "L1", "b": "L2"}, None)])
    with pytest.raises(ValueError, match="strictly increasing"):
        b.process_day(D1, [([match("e2", "c", "d", "c")], {"c": "L1", "d": "L2"}, None)])


def test_build_groups_same_date_events_automatically():
    """build() is the real caller; it must do the grouping without being asked."""
    entrants_1 = [ent("a", "L1"), ent("b", "L2")]
    entrants_2 = [ent("c", "L1"), ent("d", "L2")]
    events = [
        (D1, entrants_1, [match("e1", "a", "b", "a")]),
        (D1, entrants_2, [match("e2", "c", "d", "c")]),
    ]
    rows = list(build(iter(events)))
    assert len(rows) == 2
    assert all(r.cell_games == 0 for r in rows), "neither event may see the other"


def test_build_can_hand_back_its_state():
    """Serving needs the accumulated records; build() used to throw them away."""
    keeper = FeatureBuilder()
    events = [
        (D1, [ent("a", "L1"), ent("b", "L2")], [match("e1", "a", "b", "a")]),
    ]
    list(build(iter(events), builder=keeper))
    assert keeper.leaders["L1"].games == 1
    assert keeper.leaders["L1"].wins == 1
