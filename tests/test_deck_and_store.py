"""Tests for deck summarisation and the Parquet feature store."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from optcg_forecast.features.cards import Card, card_id_from_entry
from optcg_forecast.features.compute import FeatureRow
from optcg_forecast.features.deck import summarise
from optcg_forecast.features.store import FeatureStore


def card(cid, category="Character", cost=3, power=5000, counter=1000, trigger=False):
    return Card(
        id=cid,
        category=category,
        cost=cost,
        power=power,
        counter=counter,
        life=None,
        colors=("Red",),
        has_trigger=trigger,
    )


CATALOGUE = {
    "OP01-001": card("OP01-001", counter=2000, cost=7, power=7000),
    "OP01-002": card("OP01-002", counter=1000, cost=2, power=3000),
    "OP01-003": card("OP01-003", category="Event", cost=1, power=None, counter=None, trigger=True),
}


def deck(*pairs):
    return {"character": [{"set": "OP01", "number": n, "count": c} for n, c in pairs]}


# --------------------------------------------------------------------------- card ids


def test_card_id_zero_pads_the_number():
    """The join silently fails if the number is not padded to three digits."""
    assert card_id_from_entry({"set": "OP05", "number": "98"}) == "OP05-098"
    assert card_id_from_entry({"set": "op05", "number": 7}) == "OP05-007"
    assert card_id_from_entry({"set": "OP05"}) is None


def test_leader_life_is_not_read_as_cost():
    """The source puts a leader's LIFE in the cost field — a real trap."""
    from optcg_forecast.features.cards import _parse

    leader = _parse({"id": "ST01-001", "category": "Leader", "cost": 5, "power": 5000})
    assert leader.life == 5
    assert leader.cost is None, "reading life as cost makes every leader look expensive"


# --------------------------------------------------------------------------- summarise


def test_counts_copies_not_distinct_cards():
    s = summarise(deck(("001", 4), ("002", 4)), CATALOGUE)
    assert s.total_cards == 8
    assert s.distinct_cards == 2
    assert s.counter_2k_copies == 4


def test_same_leader_different_decks_summarise_differently():
    """The point of this module: two builds must not look identical."""
    a = summarise(deck(("001", 4), ("002", 4)), CATALOGUE)
    b = summarise(deck(("001", 1), ("002", 7)), CATALOGUE)
    assert a.counter_2k_copies != b.counter_2k_copies
    assert a.avg_cost != b.avg_cost


def test_cost_average_excludes_cards_with_no_cost():
    """Counting a 'Cost —' event as zero would drag the average down for no reason."""
    no_cost = {"X-001": card("X-001", category="Event", cost=None, power=None, counter=None)}
    s = summarise({"event": [{"set": "X", "number": "001", "count": 4}]}, no_cost)
    assert s.avg_cost == 0.0
    assert s.total_cards == 4


def test_unresolvable_cards_lower_the_resolved_fraction():
    s = summarise(deck(("001", 4), ("999", 46)), CATALOGUE)
    assert s.total_cards == 50
    assert s.resolved_fraction == pytest.approx(4 / 50)
    assert not s.is_usable, "a mostly-unresolvable list must not be trusted"


def test_missing_decklist_is_not_an_error():
    assert summarise(None, CATALOGUE).total_cards == 0
    assert not summarise(None, CATALOGUE).is_usable


# --------------------------------------------------------------------------- store


def row(event_id="e1", d=date(2026, 3, 4), rnd=1):
    return FeatureRow(
        event_id=event_id,
        event_date=d,
        round=rnd,
        p1_leader="L1",
        p2_leader="L2",
        leader_strength_diff=0.1,
        player_strength_diff=0.0,
        cell_rate=0.5,
        experience_diff=0.0,
        counter_2k_diff=1.0,
        avg_cost_diff=0.2,
        high_curve_diff=0.0,
        big_body_diff=0.0,
        event_copies_diff=0.0,
        trigger_copies_diff=0.0,
        deck_features_usable=True,
        p1_leader_games=50,
        p2_leader_games=50,
        cell_games=20,
        coverage="solid",
        label_p1_won=1,
    )


def test_write_then_read_round_trips(tmp_path: Path):
    s = FeatureStore(root=tmp_path)
    s.write([row()])
    back = s.read()
    assert len(back) == 1
    assert back[0]["p1_leader"] == "L1"
    assert back[0]["coverage"] == "solid"


def test_partitions_by_event_month(tmp_path: Path):
    s = FeatureStore(root=tmp_path)
    s.write([row(d=date(2026, 1, 5)), row(d=date(2026, 2, 5))])
    months = sorted(p.parent.name for p in s.base.glob("month=*/part.parquet"))
    assert months == ["month=2026-01", "month=2026-02"]


def test_rewriting_the_same_event_replaces_rather_than_duplicates(tmp_path: Path):
    """A retry must not double-count. This is what makes the schedule safe to miss."""
    s = FeatureStore(root=tmp_path)
    s.write([row(rnd=1), row(rnd=2)])
    s.write([row(rnd=1), row(rnd=2)])
    assert len(s.read()) == 2


def test_a_different_event_in_the_same_month_is_kept(tmp_path: Path):
    s = FeatureStore(root=tmp_path)
    s.write([row(event_id="e1")])
    s.write([row(event_id="e2")])
    assert {r["event_id"] for r in s.read()} == {"e1", "e2"}


def test_date_range_read(tmp_path: Path):
    s = FeatureStore(root=tmp_path)
    s.write([row(d=date(2026, 1, 5)), row(d=date(2026, 6, 5))])
    assert len(s.read(start=date(2026, 5, 1))) == 1
    assert len(s.read(end=date(2026, 2, 1))) == 1
    assert len(s.read()) == 2


def test_manifest_describes_what_is_stored(tmp_path: Path):
    s = FeatureStore(root=tmp_path)
    s.write([row(d=date(2026, 1, 5)), row(d=date(2026, 2, 5))])
    m = s.manifest()
    assert m["total_rows"] == 2
    assert m["feature_set_version"] == s.version
    assert "label_p1_won" in m["columns"]


def test_versions_are_isolated(tmp_path: Path):
    """Changing what a column means writes a new version rather than mixing definitions."""
    FeatureStore(root=tmp_path, version="v1").write([row()])
    v2 = FeatureStore(root=tmp_path, version="v2")
    assert v2.read() == []


def test_event_ids_lets_ingest_skip_what_is_already_stored(tmp_path: Path):
    s = FeatureStore(root=tmp_path)
    s.write([row(event_id="e1"), row(event_id="e2")])
    assert s.event_ids() == {"e1", "e2"}


# ----------------------------------------------------- a rebuild must be a rebuild


def test_replace_all_drops_events_the_rebuild_no_longer_has(tmp_path):
    """The bug that let two test fixtures live in the real store for days.

    The incremental path preserves rows for events the incoming batch does not mention, which is
    right when adding one event and wrong when rebuilding: anything that got in and then vanished
    from the source stays forever, and gets counted.
    """
    store = FeatureStore(root=tmp_path)
    store.write([row("real1", date(2026, 3, 1)), row("fixture", date(2026, 3, 2))])
    assert store.event_ids() == {"real1", "fixture"}

    store.write([row("real1", date(2026, 3, 1))])  # incremental: fixture survives
    assert store.event_ids() == {"real1", "fixture"}

    store.write([row("real1", date(2026, 3, 1))], replace_all=True)
    assert store.event_ids() == {"real1"}, "a rebuild must leave exactly what was rebuilt"


def test_replace_all_removes_a_month_that_lost_all_its_events(tmp_path):
    store = FeatureStore(root=tmp_path)
    store.write([row("a", date(2026, 3, 1)), row("b", date(2026, 4, 1))])
    assert {p.name for p in store.base.glob("month=*")} == {"month=2026-03", "month=2026-04"}

    store.write([row("a", date(2026, 3, 1))], replace_all=True)
    assert {p.name for p in store.base.glob("month=*")} == {"month=2026-03"}
    assert store.event_ids() == {"a"}


def test_replace_all_refuses_to_empty_the_store(tmp_path):
    """An empty rebuild is a bug upstream, not an instruction to delete everything."""
    store = FeatureStore(root=tmp_path)
    store.write([row("a", date(2026, 3, 1))])
    with pytest.raises(ValueError, match="refusing"):
        store.write([], replace_all=True)
    assert store.event_ids() == {"a"}


# ------------------------------------------------ the landing zone must not shrink


def test_a_shrinking_landing_zone_is_reported(tmp_path, caplog):
    """A cancelled ingest publishes a partial cache; training then silently uses less data."""
    from optcg_forecast.features.materialise import _check_the_corpus_has_not_shrunk

    _check_the_corpus_has_not_shrunk(tmp_path, 277)
    with caplog.at_level("WARNING"):
        _check_the_corpus_has_not_shrunk(tmp_path, 120)
    assert "SHRANK" in caplog.text
    assert "277" in caplog.text and "120" in caplog.text


def test_growth_is_silent_and_the_mark_rises(tmp_path, caplog):
    from optcg_forecast.features.materialise import _check_the_corpus_has_not_shrunk

    _check_the_corpus_has_not_shrunk(tmp_path, 100)
    with caplog.at_level("WARNING"):
        _check_the_corpus_has_not_shrunk(tmp_path, 277)
    assert "SHRANK" not in caplog.text
    # and the mark does not fall back after a dip
    _check_the_corpus_has_not_shrunk(tmp_path, 50)
    with caplog.at_level("WARNING"):
        caplog.clear()
        _check_the_corpus_has_not_shrunk(tmp_path, 200)
    assert "SHRANK" in caplog.text, "the high-water mark must remember 277, not 50"


def test_a_corrupt_marker_does_not_stop_the_rebuild(tmp_path, caplog):
    from optcg_forecast.features.materialise import HIGH_WATER, _check_the_corpus_has_not_shrunk

    (tmp_path / HIGH_WATER).write_text("not json at all")
    with caplog.at_level("WARNING"):
        _check_the_corpus_has_not_shrunk(tmp_path, 277)
    assert "could not read" in caplog.text
