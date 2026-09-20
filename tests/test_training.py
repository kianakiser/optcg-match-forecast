"""Tests for the training pipeline.

These are mostly about the things that would fail silently. A model that trains and produces
plausible numbers while quietly leaking the future is the failure mode worth spending tests on;
"does sklearn fit" is not.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from optcg_forecast.training.evaluate import (
    COIN_BRIER,
    calibration,
    event_cluster_bootstrap,
    score,
    worst_calibration_gap,
)
from optcg_forecast.training.registry import CHAMPION, ModelRegistry, build_card
from optcg_forecast.training.run import (
    FEATURES,
    Evaluation,
    baseline_cell_rate,
    evaluate_rolling,
    fit,
    predict,
    to_matrix,
)
from optcg_forecast.training.split import (
    assert_no_event_straddles,
    final_holdout,
    rolling_origin,
)

START = date(2026, 1, 1)


def make_rows(n_events: int = 40, per_event: int = 120, signal: float = 0.9):
    """Synthetic matches with a known, learnable signal.

    p1 wins when its leader is stronger, plus noise. A model that cannot beat a coin flip here
    is broken, which makes this a usable smoke test for the whole pipeline.
    """
    import random

    rng = random.Random(0)
    rows = []
    for e in range(n_events):
        event_date = START + timedelta(days=7 * e)
        for i in range(per_event):
            diff = rng.uniform(-0.3, 0.3)
            p_win = 0.5 + signal * diff
            row = {
                "event_id": f"E{e:03d}",
                "event_date": event_date,
                "round": 1 + i % 8,
                "p1_leader": f"L{i % 12}",
                "p2_leader": f"L{(i + 5) % 12}",
                "label_p1_won": int(rng.random() < p_win),
                "leader_strength_diff": diff,
                "cell_rate": 0.5 + 0.5 * diff,
            }
            for f in FEATURES:
                row.setdefault(f, 0.0)
            rows.append(row)
    return rows


# --------------------------------------------------------------------- scoring


def test_coin_flip_scores_exactly_the_coin_baseline():
    scores = score([0.5] * 100, [1, 0] * 50)
    assert scores.brier == pytest.approx(COIN_BRIER)
    assert scores.brier_skill == pytest.approx(0.0)


def test_perfect_predictions_score_zero_brier():
    assert score([1.0, 0.0, 1.0], [1, 0, 1]).brier == pytest.approx(0.0)


def test_log_loss_does_not_blow_up_on_certainty():
    """A confident, wrong prediction must be finite or the mean is meaningless."""
    assert math.isfinite(score([1.0], [0]).log_loss)


def test_bootstrap_resamples_events_not_matches():
    """Two events whose matches disagree must widen the interval more than 240 matches would."""
    probs = [0.9] * 120 + [0.1] * 120
    labels = [1] * 120 + [1] * 120  # second event is confidently wrong
    events = ["A"] * 120 + ["B"] * 120
    low, high = event_cluster_bootstrap(probs, labels, events, reps=200)
    assert low < 0 < high  # with only two events, nothing is established


def test_bootstrap_refuses_a_single_event():
    with pytest.raises(ValueError, match="two events"):
        event_cluster_bootstrap([0.6] * 10, [1] * 10, ["A"] * 10)


def test_calibration_gap_ignores_thin_buckets():
    """A bucket with three rows says nothing; it must not drive the reported gap."""
    probs = [0.95, 0.95, 0.95] + [0.5] * 400
    labels = [0, 0, 0] + [1, 0] * 200
    assert worst_calibration_gap(probs, labels, min_n=250) < 0.05


def test_calibration_buckets_cover_every_prediction():
    probs = [i / 100 for i in range(101)]
    labels = [i % 2 for i in range(101)]
    assert sum(b.n for b in calibration(probs, labels)) == len(probs)


# ---------------------------------------------------------------------- splits


def test_rolling_origin_never_trains_on_the_future():
    rows = make_rows()
    windows = list(rolling_origin(rows, max_windows=4, min_train_rows=100, min_test_rows=50))
    assert windows
    for w in windows:
        assert max(r["event_date"] for r in w.train) <= w.cutoff
        assert min(r["event_date"] for r in w.test) > w.cutoff


def test_rolling_origin_test_windows_are_disjoint():
    """Pooling predictions across windows is only honest if no match is scored twice."""
    rows = make_rows()
    windows = list(rolling_origin(rows, max_windows=5, min_train_rows=100, min_test_rows=50))
    seen: set[tuple] = set()
    for w in windows:
        keys = {(r["event_id"], r["round"], r["p1_leader"], id(r)) for r in w.test}
        assert not (keys & seen)
        seen |= keys


def test_no_event_straddles_the_boundary():
    rows = make_rows()
    for w in rolling_origin(rows, max_windows=4, min_train_rows=100, min_test_rows=50):
        assert_no_event_straddles(w.train, w.test)


def test_straddle_check_actually_catches_a_leak():
    row = {"event_id": "E1"}
    with pytest.raises(ValueError, match="leaks"):
        assert_no_event_straddles([row], [row])


def test_final_holdout_holds_out_the_newest_data():
    train, held = final_holdout(make_rows())
    assert held
    assert max(r["event_date"] for r in train) < min(r["event_date"] for r in held)


def test_rolling_origin_yields_nothing_without_enough_data():
    assert list(rolling_origin(make_rows(n_events=2, per_event=10))) == []


# --------------------------------------------------------------------- the model


def test_feature_matrix_never_contains_the_label():
    """The one mistake that would make every number in the report a lie."""
    assert "label_p1_won" not in FEATURES
    for banned in ("placing", "record", "drop", "winner", "label"):
        assert not any(banned in f for f in FEATURES)


def test_missing_features_become_zero_not_crashes():
    x, y, events = to_matrix([{"event_id": "E", "label_p1_won": 1}])
    assert x.shape == (1, len(FEATURES))
    assert list(y) == [1] and events == ["E"]


def test_model_beats_a_coin_flip_on_learnable_data():
    rows = make_rows()
    train, held = final_holdout(rows)
    model = fit(train, {"max_depth": 3, "learning_rate": 0.1, "max_iter": 50})
    scores = score(predict(model, held), [r["label_p1_won"] for r in held])
    assert scores.brier < COIN_BRIER


def test_cell_baseline_is_the_raw_matchup_rate():
    rows = [{"cell_rate": 0.62}, {}]
    assert baseline_cell_rate(rows) == [0.62, 0.5]


def test_rolling_evaluation_pools_every_window():
    rows = make_rows()
    ev = evaluate_rolling(
        rows, {"max_depth": 3, "learning_rate": 0.1, "max_iter": 50}, max_windows=3
    )
    assert ev is not None
    assert ev.windows >= 2
    assert ev.model_scores.n > 0
    assert "brier" in ev.summary()


def test_evaluation_reports_a_loss_against_the_baseline_honestly():
    ev = Evaluation(
        model_scores=score([0.5] * 10, [1, 0] * 5),
        cell_scores=score([0.6, 0.4] * 5, [1, 0] * 5),  # the simple baseline does better
        skill_ci=(-0.01, 0.02),
        calibration_gap=0.03,
        windows=3,
    )
    assert not ev.beats_coin
    assert not ev.beats_cell_baseline


# -------------------------------------------------------------------- registry


def _card(version: str, brier: float = 0.24):
    return build_card(
        version=version,
        feature_set_version="v1",
        feature_names=FEATURES,
        hyperparameters={"max_depth": 3},
        metrics={"brier": brier},
        training_rows=1000,
        training_events=20,
        trained_through="2026-09-01",
    )


def test_versions_are_immutable(tmp_path):
    reg = ModelRegistry(root=tmp_path)
    reg.register({"model": 1}, _card("v1"))
    with pytest.raises(FileExistsError):
        reg.register({"model": 2}, _card("v1"))


def test_promotion_is_moving_an_alias(tmp_path):
    reg = ModelRegistry(root=tmp_path)
    reg.register({"model": 1}, _card("v1"))
    reg.register({"model": 2}, _card("v2", brier=0.23))
    reg.set_alias(CHAMPION, "v1")
    assert reg.load(CHAMPION)[0] == {"model": 1}

    reg.set_alias(CHAMPION, "v2")
    assert reg.load(CHAMPION)[0] == {"model": 2}
    assert reg.resolve(CHAMPION) == "v2"

    reg.set_alias(CHAMPION, "v1")  # rollback is the same operation, backwards
    assert reg.load(CHAMPION)[0] == {"model": 1}


def test_cannot_promote_a_version_that_does_not_exist(tmp_path):
    with pytest.raises(ValueError, match="unknown version"):
        ModelRegistry(root=tmp_path).set_alias(CHAMPION, "v9")


def test_serving_gets_a_clear_error_when_nothing_is_promoted(tmp_path):
    with pytest.raises(KeyError):
        ModelRegistry(root=tmp_path).load(CHAMPION)
    assert ModelRegistry(root=tmp_path).card(CHAMPION) is None


def test_card_records_what_the_model_was_promoted_on(tmp_path):
    reg = ModelRegistry(root=tmp_path)
    reg.register({"m": 1}, _card("v1", brier=0.2412))
    _, card = reg.load("v1")
    assert card.metrics["brier"] == pytest.approx(0.2412)
    assert card.feature_set_version == "v1"
    assert card.feature_names == FEATURES
    assert card.git_sha


def test_next_version_increments_past_ten(tmp_path):
    """Sorting versions as strings would put v10 before v2."""
    reg = ModelRegistry(root=tmp_path)
    for i in range(1, 12):
        reg.register({"m": i}, _card(f"v{i}"))
    assert reg.next_version() == "v12"
    assert reg.versions()[-1] == "v11"
