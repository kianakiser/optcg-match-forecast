"""Tests for the training pipeline.

These are mostly about the things that would fail silently. A model that trains and produces
plausible numbers while quietly leaking the future is the failure mode worth spending tests on;
"does sklearn fit" is not.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pytest

from optcg_forecast.training.evaluate import (
    COIN_BRIER,
    Calibration,
    assess_calibration,
    calibration,
    event_cluster_bootstrap,
    score,
)
from optcg_forecast.training.registry import CHAMPION, ModelRegistry, build_card
from optcg_forecast.training.run import (
    FEATURES,
    MAX_CALIBRATION_Z,
    MAX_ECE,
    MIN_CALIBRATION_BUCKETS,
    SWAP,
    Evaluation,
    _copy_serving_state,
    baseline_cell_rate,
    evaluate_block,
    evaluate_rolling,
    fit,
    predict,
    run,
    swap_sides,
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
    c = assess_calibration(probs, labels, min_n=250)
    assert c.worst_gap < 0.05
    assert c.buckets == 1, "only the fat bucket may count"


def test_calibration_reports_no_evidence_rather_than_perfection():
    """It used to return 0.0 when nothing qualified, which reads as flawless calibration.

    A safety check that fails open is worse than no check, because it is trusted.
    """
    c = assess_calibration([0.9] * 10 + [0.1] * 10, [1] * 10 + [0] * 10, min_n=250)
    assert c.buckets == 0, "no bucket is big enough to say anything"
    assert c.ece == 0.0
    # and the contract must treat too few buckets as a failure, not a pass
    assert MIN_CALIBRATION_BUCKETS > 0


def test_ece_does_not_grow_with_bucket_count_but_the_max_does():
    """Why the contract gates on ECE: the max statistic measures how many buckets you have.

    Both halves here are perfectly calibrated by construction. Splitting the same predictions
    across more buckets leaves ECE alone and inflates the worst gap.
    """
    import random

    rng = random.Random(7)
    few = [0.4] * 3000 + [0.6] * 3000
    many = [0.30 + 0.05 * (i % 9) for i in range(6000)]
    c_few = assess_calibration(few, [1 if rng.random() < p else 0 for p in few])
    c_many = assess_calibration(many, [1 if rng.random() < p else 0 for p in many])

    assert c_many.buckets > c_few.buckets
    assert c_many.worst_gap > c_few.worst_gap, "the max grows with bucket count"
    assert abs(c_many.ece - c_few.ece) < 0.02, "ECE does not"


def test_a_genuinely_miscalibrated_model_is_caught():
    """The check still has to bite, or replacing the criterion would be goalpost-moving."""
    probs = [0.8] * 2000  # claims 80%...
    labels = [1] * 1000 + [0] * 1000  # ...delivers 50%
    c = assess_calibration(probs, labels)
    assert c.ece > MAX_ECE
    assert c.worst_z > MAX_CALIBRATION_Z


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
        calibration=Calibration(ece=0.02, worst_gap=0.03, worst_z=1.1, buckets=4),
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


# ------------------------------------------------------------------- the gate


def test_the_gate_is_disjoint_from_the_selection_windows():
    """The point of the final block: nothing selected on it, so it can judge the selection."""
    rows = make_rows()
    select_rows, holdout = final_holdout(rows)
    select_events = {r["event_id"] for r in select_rows}
    holdout_events = {r["event_id"] for r in holdout}
    assert holdout_events
    assert not (select_events & holdout_events)

    for w in rolling_origin(select_rows, max_windows=4, min_train_rows=100, min_test_rows=50):
        assert not ({r["event_id"] for r in w.test} & holdout_events)


def test_gate_evaluates_a_single_block():
    rows = make_rows()
    select_rows, holdout = final_holdout(rows)
    gate = evaluate_block(select_rows, holdout, {"max_depth": 3, "max_iter": 50})
    assert gate.windows == 1
    assert gate.model_scores.n == len(holdout)
    assert gate.beats_coin


def test_a_model_with_no_signal_is_refused(tmp_path, monkeypatch, caplog):
    """The outcome that matters most: noise in, nothing registered, and said out loud."""
    import random

    import optcg_forecast.training.run as train_run

    rng = random.Random(1)
    noise = make_rows(n_events=50)  # past the pipeline's minimum-corpus guard
    for row in noise:
        row["label_p1_won"] = int(rng.random() < 0.5)  # label unrelated to every feature

    class FakeStore:
        def __init__(self, **_):
            pass

        def read(self):
            return noise

    monkeypatch.setattr(train_run, "FeatureStore", FakeStore)
    monkeypatch.setattr(train_run, "SWEEP", [{"max_depth": 3, "max_iter": 50}])

    with caplog.at_level("WARNING"):
        assert (
            run(
                feature_root=tmp_path,
                model_root=tmp_path / "models",
                max_windows=3,
                promote=True,
                dry_run=False,
            )
            == 0
        )
    assert "REJECTED" in caplog.text
    assert not (tmp_path / "models" / "versions").exists()


# ------------------------------------------------------- orientation invariance


def test_every_feature_has_a_swap_rule():
    """A feature added without one would silently break the symmetry guarantee."""
    assert set(SWAP) == set(FEATURES)


def test_swapping_twice_is_the_identity():
    """The clearest evidence the transform is right: it is its own inverse."""
    rows = make_rows(n_events=3, per_event=20)
    x, _, _ = to_matrix(rows)
    assert np.allclose(swap_sides(swap_sides(x)), x)


def test_prediction_does_not_depend_on_who_is_listed_first():
    """The provider's first slot is arbitrary — it is not the player who goes first.

    Without symmetrisation a tree has no idea that exchanging the players should turn p into
    1 - p, and measurably does not.
    """
    rows = make_rows()
    train, held = final_holdout(rows)
    model = fit(train, {"max_depth": 3, "learning_rate": 0.1, "max_iter": 50})

    x, _, _ = to_matrix(held)
    mirrored = [dict(r) for r in held]
    swapped = swap_sides(x)
    for i, r in enumerate(mirrored):
        for j, f in enumerate(FEATURES):
            r[f] = swapped[i][j]

    forward = predict(model, held)
    backward = predict(model, mirrored)
    worst = max(abs(a + b - 1.0) for a, b in zip(forward, backward, strict=True))
    assert worst < 1e-9, f"p(A beats B) + p(B beats A) must be 1, worst violation {worst}"


def test_the_unsymmetrised_model_really_is_asymmetric():
    """Guards the guard: if this ever passes trivially, the test above proves nothing."""
    rows = make_rows()
    train, held = final_holdout(rows)
    model = fit(train, {"max_depth": 3, "learning_rate": 0.1, "max_iter": 50})

    x, _, _ = to_matrix(held)
    raw = model.predict_proba(x)[:, 1]
    raw_swapped = model.predict_proba(swap_sides(x))[:, 1]
    assert max(abs(a + b - 1.0) for a, b in zip(raw, raw_swapped, strict=True)) > 1e-6


# ------------------------------------------------- serving state is versioned


def _state_fixture(tmp_path, event_ids, through):
    """Write a serving-state file stamped with a given corpus."""
    from optcg_forecast.features.compute import FeatureBuilder
    from optcg_forecast.features.serving_state import STATE_FILE, stamp_for, write_state

    builder = FeatureBuilder()
    builder.players["someone"].games = 10
    builder.players["someone"].wins = 6
    return write_state(builder, stamp_for(event_ids, through), tmp_path / STATE_FILE)


class _Store:
    def __init__(self, base):
        self.base = base


def test_serving_state_is_copied_into_the_model_version(tmp_path):
    from optcg_forecast.features.serving_state import STATE_FILE

    rows = [
        {"event_id": "e1", "event_date": "2026-01-01"},
        {"event_id": "e2", "event_date": "2026-01-08"},
    ]
    _state_fixture(tmp_path, ["e1", "e2"], "2026-01-08")

    reg = ModelRegistry(root=tmp_path / "models")
    reg.register({"m": 1}, _card("v1"))
    _copy_serving_state(_Store(tmp_path), reg, "v1", rows)

    copied = tmp_path / "models" / "versions" / "v1" / STATE_FILE
    assert copied.is_file(), "rolling the champion back must roll its records back too"
    assert json.loads(copied.read_text())["players"]["someone"] == [10, 6]


def test_registering_refuses_state_from_a_different_corpus(tmp_path):
    """A model served against another corpus's records is training-serving skew, silently.

    The same handle would carry a different win rate in production than it did in training, and
    nothing would raise — accuracy would just be quietly worse than the card claims.
    """
    rows = [{"event_id": "e1", "event_date": "2026-01-01"}]
    _state_fixture(tmp_path, ["a-completely-different-event"], "2020-01-01")

    reg = ModelRegistry(root=tmp_path / "models")
    reg.register({"m": 1}, _card("v1"))
    with pytest.raises(ValueError, match="different corpus"):
        _copy_serving_state(_Store(tmp_path), reg, "v1", rows)


def test_missing_serving_state_warns_rather_than_crashing(tmp_path, caplog):
    """No state is a degraded model, not a broken pipeline — say so and carry on."""
    reg = ModelRegistry(root=tmp_path / "models")
    reg.register({"m": 1}, _card("v1"))
    with caplog.at_level("WARNING"):
        _copy_serving_state(
            _Store(tmp_path), reg, "v1", [{"event_id": "e1", "event_date": "2026-01-01"}]
        )
    assert "no serving state" in caplog.text
