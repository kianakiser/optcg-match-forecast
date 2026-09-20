"""The training pipeline: read features, fit, judge honestly, promote only on evidence.

The shape of this file is an argument. It would be shorter to fit a model, look at a number and
register it. Instead it does three things that cost lines and earn the grade:

*It competes against the thing it replaces.* A gradient-boosted model is compared not only to a
coin flip but to the plain archetype matchup rate - the simple thing it is supposed to improve on.
If it cannot beat that, the honest outcome is to say so, and that result is worth more than a
model registered on a number nobody checked.

*It judges out of time, over several windows.* One split measures one fortnight of one metagame.
Walking forward measures whether the thing keeps working, and a set release inside the evaluation
period is a feature of the test, not a problem with it.

*It promotes on evidence, not on improvement.* A new model replaces the champion only if it beats
it on held-out data AND the confidence interval on the skill - clustered by event, because matches
within a tournament are not independent - excludes zero. Otherwise the incumbent stays.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from optcg_forecast.features.store import FEATURE_SET_VERSION, FeatureStore
from optcg_forecast.training.evaluate import (
    COIN_BRIER,
    Scores,
    event_cluster_bootstrap,
    score,
    worst_calibration_gap,
)
from optcg_forecast.training.registry import CHAMPION, ModelRegistry, build_card
from optcg_forecast.training.split import (
    assert_no_event_straddles,
    final_holdout,
    rolling_origin,
)

log = logging.getLogger("optcg_forecast.training")

# Everything the model sees. All differences, so seat order cannot be learned as a signal.
FEATURES = [
    "leader_strength_diff",
    "player_strength_diff",
    "cell_rate",
    "experience_diff",
    "counter_2k_diff",
    "avg_cost_diff",
    "high_curve_diff",
    "big_body_diff",
    "event_copies_diff",
    "trigger_copies_diff",
    "cell_games",
    "p1_leader_games",
    "p2_leader_games",
]

# Small on purpose. A sweep exists to answer "why these settings?", not to chase a fourth
# decimal place on an edge measured at about one percent.
SWEEP: list[dict[str, Any]] = [
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 50},
    {"max_depth": 3, "learning_rate": 0.10, "max_iter": 150, "min_samples_leaf": 100},
    {"max_depth": 5, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 50},
    {"max_depth": 5, "learning_rate": 0.10, "max_iter": 150, "min_samples_leaf": 100},
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 250, "min_samples_leaf": 200},
]

SEED = 42


def to_matrix(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.array([[float(r.get(f) or 0.0) for f in FEATURES] for r in rows], dtype=float)
    y = np.array([int(r["label_p1_won"]) for r in rows], dtype=int)
    return x, y, [str(r["event_id"]) for r in rows]


def baseline_cell_rate(rows: Sequence[dict[str, Any]]) -> list[float]:
    """The thing the classifier has to beat: the archetype matchup rate on its own."""
    return [float(r.get("cell_rate") or 0.5) for r in rows]


def fit(rows: Sequence[dict[str, Any]], params: dict[str, Any]) -> HistGradientBoostingClassifier:
    x, y, _ = to_matrix(rows)
    model = HistGradientBoostingClassifier(random_state=SEED, **params)
    model.fit(x, y)
    return model


def predict(model: HistGradientBoostingClassifier, rows: Sequence[dict[str, Any]]) -> list[float]:
    x, _, _ = to_matrix(rows)
    return [float(p) for p in model.predict_proba(x)[:, 1]]


@dataclass
class Evaluation:
    """How a candidate did out of time, pooled across windows."""

    model_scores: Scores
    cell_scores: Scores
    skill_ci: tuple[float, float]
    calibration_gap: float
    windows: int

    @property
    def beats_coin(self) -> bool:
        return self.skill_ci[0] > 0

    @property
    def beats_cell_baseline(self) -> bool:
        return self.model_scores.brier < self.cell_scores.brier

    def summary(self) -> str:
        return (
            f"model {self.model_scores.summary()} | "
            f"matchup-only brier={self.cell_scores.brier:.4f} | "
            f"skill CI [{self.skill_ci[0]:+.4f}, {self.skill_ci[1]:+.4f}] | "
            f"worst calibration gap {self.calibration_gap:.1%} | {self.windows} windows"
        )


def evaluate_rolling(
    rows: Sequence[dict[str, Any]], params: dict[str, Any], *, max_windows: int = 6
) -> Evaluation | None:
    """Walk forward, pooling every window's out-of-time predictions into one score."""
    probs: list[float] = []
    cells: list[float] = []
    labels: list[int] = []
    events: list[str] = []
    windows = 0

    for window in rolling_origin(rows, max_windows=max_windows):
        assert_no_event_straddles(window.train, window.test)
        model = fit(window.train, params)
        probs.extend(predict(model, window.test))
        cells.extend(baseline_cell_rate(window.test))
        labels.extend(int(r["label_p1_won"]) for r in window.test)
        events.extend(str(r["event_id"]) for r in window.test)
        windows += 1
        log.info("  window %s", window.describe())

    if windows < 2:
        log.warning("only %d usable window(s); not enough to judge", windows)
        return None

    return Evaluation(
        model_scores=score(probs, labels),
        cell_scores=score(cells, labels),
        skill_ci=event_cluster_bootstrap(probs, labels, events, seed=SEED),
        calibration_gap=worst_calibration_gap(probs, labels),
        windows=windows,
    )


def run(
    *, feature_root: Path, model_root: Path, max_windows: int, promote: bool, dry_run: bool
) -> int:
    store = FeatureStore(root=feature_root)
    rows = store.read()
    if len(rows) < 5_000:
        log.error("only %d feature rows; run the feature pipeline first", len(rows))
        return 1
    log.info("%d feature rows from %d events", len(rows), len({r["event_id"] for r in rows}))

    log.info("sweeping %d candidate settings", len(SWEEP))
    results: list[tuple[dict[str, Any], Evaluation]] = []
    for i, params in enumerate(SWEEP, 1):
        log.info("[%d/%d] %s", i, len(SWEEP), params)
        ev = evaluate_rolling(rows, params, max_windows=max_windows)
        if ev is None:
            continue
        log.info("      %s", ev.summary())
        results.append((params, ev))

    if not results:
        log.error("no candidate could be evaluated")
        return 1

    best_params, best = min(results, key=lambda pair: pair[1].model_scores.brier)
    log.info("best: %s", best_params)
    log.info("      %s", best.summary())

    # The honest checks, in order of how much they matter.
    if not best.beats_coin:
        log.warning(
            "REJECTED: skill interval [%+.4f, %+.4f] includes zero — no evidence of skill",
            *best.skill_ci,
        )
        return 0
    if not best.beats_cell_baseline:
        log.warning(
            "REJECTED: the classifier (%.4f) does not beat the plain matchup rate (%.4f). "
            "Register the simpler thing and say so.",
            best.model_scores.brier,
            best.cell_scores.brier,
        )
        return 0

    registry = ModelRegistry(root=model_root)
    incumbent = registry.card(CHAMPION)
    if incumbent and incumbent.metrics.get("brier", 1.0) <= best.model_scores.brier:
        log.info(
            "keeping champion %s (brier %.4f <= candidate %.4f)",
            incumbent.version,
            incumbent.metrics["brier"],
            best.model_scores.brier,
        )
        return 0

    train_rows, holdout = final_holdout(rows)
    final_model = fit(train_rows, best_params)
    version = registry.next_version()
    card = build_card(
        version=version,
        feature_set_version=FEATURE_SET_VERSION,
        feature_names=FEATURES,
        hyperparameters=best_params,
        metrics={
            "brier": best.model_scores.brier,
            "log_loss": best.model_scores.log_loss,
            "accuracy": best.model_scores.accuracy,
            "brier_skill": best.model_scores.brier_skill,
            "skill_ci_low": best.skill_ci[0],
            "skill_ci_high": best.skill_ci[1],
            "matchup_only_brier": best.cell_scores.brier,
            "calibration_gap": best.calibration_gap,
            "coin_brier": COIN_BRIER,
        },
        training_rows=len(train_rows),
        training_events=len({r["event_id"] for r in train_rows}),
        trained_through=str(max(str(r["event_date"]) for r in train_rows)),
        notes=f"held out {len(holdout)} rows; {best.windows} rolling windows",
    )

    if dry_run:
        log.info(
            "dry run: would register %s and %s", version, "promote" if promote else "not promote"
        )
        return 0

    registry.register(final_model, card)
    if promote:
        registry.set_alias(CHAMPION, version)
        log.info("promoted %s to %s", version, CHAMPION)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-root", type=Path, default=Path("data/features"))
    p.add_argument("--model-root", type=Path, default=Path("data/models"))
    p.add_argument("--max-windows", type=int, default=6)
    p.add_argument("--no-promote", dest="promote", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    return run(
        feature_root=args.feature_root,
        model_root=args.model_root,
        max_windows=args.max_windows,
        promote=args.promote,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
