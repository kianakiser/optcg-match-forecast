"""Scoring a probabilistic forecast, and saying how sure we are of the score.

Three choices here are load-bearing and worth being able to defend.

*Proper scoring rules, not accuracy.* The model outputs a probability, and accuracy throws almost
all of that away - 0.51 and 0.99 both count as "predicted a win". Brier and log loss reward being
right about how confident to be. Accuracy is reported too, because it is what a person asks for.

*The baseline is a coin flip, not the observed base rate.* Seat 1 wins 50.56% here, but the 95%
interval on that includes 50, so quoting it would be baseline-shopping: claiming credit for a bias
that may not exist.

*Confidence intervals cluster by event.* Matches inside one tournament share players, meta and
conditions, so they are not independent observations. Treating 28,942 matches as 28,942 independent
samples makes every interval far too narrow and invites claiming significance that is not there.
The effective sample size is closer to the number of events.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

# A coin flip: the honest reference for a balanced binary outcome.
COIN_BRIER = 0.25
COIN_LOG_LOSS = math.log(2)

# Probabilities are clipped before taking a log, because a confident miss would otherwise be
# infinitely bad and one row could dominate the whole score.
_EPS = 1e-6


@dataclass(frozen=True)
class Scores:
    """How a set of predictions did."""

    n: int
    brier: float
    log_loss: float
    accuracy: float

    @property
    def brier_skill(self) -> float:
        """How much better than a coin flip. Positive is better; 0 means no skill."""
        return COIN_BRIER - self.brier

    def summary(self) -> str:
        return (
            f"n={self.n:,} brier={self.brier:.4f} (skill {self.brier_skill:+.4f}) "
            f"logloss={self.log_loss:.4f} acc={self.accuracy:.2%}"
        )


def score(probs: Sequence[float], labels: Sequence[int]) -> Scores:
    """Score predictions against outcomes."""
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} predictions for {len(labels)} labels")
    if not probs:
        raise ValueError("nothing to score")

    brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)
    ll = -sum(
        math.log(max(min(p, 1 - _EPS), _EPS)) if y else math.log(max(min(1 - p, 1 - _EPS), _EPS))
        for p, y in zip(probs, labels, strict=True)
    ) / len(probs)
    # A prediction of exactly 0.5 is not a call; count it as half right rather than
    # silently awarding it to one side.
    correct = sum(
        1.0 if (p > 0.5) == bool(y) else 0.5 if p == 0.5 else 0.0
        for p, y in zip(probs, labels, strict=True)
    )
    return Scores(n=len(probs), brier=brier, log_loss=ll, accuracy=correct / len(probs))


def event_cluster_bootstrap(
    probs: Sequence[float],
    labels: Sequence[int],
    event_ids: Sequence[str],
    *,
    reps: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval on Brier skill, resampling whole EVENTS rather than matches.

    Resampling matches would treat two games from the same tournament as independent evidence,
    which they are not. Resampling events keeps each tournament intact, so the interval reflects
    how much the answer would move if a different set of tournaments had happened.
    """
    by_event: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for p, y, e in zip(probs, labels, event_ids, strict=True):
        by_event[e].append((p, y))

    events = list(by_event)
    if len(events) < 2:
        raise ValueError("need at least two events to bootstrap over")

    rng = random.Random(seed)
    skills = []
    for _ in range(reps):
        drawn = [by_event[rng.choice(events)] for _ in events]
        flat = [pair for group in drawn for pair in group]
        if not flat:
            continue
        b = sum((p - y) ** 2 for p, y in flat) / len(flat)
        skills.append(COIN_BRIER - b)

    skills.sort()
    tail = (1 - confidence) / 2
    lo = skills[int(tail * len(skills))]
    hi = skills[min(int((1 - tail) * len(skills)), len(skills) - 1)]
    return lo, hi


@dataclass(frozen=True)
class CalibrationBucket:
    """One slice of the reliability check."""

    low: float
    high: float
    n: int
    predicted: float
    observed: float

    @property
    def gap(self) -> float:
        return abs(self.predicted - self.observed)


def calibration(
    probs: Sequence[float], labels: Sequence[int], *, bucket_width: float = 0.05
) -> list[CalibrationBucket]:
    """Are predictions of 60% actually right 60% of the time?

    A model can be well-ranked and badly calibrated - useful for choosing a favourite, useless
    for stating a probability. Since this system's whole output is a probability, calibration is
    not optional.
    """
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in zip(probs, labels, strict=True):
        buckets[min(int(p / bucket_width), int(1 / bucket_width) - 1)].append((p, y))

    out = []
    for idx in sorted(buckets):
        pairs = buckets[idx]
        out.append(
            CalibrationBucket(
                low=idx * bucket_width,
                high=(idx + 1) * bucket_width,
                n=len(pairs),
                predicted=sum(p for p, _ in pairs) / len(pairs),
                observed=sum(y for _, y in pairs) / len(pairs),
            )
        )
    return out


def worst_calibration_gap(
    probs: Sequence[float], labels: Sequence[int], *, min_n: int = 250
) -> float:
    """Largest miscalibration among buckets with enough data to mean anything."""
    gaps = [b.gap for b in calibration(probs, labels) if b.n >= min_n]
    return max(gaps) if gaps else 0.0
