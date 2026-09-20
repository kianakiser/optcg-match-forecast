"""Splitting the data for honest evaluation.

The features are already point-in-time correct, so a row never contains information from after
its own event. That is necessary but not sufficient: you can still cheat at evaluation time by
training on July and testing on June.

So the split is by event DATE, and always forward. Two further rules matter:

*Never split by row.* Matches inside one tournament share players, decks and conditions. Put some
of an event in train and the rest in test and the model has effectively seen the test set. This is
not a subtle effect - the project's own history has a measured example, where a random shuffle made
a per-player term look essential while out of time it was worth nothing at all.

*Rolling origin, not one split.* A single train/test split measures one fortnight of one metagame.
Walking forward through several windows measures whether the thing keeps working, which is the
actual question, and it is the only way to see a set release happen inside the evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

DEFAULT_WINDOW_DAYS = 28
# Below this a window's score is mostly noise and reporting it invites over-reading.
MIN_TEST_ROWS = 300
# A model needs some history before it can say anything; fitting on a fortnight is not a test
# of the pipeline, it is a test of luck.
MIN_TRAIN_ROWS = 2_000


def _row_date(row: dict[str, Any]) -> date:
    value = row["event_date"]
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class Window:
    """One rolling-origin fold: train on everything before, test on the next stretch."""

    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    cutoff: date
    test_end: date

    @property
    def test_events(self) -> int:
        return len({r["event_id"] for r in self.test})

    def describe(self) -> str:
        return (
            f"train<={self.cutoff} ({len(self.train):,} rows) "
            f"test {self.cutoff}..{self.test_end} ({len(self.test):,} rows, "
            f"{self.test_events} events)"
        )


def rolling_origin(
    rows: Sequence[dict[str, Any]],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_windows: int | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_test_rows: int = MIN_TEST_ROWS,
) -> Iterator[Window]:
    """Walk forward through the history, yielding train/test folds.

    Windows are yielded oldest first. Each test window is disjoint from the others, so pooling
    their predictions gives one honest out-of-time score over the whole period.
    """
    if not rows:
        return

    ordered = sorted(rows, key=_row_date)
    first, last = _row_date(ordered[0]), _row_date(ordered[-1])

    cutoffs = []
    cutoff = last - timedelta(days=window_days)
    while cutoff > first:
        cutoffs.append(cutoff)
        cutoff -= timedelta(days=window_days)
    cutoffs.reverse()
    if max_windows is not None:
        cutoffs = cutoffs[-max_windows:]

    for cut in cutoffs:
        end = cut + timedelta(days=window_days)
        train = [r for r in ordered if _row_date(r) <= cut]
        test = [r for r in ordered if cut < _row_date(r) <= end]
        if len(train) < min_train_rows or len(test) < min_test_rows:
            continue
        yield Window(train=train, test=test, cutoff=cut, test_end=end)


def final_holdout(
    rows: Sequence[dict[str, Any]], *, window_days: int = DEFAULT_WINDOW_DAYS
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The most recent stretch, held out. Used for the promotion decision.

    Deliberately the newest data: a model is promoted on how it does on the metagame it is about
    to face, not on how it would have done last spring.
    """
    if not rows:
        return [], []
    ordered = sorted(rows, key=_row_date)
    cut = _row_date(ordered[-1]) - timedelta(days=window_days)
    return (
        [r for r in ordered if _row_date(r) <= cut],
        [r for r in ordered if _row_date(r) > cut],
    )


def assert_no_event_straddles(train: Sequence[dict], test: Sequence[dict]) -> None:
    """Fail if any event appears on both sides.

    Cheap, and it catches the single most damaging mistake available here.
    """
    overlap = {r["event_id"] for r in train} & {r["event_id"] for r in test}
    if overlap:
        raise ValueError(
            f"{len(overlap)} event(s) appear in both train and test, so the split leaks: "
            f"{sorted(overlap)[:3]}"
        )
