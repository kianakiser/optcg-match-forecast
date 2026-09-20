"""Test-suite guards.

There is one guard here and it exists because the suite silently corrupted real data.

Several pipeline tests called the feature pipeline's `main()` with a temporary landing directory
but left `--feature-root` at its default, which is the REAL `data/features`. Every test run wrote
two synthetic fixture events into the live feature store. They then survived every subsequent
rebuild, because the incremental write preserves rows for events the incoming batch does not
mention. Two fake matches were trained on, and were counted in figures that went into the README
and the proposal.

Nothing failed. That is the whole problem: the tests passed, the pipeline was green, and the
corpus was wrong by two events for days.

So the fix is not only to pass `--feature-root` in those tests - it is to make the mistake
impossible to repeat quietly. This fixture fails the run if the suite touches the real store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REAL_STORE = Path("data/features")
REAL_LANDING = Path("data/landing")


def _fingerprint(root: Path) -> str | None:
    """A cheap, order-independent hash of what is on disk. None when there is nothing."""
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            h.update(str(path).encode())
            h.update(str(stat.st_size).encode())
            h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def real_data_is_read_only():
    """Fail the session if any test writes to the real landing zone or feature store.

    Session-scoped so it costs one fingerprint either side of the whole run, and autouse so a
    test written next year is covered without anyone remembering this happened.
    """
    before = {p: _fingerprint(p) for p in (REAL_STORE, REAL_LANDING)}
    yield
    for path, was in before.items():
        now = _fingerprint(path)
        if was != now:
            pytest.fail(
                f"the test suite modified {path}, which holds real data.\n"
                f"Some test is using a production default instead of tmp_path - most likely a "
                f"pipeline entry point called without --feature-root or --out-dir.\n"
                f"Find it, give it a temporary directory, and rebuild the store from the "
                f"landing zone:\n"
                f"    uv run python -m optcg_forecast.features.materialise",
                pytrace=False,
            )
