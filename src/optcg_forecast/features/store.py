"""The feature store, as versioned Parquet.

The course allows either a managed feature store or "versioned Parquet/CSV on GCS" - the FAQ
calls the latter a poor man's feature store and blesses it explicitly. Parquet first, because it
has almost nothing that can break in November; Hopsworks is a migration we can make later if
there is time, and the migration itself is a good milestone-summary story.

What makes this a feature *store* rather than a pile of files:

*Partitioned by event month*, so a backfill rewrites one partition instead of the world, and a
reader can take a date range without loading everything.

*Versioned by feature-set name.* Change what a column means and you write to a new version rather
than silently mixing two definitions in one dataset - the mistake that makes an old model
inexplicable six weeks later.

*Idempotent.* Re-running over the same events replaces those rows rather than appending
duplicates, because a pipeline that double-counts on a retry is worse than one that fails.

The layout is deliberately plain - directories and Parquet files - so that moving it to GCS is a
path change, and moving it to Hopsworks is a writer change.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from optcg_forecast.features.compute import FeatureRow

log = logging.getLogger(__name__)

# Bump when a column's MEANING changes, not when one is added. Readers pin to a version, so an
# old model can always be explained against the features it actually saw.
# v2, 2026-09-20: events sharing a date are now emitted from one pre-date snapshot instead of
# one at a time, so a later event on a date no longer trains on an earlier one. That changes the
# VALUE of leader_strength_diff, player_strength_diff and cell_rate on 12,486 rows, which means a
# v1 store and a v2 store built from the same landing zone disagree. The version has to move or a
# model card saying "feature_set_version: v1" is ambiguous about which of the two it means.
FEATURE_SET_VERSION = "v2"


@dataclass
class FeatureStore:
    """Versioned, partitioned Parquet on a local or mounted path."""

    root: Path = Path("data/features")
    version: str = FEATURE_SET_VERSION

    @property
    def base(self) -> Path:
        return self.root / self.version

    def _partition(self, event_date: date) -> Path:
        return self.base / f"month={event_date:%Y-%m}" / "part.parquet"

    # ------------------------------------------------------------------ write

    def write(self, rows: Sequence[FeatureRow], *, replace_all: bool = False) -> list[Path]:
        """Write rows, replacing any existing rows for the same events.

        Replacing rather than appending is what makes a retry safe: the same event ingested
        twice yields one copy, not two.

        `replace_all` makes the store hold exactly these rows and nothing else, which is what a
        full rebuild means and what the incremental path quietly does NOT do. Incrementally,
        rows for events absent from `rows` are preserved - correct when adding one event, wrong
        when rebuilding, because anything that got in and then vanished from the source stays
        forever. Two synthetic events from a test fixture survived in this store for exactly
        that reason, and were counted in published figures.
        """
        if not rows:
            if replace_all:
                raise ValueError(
                    "refusing to replace the whole store with nothing - this is either a bug "
                    "upstream or an empty landing zone, and silently emptying the store would "
                    "hide both"
                )
            log.info("no feature rows to write")
            return []

        by_month: dict[str, list[FeatureRow]] = {}
        for row in rows:
            by_month.setdefault(f"{row.event_date:%Y-%m}", []).append(row)

        written = []
        for month, month_rows in sorted(by_month.items()):
            path = self.base / f"month={month}" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            incoming_events = {r.event_id for r in month_rows}
            keep: list[dict[str, Any]] = []
            if path.is_file() and not replace_all:
                existing = pq.read_table(path).to_pylist()
                keep = [r for r in existing if r.get("event_id") not in incoming_events]
                if len(keep) != len(existing):
                    log.info(
                        "replacing %d existing row(s) for %d re-ingested event(s) in %s",
                        len(existing) - len(keep),
                        len(incoming_events),
                        month,
                    )

            records = keep + [r.as_dict() for r in month_rows]
            records.sort(key=lambda r: (str(r["event_date"]), str(r["event_id"]), r["round"]))
            pq.write_table(pa.Table.from_pylist(records), path, compression="snappy")
            written.append(path)
            log.info("wrote %d row(s) to %s", len(records), path)

        if replace_all:
            for stale in sorted(self.base.glob("month=*")):
                if stale / "part.parquet" not in written:
                    log.warning("dropping stale partition %s: no rows in the rebuild", stale.name)
                    for f in sorted(stale.iterdir()):
                        f.unlink()
                    stale.rmdir()

        self._write_manifest()
        return written

    def _write_manifest(self) -> None:
        """A small manifest so a reader can see what exists without scanning every file."""
        partitions = []
        for part in sorted(self.base.glob("month=*/part.parquet")):
            meta = pq.read_metadata(part)
            partitions.append({"month": part.parent.name.split("=", 1)[1], "rows": meta.num_rows})
        manifest = {
            "feature_set_version": self.version,
            "partitions": partitions,
            "total_rows": sum(p["rows"] for p in partitions),
            "columns": [f.name for f in FeatureRow.__dataclass_fields__.values()],
        }
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # ------------------------------------------------------------------ read

    def manifest(self) -> dict[str, Any]:
        path = self.base / "manifest.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    def read(self, *, start: date | None = None, end: date | None = None) -> list[dict[str, Any]]:
        """Read rows, optionally within a date range.

        The range filter is applied per partition first, so a training run asking for one
        month does not pay to read fourteen.
        """
        rows: list[dict[str, Any]] = []
        for part in sorted(self.base.glob("month=*/part.parquet")):
            month = part.parent.name.split("=", 1)[1]
            if start and month < f"{start:%Y-%m}":
                continue
            if end and month > f"{end:%Y-%m}":
                continue
            rows.extend(pq.read_table(part).to_pylist())

        if start:
            rows = [r for r in rows if _as_date(r["event_date"]) >= start]
        if end:
            rows = [r for r in rows if _as_date(r["event_date"]) <= end]
        rows.sort(key=lambda r: (str(r["event_date"]), str(r["event_id"]), r["round"]))
        return rows

    def event_ids(self) -> set[str]:
        """Which events are already in the store, so ingest can skip them."""
        return {r["event_id"] for r in self.read()}


def _as_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def write_feature_rows(rows: Iterable[FeatureRow], root: Path | None = None) -> list[Path]:
    """Convenience wrapper used by the pipeline entry point."""
    store = FeatureStore(root=root) if root else FeatureStore()
    return store.write(list(rows))
