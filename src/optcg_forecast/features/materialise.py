"""Landing zone to feature store: the stage that turns clean rows into model inputs.

Kept separate from ingest for one reason. Ingest talks to a rate-limited API and its output is
immutable; this stage is pure computation over files already on disk. Splitting them means a
change to the feature definitions is a rerun of this stage over cached landing data - seconds,
no API traffic - rather than a fresh crawl of fourteen months of tournaments.

It rebuilds the whole store from scratch every time, which looks wasteful and is not. The
features are cumulative: a leader's strength before event N depends on every event before it.
Appending one event to existing state would work only if the state were persisted and exactly
in step with the store, which is a synchronisation bug waiting to happen. Rebuilding is 29k rows
in a few seconds, it is trivially idempotent, and it means the store always reflects the current
feature code rather than a mixture of whatever versions ran that month.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from optcg_forecast.features.cards import Card, CardCatalogue
from optcg_forecast.features.compute import FeatureBuilder, FeatureRow, build
from optcg_forecast.features.ingest import Entrant, Match
from optcg_forecast.features.serving_state import STATE_FILE, stamp_for, write_state
from optcg_forecast.features.store import FeatureStore

log = logging.getLogger(__name__)


def _entrant(row: dict[str, Any]) -> Entrant:
    return Entrant(
        event_id=str(row["event_id"]),
        event_date=date.fromisoformat(str(row["event_date"])[:10]),
        player=str(row["player"]),
        leader_id=row.get("leader_id"),
        country=row.get("country"),
        decklist=row.get("decklist"),
        did_drop=bool(row.get("did_drop")),
    )


def _match(row: dict[str, Any]) -> Match:
    return Match(
        event_id=str(row["event_id"]),
        event_date=date.fromisoformat(str(row["event_date"])[:10]),
        round=int(row["round"]),
        table=int(row["table"]),
        player1=str(row["player1"]),
        player2=str(row["player2"]),
        winner=str(row["winner"]),
    )


def read_landing(landing: Path) -> Iterator[tuple[date, list[Entrant], list[Match]]]:
    """Yield every landed event in date order.

    Date order is not a preference. compute.build raises if events arrive out of order,
    because processing them out of order would let a model see the future.
    """
    records = []
    for path in sorted(landing.rglob("*.json")):
        if path.name.startswith("_"):
            continue  # bookkeeping, by convention - see already_ingested()
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("skipping unreadable landing file %s", path)
            continue
        # Not every .json under this tree is a landing record. A store manifest, an editor
        # backup or a stray export would otherwise be treated as an event and take the whole
        # rebuild down with a KeyError - which is exactly what happened the first time a test
        # pointed the feature store at a directory inside the landing zone.
        if not isinstance(record, dict) or not {"event_id", "event_date"} <= record.keys():
            log.warning("skipping %s: not a landing record", path)
            continue
        records.append(record)

    records.sort(key=lambda r: (str(r["event_date"]), str(r["event_id"])))
    _check_the_corpus_has_not_shrunk(landing, len(records))
    for record in records:
        event_date = date.fromisoformat(str(record["event_date"])[:10])
        yield (
            event_date,
            [_entrant(e) for e in record.get("entrants", [])],
            [_match(m) for m in record.get("matches", [])],
        )


HIGH_WATER = "_high_water.json"


def _check_the_corpus_has_not_shrunk(landing: Path, events: int) -> None:
    """Shout if the landing zone has fewer events than it has ever had.

    Events are immutable and ingest only ever adds, so the count is monotonic and a drop always
    means something went wrong. The way it goes wrong in practice: the scheduled feature run
    saves its cache with `if: always()`, so a run cancelled mid-ingest publishes a PARTIAL
    landing zone under the newest key, and the training job restores by prefix and trains on
    whatever it finds. Fewer events, a worse model, a green tick.

    A warning rather than an error, because there is one legitimate way for the count to fall:
    GitHub evicts caches after a week of inactivity, and a from-scratch re-ingest genuinely
    starts small. That case is recoverable and common enough that failing would be wrong - but
    it should never pass silently either.
    """
    marker = landing / HIGH_WATER
    previous = 0
    if marker.is_file():
        try:
            previous = int(json.loads(marker.read_text()).get("events", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            log.warning("could not read %s; treating the corpus as new", marker)
    if events < previous:
        log.warning(
            "THE LANDING ZONE SHRANK: %d events now, %d before. Events are immutable and ingest "
            "only adds, so this is a partial restore (a cancelled ingest publishing its cache) "
            "or a cache that was evicted and is refilling. Anything trained on this corpus is "
            "trained on less than it should be.",
            events,
            previous,
        )
    marker.write_text(json.dumps({"events": max(events, previous)}))


def materialise(
    landing: Path, store: FeatureStore, catalogue: dict[str, Card] | None = None
) -> list[FeatureRow]:
    """Rebuild the feature store from the landing zone. Returns what was written."""
    cat = catalogue if catalogue is not None else CardCatalogue().load()
    events = list(read_landing(landing))
    if not events:
        log.warning("landing zone %s is empty; nothing to materialise", landing)
        return []

    # Hold the builder rather than letting build() construct and discard one: the leader,
    # player and matchup records it accumulates are the only place a prediction's inputs can
    # come from, and the feature store deliberately keeps no player identifiers to rebuild
    # them from later.
    builder = FeatureBuilder()
    rows = list(build(iter(events), cat, builder=builder))
    log.info(
        "computed %d feature row(s) from %d event(s) (%s..%s)",
        len(rows),
        len(events),
        events[0][0],
        events[-1][0],
    )
    # replace_all, because this IS the full rebuild. Anything not derivable from the
    # landing zone has no business being in the store.
    store.write(rows, replace_all=True)

    stamp = stamp_for([r.event_id for r in rows], str(events[-1][0]))
    write_state(builder, stamp, store.base / STATE_FILE)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Rebuild the feature store without touching the API.

    Its own entry point because the training job needs features but has no business fetching
    tournaments, and because a change to a feature definition is rerun from here in seconds.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild the feature store from the landing zone")
    parser.add_argument("--landing", type=Path, default=Path("data/landing"))
    parser.add_argument("--feature-root", type=Path, default=Path("data/features"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    rows = materialise(args.landing, FeatureStore(root=args.feature_root))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
