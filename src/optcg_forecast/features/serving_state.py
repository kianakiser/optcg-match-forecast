"""The records a prediction needs, kept so they survive past the pipeline that built them.

This module exists because of a decision about the product. The UI asks for two decks *and both
players' Limitless handles*, because measured over six held-out windows the player-history terms
are the signal: 0.2314 Brier with them against 0.2441 without, while a deck-only model scores
0.2441 against 0.2452 for a plain matchup lookup and is not distinguishable from it.

Asking for handles is only possible if the numbers behind those terms still exist at predict
time, and they did not. `FeatureBuilder` accumulates leader, player and matchup records while
walking the corpus, and `build()` threw the builder away when the walk finished. The feature
store keeps the *rows* - and deliberately keeps no player identifiers in them, so the state
cannot be reconstructed from the store afterwards. Given two handles, serving had nothing to
look them up in.

So the state is written out beside the feature store, and copied into a model version when one
is registered. Two properties matter:

*It is stamped with the corpus it came from.* A model served against state built from a
different corpus is training-serving skew of the most literal kind - the same handle scoring
differently in training and in production - and it would show up as a quiet accuracy loss rather
than an error. `Stamp` records the feature-set version, the last event date, the event count and
a digest of the event ids; the training pipeline refuses to register unless it matches what it
trained on.

*It is versioned with the model, not with the service.* Rolling the champion back has to roll
the state back with it, or the old model starts reading records it never saw.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optcg_forecast.features.compute import FeatureBuilder, _Record
from optcg_forecast.features.store import FEATURE_SET_VERSION

log = logging.getLogger(__name__)

STATE_FILE = "serving_state.json"
CELL_SEPARATOR = "|"


@dataclass(frozen=True)
class Stamp:
    """What corpus this state was built from. Compared, not just recorded."""

    feature_set_version: str
    through: str
    events: int
    event_digest: str

    def matches(self, other: Stamp) -> bool:
        return (
            self.feature_set_version == other.feature_set_version
            and self.event_digest == other.event_digest
        )

    def describe(self) -> str:
        return (
            f"{self.events} events through {self.through} "
            f"(feature set {self.feature_set_version}, digest {self.event_digest[:12]})"
        )


def digest_events(event_ids: list[str]) -> str:
    """Order-independent fingerprint of exactly which events are in a corpus."""
    h = hashlib.sha256()
    for event_id in sorted(set(event_ids)):
        h.update(event_id.encode())
        h.update(b"\x00")
    return h.hexdigest()


def stamp_for(event_ids: list[str], through: str) -> Stamp:
    unique = sorted(set(event_ids))
    return Stamp(
        feature_set_version=FEATURE_SET_VERSION,
        through=through,
        events=len(unique),
        event_digest=digest_events(unique),
    )


def _records_to_json(records: dict[Any, _Record], key: Any = str) -> dict[str, list[int]]:
    """Two ints per entry rather than an object, because there are tens of thousands."""
    return {key(k): [v.games, v.wins] for k, v in records.items() if v.games > 0}


def _records_from_json(raw: dict[str, list[int]], key: Any = str) -> dict[Any, _Record]:
    return {key(k): _Record(games=int(v[0]), wins=int(v[1])) for k, v in raw.items()}


def write_state(builder: FeatureBuilder, stamp: Stamp, target: Path) -> Path:
    """Write the accumulated records, stamped with the corpus that produced them."""
    target.parent.mkdir(parents=True, exist_ok=True)
    leaders = _records_to_json(builder.leaders)
    players = _records_to_json(builder.players)
    cells = _records_to_json(builder.cells, key=lambda k: f"{k[0]}{CELL_SEPARATOR}{k[1]}")
    payload: dict[str, Any] = {
        "stamp": {
            "feature_set_version": stamp.feature_set_version,
            "through": stamp.through,
            "events": stamp.events,
            "event_digest": stamp.event_digest,
        },
        "leaders": leaders,
        "players": players,
        "cells": cells,
    }
    target.write_text(json.dumps(payload, separators=(",", ":")))
    log.info(
        "wrote serving state to %s: %d leaders, %d players, %d cells, %s",
        target,
        len(leaders),
        len(players),
        len(cells),
        stamp.describe(),
    )
    return target


def read_state(source: Path) -> tuple[FeatureBuilder, Stamp]:
    """Load records back into a builder that can emit features for a query."""
    payload = json.loads(source.read_text())
    raw = payload["stamp"]
    stamp = Stamp(
        feature_set_version=raw["feature_set_version"],
        through=raw["through"],
        events=int(raw["events"]),
        event_digest=raw["event_digest"],
    )

    builder = FeatureBuilder()
    builder.leaders.update(_records_from_json(payload["leaders"]))
    builder.players.update(_records_from_json(payload["players"]))
    builder.cells.update(
        _records_from_json(
            payload["cells"],
            key=lambda k: tuple(k.split(CELL_SEPARATOR, 1)),
        )
    )
    # The walk is finished; refuse to let anything append to loaded state, because a builder
    # resumed from disk has no idea which date it stopped at and could silently go backwards.
    builder._last_date = None
    return builder, stamp
