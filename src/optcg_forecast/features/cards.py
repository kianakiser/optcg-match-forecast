"""The card catalogue: what each of the ~2,800 printed cards actually is.

Needed because a decklist is 50 card ids, and an id on its own says nothing. To know that a
deck runs a heavy curve or a lot of 2000-counters, the ids have to be joined to their printed
values.

Source is buhbbl/punk-records, a versioned JSON dump of the official card list. It is public,
which matters: the graded repository has to stand on its own, and a reviewer cloning it must be
able to run everything without access to any private project of ours.

The catalogue is fetched per pack and cached on disk. It changes only when a set releases - every
two to three months - so re-fetching on every pipeline run would be wasteful and rude.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RAW_BASE = "https://raw.githubusercontent.com/buhbbl/punk-records/main"
LANGUAGE = "english"
USER_AGENT = (
    "optcg-match-forecast/0.1 (HSLU MLOps student project; "
    "+https://github.com/kianakiser/optcg-match-forecast)"
)


class CardDataError(RuntimeError):
    """The card catalogue could not be fetched or parsed."""


@dataclass(frozen=True)
class Card:
    """One printed card, reduced to the fields the features actually use."""

    id: str
    category: str  # Leader | Character | Event | Stage
    cost: int | None  # DON!! cost; None for "Cost —" events
    power: int | None  # None for cards with no power box
    counter: int | None  # only ever 1000, 2000 or None across the whole catalogue
    life: int | None  # leaders only
    colors: tuple[str, ...]
    has_trigger: bool

    @property
    def is_leader(self) -> bool:
        return self.category == "Leader"


def _parse(raw: dict[str, Any]) -> Card | None:
    card_id = raw.get("id")
    if not card_id:
        return None
    category = str(raw.get("category") or "")
    cost = raw.get("cost")
    # The source puts a leader's LIFE in the cost field, which is a real trap: read it as
    # cost and every leader looks like an expensive card.
    life = cost if category == "Leader" else None
    return Card(
        id=str(card_id),
        category=category,
        cost=None if category == "Leader" else (int(cost) if cost is not None else None),
        power=int(raw["power"]) if raw.get("power") is not None else None,
        counter=int(raw["counter"]) if raw.get("counter") is not None else None,
        life=int(life) if life is not None else None,
        colors=tuple(raw.get("colors") or ()),
        has_trigger=bool(raw.get("trigger")),
    )


@dataclass
class CardCatalogue:
    """Card lookup by id, backed by an on-disk cache."""

    cache_dir: Path = Path("data/cards")
    timeout_s: float = 30.0
    _by_id: dict[str, Card] = field(default_factory=dict)

    def _get_json(self, path: str) -> Any:
        url = f"{RAW_BASE}/{path}"
        try:
            with urlopen(
                Request(url, headers={"User-Agent": USER_AGENT}), timeout=self.timeout_s
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CardDataError(f"{url}: {exc}") from exc

    def load(self, *, refresh: bool = False) -> dict[str, Card]:
        """Return every card, from cache when possible."""
        if self._by_id and not refresh:
            return self._by_id

        cached = self.cache_dir / f"{LANGUAGE}_cards.json"
        if cached.is_file() and not refresh:
            raw_cards = json.loads(cached.read_text())
        else:
            packs = self._get_json(f"{LANGUAGE}/packs.json")
            # packs.json is a dict keyed by pack id; accept a list too, in case that changes.
            if isinstance(packs, dict):
                pack_ids = list(packs.keys())
            elif isinstance(packs, list):
                pack_ids = [p.get("id") if isinstance(p, dict) else p for p in packs]
            else:
                raise CardDataError("packs.json was neither a dict nor a list")

            raw_cards = []
            for pack_id in pack_ids:
                if pack_id is None:
                    continue
                entries = self._get_json(f"{LANGUAGE}/data/{pack_id}.json")
                if isinstance(entries, list):
                    raw_cards.extend(entries)
            if not raw_cards:
                raise CardDataError("no cards found in any pack")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(raw_cards))

        self._by_id = {}
        for raw in raw_cards:
            card = _parse(raw) if isinstance(raw, dict) else None
            if card:
                self._by_id[card.id] = card
        return self._by_id

    def get(self, card_id: str) -> Card | None:
        return self._by_id.get(card_id) or self.load().get(card_id)


def card_id_from_entry(entry: dict[str, Any]) -> str | None:
    """Build a card id from a decklist entry.

    Decklists carry {set, number}; the catalogue keys on "OP05-098". The number is
    zero-padded to three digits, which is the join that silently fails if you forget it.
    """
    set_code, number = entry.get("set"), entry.get("number")
    if not set_code or number is None:
        return None
    return f"{str(set_code).upper()}-{str(number).zfill(3)}"
