"""Summarise a 50-card decklist into a handful of numbers.

Why this exists: two players can bring the same leader and very different decks. Using leader
identity alone would give them identical features, which is wrong in principle even where it
turns out to be nearly right in practice.

How wrong it is in practice is worth stating, because it sets expectations. Decks sharing a
leader share about 42 of their 50 slots, and most of the remaining variation tracks *when* the
deck was played rather than who built it - a new set arrives, everyone adopts the new cards, and
the archetype moves as a block. So these features are expected to add a little, not a lot. They
are cheap, so the classifier can decide; the alternative is asserting it without checking.

Everything here is an additive aggregate, which means it cannot represent a combo - two cards
worth more together than apart. That is a real limitation and it was tested directly: no card
pair survived multiple-comparison correction, because any combo strong enough to matter is in
nearly every list within a few weeks and so stops distinguishing decks at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optcg_forecast.features.cards import Card, card_id_from_entry

# Cards costing this or more are the top of the curve.
HIGH_CURVE_COST = 6
# Characters at or above this power are the bodies that trade up.
BIG_BODY_POWER = 6000


@dataclass(frozen=True)
class DeckSummary:
    """A decklist reduced to numbers. All zero when the list cannot be read."""

    total_cards: int = 0
    distinct_cards: int = 0
    counter_2k_copies: int = 0
    counter_total: int = 0
    avg_cost: float = 0.0
    high_curve_copies: int = 0
    big_body_copies: int = 0
    event_copies: int = 0
    trigger_copies: int = 0
    resolved_fraction: float = 0.0

    @property
    def is_usable(self) -> bool:
        """A list we could mostly resolve. Below this the numbers mislead more than they help."""
        return self.total_cards >= 40 and self.resolved_fraction >= 0.9


def _entries(decklist: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the non-leader entries out of a decklist, whatever shape it arrives in."""
    out: list[dict[str, Any]] = []
    for key in ("character", "event", "stage"):
        section = decklist.get(key)
        if isinstance(section, list):
            out.extend(e for e in section if isinstance(e, dict))
    return out


def summarise(decklist: dict[str, Any] | None, catalogue: dict[str, Card]) -> DeckSummary:
    """Turn one decklist into a DeckSummary, joining each entry to the card catalogue."""
    if not decklist:
        return DeckSummary()

    total = resolved = distinct = 0
    counter_2k = counter_total = high_curve = big_body = events = triggers = 0
    cost_sum = cost_count = 0

    for entry in _entries(decklist):
        count = int(entry.get("count") or 0)
        if count <= 0:
            continue
        total += count
        distinct += 1

        card_id = card_id_from_entry(entry)
        card = catalogue.get(card_id) if card_id else None
        if card is None:
            continue
        resolved += count

        if card.counter == 2000:
            counter_2k += count
        if card.counter:
            counter_total += card.counter * count
        if card.cost is not None:
            # "Cost —" events are excluded from the average rather than counted as zero,
            # which would drag every deck running them downward for no reason.
            cost_sum += card.cost * count
            cost_count += count
            if card.cost >= HIGH_CURVE_COST:
                high_curve += count
        if card.category == "Character" and (card.power or 0) >= BIG_BODY_POWER:
            big_body += count
        if card.category == "Event":
            events += count
        if card.has_trigger:
            triggers += count

    return DeckSummary(
        total_cards=total,
        distinct_cards=distinct,
        counter_2k_copies=counter_2k,
        counter_total=counter_total,
        avg_cost=(cost_sum / cost_count) if cost_count else 0.0,
        high_curve_copies=high_curve,
        big_body_copies=big_body,
        event_copies=events,
        trigger_copies=triggers,
        resolved_fraction=(resolved / total) if total else 0.0,
    )
