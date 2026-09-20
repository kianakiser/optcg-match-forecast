"""Turn ingested events into model-ready feature rows, point-in-time correctly.

The whole design of this module exists to make one bug impossible.

A feature for a match played on 3 March must be computed from what was known on 2 March. Get that
wrong and the model learns from its own future: scores look excellent in development and collapse
in production. It is the single most common way a project like this fails, and it fails silently.

Rather than compute features and then try to prove no future leaked in, this does a single forward
pass in date order. State is read to emit a row, and only updated *afterwards*, so a match can
never see its own event - let alone a later one. Point-in-time correctness is then a property of
the loop's shape rather than a rule someone has to remember.

One consequence worth stating: every match inside an event sees the same snapshot, taken before the
event began. That is deliberate. Updating within an event would let round 1 results inform round 4
features, which is exactly the in-event leakage the `table` field was rejected for.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
from typing import Any

from optcg_forecast.features.cards import Card
from optcg_forecast.features.deck import DeckSummary, summarise
from optcg_forecast.features.ingest import Entrant, Match

# An archetype needs some history before its strength estimate means anything. Below this we
# say so through the coverage flag rather than pretending. 40 is where the measured accuracy
# stops improving steeply; it is a knob, not a law.
MIN_GAMES_FOR_STRENGTH = 40

# Shrinkage: an archetype at 3-0 is not a 100% deck. Pull rates toward 0.5 by pretending we
# also saw PRIOR_GAMES coin flips. Larger = more sceptical of thin evidence.
PRIOR_GAMES = 20.0

# A pairing needs this many games before its head-to-head rate is worth more than the two
# individual strengths.
MIN_GAMES_FOR_CELL = 12


@dataclass
class _Record:
    """Wins and games for one archetype or player, as known so far."""

    games: int = 0
    wins: int = 0

    @property
    def shrunk_rate(self) -> float:
        """Win rate pulled toward 0.5, so thin evidence cannot shout."""
        return (self.wins + 0.5 * PRIOR_GAMES) / (self.games + PRIOR_GAMES)


@dataclass
class FeatureRow:
    """One match, ready for the model. Every field known before the event started."""

    event_id: str
    event_date: date
    round: int
    p1_leader: str
    p2_leader: str

    # Differences, not levels: the model must not be able to learn "seat 1 is better",
    # because it is not - seat 1 wins 50.56%, which is a coin flip.
    leader_strength_diff: float
    player_strength_diff: float
    cell_rate: float
    experience_diff: float

    # Deck contents. Two players can bring the same leader and different decks - measured,
    # about 23% of the variation in counter count is between decks sharing a leader - so
    # leader identity alone would call them identical. These are differences like the rest.
    counter_2k_diff: float
    avg_cost_diff: float
    high_curve_diff: float
    big_body_diff: float
    event_copies_diff: float
    trigger_copies_diff: float
    deck_features_usable: bool

    # Coverage, carried as features AND surfaced to the caller. The model can learn to
    # distrust thin evidence; the service can refuse to answer on no evidence.
    p1_leader_games: int
    p2_leader_games: int
    cell_games: int
    coverage: str

    label_p1_won: int

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _coverage(p1_games: int, p2_games: int, cell_games: int) -> str:
    """How much evidence stands behind this row.

    Measured on the backfill: `solid` scores ~0.2445 Brier, `thin` ~0.2477, `prior_only`
    ~0.2486, and `cold` is genuinely uninformative - a leader-attribute backoff was measured
    at 0.2513, worse than a coin flip, which is why cold returns 0.5 rather than guessing.
    """
    if cell_games >= MIN_GAMES_FOR_CELL:
        return "solid"
    if min(p1_games, p2_games) >= MIN_GAMES_FOR_STRENGTH:
        return "prior_only"
    if cell_games > 0 or max(p1_games, p2_games) >= MIN_GAMES_FOR_STRENGTH:
        return "thin"
    return "cold"


@dataclass
class FeatureBuilder:
    """Walks events in date order, emitting rows then learning from them."""

    leaders: dict[str, _Record] = field(default_factory=lambda: defaultdict(_Record))
    players: dict[str, _Record] = field(default_factory=lambda: defaultdict(_Record))
    cells: dict[tuple[str, str], _Record] = field(default_factory=lambda: defaultdict(_Record))
    _last_date: date | None = None

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        """Order-independent pairing key, so A-vs-B and B-vs-A share evidence."""
        return (a, b) if a <= b else (b, a)

    def _cell_rate_for(self, p1: str, p2: str) -> tuple[float, int]:
        """Head-to-head rate from p1's perspective, shrunk toward 0.5."""
        key = self._key(p1, p2)
        rec = self.cells.get(key)
        if rec is None or rec.games == 0:
            return 0.5, 0
        # cells store wins for the lexicographically first archetype
        wins = rec.wins if key[0] == p1 else rec.games - rec.wins
        return (wins + 0.5 * PRIOR_GAMES) / (rec.games + PRIOR_GAMES), rec.games

    def features_for(
        self,
        match: Match,
        p1_leader: str,
        p2_leader: str,
        p1_deck: DeckSummary | None = None,
        p2_deck: DeckSummary | None = None,
    ) -> FeatureRow:
        """Emit one row from state as it stands BEFORE this match's event."""
        d1 = p1_deck or DeckSummary()
        d2 = p2_deck or DeckSummary()
        deck_ok = d1.is_usable and d2.is_usable
        l1, l2 = self.leaders.get(p1_leader, _Record()), self.leaders.get(p2_leader, _Record())
        u1, u2 = (
            self.players.get(match.player1, _Record()),
            self.players.get(match.player2, _Record()),
        )
        cell_rate, cell_games = self._cell_rate_for(p1_leader, p2_leader)

        return FeatureRow(
            event_id=match.event_id,
            event_date=match.event_date,
            round=match.round,
            p1_leader=p1_leader,
            p2_leader=p2_leader,
            leader_strength_diff=l1.shrunk_rate - l2.shrunk_rate,
            player_strength_diff=u1.shrunk_rate - u2.shrunk_rate,
            cell_rate=cell_rate,
            experience_diff=float(u1.games - u2.games),
            p1_leader_games=l1.games,
            p2_leader_games=l2.games,
            cell_games=cell_games,
            coverage=_coverage(l1.games, l2.games, cell_games),
            counter_2k_diff=float(d1.counter_2k_copies - d2.counter_2k_copies) if deck_ok else 0.0,
            avg_cost_diff=(d1.avg_cost - d2.avg_cost) if deck_ok else 0.0,
            high_curve_diff=float(d1.high_curve_copies - d2.high_curve_copies) if deck_ok else 0.0,
            big_body_diff=float(d1.big_body_copies - d2.big_body_copies) if deck_ok else 0.0,
            event_copies_diff=float(d1.event_copies - d2.event_copies) if deck_ok else 0.0,
            trigger_copies_diff=float(d1.trigger_copies - d2.trigger_copies) if deck_ok else 0.0,
            deck_features_usable=deck_ok,
            label_p1_won=int(match.player1_won),
        )

    def learn(self, match: Match, p1_leader: str, p2_leader: str) -> None:
        """Fold one finished match into state. Only ever called after features are emitted."""
        won = match.player1_won
        self.leaders[p1_leader].games += 1
        self.leaders[p1_leader].wins += won
        self.leaders[p2_leader].games += 1
        self.leaders[p2_leader].wins += 1 - won
        self.players[match.player1].games += 1
        self.players[match.player1].wins += won
        self.players[match.player2].games += 1
        self.players[match.player2].wins += 1 - won

        key = self._key(p1_leader, p2_leader)
        rec = self.cells[key]
        rec.games += 1
        # store wins for the lexicographically first archetype, so the key stays symmetric
        rec.wins += won if key[0] == p1_leader else 1 - won

    def process_day(
        self,
        event_date: date,
        events: Sequence[tuple[Iterable[Match], dict[str, str], dict[str, DeckSummary] | None]],
    ) -> list[FeatureRow]:
        """Emit rows for every event on one calendar date, then learn from all of them.

        The unit is the DAY, not the event, and that is the whole point of this method.

        The source gives each event a date and no time. On 21 dates in the current corpus two or
        more events share a date, covering 23,915 matches. Processing them one event at a time -
        emit, learn, emit, learn - lets the second event of a day train on the first event's
        results, and the order it happens to walk them in comes from the event id, which is not
        chronological and carries no information about which tournament actually finished first.

        That is leakage in the strict sense: 12,486 rows could see results the model would not
        have had. It is also the quiet kind, because it inflates the score rather than breaking
        anything, and an inflated score looks like success.

        So the state a row sees is the state as of the START of its date, for every event on that
        date, and the day's results are folded in only once all of them have been emitted. Within
        a day the ordering question disappears, because no ordering is used.
        """
        if self._last_date is not None and event_date <= self._last_date:
            raise ValueError(
                f"days must arrive in strictly increasing date order for point-in-time "
                f"correctness, and each date exactly once: {event_date} came after "
                f"{self._last_date}"
            )
        self._last_date = event_date

        prepared = [
            (
                [m for m in matches if leader_of.get(m.player1) and leader_of.get(m.player2)],
                leader_of,
                deck_of or {},
            )
            for matches, leader_of, deck_of in events
        ]

        # Emit everything first, from one shared snapshot of state.
        rows = [
            self.features_for(
                m,
                leader_of[m.player1],
                leader_of[m.player2],
                decks.get(m.player1),
                decks.get(m.player2),
            )
            for usable, leader_of, decks in prepared
            for m in usable
        ]
        # Only then learn, from the whole day at once.
        for usable, leader_of, _ in prepared:
            for m in usable:
                self.learn(m, leader_of[m.player1], leader_of[m.player2])
        return rows

    def process_event(
        self,
        event_date: date,
        matches: Iterable[Match],
        leader_of: dict[str, str],
        deck_of: dict[str, DeckSummary] | None = None,
    ) -> list[FeatureRow]:
        """One event that is the only event of its date.

        A thin wrapper over process_day. Calling it twice for the same date raises, because that
        is precisely the leak process_day exists to prevent - group the day's events and pass
        them together instead.
        """
        return self.process_day(event_date, [(matches, leader_of, deck_of)])


def build(
    events: Iterable[tuple[date, list[Entrant], list[Match]]],
    catalogue: dict[str, Card] | None = None,
    builder: FeatureBuilder | None = None,
) -> Iterator[FeatureRow]:
    """Walk events in date order, a day at a time, and yield feature rows.

    `events` must be sorted by date; process_day raises if it is not, because silently accepting
    out-of-order events would reintroduce exactly the leakage this module prevents. Events
    sharing a date are grouped and processed together - see process_day for why that matters.

    Pass `builder` to keep the accumulated state after the walk. Serving needs it: the leader,
    player and matchup records are the only place the model's inputs can come from at predict
    time, and they are otherwise discarded here.
    """
    state = builder if builder is not None else FeatureBuilder()
    cat = catalogue or {}
    for event_date, group in groupby(events, key=lambda e: e[0]):
        day = []
        for _, entrants, matches in group:
            leader_of = {e.player: e.leader_id for e in entrants if e.leader_id}
            deck_of = (
                {e.player: summarise(e.decklist, cat) for e in entrants if e.decklist}
                if cat
                else {}
            )
            day.append((matches, leader_of, deck_of))
        yield from state.process_day(event_date, day)
