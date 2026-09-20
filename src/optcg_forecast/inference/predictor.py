"""Answering one question, from the champion and the records it was registered with.

The central decision here is that serving does NOT build its own features. It constructs a
synthetic Match and hands it to `FeatureBuilder.features_for` - the same method, on the same
class, that produced every training row. Training-serving skew is the classic way a project like
this fails silently, and the cheapest defence is not having two implementations to keep in step.

The second decision is that the model is loaded once, at startup, and never reloaded. A request
that quietly began answering with a different model halfway through a session would make every
logged prediction unattributable. Moving the champion alias is a deploy, not a hot-swap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from optcg_forecast.features.compute import FeatureBuilder
from optcg_forecast.features.ingest import Match
from optcg_forecast.features.serving_state import STATE_FILE, Stamp, read_state
from optcg_forecast.training.registry import CHAMPION, ModelCard, ModelRegistry
from optcg_forecast.training.run import predict

log = logging.getLogger(__name__)

# A synthetic match carries no outcome. These stand in for fields features_for reads but a
# query cannot supply; none of them reaches a feature.
QUERY_EVENT = "__query__"
QUERY_ROUND = 1


class NotReady(RuntimeError):
    """No champion, or no records to serve it with. The service says so rather than guessing."""


@dataclass(frozen=True)
class Prediction:
    """Both answers, and how much evidence is behind them."""

    deck_probability: float
    match_probability: float | None
    coverage: str
    leader_a_games: int
    leader_b_games: int
    pairing_games: int
    player_a_games: int | None
    player_b_games: int | None
    used_handles: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "deck_probability": round(self.deck_probability, 4),
            "match_probability": (
                round(self.match_probability, 4) if self.match_probability is not None else None
            ),
            "coverage": self.coverage,
            "evidence": {
                "leader_a_games": self.leader_a_games,
                "leader_b_games": self.leader_b_games,
                "pairing_games": self.pairing_games,
                "player_a_games": self.player_a_games,
                "player_b_games": self.player_b_games,
            },
            "used_handles": self.used_handles,
        }


@dataclass
class Predictor:
    """The champion, its records, and the catalogue needed to name a leader."""

    model: Any
    state: FeatureBuilder
    stamp: Stamp
    card: ModelCard
    leaders: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, model_root: Path, ref: str = CHAMPION) -> Predictor:
        registry = ModelRegistry(root=model_root)
        version = registry.resolve(ref)
        if version is None:
            raise NotReady(
                f"nothing is registered under {ref!r} in {model_root}. Run the training "
                f"pipeline, or restore the model registry."
            )
        model, card = registry.load(ref)

        state_path = registry.root / "versions" / version / STATE_FILE
        if not state_path.is_file():
            raise NotReady(
                f"{version} has no {STATE_FILE}, so it cannot answer anything: the leader, "
                f"player and matchup records are where every feature comes from."
            )
        loaded = read_state(state_path)
        if not loaded.leader_names:
            log.warning(
                "%s ships no leader names, so the UI will offer card ids. Re-run the feature "
                "pipeline to bake them in.",
                version,
            )

        log.info("serving %s (%s), records: %s", version, card.git_sha, loaded.stamp.describe())
        return cls(
            model=model,
            state=loaded.builder,
            stamp=loaded.stamp,
            card=card,
            leaders=loaded.leader_names,
        )

    # ---------------------------------------------------------------- catalogue

    def known_leaders(self) -> list[dict[str, Any]]:
        """Every leader with history, commonest first, so the UI can offer real choices."""
        ranked = sorted(
            ((lid, rec.games) for lid, rec in self.state.leaders.items() if rec.games > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [
            {"id": lid, "name": self.leaders.get(lid, lid), "games": games} for lid, games in ranked
        ]

    def knows_player(self, handle: str) -> bool:
        return self.state.players.get(handle) is not None and self.state.players[handle].games > 0

    # ------------------------------------------------------------------ predict

    def predict(
        self,
        leader_a: str,
        leader_b: str,
        handle_a: str | None = None,
        handle_b: str | None = None,
    ) -> Prediction:
        """Answer the deck question always, and the match question when handles are given."""
        if leader_a not in self.state.leaders and leader_b not in self.state.leaders:
            raise NotReady(
                f"neither {leader_a} nor {leader_b} appears anywhere in the records, so there "
                f"is nothing to base an answer on."
            )

        a = (handle_a or "").strip()
        b = (handle_b or "").strip()
        # Both or neither. One handle would put a rated player against an unrated one and read
        # the difference as a skill gap, which is a measurement of our ignorance, not of them.
        use_handles = bool(a and b)

        row = self.state.features_for(
            Match(
                event_id=QUERY_EVENT,
                event_date=date.fromisoformat(self.stamp.through),
                round=QUERY_ROUND,
                table=0,
                player1=a or "__a__",
                player2=b or "__b__",
                winner=a or "__a__",  # arbitrary: the label is not read for a prediction
            ),
            leader_a,
            leader_b,
        )
        features = row.as_dict()

        deck = predict(self.model, [features], pilot_neutral=True)[0]
        match = predict(self.model, [features])[0] if use_handles else None

        pa = self.state.players.get(a) if use_handles else None
        pb = self.state.players.get(b) if use_handles else None
        return Prediction(
            deck_probability=deck,
            match_probability=match,
            coverage=row.coverage,
            leader_a_games=row.p1_leader_games,
            leader_b_games=row.p2_leader_games,
            pairing_games=row.cell_games,
            player_a_games=pa.games if pa else (0 if use_handles else None),
            player_b_games=pb.games if pb else (0 if use_handles else None),
            used_handles=use_handles,
        )
