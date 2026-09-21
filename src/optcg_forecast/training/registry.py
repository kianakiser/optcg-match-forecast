"""A model registry: versioned models, and an alias saying which one is live.

The alias is the point. Serving loads whatever currently answers to `champion`, so promoting a
model means moving a pointer - no redeploy, no code change, and an instant rollback by moving it
back. Lab 10 teaches exactly this with MLflow; the mechanism matters more than the vendor.

Local files for now, matching the feature store's approach: Hopsworks is a migration we can make
when there is time, not a dependency we take on in November. What has to survive that migration is
the discipline, which is:

- a model version is immutable once written
- every version records the metrics it was promoted on, the feature-set version it was trained
  against, and the git commit that produced it
- promotion is a separate, logged decision, never a side effect of training
"""

from __future__ import annotations

import json
import logging
import pickle
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CHAMPION = "champion"


def _git_sha() -> str:
    """Which commit produced this model. Unknown is fine; wrong is not.

    The dirty check is the point. Recording a bare HEAD from a modified working tree names a
    commit that did not produce the model, which is worse than recording nothing because it
    looks reproducible. This has already happened once here: a card claimed a commit whose
    successor, landed sixty seconds later, contained the code that actually made its numbers.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        sha = out.stdout.strip()[:12]
    except Exception:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:
        return sha
    if dirty:
        log.warning(
            "working tree is dirty: this model cannot be reproduced from %s alone (%d changed "
            "path(s))",
            sha,
            len(dirty.splitlines()),
        )
        return f"{sha}-dirty"
    return sha


@dataclass
class ModelCard:
    """What a model is, and what it was promoted on. Written beside every version."""

    version: str
    created_at: str
    git_sha: str
    feature_set_version: str
    feature_names: list[str]
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    training_rows: int
    training_events: int
    trained_through: str
    notes: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class ModelRegistry:
    """Versioned model storage with movable aliases."""

    root: Path = Path("data/models")

    def __post_init__(self) -> None:
        # A str is the obvious thing to pass and fails later, inside a path join, with a
        # TypeError that names neither the argument nor the caller. Coerce and move on.
        self.root = Path(self.root)

    @property
    def _aliases_file(self) -> Path:
        return self.root / "aliases.json"

    def _version_dir(self, version: str) -> Path:
        return self.root / "versions" / version

    def next_version(self) -> str:
        existing = sorted(
            int(p.name[1:]) for p in (self.root / "versions").glob("v*") if p.name[1:].isdigit()
        )
        return f"v{(existing[-1] + 1) if existing else 1}"

    # ------------------------------------------------------------------ write

    def register(self, model: Any, card: ModelCard, gate_model: Any | None = None) -> str:
        """Store a model version. Immutable once written.

        Two models, not one, and the second is the interesting one.

        `model` is refitted on the whole corpus and is what gets served - a model answering
        tomorrow's questions should know about last week. That makes it useless for judging
        against future data, because it has already seen the newest 28 days.

        `gate_model` is the same settings fitted only on data up to the selection cutoff. It is
        never served. It exists so that NEXT week's run can score this week's champion on a block
        neither of them has seen, which is the only way a champion-versus-challenger comparison
        means anything. Without it the comparison is between two numbers measured on different
        fortnights of different metagames.
        """
        target = self._version_dir(card.version)
        if target.exists():
            raise FileExistsError(f"{card.version} already exists; versions are immutable")
        target.mkdir(parents=True)
        (target / "model.pkl").write_bytes(pickle.dumps(model))
        if gate_model is not None:
            (target / "gate_model.pkl").write_bytes(pickle.dumps(gate_model))
        (target / "card.json").write_text(json.dumps(asdict(card), indent=1, default=str))
        log.info("registered %s", card.version)
        return card.version

    def load_gate_model(self, ref: str = CHAMPION) -> Any | None:
        """The un-refitted model, for comparisons against data it has not seen. None if absent."""
        version = self.resolve(ref)
        if version is None:
            return None
        path = self._version_dir(version) / "gate_model.pkl"
        return pickle.loads(path.read_bytes()) if path.is_file() else None

    def set_alias(self, alias: str, version: str) -> None:
        """Point an alias at a version. This is what promotion actually is."""
        if not self._version_dir(version).is_dir():
            raise ValueError(f"cannot alias {alias} to unknown version {version}")
        aliases = self.aliases()
        previous = aliases.get(alias)
        aliases[alias] = version
        self.root.mkdir(parents=True, exist_ok=True)
        self._aliases_file.write_text(json.dumps(aliases, indent=1))
        log.info("alias %s: %s -> %s", alias, previous or "(unset)", version)

    # ------------------------------------------------------------------ read

    def aliases(self) -> dict[str, str]:
        return json.loads(self._aliases_file.read_text()) if self._aliases_file.is_file() else {}

    def resolve(self, ref: str) -> str | None:
        """Turn 'champion' or 'v3' into a concrete version."""
        if ref.startswith("v") and ref[1:].isdigit():
            return ref if self._version_dir(ref).is_dir() else None
        return self.aliases().get(ref)

    def load(self, ref: str = CHAMPION) -> tuple[Any, ModelCard]:
        """Load a model by alias or version. Serving calls this with 'champion'."""
        version = self.resolve(ref)
        if version is None:
            raise KeyError(f"nothing registered under {ref!r}")
        target = self._version_dir(version)
        model = pickle.loads((target / "model.pkl").read_bytes())
        card = ModelCard(**json.loads((target / "card.json").read_text()))
        return model, card

    def card(self, ref: str = CHAMPION) -> ModelCard | None:
        try:
            return self.load(ref)[1]
        except (KeyError, FileNotFoundError):
            return None

    def versions(self) -> list[str]:
        root = self.root / "versions"
        if not root.is_dir():
            return []
        return sorted(
            (p.name for p in root.glob("v*") if p.name[1:].isdigit()),
            key=lambda v: int(v[1:]),
        )


def build_card(
    *,
    version: str,
    feature_set_version: str,
    feature_names: list[str],
    hyperparameters: dict[str, Any],
    metrics: dict[str, float],
    training_rows: int,
    training_events: int,
    trained_through: str,
    notes: str = "",
) -> ModelCard:
    return ModelCard(
        version=version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        feature_set_version=feature_set_version,
        feature_names=list(feature_names),
        hyperparameters=dict(hyperparameters),
        metrics=dict(metrics),
        training_rows=training_rows,
        training_events=training_events,
        trained_through=trained_through,
        notes=notes,
    )
