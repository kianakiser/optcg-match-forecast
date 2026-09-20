"""Tests for the inference pipeline.

The two that matter are `test_serving_matches_what_training_would_have_computed`, which is the
training-serving skew guard at the HTTP boundary rather than at the function boundary, and
`test_the_answer_does_not_depend_on_which_deck_you_pick_first`, which is the property a user
would notice being broken.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the serve extra is optional")
from fastapi.testclient import TestClient  # noqa: E402

from optcg_forecast.features.compute import FeatureBuilder  # noqa: E402
from optcg_forecast.features.ingest import Match  # noqa: E402
from optcg_forecast.features.serving_state import STATE_FILE, stamp_for, write_state  # noqa: E402
from optcg_forecast.inference.predictor import NotReady, Predictor  # noqa: E402
from optcg_forecast.inference.serve import build_app  # noqa: E402
from optcg_forecast.training.registry import CHAMPION, ModelRegistry, build_card  # noqa: E402
from optcg_forecast.training.run import FEATURES, fit  # noqa: E402

D = date(2026, 9, 17)


def a_registry(tmp_path: Path, *, with_state: bool = True) -> Path:
    """A champion trained on synthetic history, with records to serve it from."""
    import random

    rng = random.Random(0)
    builder = FeatureBuilder()
    rows = []
    for day in range(30):
        matches, leader_of = [], {}
        for i in range(12):
            p1, p2 = f"player{i}", f"player{(i + 5) % 12}"
            leader_of[p1] = f"L{i % 4}"
            leader_of[p2] = f"L{(i + 5) % 4}"
            matches.append(
                Match(
                    event_id=f"e{day}",
                    event_date=date(2026, 1, 1),
                    round=1 + i % 4,
                    table=1,
                    player1=p1,
                    player2=p2,
                    winner=p1 if rng.random() < 0.5 + 0.2 * (i % 4 - 1.5) else p2,
                )
            )
        rows.extend(
            builder.process_day(
                date(2026, 1, 1) + __import__("datetime").timedelta(days=day),
                [(matches, leader_of, None)],
            )
        )

    root = tmp_path / "models"
    registry = ModelRegistry(root=root)
    training = [r.as_dict() for r in rows]
    model = fit(training, {"max_depth": 3, "max_iter": 40})
    registry.register(
        model,
        build_card(
            version="v1",
            feature_set_version="v2",
            feature_names=FEATURES,
            hyperparameters={"max_depth": 3},
            metrics={"gate_brier": 0.24},
            training_rows=len(training),
            training_events=30,
            trained_through=str(D),
        ),
        gate_model=model,
    )
    registry.set_alias(CHAMPION, "v1")
    if with_state:
        write_state(
            builder,
            stamp_for([f"e{d}" for d in range(30)], str(D)),
            root / "versions" / "v1" / STATE_FILE,
        )
    return root


@pytest.fixture
def client(tmp_path):
    with TestClient(build_app(a_registry(tmp_path))) as c:
        yield c


# ------------------------------------------------------------------ the basics


def test_health_reports_the_champion(client):
    body = client.get("/health").json()
    assert body["ready"] is True
    assert body["champion"] == "v1"


def test_version_says_exactly_what_is_live(client):
    """'Which model is answering right now' must not require reading a deploy log."""
    v = client.get("/version").json()
    assert v["version"] == "v1"
    assert v["feature_set_version"] == "v2"
    assert v["trained_through"] == str(D)
    assert v["git_sha"]
    assert "252 events" not in v["records"]  # it describes THIS corpus, not a hardcoded one


def test_leaders_are_offered_commonest_first(client):
    leaders = client.get("/leaders").json()["leaders"]
    assert leaders
    assert [x["games"] for x in leaders] == sorted((x["games"] for x in leaders), reverse=True)
    assert all(x["games"] > 0 for x in leaders), "a leader with no history is not a choice"


def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "match forecast" in r.text.lower()


# ------------------------------------------------------------ the two questions


def test_deck_question_needs_no_handles(client):
    d = client.post("/predict", json={"leader_a": "L0", "leader_b": "L1"}).json()
    assert 0.0 < d["deck_probability"] < 1.0
    assert d["match_probability"] is None
    assert d["used_handles"] is False


def test_handles_add_the_match_question(client):
    d = client.post(
        "/predict",
        json={"leader_a": "L0", "leader_b": "L1", "handle_a": "player1", "handle_b": "player2"},
    ).json()
    assert d["used_handles"] is True
    assert d["match_probability"] is not None
    assert d["evidence"]["player_a_games"] > 0


def test_one_handle_is_treated_as_none(client):
    """One rated player against an unrated one would read our ignorance as a skill gap."""
    d = client.post(
        "/predict", json={"leader_a": "L0", "leader_b": "L1", "handle_a": "player1"}
    ).json()
    assert d["used_handles"] is False
    assert d["match_probability"] is None


def test_an_unknown_handle_still_answers(client):
    """Refusing is worse than saying there is no evidence for this player."""
    d = client.post(
        "/predict",
        json={"leader_a": "L0", "leader_b": "L1", "handle_a": "nobody", "handle_b": "player1"},
    ).json()
    assert d["used_handles"] is True
    assert d["evidence"]["player_a_games"] == 0


def test_the_answer_does_not_depend_on_which_deck_you_pick_first(client):
    """The property a user would notice: A-vs-B and B-vs-A must sum to 1."""
    fwd = client.post("/predict", json={"leader_a": "L0", "leader_b": "L2"}).json()
    rev = client.post("/predict", json={"leader_a": "L2", "leader_b": "L0"}).json()
    assert fwd["deck_probability"] + rev["deck_probability"] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------- skew, at the HTTP edge


def test_serving_matches_what_training_would_have_computed(tmp_path):
    """The features behind an HTTP answer must be the ones the pipeline itself would emit."""
    root = a_registry(tmp_path)
    p = Predictor.load(root)

    served = p.state.features_for(
        Match(
            event_id="__query__",
            event_date=D,
            round=1,
            table=0,
            player1="player1",
            player2="player2",
            winner="player1",
        ),
        "L0",
        "L1",
    )
    assert served.p1_leader_games == p.state.leaders["L0"].games
    assert served.p2_leader_games == p.state.leaders["L1"].games
    assert served.player_strength_diff == pytest.approx(
        p.state.players["player1"].shrunk_rate - p.state.players["player2"].shrunk_rate
    )


# --------------------------------------------------------------- failure modes


def test_a_service_with_no_champion_starts_and_says_so(tmp_path):
    """Crash-looping tells Cloud Run to retry for ever and tells a human nothing."""
    with TestClient(build_app(tmp_path / "empty")) as c:
        health = c.get("/health")
        assert health.status_code == 503
        assert health.json()["ready"] is False
        assert c.post("/predict", json={"leader_a": "L0", "leader_b": "L1"}).status_code == 503


def test_a_champion_without_records_will_not_load(tmp_path):
    """The records ARE the features; serving without them would answer from nothing."""
    root = a_registry(tmp_path, with_state=False)
    with pytest.raises(NotReady, match="serving_state"):
        Predictor.load(root)


def test_two_unknown_leaders_are_refused_not_guessed(client):
    r = client.post("/predict", json={"leader_a": "ZZ99-999", "leader_b": "YY88-888"})
    assert r.status_code == 422
    assert "nothing to base an answer on" in r.json()["detail"]


def test_a_malformed_query_is_rejected(client):
    assert client.post("/predict", json={"leader_a": ""}).status_code == 422


# ------------------------------------------------------------- observability


def test_every_prediction_is_logged_with_its_model_version(client, caplog):
    """Without the version on the line, a logged prediction cannot be attributed later."""
    with caplog.at_level("INFO", logger="optcg_forecast.predictions"):
        client.post("/predict", json={"leader_a": "L0", "leader_b": "L1"})
    lines = [
        json.loads(r.message) for r in caplog.records if r.name == "optcg_forecast.predictions"
    ]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["event"] == "prediction"
    assert entry["model_version"] == "v1"
    assert entry["feature_set_version"] == "v2"
    assert entry["leader_a"] == "L0"
    assert "deck_probability" in entry


def test_the_response_names_the_model_that_produced_it(client):
    assert (
        client.post("/predict", json={"leader_a": "L0", "leader_b": "L1"}).json()["model_version"]
        == "v1"
    )
