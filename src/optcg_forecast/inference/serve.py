"""The inference pipeline: one champion, two questions, over HTTP.

Third of the three FTI pipelines. It reads the model registry and nothing else - never the
tournament API, never the landing zone - which is what decoupling buys: this process can serve
while the feature pipeline is mid-ingest and the training pipeline is mid-sweep.

Three things here are deliberate.

*The model loads once, at startup.* If it reloaded on a timer, two requests a minute apart could
be answered by different models and every logged prediction would be unattributable. Moving the
champion alias is a deploy.

*Every prediction is logged as one JSON line on stdout.* On Cloud Run that goes straight to Cloud
Logging, which survives the container and costs nothing to operate. It carries the model version
and the inputs, so when those matches resolve the nightly scorer can join predictions to outcomes
and say how the deployed model actually did - which is the difference between monitoring and
hoping.

*The version is an endpoint, not a comment.* "Which model is live right now" has to be answerable
without reading a deploy log.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from optcg_forecast.inference.predictor import NotReady, Predictor

log = logging.getLogger("optcg_forecast.inference")
predictions = logging.getLogger("optcg_forecast.predictions")

STATIC = Path(__file__).parent / "static"
DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "data/models"))


class Query(BaseModel):
    leader_a: str = Field(min_length=1, max_length=32)
    leader_b: str = Field(min_length=1, max_length=32)
    handle_a: str | None = Field(default=None, max_length=64)
    handle_b: str | None = Field(default=None, max_length=64)


def build_app(model_root: Path = DEFAULT_MODEL_ROOT) -> FastAPI:
    # Held in a dict rather than a module global so a test can build one app per model root.
    state: dict[str, Predictor | None] = {"predictor": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            state["predictor"] = Predictor.load(model_root)
        except NotReady as exc:
            # Start anyway. A container that refuses to boot tells Cloud Run to retry for ever
            # and tells a human nothing; /health answering "no champion" tells them exactly what
            # is wrong. Failing to serve is a state worth reporting, not worth hiding in a crash
            # loop.
            log.error("starting without a champion: %s", exc)
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="OPTCG match forecast",
        description="Deck-vs-deck and match probabilities for One Piece TCG swiss rounds.",
        version="1",
    )

    def predictor() -> Predictor:
        loaded = state["predictor"]
        if loaded is None:
            raise HTTPException(
                status_code=503,
                detail="no champion is loaded; the service is running but cannot answer",
            )
        return loaded

    @app.get("/health")
    def health() -> JSONResponse:
        loaded = state["predictor"]
        ready = loaded is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "champion": loaded.card.version if loaded else None,
                "detail": None if ready else "no champion loaded",
            },
        )

    @app.get("/version")
    def version() -> dict[str, Any]:
        """What is live, on what data, judged how. Observability, not decoration."""
        p = predictor()
        return {
            "version": p.card.version,
            "git_sha": p.card.git_sha,
            "feature_set_version": p.card.feature_set_version,
            "trained_through": p.card.trained_through,
            "training_rows": p.card.training_rows,
            "training_events": p.card.training_events,
            "records": p.stamp.describe(),
            "metrics": p.card.metrics,
            "notes": p.card.notes,
        }

    @app.get("/leaders")
    def leaders() -> dict[str, Any]:
        p = predictor()
        return {"leaders": p.known_leaders()}

    @app.post("/predict")
    def predict(query: Query) -> dict[str, Any]:
        p = predictor()
        try:
            result = p.predict(query.leader_a, query.leader_b, query.handle_a, query.handle_b)
        except NotReady as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        payload = result.as_dict()
        # One line, one prediction. Joined to outcomes later by (leaders, handles, version).
        predictions.info(
            json.dumps(
                {
                    "event": "prediction",
                    "model_version": p.card.version,
                    "feature_set_version": p.card.feature_set_version,
                    "leader_a": query.leader_a,
                    "leader_b": query.leader_b,
                    "handle_a": query.handle_a or None,
                    "handle_b": query.handle_b or None,
                    **payload,
                },
                separators=(",", ":"),
            )
        )
        return {"model_version": p.card.version, **payload}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = STATIC / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>OPTCG match forecast</h1><p>UI not bundled.</p>", 200)
        return HTMLResponse(page.read_text())

    return app


app = build_app()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the champion model")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - Cloud Run requires it
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    import uvicorn

    uvicorn.run(build_app(args.model_root), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
