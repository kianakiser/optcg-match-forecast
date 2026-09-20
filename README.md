<!--
README requirements (Repository Guide, "By MS4"):
  prediction · FTI diagram · clone-and-run steps · live URL · video link
Video link goes at the top. Every TODO must be gone by 2027-01-10.
-->

# optcg-match-forecast

> Predicts the winner of individual swiss-round matches in competitive One Piece TCG tournaments,
> from the two submitted decklists and each player's prior tournament history.

<!-- MS4: unlisted YouTube / SWITCHtube link here. No video files in the repo. -->
**Live system:** _TODO (MS4)_ · **Video (3 min):** _TODO (MS4)_

Semester project for **I.BA_MLOPS** (MLOps), BSc Artificial Intelligence & Machine Learning,
Hochschule Luzern — HS26.

---

## What it predicts

Given two registered decklists and both pilots' results at strictly earlier events, the system
returns **P(player 1 wins)** for that match. It is served on demand through a UI, and scored
nightly against every match that resolved since the previous run.

The provider lists a tournament only **after it finishes** (verified: the newest event in the live
index is a day old), so there is no in-progress feed to predict against in real time. The honest
architecture is therefore on-demand serving plus nightly scoring — not live in-tournament
inference. The label still arrives on its own, with no manual annotation anywhere in the loop.

**Why this is hard, and why it is interesting:** strong players pick strong decks, so a naive
archetype win rate credits the *deck* for the *pilot*. Separating the two is the modelling
question — and out of time the per-player strength term turns out to contribute almost nothing
(Brier 0.2464 with it, 0.2465 without), while a random split makes it look essential. The signal
lives in archetype-vs-archetype matchup cells.

**The label is not derivable from the features.** It is the outcome of a game between two humans.
Player 1 wins 50.19% of 68,322 decided swiss matches — a Wilson CI of [49.82%, 50.57%], so seat
position is not distinguishable from a coin flip and carries no free signal.

**Success criterion.** Pooled **Brier ≤ 0.2490** against the coin-flip 0.2500, over ≥ 6 held-out
28-day windows (≥ 8,000 matches), with the 95% event-cluster bootstrap CI on the skill excluding
zero — plus calibration within 3 pp in every 5-pp favourite bucket with n ≥ 250.

**Current champion (`v1`, trained 2026-09-20).** Pooled over 6 held-out 28-day windows and 12,700
out-of-time matches: Brier **0.2325**, accuracy **60.98%**, skill **+0.0175** with an event-cluster
CI of **[+0.0152, +0.0202]**, worst calibration gap **1.7%**. It clears the criterion on every
term.

The comparison that matters is not against the coin flip but against the plain archetype matchup
rate, which scores **0.2449** on the same matches. The model beats the simple thing it replaces,
and the pipeline refuses to register a candidate that does not — see
[`training/run.py`](src/optcg_forecast/training/run.py).

## Data

| | |
|---|---|
| Source | Limitless TCG tournament API (`play.limitlesstcg.com/api`) — keyless, community-run |
| Corpus (2024-09-07 to 2026-09-17) | 277 events · 25,206 entrants · **23,733 with full 50-card decklists** (94.2%) |
| Usable matches | **68,322** decided swiss matches from 254 events; 11,685 of them round 1 |
| Concentration | largest organiser is 34.0% of matches (61.4% of events) — down from 80.2% before the two-year backfill |
| Leaders seen | 133 distinct |
| Update | event-driven; new events are immutable once finished, so re-fetch by id is safe |

Ingest is deliberately polite: finished events are cached permanently because they never change,
and the request budget is throttled well inside the documented limit.

## Leakage

The API returns each entrant's decklist **directly alongside the event's own outcome**:

```jsonc
{ "player": "...", "decklist": {...}, "deck": {...},   // known before play — features
  "placing": 12, "record": {...}, "drop": null }       // outcomes OF THIS EVENT — never features
```

Two traps, both handled at the ingest boundary in
[`features/ingest.py`](src/optcg_forecast/features/ingest.py):

1. **Post-hoc fields.** `placing`, `record` and `drop` are stripped *physically* before anything
   downstream sees them — not filtered at training time, which would leave the footgun loaded for
   the next caller. `assert_no_leakage()` fails the pipeline if one ever survives.

2. **Selection on the outcome.** Players who drop out mid-event are **kept**. Dropping out is
   itself an outcome, and droppers win far fewer matches than finishers, so the obvious
   `WHERE placing IS NOT NULL` would condition the population on *finishing* and delete most of
   the true negatives. Measured on the backfill: **10,027 of 25,206 entrants (39.8%) dropped**, and
   some of them still received a placing — so `drop IS NOT NULL` is the correct test, not
   `placing IS NULL`.

3. **Top-cut brackets mislabelled as swiss.** Two events are pure single-elimination brackets
   whose rows all carry `phase: 1`. The `match` field (`"T32-16"`) is the reliable marker, so a row
   carrying one is excluded regardless of its phase — 62 matches that would otherwise have been
   counted as swiss.

Splits are **out of time**, by event date, with events as the clustering unit for bootstrap
intervals. Matches from one event never straddle the split.

## Architecture

![FTI architecture](docs/src/architecture.png)

Three decoupled pipelines — they never call each other, only the feature store and the registry.

| | Pipeline | Trigger | Reads | Writes |
|---|---|---|---|---|
| 1 | **Feature** | GitHub Actions, daily + on-demand backfill | Limitless API | Hopsworks |
| 2 | **Training** | scheduled / manual | Hopsworks feature view | Hopsworks model registry |
| 3 | **Inference** | on demand (UI) + nightly scoring | registry + feature store | predictions / UI |

Regenerate the diagram after any stack change:

```bash
./scripts/build_diagram.sh
```

## Stack

| Concern | Choice | Why |
|---|---|---|
| Feature store | Hopsworks | point-in-time-correct joins — training and serving read one feature definition, so a player-history feature can never silently include the event being predicted |
| Model registry | Hopsworks | the same hosted service as the feature store; promotion moves the `champion` alias rather than redeploying |
| Experiment tracking | Weights & Biases | hosted run tracking, so there is no MLflow server to operate for a project this size |
| Orchestration | GitHub Actions | scheduled pipelines live beside the code, and the runner stays the ingest edge rather than a cloud IP range |
| Serving | Google Cloud Run | container deploy, scales to zero between events |
| Storage | Google Cloud Storage | immutable partitioned landing zone for raw payloads, so features can always be rebuilt |

**Stretch, explicitly optional:** drift monitoring on decklist composition, a second tournament
feed to reduce organiser concentration, and calibration dashboards. Core FTI ships first.

## Clone and run

```bash
git clone https://github.com/kianakiser/optcg-match-forecast.git
cd optcg-match-forecast

uv sync                      # exact versions from uv.lock
cp .env.example .env         # then fill it in — see the table below

uv run python -m optcg_forecast.features.run                       # ingest + write features
uv run python -m optcg_forecast.features.backfill --start 2025-07-06 --end 2026-09-18
uv run python -m optcg_forecast.training.run                       # train, evaluate, register
uv run python -m optcg_forecast.inference.serve                    # serve predictions
```

Backfill runs through the **same** feature pipeline as live ingest — one code path, so a repaired
gap and a fresh event produce identical features.

With Docker:

```bash
docker build -t optcg-match-forecast .
docker run --rm --env-file .env -p 8080:8080 optcg-match-forecast
```

### Environment variables

Names only — see [`.env.example`](.env.example). Never commit `.env`; the same names must exist as
GitHub Actions secrets for the scheduled pipelines.

| Variable | What it is |
|---|---|
| `SOURCE_API_BASE_URL` / `SOURCE_API_KEY` | tournament API endpoint and key, if one is issued |
| `HOPSWORKS_API_KEY` / `HOPSWORKS_PROJECT` | feature store access |
| `WANDB_API_KEY` | experiment tracking |
| `GCP_PROJECT_ID` / `GCS_BUCKET` / `GCP_REGION` | serving and raw-payload storage |

## Tests

```bash
uv run pytest        # includes repo-hygiene checks that enforce the course rules
uv run ruff check .
```

The hygiene suite fails the build on the course's own rules: no `.env` tracked, no data or model
artifacts in git, no `mlruns/`, no secret patterns, no pipeline importing from a notebook.

## Layout

```
├── src/optcg_forecast/
│   ├── features/     ingest boundary, feature computation, backfill
│   ├── training/     train, evaluate out-of-time, register
│   ├── inference/    load champion, predict, serve
│   └── common/       config and the single source of feature definitions
├── docs/             proposal.pdf, ms*_summary.pdf, architecture diagram
├── tests/            unit tests, run in CI
├── scripts/          diagram generation
└── .github/workflows/  CI + scheduled pipelines
```

Feature definitions live in exactly one place under `common/`, used by both training and
inference. That is what keeps training–serving skew out.

## Milestones

| | Due | Status |
|---|---|---|
| MS1 Proposal | 2026-10-01 | ☐ |
| MS2 Feature pipeline | 2026-11-05 | ☐ |
| MS3 Training pipeline | 2026-12-03 | ☐ |
| MS4 Live system | 2027-01-10 | ☐ |

## Acknowledgements

Tournament data from the community-run [Limitless TCG](https://play.limitlesstcg.com) platform.
Non-commercial student project; One Piece Card Game is a trademark of its respective owner.
