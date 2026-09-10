# Tactical AI — RL Football Tactical Decision Making

A runnable MVP for the project:

**Reinforcement Learning for Tactical Decision Making in Football**

The project models a football manager as an RL agent. Every tactical decision interval, the manager chooses a compact tactical plan while an opponent-style estimator maintains a Bayesian belief over hidden opponent strategies.

## What this MVP includes

- Gymnasium-compatible tactical football environment
- Hidden opponent styles:
  - high press
  - possession
  - counterattack
- Tactical manager action space:
  - formation
  - defensive line
  - pressing intensity
  - attacking tempo
  - substitution decision
- Fatigue, possession, territory, pressure, score and xG-like match dynamics
- Bayesian opponent-style filter
- Three observation modes:
  - `blind`
  - `inference`
  - `full_info`
- MaskablePPO training
- Legal substitution action masking
- Rule-based baselines
- Multi-seed evaluation
- CSV result export
- FastAPI demo API
- Unit tests
- GRF backend interface stub for later integration

## Important design note

The current simulator is an **MVP research environment**, not a claim of realistic football physics. It exists so the POMDP formulation, PPO pipeline, Bayesian inference and experimental comparison can be developed before Google Research Football integration is completed.

The `GRFBackend` class deliberately raises `NotImplementedError` until the exact GRF tactical-control mapping is validated. Do not claim GRF tactical integration is complete until that adapter is implemented and tested.

## Setup

Recommended: Python 3.10 or 3.11.

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

## Install the project itself

After installing requirements, install the local package in editable mode so scripts can import `tactical_ai`:

```bash
pip install -e .
```

## Smoke test

```bash
python scripts/smoke_test.py
```

## Train an inference agent

```bash
python scripts/train.py --mode inference --timesteps 50000 --seed 1
```

Train blind and full-information agents:

```bash
python scripts/train.py --mode blind --timesteps 50000 --seed 1
python scripts/train.py --mode full_info --timesteps 50000 --seed 1
```

## Evaluate

```bash
python scripts/evaluate.py \
  --blind models/blind_seed1.zip \
  --inference models/inference_seed1.zip \
  --full-info models/full_info_seed1.zip \
  --matches 100
```

## Run API

```bash
uvicorn tactical_ai.api:app --reload
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Core research comparison

The central experiment is:

1. Blind agent — receives no explicit opponent-style belief.
2. Inference agent — receives Bayesian belief probabilities.
3. Full-information agent — receives the opponent's true scripted style.

Use matched network capacity, training budgets, seeds and evaluation opponent pools.

## Proposed next integration milestone

Replace `ToyMatchBackend` with `GRFBackend` while preserving the same high-level environment interface. Validate one tactical dimension at a time:

1. pressing intensity
2. defensive-line height
3. attacking tempo
4. formation
5. substitutions

Only after controlled tests demonstrate measurable behavioural effects should those controls be used in final PPO experiments.

---

# Stage 2 — Real Google Research Football adapter

The project now contains an optional real-GRF backend in `tactical_ai/grf_backend.py`.

The adapter uses GRF's `raw` observations and keeps the manager action space unchanged. It deliberately **does not pretend that GRF has native live formation/pressing sliders**. Instead, tactical decisions are translated into low-level player actions by `TacticalLowLevelController`.

## Validate GRF

```bash
python scripts/grf_smoke_test.py
```

Then validate full-team control:

```bash
python scripts/grf_smoke_test.py --all-left
```

Then run the first tactical A/B experiment:

```bash
python scripts/grf_pressing_ab_test.py --matches 3
```

See:

- `docs/GRF_SETUP_WINDOWS.md`
- `docs/GRF_VALIDATION_CHECKLIST.md`

The final RL experiments should stay on the toy backend until the GRF validation gates are passed. Once those gates pass, train with `backend="grf"` through the same `TacticalFootballEnv` interface.


## Stage 3 — Browser UI + live formation changes

Run:

```bash
uvicorn tactical_ai.api:app --reload
```

Open `http://127.0.0.1:8000/`.

The UI contains the score/time header, 2D pitch, opponent-style Bayesian bars,
agent-decision history, match stats, replay slider, Play/Pause, New Match,
formation, pressing, defensive-line, tempo and substitution controls.

Formation changes happen inside the same match. For example:

```text
0'   4-2-3-1
55'  4-3-3
70'  3-5-2
```

The episode is not reset. In the GRF backend, the selected formation is
translated by `TacticalLowLevelController` into updated positional targets.
GRF itself is not claimed to expose a native `change_formation()` command.

UI files:

```text
tactical_ai/ui/index.html
tactical_ai/ui/styles.css
tactical_ai/ui/app.js
```
