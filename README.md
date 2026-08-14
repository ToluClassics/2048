# Getting LLMs to play 2048

A 2048 game engine and testbed for evaluating agent strategies, from rule-based search to LLMs that are given just enough information to understand the game and must discover strong play for themselves.

## Overview

This project builds a 2048 environment and uses it to benchmark different agents on multi-turn gameplay. For the LLM agents, we provide the board state and the rules of the game, but not a handcrafted strategy. The point is to see whether the model can infer good long-horizon play on its own and drive the board toward very high scores, with a rule-based Expectimax agent as a baseline.


## How It Works

The default `2048-canonical-v1` environment runs on a 4×4 board, starts with two tiles, and spawns a `2` with probability 0.9 or a `4` with probability 0.1 after each valid move. Each turn, the player chooses LEFT, RIGHT, UP, or DOWN. All tiles slide in that direction, and adjacent equal tiles merge into one. The game ends when no valid moves remain.

**Scoring** — each merge contributes the resulting tile's value to the score. E.g. merging two `64` tiles gives `+128`.

**Board moves** are implemented with three primitives:
- `compress` — slide all non-zero tiles to one side
- `merge` — combine equal adjacent tiles
- `compress` again — close the gap left by merges

UP/DOWN moves reuse the LEFT/RIGHT logic via board transposition. RIGHT reuses LEFT logic via row reversal.

## Legacy exploratory leaderboard

Best scores achieved by each agent. Rule-based agents use seeds `42..43` with `1000` max turns per game. LLM agents are evaluated with the same turn limit across multiple games.

> **Evidence status:** these rows came from the earlier `2048-lite-v0.1` environment, with only two rule-based games and four games per listed LLM. They are not comparable with the canonical environment, sufficiently powered, or fully provenance-bound. The bundled transcripts predate the versioned replay contract below.

| Agent | Games | Avg Score | Median Score | Min Score | Max Score | Avg Max Tile | Best Tile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 2 | 884 | 884 | 600 | 1168 | 96 | 128 |
| expectimax | 2 | 16064 | 16064 | 15980 | 16148 | 1024 | 1024 |
| GPT-5.4 Mini | 4 | 3879 | 3398 | 2800 | 5920 | 320 | 512 |
| GPT-5.4 | 4 | 8315 | 7944 | 5584 | 11788 | 640 | 1024 |
| Qwen3.5-9B | 4 | 2023 | 1880 | 1476 | 2856 | 192 | 256 |
| Qwen3.5-2B | 4 | 1128 | 1198 | 664 | 1452 | 96 | 128 |
| GPT-OSS-20B | 4 | 3281 | 3014 | 1184 | 5912 | 288 | 512 |
| Qwen3.5-0.8B | 4 | 1385 | 908 | 556 | 3168 | 128 | 256 |

## Game Engine

The core interface is the `Game2048` class in [engine.py](engine.py):

```python
game = Game2048(random_seed=42)

# Make a move — returns True if the move changed the board
game.step("LEFT")

# Check if the game is over
game.is_over()

# Current score
game.score

# Current board (4x4 list of ints, 0 = empty)
game.board
```

## Agents

- `RandomAgent` picks one of the four moves uniformly at random. It is a lightweight baseline for checking that the game loop and scoring work as expected.
- `ExpectimaxAgent` looks ahead several turns with expectimax search and scores future boards with heuristics such as open space, smoothness, monotonicity, corner control, and large-tile growth.

## Running Agents

From the repo root, you can run either agent with `play_game.py`:

```bash
python3 play_game.py --agent random --max-turns 1000 --seed 42
python3 play_game.py --agent expectimax --max-turns 1000 --seed 42
```

To benchmark both agents across multiple seeds:

```bash
python3 benchmark_agents.py \
  --agents random expectimax \
  --num-games 5 \
  --start-seed 100 \
  --max-turns 1000 \
  --output-dir results/canonical-baselines
```

The evaluator writes one replay per episode plus deterministic `summary.json` and `leaderboard.md` files. Median score is the predeclared primary metric; the raw five episode scores remain linked in the summary.

For prompted agents, one turn permits exactly one model inference. A response without a parseable final action is recorded as one invalid `NONE` action; the evaluator does not silently spend extra inference calls repairing it. Transport failures may still use the narrowly scoped HTTP retry policy.

## Reproducible Episode Replays

Add `--record` to produce a versioned JSONL evidence artifact:

```bash
python3 play_game.py \
  --agent random \
  --seed 42 \
  --max-turns 1000 \
  --record results/random-seed-42.jsonl
```

Validate the artifact by recomputing every move, score delta, spawn, cumulative score, and final summary:

```bash
python3 replay.py results/random-seed-42.jsonl
```

The first JSONL record is a run manifest. It binds the episode to:

- schema version `2048.replay.v1`;
- the complete versioned environment contract;
- the seed, maximum turns, agent class, and behavior-affecting configuration;
- the source repository revision when run from a Git checkout; and
- the exact initial board.

Each turn then records the board before the action, requested action, validity, board after movement, exact spawned tile, final board, score delta, cumulative score, maximum tile, terminal state, separate model reasoning and final output when available, response metadata, and any numeric decision value. The validator reproduces the initial board and every spawn from the declared seed. The final record contains the deterministic episode summary.

New runs default to `2048-canonical-v1`. The earlier one-initial-tile, 2-only variant remains available as `2048-lite-v0.1` solely so its historical artifacts can still be identified and validated. Results from the two contracts must not be compared.

For compatible hosted or local endpoints, cap and record generation behavior explicitly:

```bash
python3 play_game.py \
  --agent vllm \
  --model qwen3.5:0.8b \
  --api-base-url http://localhost:11434/v1 \
  --max-output-tokens 1024 \
  --reasoning-effort low \
  --seed 100 \
  --max-turns 20 \
  --record results/qwen-smoke.jsonl
```

### Local replay viewer

Serve the repository as static files:

```bash
python3 -m http.server 8000
```

Then open [http://localhost:8000/viewer/](http://localhost:8000/viewer/). The viewer loads a bundled legacy-lite demonstrative episode by default and can open any supported `2048.replay.v1` JSONL artifact. It provides turn-addressable URLs, playback controls, action validity, score/spawn evidence, separate reasoning and final-output panels, manifest provenance, and JSONL download. It only renders recorded state; validation remains the responsibility of `replay.py`.

### Published replay catalog and GitHub Pages

Store each completed five-seed evaluation beneath `viewer/evaluations/`. For example:

```bash
python3 benchmark_agents.py \
  --agents vllm \
  --model openai/gpt-5.4-mini \
  --api-base-url https://openrouter.ai/api/v1 \
  --num-games 5 \
  --start-seed 100 \
  --max-turns 1000 \
  --max-output-tokens 1024 \
  --reasoning-effort low \
  --output-dir viewer/evaluations/gpt-5.4-mini
```

After all intended models have the same environment, seed set, and turn limit, validate them and generate the viewer catalog plus leaderboard:

```bash
python3 build_catalog.py viewer/evaluations \
  --output viewer/catalog.json \
  --seeds 100 101 102 103 104
```

The catalog builder rejects invalid replays, mismatched seed sets/contracts, dirty or unknown source revisions, and episode summaries that disagree with their JSONL evidence. Commit the generated `viewer/catalog.json`, `viewer/catalog.md`, and referenced evaluation artifacts only after reviewing them.

Once that evidence commit is merged, enable GitHub Pages in the repository settings with **Deploy from a branch**, select the publication branch, and choose the repository root. The root `index.html` redirects to the static replay viewer. No server, API key, or inference endpoint is exposed by the page; it only serves committed artifacts.

## Focused Checks

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile *.py tests/*.py
git diff --check
```
