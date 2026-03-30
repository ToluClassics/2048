# Getting LLMs to play 2048

![2048 demo](assets/demo_video.gif)

A 2048 game engine and testbed for evaluating agent strategies — from rule-based to LLM-powered.

## Overview

This project builds a 2048 environment and uses it to benchmark different agents on multi-turn gameplay. The end goal is a demo comparing how well different LLMs play 2048, with a rule-based Expectimax agent as a baseline.


## How It Works

The game runs on a 4×4 board. Each turn, the player chooses a direction (LEFT, RIGHT, UP, DOWN). All tiles slide in that direction, and any adjacent equal tiles merge into one (doubling their value and adding to the score). After each valid move, a new tile (value `2`) is placed randomly on an empty cell. The game ends when no valid moves remain.

**Scoring** — each merge contributes the resulting tile's value to the score. E.g. merging two `64` tiles gives `+128`.

**Board moves** are implemented with three primitives:
- `compress` — slide all non-zero tiles to one side
- `merge` — combine equal adjacent tiles
- `compress` again — close the gap left by merges

UP/DOWN moves reuse the LEFT/RIGHT logic via board transposition. RIGHT reuses LEFT logic via row reversal.

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
python3 2048/play_game.py --agent random --max-turns 1000 --seed 42
python3 2048/play_game.py --agent expectimax --max-turns 1000 --seed 42
```

To benchmark both agents across multiple seeds:

```bash
python3 2048/benchmark_agents.py --agents random expectimax --num-games 2 --start-seed 42 --max-turns 1000
```

## Benchmark Snapshot

Using seeds `42..43` and `1000` max turns per game:

| Agent | Games | Avg Score | Median Score | Min Score | Max Score | Avg Max Tile | Best Tile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 2 | 884 | 884 | 600 | 1168 | 96 | 128 |
| expectimax | 2 | 16064 | 16064 | 15980 | 16148 | 1024 | 1024 |