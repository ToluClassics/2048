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

The core interface is the `Game2048` class in [2048.py](2048.py):

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

To watch a random-move game play out in the terminal:

```bash
python 2048.py
```

## Project Structure

```
2048/
├── 2048.py        # Game engine and board logic
├── main.py        # Entry point
├── utils.py       # Utility functions
└── pyproject.toml
```
