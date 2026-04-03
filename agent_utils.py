import re
from typing import Optional

from base_agent import Board
from engine import POSSIBLE_MOVES

ACTION_TAG_PATTERN = re.compile(r"<action>\s*(LEFT|RIGHT|UP|DOWN|NONE)\s*</action>", re.IGNORECASE)
MAX_RESPONSE_PARSE_ATTEMPTS = 5

DEFAULT_SYSTEM_PROMPT = """
You are the decision-making agent for the game 2048.

Your job on every turn is to read the current 4x4 board and choose exactly one move:
"UP", "DOWN", "LEFT", or "RIGHT".

Board interpretation:
- The board is a 4x4 grid given as 4 rows from top to bottom.
- Each number is a tile value.
- 0 means the cell is empty.

How the game works:
- A move shifts all tiles as far as possible in the chosen direction.
- After sliding, adjacent equal tiles merge into one tile with double the value.
- A tile can merge at most once on a move.
- After merges, tiles compress again toward the move direction.
- Example: moving LEFT on [2, 0, 2, 4] produces [4, 4, 0, 0].
- Example: moving LEFT on [2, 2, 2, 2] produces [4, 4, 0, 0], not [8, 0, 0, 0].
- In this implementation, every valid move adds one new tile with value 2 in a random empty cell.
- A move is invalid if it does not change the board state.
- The game ends when no valid moves remain.

Scoring and objective:
- When tiles merge, the reward gained is the value of the merged tile.
- Your goal is to maximize cumulative reward over the full episode.

How to choose actions:
- Infer which moves are legal by checking whether a move would change the board.
- Avoid invalid moves that leave the board unchanged.
- Prefer moves that create merges, preserve empty cells, avoid locking the board, and keep large tiles stable and easy to combine later.
- Avoid moves that break a strong structure unless they clearly improve the position.

Output requirements:
- Return exactly one XML block and nothing else.
- Keep the reasoning brief and specific to the current board.
- The action must be exactly one of: LEFT, RIGHT, UP, DOWN.

Respond using XML in exactly this structure:
<response>
  <reasoning>brief explanation here</reasoning>
  <action>move</action>
</response>
""".strip()


def format_board(board: Board) -> str:
    if not board or not board[0]:
        return "<empty board>"

    max_value = max(max(row) for row in board)
    cell_width = max(5, len(str(max_value)))
    horizontal_border = "+" + "+".join("-" * (cell_width + 2) for _ in board[0]) + "+"

    lines = [horizontal_border]
    for row in board:
        cells = [f"{value:{cell_width}d}" if value else " " * cell_width for value in row]
        lines.append("| " + " | ".join(cells) + " |")
        lines.append(horizontal_border)

    return "\n".join(lines)


def make_observation(board: Board) -> dict[str, object]:
    return {"board": board, "done": False}


def build_user_prompt(
    observation_history: list[dict[str, object]],
    current_observation: dict[str, object],
    history_size: int,
    invalid_response: Optional[str] = None,
) -> str:
    previous_observations = observation_history[-history_size:] if history_size > 0 else []
    lines = ["Observation:"]

    if previous_observations:
        lines.append("Previous observations:")
        for index, observation in enumerate(previous_observations, start=1):
            lines.append(f"- Observation {index}:")
            lines.append(format_board(observation["board"]))
            lines.append(f"done: {observation['done']}")
    else:
        lines.append("Previous observations: none")

    lines.append("Current observation:")
    lines.append(format_board(current_observation["board"]))
    lines.append(f"done: {current_observation['done']}")

    prompt = "\n".join(lines)
    if invalid_response:
        prompt += (
            "\n\nYour previous response was invalid for this state. "
            "Reply again using exactly one <response> block with an <action> tag "
            "containing one of: LEFT, RIGHT, UP, DOWN, NONE."
        )
    return prompt


def parse_move(raw_response: str) -> Optional[str]:
    match = ACTION_TAG_PATTERN.search(raw_response)
    if not match:
        return None

    action = match.group(1).upper()
    if action == "NONE" or action in POSSIBLE_MOVES:
        return action
    return None
