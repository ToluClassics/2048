"""Versioned, deterministic episode recordings for the current 2048 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from base_agent import BaseAgent, Board
from engine import (
    DEFAULT_ENVIRONMENT_ID,
    ENVIRONMENT_CONTRACTS,
    Game2048,
    MOVE_FUNCTIONS,
    POSSIBLE_MOVES,
    check_boards_equal,
    is_game_over,
    place_tile,
)

SCHEMA_VERSION = "2048.replay.v1"
ENVIRONMENT_CONTRACT = ENVIRONMENT_CONTRACTS[DEFAULT_ENVIRONMENT_ID]


class ReplayValidationError(ValueError):
    """Raised when a replay does not reproduce under its declared contract."""


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def describe_agent(agent: BaseAgent) -> dict[str, Any]:
    """Return a secret-free description of the configuration that affects play."""
    configuration = {}
    for name in (
        "model",
        "temperature",
        "history_size",
        "max_depth",
        "max_tokens",
        "max_output_tokens",
        "max_response_attempts",
        "inference_seed",
        "reasoning_effort",
        "provider",
        "allow_provider_fallbacks",
    ):
        value = getattr(agent, name, None)
        if value is not None:
            configuration[name] = value
    api_base_url = getattr(agent, "api_base_url", None)
    if api_base_url:
        configuration["api_base_url"] = api_base_url
    return {
        "name": agent.name,
        "class": type(agent).__name__,
        "configuration": configuration,
    }


def detect_spawn(
    board_after_move: Board,
    board_after_spawn: Board,
    spawn_distribution: dict[str, float],
) -> dict[str, int] | None:
    differences = []
    for row in range(4):
        for col in range(4):
            before = board_after_move[row][col]
            after = board_after_spawn[row][col]
            if before != after:
                differences.append((row, col, before, after))
    if not differences:
        return None
    if len(differences) != 1:
        raise ReplayValidationError("a valid move must create exactly one spawn difference")
    row, col, before, after = differences[0]
    if before != 0 or str(after) not in spawn_distribution:
        raise ReplayValidationError("spawn violates the declared environment distribution")
    return {"row": row, "col": col, "value": after}


@dataclass
class EpisodeRecorder:
    output_path: Path
    seed: int
    max_turns: int
    agent: BaseAgent
    initial_board: Board
    environment_contract: dict[str, Any] = field(
        default_factory=lambda: dict(ENVIRONMENT_CONTRACT)
    )
    source_revision: str = "unknown"
    records: list[dict[str, Any]] = field(default_factory=list)
    valid_actions: int = 0
    invalid_actions: int = 0

    def __post_init__(self) -> None:
        identity = {
            "schema_version": SCHEMA_VERSION,
            "environment": self.environment_contract,
            "seed": self.seed,
            "max_turns": self.max_turns,
            "agent": describe_agent(self.agent),
            "initial_board": self.initial_board,
            "source_revision": self.source_revision,
        }
        episode_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]
        self.records.append(
            {
                "record_type": "manifest",
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode_id,
                "environment": self.environment_contract,
                "seed": self.seed,
                "max_turns": self.max_turns,
                "agent": describe_agent(self.agent),
                "source": {
                    "repository": "ToluClassics/2048",
                    "revision": self.source_revision,
                },
                "initial_board": self.initial_board,
            }
        )

    def record_turn(
        self,
        *,
        turn: int,
        board_before: Board,
        requested_action: str,
        action_valid: bool,
        board_after_move: Board,
        board_after: Board,
        score_delta: int,
        score: int,
        terminal: bool,
        decision_value: object = None,
    ) -> None:
        if action_valid:
            self.valid_actions += 1
        else:
            self.invalid_actions += 1
        agent_event: dict[str, Any] = {}
        visible_output = getattr(self.agent, "last_response", "")
        if isinstance(visible_output, str) and visible_output.strip():
            agent_event["visible_output"] = visible_output.strip()
        visible_reasoning = getattr(self.agent, "last_reasoning", "")
        if isinstance(visible_reasoning, str) and visible_reasoning.strip():
            agent_event["visible_reasoning"] = visible_reasoning.strip()
        response_metadata = getattr(self.agent, "last_response_metadata", None)
        if isinstance(response_metadata, dict) and response_metadata:
            agent_event["response_metadata"] = response_metadata
        if isinstance(decision_value, (int, float)) and math.isfinite(decision_value):
            agent_event["decision_value"] = decision_value
        self.records.append(
            {
                "record_type": "turn",
                "turn": turn,
                "board_before": board_before,
                "requested_action": requested_action,
                "action_valid": action_valid,
                "board_after_move": board_after_move,
                "spawn": (
                    detect_spawn(
                        board_after_move,
                        board_after,
                        self.environment_contract["spawn_distribution"],
                    )
                    if action_valid
                    else None
                ),
                "board_after": board_after,
                "score_delta": score_delta,
                "score": score,
                "max_tile": max(max(row) for row in board_after),
                "terminal": terminal,
                "agent_event": agent_event,
            }
        )

    def finish(self, *, final_board: Board, score: int, termination_reason: str) -> None:
        self.records.append(
            {
                "record_type": "summary",
                "turns": len(self.records) - 1,
                "score": score,
                "max_tile": max(max(row) for row in final_board),
                "final_board": final_board,
                "valid_actions": self.valid_actions,
                "invalid_actions": self.invalid_actions,
                "termination_reason": termination_reason,
            }
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        contents = "".join(canonical_json(record) + "\n" for record in self.records)
        self.output_path.write_text(contents, encoding="utf-8")


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(f"line {line_number} is not valid JSON") from exc
        if not isinstance(record, dict):
            raise ReplayValidationError(f"line {line_number} must be a JSON object")
        records.append(record)
    return records


def validate_replay(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recompute every transition and return the verified summary."""
    items = list(records)
    if len(items) < 2 or items[0].get("record_type") != "manifest":
        raise ReplayValidationError("replay must begin with a manifest")
    manifest = items[0]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReplayValidationError("unsupported replay schema")
    declared_environment = manifest.get("environment")
    environment_id = (
        declared_environment.get("id") if isinstance(declared_environment, dict) else None
    )
    environment_contract = ENVIRONMENT_CONTRACTS.get(environment_id)
    if declared_environment != environment_contract:
        raise ReplayValidationError("unsupported environment contract")

    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReplayValidationError("manifest seed must be an integer")

    current_board = manifest.get("initial_board")
    _validate_board(current_board, "initial_board")
    initial_tiles = [value for row in current_board for value in row if value]
    allowed_tiles = {int(value) for value in environment_contract["spawn_distribution"]}
    if (
        len(initial_tiles) != environment_contract["initial_tiles"]
        or any(value not in allowed_tiles for value in initial_tiles)
    ):
        raise ReplayValidationError("initial_board violates the declared environment contract")
    seeded_game = Game2048(random_seed=seed, environment_id=environment_id)
    if seeded_game.board != current_board:
        raise ReplayValidationError("initial_board does not match the declared seed")
    current_score = 0
    turn_count = 0
    valid_actions = 0
    invalid_actions = 0

    for record in items[1:-1]:
        if record.get("record_type") != "turn":
            raise ReplayValidationError("only turn records may appear before the summary")
        turn_count += 1
        if record.get("turn") != turn_count:
            raise ReplayValidationError("turn numbers must be consecutive and one-based")
        if record.get("board_before") != current_board:
            raise ReplayValidationError(f"turn {turn_count} does not begin at the previous board")

        action = record.get("requested_action")
        action_valid = record.get("action_valid") is True
        expected_after_move = current_board
        expected_delta = 0
        if action in POSSIBLE_MOVES:
            candidate, candidate_delta = MOVE_FUNCTIONS[action](current_board)
            expected_valid = not check_boards_equal(candidate, current_board)
            if expected_valid != action_valid:
                raise ReplayValidationError(f"turn {turn_count} has incorrect action validity")
            if expected_valid:
                expected_after_move = candidate
                expected_delta = candidate_delta
            if seeded_game.step(action) is not expected_valid:
                raise ReplayValidationError(f"turn {turn_count} disagrees with the seeded engine")
        elif action_valid:
            raise ReplayValidationError(f"turn {turn_count} applies an unknown action")

        if record.get("board_after_move") != expected_after_move:
            raise ReplayValidationError(f"turn {turn_count} has an incorrect post-move board")
        if record.get("score_delta") != expected_delta:
            raise ReplayValidationError(f"turn {turn_count} has an incorrect score delta")

        expected_after = expected_after_move
        spawn = record.get("spawn")
        if action_valid:
            if not isinstance(spawn, dict) or set(spawn) != {"row", "col", "value"}:
                raise ReplayValidationError(f"turn {turn_count} is missing its exact spawn")
            if str(spawn["value"]) not in environment_contract["spawn_distribution"]:
                raise ReplayValidationError(f"turn {turn_count} violates the spawn contract")
            try:
                if expected_after_move[spawn["row"]][spawn["col"]] != 0:
                    raise ReplayValidationError(f"turn {turn_count} spawns into a non-empty cell")
                expected_after = place_tile(
                    expected_after_move, spawn["value"], spawn["row"], spawn["col"]
                )
            except (IndexError, TypeError) as exc:
                raise ReplayValidationError(f"turn {turn_count} has invalid spawn coordinates") from exc
            valid_actions += 1
        else:
            if spawn is not None:
                raise ReplayValidationError(f"turn {turn_count} spawns after an invalid action")
            invalid_actions += 1

        if record.get("board_after") != expected_after:
            raise ReplayValidationError(f"turn {turn_count} has an incorrect final board")
        if seeded_game.board != expected_after:
            raise ReplayValidationError(
                f"turn {turn_count} spawn does not match the declared seed"
            )
        current_score += expected_delta
        if record.get("score") != current_score:
            raise ReplayValidationError(f"turn {turn_count} has an incorrect cumulative score")
        if seeded_game.score != current_score:
            raise ReplayValidationError(f"turn {turn_count} score disagrees with the seeded engine")
        expected_max_tile = max(max(row) for row in expected_after)
        if record.get("max_tile") != expected_max_tile:
            raise ReplayValidationError(f"turn {turn_count} has an incorrect maximum tile")
        if record.get("terminal") is not is_game_over(expected_after):
            raise ReplayValidationError(f"turn {turn_count} has an incorrect terminal flag")
        current_board = expected_after

    summary = items[-1]
    if summary.get("record_type") != "summary":
        raise ReplayValidationError("replay must end with a summary")
    expected_summary = {
        "turns": turn_count,
        "score": current_score,
        "max_tile": max(max(row) for row in current_board),
        "final_board": current_board,
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ReplayValidationError(f"summary has an incorrect {key}")
    if is_game_over(current_board):
        expected_reason = "game_over"
    elif turn_count >= manifest.get("max_turns", -1):
        expected_reason = "max_turns"
    else:
        expected_reason = "invalid_agent_action"
    if summary.get("termination_reason") != expected_reason:
        raise ReplayValidationError("summary has an incorrect termination_reason")
    return summary


def _validate_board(board: object, label: str) -> None:
    if not isinstance(board, list) or len(board) != 4:
        raise ReplayValidationError(f"{label} must be a 4x4 board")
    if any(not isinstance(row, list) or len(row) != 4 for row in board):
        raise ReplayValidationError(f"{label} must be a 4x4 board")
    if any(not isinstance(value, int) or value < 0 for row in board for value in row):
        raise ReplayValidationError(f"{label} contains an invalid tile")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and deterministically replay a 2048 JSONL episode.")
    parser.add_argument("replay", type=Path, help="Path to a 2048.replay.v1 JSONL artifact.")
    args = parser.parse_args()
    print(canonical_json(validate_replay(load_records(args.replay))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
