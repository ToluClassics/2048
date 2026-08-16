import argparse
import copy
import random
import subprocess
import time
from pathlib import Path
from typing import Callable

from base_agent import BaseAgent
from engine import (
    DEFAULT_ENVIRONMENT_ID,
    ENVIRONMENT_CONTRACTS,
    Game2048,
    MOVE_FUNCTIONS,
    POSSIBLE_MOVES,
    print_board,
)
from random_agent import RandomAgent
from replay import EpisodeRecorder


def make_expectimax_agent(_seed: int | None) -> BaseAgent:
    from expectimax_agent import ExpectimaxAgent

    return ExpectimaxAgent()


def make_openai_agent(_seed: int | None) -> BaseAgent:
    from openai_agent import OpenAIAgent

    return OpenAIAgent()


def make_claude_agent(_seed: int | None) -> BaseAgent:
    from claude_agent import ClaudeAgent

    return ClaudeAgent()


def make_vllm_agent(_seed: int | None) -> BaseAgent:
    from vllm_agent import VLLMAgent

    return VLLMAgent()

AGENT_FACTORIES: dict[str, Callable[[int | None], BaseAgent]] = {
    "random": lambda seed: RandomAgent(random_seed=seed),
    "expectimax": make_expectimax_agent,
    "openai": make_openai_agent,
    "claude": make_claude_agent,
    "vllm": make_vllm_agent,
}


def play_game(
    agent: BaseAgent,
    max_turns: int,
    random_seed: int = 42,
    sleep_seconds: float = 0.0,
    verbose: bool = True,
    replay_path: Path | None = None,
    source_revision: str | None = None,
    environment_id: str = DEFAULT_ENVIRONMENT_ID,
) -> Game2048:
    reset_agent = getattr(agent, "reset", None)
    if callable(reset_agent):
        reset_agent()

    game = Game2048(random_seed=random_seed, environment_id=environment_id)
    recorder = None
    if replay_path is not None:
        recorder = EpisodeRecorder(
            output_path=replay_path,
            seed=random_seed,
            max_turns=max_turns,
            agent=agent,
            initial_board=copy.deepcopy(game.board),
            environment_contract=copy.deepcopy(game.environment_contract),
            source_revision=source_revision or current_git_revision(),
        )
    if verbose:
        print(f"Agent: {agent.name}")
        print_board(game.board)
        print(f"Score: {game.score}")

    turn = 0
    termination_reason = "max_turns"
    while turn < max_turns and not game.is_over():
        if verbose:
            print(f"============================================= Turn {turn + 1} =============================================")
        board_before = copy.deepcopy(game.board)
        move, decision_value = agent.get_move(copy.deepcopy(game.board))
        if move == "NONE":
            if verbose:
                print_agent_trace(agent)
                    
                print("Move: NONE")
                print("Result: board unchanged")
                print_board(game.board)
                print(f"Score: {game.score}")
            if recorder is not None:
                recorder.record_turn(
                    turn=turn + 1,
                    board_before=board_before,
                    requested_action=move,
                    action_valid=False,
                    board_after_move=copy.deepcopy(game.board),
                    board_after=copy.deepcopy(game.board),
                    score_delta=0,
                    score=game.score,
                    terminal=game.is_over(),
                    decision_value=decision_value,
                )
            turn += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        if move not in POSSIBLE_MOVES:
            termination_reason = "invalid_agent_action"
            if verbose:
                print(f"Invalid move from agent: {move}. Ending game.")
            if recorder is not None:
                recorder.record_turn(
                    turn=turn + 1,
                    board_before=board_before,
                    requested_action=str(move),
                    action_valid=False,
                    board_after_move=copy.deepcopy(game.board),
                    board_after=copy.deepcopy(game.board),
                    score_delta=0,
                    score=game.score,
                    terminal=False,
                    decision_value=decision_value,
                )
            turn += 1
            break

        board_after_move, score_delta = MOVE_FUNCTIONS[move](board_before)
        moved = game.step(move)
        if not moved:
            board_after_move = board_before
            score_delta = 0
        if recorder is not None:
            recorder.record_turn(
                turn=turn + 1,
                board_before=board_before,
                requested_action=move,
                action_valid=moved,
                board_after_move=board_after_move,
                board_after=copy.deepcopy(game.board),
                score_delta=score_delta,
                score=game.score,
                terminal=game.is_over(),
                decision_value=decision_value,
            )
        if verbose:
            print_agent_trace(agent)
            print(f"Move: {move}")
            if not moved:
                print("Result: move did not change the board")

            print_board(game.board)
            print(f"Score: {game.score}")

        turn += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if recorder is not None:
        if game.is_over():
            termination_reason = "game_over"
        recorder.finish(
            final_board=copy.deepcopy(game.board),
            score=game.score,
            termination_reason=termination_reason,
        )

    if verbose:
        if game.is_over():
            print(f"Game over! Final score: {game.score}")
        else:
            print(f"Stopped after {turn} turns. Final score: {game.score}")

    return game


def print_agent_trace(agent: BaseAgent) -> None:
    reasoning = getattr(agent, "last_reasoning", "")
    if isinstance(reasoning, str) and reasoning.strip():
        print("Model reasoning:")
        print(reasoning.strip())
    response = getattr(agent, "last_response", "")
    if isinstance(response, str) and response.strip():
        print("Model response:")
        print(response.strip())


def current_git_revision(ignore_paths: list[Path] | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    revision = result.stdout.strip()
    status_command = ["git", "status", "--porcelain", "--untracked-files=all"]
    if ignore_paths:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        if root_result.returncode == 0:
            root = Path(root_result.stdout.strip()).resolve()
            exclusions = []
            for path in ignore_paths:
                try:
                    relative_path = path.resolve().relative_to(root)
                except ValueError:
                    continue
                exclusions.append(f":(exclude){relative_path.as_posix()}")
            if exclusions:
                status_command.extend(["--", ".", *exclusions])
    status = subprocess.run(
        status_command,
        check=False,
        capture_output=True,
        text=True,
    )
    return f"{revision}-dirty" if status.stdout.strip() else revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2048 agent for a fixed number of turns.")
    parser.add_argument(
        "--agent",
        choices=sorted(AGENT_FACTORIES),
        default="random",
        help="Agent to use for gameplay.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="Maximum number of turns to play before stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the game and seeded agents.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between turns in seconds.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for the LLM agent.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="API base URL for the LLM agent.",
    )
    parser.add_argument(
        "--environment",
        choices=sorted(ENVIRONMENT_CONTRACTS),
        default=DEFAULT_ENVIRONMENT_ID,
        help="Versioned game environment to run.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per model decision.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
        help="Requested reasoning effort for compatible model endpoints.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional OpenRouter provider slug to pin.",
    )
    parser.add_argument(
        "--allow-provider-fallbacks",
        action="store_true",
        help="Allow OpenRouter to fall back from the pinned provider.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Write a versioned episode JSONL replay to this path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    if args.agent == "openai":
        from openai_agent import OpenAIAgent

        agent = OpenAIAgent(
            model=args.model,
            api_base_url=args.api_base_url,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
    elif args.agent == "vllm":
        from vllm_agent import VLLMAgent

        agent = VLLMAgent(
            model=args.model,
            api_base_url=args.api_base_url,
            max_output_tokens=args.max_output_tokens,
            inference_seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            provider=args.provider,
            allow_provider_fallbacks=args.allow_provider_fallbacks,
        )
    else:
        agent = AGENT_FACTORIES[args.agent](args.seed)

    play_game(
        agent=agent,
        max_turns=args.max_turns,
        random_seed=args.seed,
        sleep_seconds=args.sleep,
        replay_path=args.record,
        environment_id=args.environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
