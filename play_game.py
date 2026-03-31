import argparse
import copy
import random
import time
from typing import Callable

from base_agent import BaseAgent
from engine import Game2048, POSSIBLE_MOVES, print_board
from expectimax_agent import ExpectimaxAgent
from llm_agent import LLMAgent
from random_agent import RandomAgent

AGENT_FACTORIES: dict[str, Callable[[int | None], BaseAgent]] = {
    "random": lambda seed: RandomAgent(random_seed=seed),
    "expectimax": lambda seed: ExpectimaxAgent(),
    "llm": lambda seed: LLMAgent(),
}


def play_game(
    agent: BaseAgent,
    max_turns: int,
    random_seed: int = 42,
    sleep_seconds: float = 0.0,
    verbose: bool = True,
) -> Game2048:
    reset_agent = getattr(agent, "reset", None)
    if callable(reset_agent):
        reset_agent()

    game = Game2048(random_seed=random_seed)
    if verbose:
        print(f"Agent: {agent.name}")
        print_board(game.board)
        print(f"Score: {game.score}")

    turn = 0
    while turn < max_turns and not game.is_over():
        if verbose:
            print(f"============================================= Turn {turn + 1} =============================================")
        move, _ = agent.get_move(copy.deepcopy(game.board))
        if move == "NONE":
            if verbose:
                if isinstance(agent, LLMAgent) and agent.last_response.strip():
                    print("LLM response:")
                    print(agent.last_response.strip())
                    
                print("Move: NONE")
                print("Result: board unchanged")
                print_board(game.board)
                print(f"Score: {game.score}")
            turn += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        if move not in POSSIBLE_MOVES:
            if verbose:
                print(f"Invalid move from agent: {move}. Ending game.")
            break

        moved = game.step(move)
        if verbose:
            if isinstance(agent, LLMAgent) and agent.last_response.strip():
                print("LLM response:")
                print(agent.last_response.strip())
            print(f"Move: {move}")
            if not moved:
                print("Result: move did not change the board")

            print_board(game.board)
            print(f"Score: {game.score}")

        turn += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if verbose:
        if game.is_over():
            print(f"Game over! Final score: {game.score}")
        else:
            print(f"Stopped after {turn} turns. Final score: {game.score}")

    return game


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    if args.agent == "llm":
        agent = LLMAgent(
            model=args.model,
            api_base_url=args.api_base_url,
        )
    else:
        agent = AGENT_FACTORIES[args.agent](args.seed)

    play_game(
        agent=agent,
        max_turns=args.max_turns,
        random_seed=args.seed,
        sleep_seconds=args.sleep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
