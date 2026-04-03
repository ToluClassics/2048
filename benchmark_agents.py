import argparse
import statistics
from typing import Iterable

from engine import Board
from openai_agent import OpenAIAgent
from play_game import AGENT_FACTORIES, play_game
from vllm_agent import VLLMAgent


def max_tile(board: Board) -> int:
    return max(max(row) for row in board)


def format_float(value: float) -> str:
    return f"{value:.2f}"


def summarize(values: list[int]) -> dict[str, float]:
    return {
        "avg": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _make_agent_factory(
    agent_name: str,
    model: str | None = None,
    api_base_url: str | None = None,
):
    if agent_name == "openai":
        return lambda _seed: OpenAIAgent(model=model, api_base_url=api_base_url)
    if agent_name == "vllm":
        return lambda _seed: VLLMAgent(model=model, api_base_url=api_base_url)
    return AGENT_FACTORIES[agent_name]


def run_benchmark(
    agent_names: list[str],
    seeds: Iterable[int],
    max_turns: int,
    model: str | None = None,
    api_base_url: str | None = None,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}

    seed_list = list(seeds)
    for agent_name in agent_names:
        scores: list[int] = []
        max_tiles: list[int] = []
        factory = _make_agent_factory(agent_name, model=model, api_base_url=api_base_url)

        for seed in seed_list:
            agent = factory(seed)
            game = play_game(
                agent=agent,
                max_turns=max_turns,
                random_seed=seed,
                sleep_seconds=0.0,
                verbose=True,
            )
            scores.append(game.score)
            max_tiles.append(max_tile(game.board))

        results[agent_name] = {
            "games": len(seed_list),
            "score_summary": summarize(scores),
            "tile_summary": summarize(max_tiles),
        }

    return results


def print_markdown_table(results: dict[str, dict[str, object]]) -> None:
    print("| Agent | Games | Avg Score | Median Score | Min Score | Max Score | Avg Max Tile | Best Tile |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for agent_name, result in results.items():
        score_summary = result["score_summary"]
        tile_summary = result["tile_summary"]
        print(
            f"| {agent_name} "
            f"| {result['games']} "
            f"| {format_float(score_summary['avg'])} "
            f"| {format_float(score_summary['median'])} "
            f"| {score_summary['min']} "
            f"| {score_summary['max']} "
            f"| {format_float(tile_summary['avg'])} "
            f"| {tile_summary['max']} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 2048 agents across multiple random seeds.")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=sorted(AGENT_FACTORIES),
        default=["random", "expectimax"],
        help="Agents to benchmark.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=20,
        help="Number of seeds/games to run per agent.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="First seed in the benchmark range.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="Maximum turns per game.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for LLM agents (openai/vllm). Can also be set via LLM_MODEL env var.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="API base URL for LLM agents (openai/vllm).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = range(args.start_seed, args.start_seed + args.num_games)
    results = run_benchmark(args.agents, seeds, args.max_turns, model=args.model, api_base_url=args.api_base_url)

    print(f"Benchmark seeds: {args.start_seed}..{args.start_seed + args.num_games - 1}")
    print(f"Max turns per game: {args.max_turns}")
    print()
    print_markdown_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
