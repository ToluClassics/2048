import argparse
import json
import re
import shlex
import statistics
from pathlib import Path
from typing import Iterable

from engine import DEFAULT_ENVIRONMENT_ID, ENVIRONMENT_CONTRACTS, Board
from play_game import AGENT_FACTORIES, current_git_revision, play_game
from replay import describe_agent, load_records, validate_replay


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
    max_output_tokens: int = 1024,
    reasoning_effort: str = "low",
    provider: str | None = None,
    allow_provider_fallbacks: bool = False,
):
    if agent_name == "openai":
        from openai_agent import OpenAIAgent

        return lambda _seed: OpenAIAgent(
            model=model,
            api_base_url=api_base_url,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
    if agent_name == "claude":
        from claude_agent import ClaudeAgent

        return lambda _seed: ClaudeAgent(model=model)
    if agent_name == "vllm":
        from vllm_agent import VLLMAgent

        return lambda seed: VLLMAgent(
            model=model,
            api_base_url=api_base_url,
            max_output_tokens=max_output_tokens,
            inference_seed=seed,
            reasoning_effort=reasoning_effort,
            provider=provider,
            allow_provider_fallbacks=allow_provider_fallbacks,
        )
    return AGENT_FACTORIES[agent_name]


def run_benchmark(
    agent_names: list[str],
    seeds: Iterable[int],
    max_turns: int,
    model: str | None = None,
    api_base_url: str | None = None,
    output_dir: Path | None = None,
    environment_id: str = DEFAULT_ENVIRONMENT_ID,
    max_output_tokens: int = 1024,
    reasoning_effort: str = "low",
    provider: str | None = None,
    allow_provider_fallbacks: bool = False,
    source_revision: str | None = None,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, dict[str, object]]:
    if resume and output_dir is None:
        raise ValueError("resume requires an output directory")
    results: dict[str, dict[str, object]] = {}
    evaluation_revision = source_revision or current_git_revision()

    seed_list = list(seeds)
    for agent_name in agent_names:
        scores: list[int] = []
        max_tiles: list[int] = []
        factory = _make_agent_factory(
            agent_name,
            model=model,
            api_base_url=api_base_url,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            provider=provider,
            allow_provider_fallbacks=allow_provider_fallbacks,
        )
        episodes = []
        run_slug = safe_slug(model or agent_name)

        for seed in seed_list:
            agent = factory(seed)
            replay_path = (
                output_dir / "replays" / run_slug / f"seed-{seed}.jsonl"
                if output_dir is not None
                else None
            )
            if resume and replay_path is not None and replay_path.is_file():
                validated, episode_revision = _load_resumable_episode(
                    replay_path,
                    agent=agent,
                    seed=seed,
                    max_turns=max_turns,
                    environment_id=environment_id,
                )
                score = validated["score"]
                episode_max_tile = validated["max_tile"]
                if verbose:
                    print(f"Reusing validated replay: {replay_path}")
            else:
                game = play_game(
                    agent=agent,
                    max_turns=max_turns,
                    random_seed=seed,
                    sleep_seconds=0.0,
                    verbose=verbose,
                    replay_path=replay_path,
                    source_revision=evaluation_revision,
                    environment_id=environment_id,
                )
                score = game.score
                episode_max_tile = max_tile(game.board)
                episode_revision = evaluation_revision
            scores.append(score)
            max_tiles.append(episode_max_tile)
            episodes.append(
                {
                    "seed": seed,
                    "score": score,
                    "max_tile": episode_max_tile,
                    "source_revision": episode_revision,
                    "replay": (
                        replay_path.relative_to(output_dir).as_posix()
                        if replay_path is not None and output_dir is not None
                        else None
                    ),
                }
            )

        results[agent_name] = {
            "model": model,
            "games": len(seed_list),
            "score_summary": summarize(scores),
            "tile_summary": summarize(max_tiles),
            "episodes": episodes,
        }

    return results


def _load_resumable_episode(
    replay_path: Path,
    *,
    agent: object,
    seed: int,
    max_turns: int,
    environment_id: str,
) -> tuple[dict[str, object], str]:
    try:
        records = load_records(replay_path)
        validated = validate_replay(records)
        manifest = records[0]
        if manifest.get("seed") != seed:
            raise ValueError("seed does not match")
        if manifest.get("max_turns") != max_turns:
            raise ValueError("turn limit does not match")
        if manifest.get("environment") != ENVIRONMENT_CONTRACTS[environment_id]:
            raise ValueError("environment contract does not match")
        if manifest.get("agent") != describe_agent(agent):
            raise ValueError("agent configuration does not match")
        revision = manifest.get("source", {}).get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("source revision is missing")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot resume from {replay_path}: {exc}") from exc
    return validated, revision


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_generation_command(
    output_dir: Path,
    *,
    agent_names: list[str],
    model: str | None,
    seeds: list[int],
    max_turns: int,
    environment_id: str,
    api_base_url: str | None,
    max_output_tokens: int,
    reasoning_effort: str,
    provider: str | None,
    allow_provider_fallbacks: bool,
    resume: bool = False,
) -> str:
    command = [
        "python3",
        "benchmark_agents.py",
        "--agents",
        *agent_names,
        "--num-games",
        str(len(seeds)),
        "--start-seed",
        str(seeds[0]),
        "--max-turns",
        str(max_turns),
        "--environment",
        environment_id,
    ]
    if model:
        command.extend(["--model", model])
    if api_base_url:
        command.extend(["--api-base-url", api_base_url])
    command.extend(["--max-output-tokens", str(max_output_tokens)])
    command.extend(["--reasoning-effort", reasoning_effort])
    if provider:
        command.extend(["--provider", provider])
    if allow_provider_fallbacks:
        command.append("--allow-provider-fallbacks")
    if resume:
        command.append("--resume")
    command.extend(["--output-dir", output_dir.as_posix()])
    return shlex.join(command)


def write_evaluation_artifacts(
    output_dir: Path,
    *,
    agent_names: list[str],
    model: str | None,
    seeds: list[int],
    max_turns: int,
    environment_id: str,
    max_output_tokens: int,
    reasoning_effort: str,
    provider: str | None,
    allow_provider_fallbacks: bool,
    source_revision: str,
    results: dict[str, dict[str, object]],
    api_base_url: str | None = None,
    resume: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "2048.evaluation.v1",
        "environment": ENVIRONMENT_CONTRACTS[environment_id],
        "agents": agent_names,
        "model": model,
        "seeds": seeds,
        "max_turns": max_turns,
        "inference": {
            "api_base_url": api_base_url,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "provider": provider,
            "allow_provider_fallbacks": allow_provider_fallbacks,
        },
        "source_revision": source_revision,
        "source_revisions": sorted(
            {
                episode["source_revision"]
                for result in results.values()
                for episode in result["episodes"]
            }
        ),
        "primary_metric": "median_score",
        "generation_command": build_generation_command(
            output_dir,
            agent_names=agent_names,
            model=model,
            seeds=seeds,
            max_turns=max_turns,
            environment_id=environment_id,
            api_base_url=api_base_url,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            provider=provider,
            allow_provider_fallbacks=allow_provider_fallbacks,
            resume=resume,
        ),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "| Agent | Model | Games | Median score | Mean score | Min | Max | Median max tile | Best tile |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for agent_name, result in results.items():
        score = result["score_summary"]
        tile = result["tile_summary"]
        lines.append(
            f"| {agent_name} | {result['model'] or '—'} | {result['games']} "
            f"| {format_float(score['median'])} | {format_float(score['avg'])} "
            f"| {score['min']} | {score['max']} | {format_float(tile['median'])} "
            f"| {tile['max']} |"
        )
    (output_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default=5,
        help="Number of seeds/games to run per agent.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=100,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory for episode JSONL, summary JSON, and generated leaderboard Markdown.",
    )
    parser.add_argument(
        "--environment",
        choices=sorted(ENVIRONMENT_CONTRACTS),
        default=DEFAULT_ENVIRONMENT_ID,
        help="Versioned environment contract.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--provider", default=None)
    parser.add_argument("--allow-provider-fallbacks", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching, validated episode replays and run only missing seeds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = range(args.start_seed, args.start_seed + args.num_games)
    source_revision = current_git_revision(ignore_paths=[args.output_dir])
    results = run_benchmark(
        args.agents,
        seeds,
        args.max_turns,
        model=args.model,
        api_base_url=args.api_base_url,
        output_dir=args.output_dir,
        environment_id=args.environment,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        provider=args.provider,
        allow_provider_fallbacks=args.allow_provider_fallbacks,
        source_revision=source_revision,
        resume=args.resume,
    )
    write_evaluation_artifacts(
        args.output_dir,
        agent_names=args.agents,
        model=args.model,
        seeds=list(seeds),
        max_turns=args.max_turns,
        environment_id=args.environment,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        provider=args.provider,
        allow_provider_fallbacks=args.allow_provider_fallbacks,
        source_revision=source_revision,
        results=results,
        api_base_url=args.api_base_url,
        resume=args.resume,
    )

    print(f"Benchmark seeds: {args.start_seed}..{args.start_seed + args.num_games - 1}")
    print(f"Max turns per game: {args.max_turns}")
    print()
    print_markdown_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
