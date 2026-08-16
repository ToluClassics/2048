"""Build a validated replay catalog and leaderboard from evaluation summaries."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from replay import load_records, validate_replay


class CatalogError(ValueError):
    pass


def build_catalog(
    evaluations_dir: Path,
    output_path: Path,
    *,
    expected_seeds: list[int],
    require_bound_source: bool = True,
) -> dict[str, Any]:
    summaries = sorted(evaluations_dir.glob("**/summary.json"))
    if not summaries:
        raise CatalogError("no evaluation summaries found")

    entries = []
    environment = None
    max_turns = None
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != "2048.evaluation.v1":
            raise CatalogError(f"unsupported summary schema: {summary_path}")
        if summary.get("seeds") != expected_seeds:
            raise CatalogError(f"seed set mismatch: {summary_path}")
        if require_bound_source and _is_unbound(summary.get("source_revision")):
            raise CatalogError(f"summary is not bound to a clean source revision: {summary_path}")
        if environment is None:
            environment = summary["environment"]
            max_turns = summary["max_turns"]
        elif summary["environment"] != environment or summary["max_turns"] != max_turns:
            raise CatalogError("all leaderboard entries must share an environment and turn limit")

        declared_revisions = summary.get("source_revisions", [summary["source_revision"]])
        if require_bound_source and any(_is_unbound(revision) for revision in declared_revisions):
            raise CatalogError(f"summary contains an unbound episode revision: {summary_path}")
        all_episode_source_revisions = set()
        for agent_name, result in summary["results"].items():
            episodes = []
            failed_turns = 0
            total_turns = 0
            agent_configuration = None
            episode_source_revisions = set()
            for episode in result["episodes"]:
                replay_path = (summary_path.parent / episode["replay"]).resolve()
                if not replay_path.is_relative_to(summary_path.parent.resolve()):
                    raise CatalogError(f"replay escapes its evaluation directory: {replay_path}")
                records = load_records(replay_path)
                validated = validate_replay(records)
                manifest = records[0]
                agent_configuration = manifest["agent"].get("configuration", {})
                episode_source_revisions.add(manifest["source"]["revision"])
                all_episode_source_revisions.add(manifest["source"]["revision"])
                if require_bound_source and _is_unbound(manifest["source"]["revision"]):
                    raise CatalogError(f"replay is not bound to a clean source revision: {replay_path}")
                if validated["score"] != episode["score"] or validated["max_tile"] != episode["max_tile"]:
                    raise CatalogError(f"episode summary mismatch: {replay_path}")
                failed_turns += validated["invalid_actions"]
                total_turns += validated["turns"]
                try:
                    published_path = replay_path.relative_to(output_path.parent.resolve())
                except ValueError as exc:
                    raise CatalogError(
                        "published replays must live beneath the catalog directory"
                    ) from exc
                episodes.append(
                    {
                        **episode,
                        "replay": published_path.as_posix(),
                        "episode_id": manifest["episode_id"],
                        "failed_turns": validated["invalid_actions"],
                        "turns": validated["turns"],
                    }
                )

            entries.append(
                {
                    "agent": agent_name,
                    "model": result["model"],
                    "inference": summary["inference"],
                    "generation_command": summary.get("generation_command")
                    or _legacy_generation_command(summary, summary_path, agent_configuration or {}),
                    "source_revision": summary["source_revision"],
                    "source_revisions": sorted(episode_source_revisions),
                    "score_summary": result["score_summary"],
                    "tile_summary": result["tile_summary"],
                    "failure_summary": {
                        "failed_turns": failed_turns,
                        "total_turns": total_turns,
                        "rate": failed_turns / total_turns if total_turns else 0.0,
                    },
                    "episodes": episodes,
                }
            )
        if set(declared_revisions) != all_episode_source_revisions:
            raise CatalogError(f"episode source revision set mismatch: {summary_path}")

    entries.sort(
        key=lambda entry: (
            -entry["score_summary"]["median"],
            -entry["score_summary"]["avg"],
            entry["model"] or entry["agent"],
        )
    )
    catalog = {
        "schema_version": "2048.catalog.v1",
        "environment": environment,
        "seeds": expected_seeds,
        "max_turns": max_turns,
        "primary_metric": "median_score",
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_path.with_suffix(".md"), entries)
    return catalog


def _is_unbound(revision: object) -> bool:
    return not isinstance(revision, str) or revision == "unknown" or revision.endswith("-dirty")


def _legacy_generation_command(
    summary: dict[str, Any],
    summary_path: Path,
    agent_configuration: dict[str, Any],
) -> str:
    seeds = summary["seeds"]
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise CatalogError(f"seed set is not a contiguous benchmark range: {summary_path}")
    if summary_path.parent.is_absolute():
        try:
            output_dir = summary_path.parent.relative_to(Path.cwd())
        except ValueError:
            output_dir = Path(summary_path.parent.name)
    else:
        output_dir = summary_path.parent
    inference = summary["inference"]
    command = [
        "python3",
        "benchmark_agents.py",
        "--agents",
        *summary["agents"],
        "--num-games",
        str(len(seeds)),
        "--start-seed",
        str(seeds[0]),
        "--max-turns",
        str(summary["max_turns"]),
        "--environment",
        summary["environment"]["id"],
    ]
    if summary.get("model"):
        command.extend(["--model", summary["model"]])
    api_base_url = inference.get("api_base_url") or agent_configuration.get("api_base_url")
    if api_base_url:
        command.extend(["--api-base-url", api_base_url])
    command.extend(["--max-output-tokens", str(inference["max_output_tokens"])])
    command.extend(["--reasoning-effort", inference["reasoning_effort"]])
    if inference.get("provider"):
        command.extend(["--provider", inference["provider"]])
    if inference.get("allow_provider_fallbacks"):
        command.append("--allow-provider-fallbacks")
    command.extend(["--output-dir", output_dir.as_posix()])
    return shlex.join(command)


def _write_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "| Rank | Agent | Model | Median score | Mean score | Scores | Median max tile | Failed turns |",
        "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for rank, entry in enumerate(entries, start=1):
        scores = ", ".join(str(episode["score"]) for episode in entry["episodes"])
        lines.append(
            f"| {rank} | {entry['agent']} | {entry['model'] or '—'} "
            f"| {entry['score_summary']['median']:.2f} | {entry['score_summary']['avg']:.2f} "
            f"| {scores} | {entry['tile_summary']['median']:.2f} "
            f"| {entry['failure_summary']['failed_turns']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate evaluations and build the replay catalog.")
    parser.add_argument("evaluations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("viewer/catalog.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument(
        "--allow-unbound-source",
        action="store_true",
        help="Allow unknown or dirty source revisions for local pilot catalogs only.",
    )
    args = parser.parse_args()
    build_catalog(
        args.evaluations,
        args.output,
        expected_seeds=args.seeds,
        require_bound_source=not args.allow_unbound_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
