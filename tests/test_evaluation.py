import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_agents import run_benchmark, write_evaluation_artifacts
from build_catalog import build_catalog
from replay import load_records, validate_replay


class EvaluationCapabilityTests(unittest.TestCase):
    def test_seed_set_produces_valid_episodes_and_generated_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            seeds = [100, 101]
            source_revision = "test-revision"
            with patch(
                "benchmark_agents.current_git_revision",
                return_value=source_revision,
            ) as revision_snapshot:
                results = run_benchmark(
                    ["random"],
                    seeds,
                    max_turns=5,
                    output_dir=output_dir,
                    verbose=False,
                )
            revision_snapshot.assert_called_once_with()
            write_evaluation_artifacts(
                output_dir,
                agent_names=["random"],
                model=None,
                seeds=seeds,
                max_turns=5,
                environment_id="2048-canonical-v1",
                max_output_tokens=1024,
                reasoning_effort="low",
                provider=None,
                allow_provider_fallbacks=False,
                source_revision=source_revision,
                results=results,
            )

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            episodes = summary["results"]["random"]["episodes"]
            self.assertEqual(summary["seeds"], seeds)
            self.assertEqual(summary["primary_metric"], "median_score")
            self.assertEqual(summary["source_revision"], source_revision)
            self.assertEqual(summary["source_revisions"], [source_revision])
            self.assertIn("python3 benchmark_agents.py", summary["generation_command"])
            self.assertIn("--start-seed 100", summary["generation_command"])
            self.assertEqual(len(episodes), 2)
            for episode in episodes:
                records = load_records(output_dir / episode["replay"])
                validated = validate_replay(records)
                self.assertEqual(records[0]["source"]["revision"], source_revision)
                self.assertEqual(validated["score"], episode["score"])
            self.assertIn("| random |", (output_dir / "leaderboard.md").read_text())

            catalog = build_catalog(
                output_dir,
                output_dir / "catalog.json",
                expected_seeds=seeds,
                require_bound_source=False,
            )
            self.assertEqual(catalog["entries"][0]["agent"], "random")
            self.assertEqual(len(catalog["entries"][0]["episodes"]), 2)
            self.assertIn("generation_command", catalog["entries"][0])
            self.assertEqual(
                catalog["entries"][0]["failure_summary"]["total_turns"],
                sum(episode["turns"] for episode in catalog["entries"][0]["episodes"]),
            )

    def test_resume_reuses_a_valid_episode_and_runs_only_missing_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            first_results = run_benchmark(
                ["random"],
                [100],
                max_turns=5,
                output_dir=output_dir,
                source_revision="first-revision",
                verbose=False,
            )
            first_path = output_dir / first_results["random"]["episodes"][0]["replay"]
            first_contents = first_path.read_bytes()

            resumed_results = run_benchmark(
                ["random"],
                [100, 101],
                max_turns=5,
                output_dir=output_dir,
                source_revision="second-revision",
                verbose=False,
                resume=True,
            )

            self.assertEqual(first_path.read_bytes(), first_contents)
            self.assertEqual(
                [episode["source_revision"] for episode in resumed_results["random"]["episodes"]],
                ["first-revision", "second-revision"],
            )
            write_evaluation_artifacts(
                output_dir,
                agent_names=["random"],
                model=None,
                seeds=[100, 101],
                max_turns=5,
                environment_id="2048-canonical-v1",
                max_output_tokens=1024,
                reasoning_effort="low",
                provider=None,
                allow_provider_fallbacks=False,
                source_revision="second-revision",
                results=resumed_results,
                resume=True,
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["source_revisions"], ["first-revision", "second-revision"])
            self.assertIn("--resume", summary["generation_command"])


if __name__ == "__main__":
    unittest.main()
