import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine import Game2048
from expectimax_agent import ExpectimaxAgent, evaluate
from play_game import play_game
from random_agent import RandomAgent
from replay import ReplayValidationError, describe_agent, load_records, validate_replay
from vllm_agent import VLLMAgent


class ReplayCapabilityTests(unittest.TestCase):
    def test_canonical_environment_is_seeded_and_supports_four_spawns(self):
        self.assertEqual(
            Game2048(random_seed=7).board,
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 0, 0]],
        )
        self.assertEqual(
            Game2048(random_seed=7).board,
            Game2048(random_seed=7).board,
        )

    def test_expectimax_chance_node_includes_two_and_four_spawns(self):
        board = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 0],
        ]
        board_with_2 = copy.deepcopy(board)
        board_with_2[3][3] = 2
        board_with_4 = copy.deepcopy(board)
        board_with_4[3][3] = 4
        expected = 0.9 * evaluate(board_with_2) + 0.1 * evaluate(board_with_4)

        actual = ExpectimaxAgent().explore_future_value("LEFT", board, 1, is_chance=True)

        self.assertAlmostEqual(actual, expected)

    def test_model_generation_limits_are_part_of_replay_provenance(self):
        agent = VLLMAgent(
            model="qwen3.5:0.8b",
            max_output_tokens=1024,
            reasoning_effort="low",
        )
        description = describe_agent(agent)

        self.assertEqual(description["configuration"]["max_output_tokens"], 1024)
        self.assertEqual(description["configuration"]["max_response_attempts"], 1)
        self.assertEqual(description["configuration"]["reasoning_effort"], "low")

    def test_reasoning_only_response_is_one_invalid_action_not_a_final_answer(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            message = SimpleNamespace(content=None, reasoning="still thinking")
            return SimpleNamespace(
                id="pilot-response",
                model="qwen3.5:0.8b",
                choices=[SimpleNamespace(message=message, finish_reason="length")],
                usage=None,
            )

        agent = VLLMAgent(model="qwen3.5:0.8b", max_output_tokens=1024)
        agent._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        move, _ = agent.get_move(
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]]
        )

        self.assertEqual(move, "NONE")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 1024)
        self.assertEqual(agent.last_reasoning, "still thinking")
        self.assertEqual(agent.last_response, "")
        self.assertEqual(agent.last_response_metadata["finish_reason"], "length")

    def test_seeded_episode_records_and_replays_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.jsonl"
            second_path = Path(directory) / "second.jsonl"

            first_game = play_game(
                RandomAgent(random_seed=42),
                max_turns=40,
                random_seed=42,
                verbose=False,
                replay_path=first_path,
                source_revision="test-revision",
            )
            second_game = play_game(
                RandomAgent(random_seed=42),
                max_turns=40,
                random_seed=42,
                verbose=False,
                replay_path=second_path,
                source_revision="test-revision",
            )

            summary = validate_replay(load_records(first_path))
            manifest = load_records(first_path)[0]
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(manifest["environment"]["id"], "2048-canonical-v1")
            self.assertEqual(summary["final_board"], first_game.board)
            self.assertEqual(summary["score"], first_game.score)
            self.assertEqual(second_game.board, first_game.board)

    def test_replay_rejects_a_changed_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            play_game(
                RandomAgent(random_seed=7),
                max_turns=5,
                random_seed=7,
                verbose=False,
                replay_path=path,
                source_revision="test-revision",
            )
            records = load_records(path)
            changed = copy.deepcopy(records)
            turn = next(record for record in changed if record["record_type"] == "turn" and record["spawn"])
            turn["spawn"]["value"] = 8

            with self.assertRaisesRegex(ReplayValidationError, "spawn contract"):
                validate_replay(changed)

    def test_replay_rejects_a_plausible_spawn_that_disagrees_with_the_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            play_game(
                RandomAgent(random_seed=42),
                max_turns=1,
                random_seed=42,
                verbose=False,
                replay_path=path,
                source_revision="test-revision",
            )
            changed = copy.deepcopy(load_records(path))
            turn = changed[1]
            original_spawn = (turn["spawn"]["row"], turn["spawn"]["col"])
            replacement = next(
                (row, col)
                for row in range(4)
                for col in range(4)
                if turn["board_after_move"][row][col] == 0
                and (row, col) != original_spawn
            )
            changed_board = copy.deepcopy(turn["board_after_move"])
            changed_board[replacement[0]][replacement[1]] = 2
            turn["spawn"] = {"row": replacement[0], "col": replacement[1], "value": 2}
            turn["board_after"] = changed_board
            turn["max_tile"] = max(max(row) for row in changed_board)
            turn["terminal"] = False
            changed[-1]["final_board"] = changed_board
            changed[-1]["max_tile"] = turn["max_tile"]

            with self.assertRaisesRegex(ReplayValidationError, "declared seed"):
                validate_replay(changed)

    def test_static_viewer_is_wired_to_a_valid_sample(self):
        root = Path(__file__).resolve().parents[1]
        root_html = (root / "index.html").read_text(encoding="utf-8")
        html = (root / "viewer" / "index.html").read_text(encoding="utf-8")
        script = (root / "viewer" / "app.js").read_text(encoding="utf-8")
        summary = validate_replay(load_records(root / "viewer" / "sample_episode.jsonl"))

        self.assertIn('url=viewer/', root_html)
        self.assertIn('id="replay-file"', html)
        self.assertIn('id="board"', html)
        self.assertIn('id="catalog-episode"', html)
        self.assertIn('id="agent-reasoning"', html)
        self.assertIn('fetch("catalog.json")', script)
        self.assertIn('requestedEpisode || "sample_episode.jsonl"', script)
        self.assertGreater(summary["turns"], 0)


if __name__ == "__main__":
    unittest.main()
