import copy
import tempfile
import unittest
from pathlib import Path

from play_game import play_game
from random_agent import RandomAgent
from replay import ReplayValidationError, load_records, validate_replay


class ReplayCapabilityTests(unittest.TestCase):
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
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
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
            turn["spawn"]["value"] = 4

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
        html = (root / "viewer" / "index.html").read_text(encoding="utf-8")
        script = (root / "viewer" / "app.js").read_text(encoding="utf-8")
        summary = validate_replay(load_records(root / "viewer" / "sample_episode.jsonl"))

        self.assertIn('id="replay-file"', html)
        self.assertIn('id="board"', html)
        self.assertIn('fetch("sample_episode.jsonl")', script)
        self.assertGreater(summary["turns"], 0)


if __name__ == "__main__":
    unittest.main()
