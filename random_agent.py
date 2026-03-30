import random
from typing import Optional, Tuple

from base_agent import BaseAgent, Board

POSSIBLE_MOVES = ["LEFT", "RIGHT", "UP", "DOWN"]


class RandomAgent(BaseAgent):
    def __init__(self, name: str = "RandomAgent", random_seed: Optional[int] = None):
        super().__init__(name)
        self.rng = random.Random(random_seed)
    
    def get_move(self, board: Board) -> Tuple[str, int]:
        return self.rng.choice(POSSIBLE_MOVES), -1
