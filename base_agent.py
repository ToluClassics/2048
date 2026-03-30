from abc import ABC, abstractmethod
from typing import List, Tuple

Row = List[int]
Board = List[Row]

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, board: Board)-> Tuple[str, int]:
        raise NotImplementedError("Must be implemented by subclass")
