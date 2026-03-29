import copy
import random
from typing import List

Board = List[List[int]]
POSSIBLE_MOVES = ["LEFT", "RIGHT", "UP", "DOWN"]

def clone_board(board: Board) -> Board:
    return copy.deepcopy(board)

def place_tile(board: Board, value: int, row: int, col: int) -> Board:
    new_board = clone_board(board)
    new_board[row][col] = value
    return new_board

def print_board(board: Board) -> None:
    """Pretty-print the board."""
    for row in board:
        print(" ".join(f"{x:5d}" for x in row))
    print()

def check_empty_tiles(board: Board) -> List[tuple]:
    """
    Check and list all the cells on the baord that's empty

    Args:
        board (Board):
    """
    empties = []
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 0:
                empties.append((r, c))
    
    return empties


def add_random_tile_to_board(board: Board, rng: random.Random) -> Board:
    """
    Add a random tile to an existing board. First check what the empty cells are 
    on the board and then randomly sample one of them to insert a tile

    Args:
        board (Board): 
        rng (random.Random):
    """
    empties = check_empty_tiles(board)

    if not empties:
        return clone_board(board)

    cell = rng.choice(empties)
    board = place_tile(board, 2, cell[0], cell[1])
    return board


class Game2048:
    def __init__(self, random_seed: int = 42):

        self.rng  = random.Random(random_seed)
        #initialize board
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.board = add_random_tile_to_board(self.board, self.rng)


if __name__ == "__main__":
    game = Game2048()

    print_board(game.board)

