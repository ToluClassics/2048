import copy
import random
from typing import List, Tuple

Row = List[int]
Board = List[Row]
POSSIBLE_MOVES = ["LEFT", "RIGHT", "UP", "DOWN"]

def clone_board(board: Board) -> Board:
    return copy.deepcopy(board)

def place_tile(board: Board, value: int, row: int, col: int) -> Board:
    new_board = clone_board(board)
    new_board[row][col] = value
    return new_board

def print_board(board: Board) -> None:
    """Pretty-print the board in a boxed grid."""
    if not board or not board[0]:
        print("<empty board>")
        print()
        return

    max_value = max(max(row) for row in board)
    cell_width = max(5, len(str(max_value)))
    horizontal_border = "+" + "+".join("-" * (cell_width + 2) for _ in board[0]) + "+"

    print(horizontal_border)
    for row in board:
        cells = [f"{value:{cell_width}d}" if value else " " * cell_width for value in row]
        print("| " + " | ".join(cells) + " |")
        print(horizontal_border)
    print()

def check_empty_tiles(board: Board) -> List[Tuple]:
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

def check_boards_equal(a: Board, b: Board) -> bool:
    return a == b

def check_move_valid(board: Board, move: str) -> bool:
    move_func = MOVE_FUNCTIONS[move]
    new_board, _ = move_func(board)
    return not check_boards_equal(board, new_board)

def is_game_over(board: Board) -> bool:
    for move in POSSIBLE_MOVES:
        if check_move_valid(board, move):
            return False
    return True

def reverse_rows(board: Board) -> Board:
    """Reverse each row. Useful for implementing RIGHT moves."""
    return [row[::-1] for row in board]

def compress_row_left(row: Row) -> Row:
    "Move all non-empty tiles to left"

    tiles = [x for x in row if x > 0]
    while len(tiles) < len(row):
        tiles.append(0)
    
    return tiles

def transpose_board(board: Board) -> Board:
    """Transpose the board. Useful for implementing UP and DOWN moves."""
    transposed_board = [[0] * len(board) for _ in range(len(board[0]))]
    for r in range(len(board)):
        for c in range(len(board[0])):
            transposed_board[c][r] = board[r][c]
    return transposed_board

def merge_row_left(row: List[int]) -> Tuple[List[int], int]:
    """
    Merge equal adjacent tiles from left to right.
    Assumes row has already been compressed left.

    Example:
        [2, 2, 4, 0] -> [4, 0, 4, 0], score_gain = 4
    """
    row = row[:]  # copy
    score_gain = 0

    for i in range(3):
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i] *= 2
            row[i + 1] = 0
            score_gain += row[i]

    return row, score_gain

def move_row_left(row: List[int]) -> Tuple[List[int], int]:
    """
    Full left-move logic for one row:
    1. compress left
    2. merge
    3. compress again

    Example:
        [2, 0, 2, 4] -> [4, 4, 0, 0], score_gain = 4
    """
    step1 = compress_row_left(row)
    step2, score_gain = merge_row_left(step1)
    step3 = compress_row_left(step2)
    return step3, score_gain

def move_left(board: Board) -> Tuple[Board, int]:
    """Move the whole board left."""
    new_board = []
    total_score_gain = 0

    for row in board:
        new_row, row_score = move_row_left(row)
        new_board.append(new_row)
        total_score_gain += row_score

    return new_board, total_score_gain

def move_right(board: Board) -> Tuple[Board, int]:
    """Move the whole board right by reversing, moving left, then reversing back."""
    reversed_board = reverse_rows(board)
    moved_board, score_gain = move_left(reversed_board)
    restored_board = reverse_rows(moved_board)
    return restored_board, score_gain

def move_up(board: Board) -> Tuple[Board, int]:
    """Move the whole board up by transposing, moving left, then transposing back."""
    transposed_board = transpose_board(board)
    moved_board, score_gain = move_left(transposed_board)
    restored_board = transpose_board(moved_board)
    return restored_board, score_gain

def move_down(board: Board) -> Tuple[Board, int]:
    """Move the whole board down by transposing, moving right, then transposing back."""
    transposed_board = transpose_board(board)
    moved_board, score_gain = move_right(transposed_board)
    restored_board = transpose_board(moved_board)
    return restored_board, score_gain

MOVE_FUNCTIONS = {
    "LEFT": move_left,
    "RIGHT": move_right,
    "UP": move_up,
    "DOWN": move_down
}


class Game2048:
    def __init__(self, random_seed: int = 42):
        self.rng  = random.Random(random_seed)
        #initialize board
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.board = add_random_tile_to_board(self.board, self.rng)
    
    def is_over(self) -> bool:
        return is_game_over(self.board)
    
    def step(self, move: POSSIBLE_MOVES):
        new_board, score_gain = MOVE_FUNCTIONS[move](self.board)
        if check_boards_equal(new_board, self.board):
            return False

        self.board = new_board
        self.score += score_gain
        self.board = add_random_tile_to_board(self.board, self.rng)
        return True



if __name__ == "__main__":
    import time

    max_turns = 1000
    game = Game2048()
    print_board(game.board)
    print(f"Score: {game.score}")

    i=0
    while not game.is_over():
        print(f"============================================= Turn {i+1} =============================================")
        move = random.choice(POSSIBLE_MOVES)
        print(f"Move: {move}")
        game.step(move)
        print_board(game.board)
        print(f"Score: {game.score}")
        i+=1
        if i > max_turns:
            break

        # time.sleep(2.0)
    
    print(f"Game over! Final score: {game.score}")

