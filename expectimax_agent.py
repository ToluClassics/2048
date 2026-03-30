from typing import Tuple

from base_agent import BaseAgent, Board
from engine import check_move_valid, POSSIBLE_MOVES, MOVE_FUNCTIONS, board_to_key, is_game_over, check_empty_tiles, place_tile

def count_empty(board: Board) -> int:
    return len(check_empty_tiles(board))


def max_tile(board: Board) -> int:
    return max(max(row) for row in board)


def corner_max_bonus(board: Board) -> int:
    """
    Return 1 if the largest tile is in a corner, else 0.
    """
    m = max_tile(board)
    corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
    return 1 if m in corners else 0


def smoothness(board: Board) -> float:
    """
    Penalize large differences between neighboring tiles.
    We ignore zeros when comparing.
    """
    penalty = 0.0

    # Horizontal neighbors
    for r in range(4):
        for c in range(3):
            a, b = board[r][c], board[r][c + 1]
            if a != 0 and b != 0:
                penalty -= abs(a - b)

    # Vertical neighbors
    for r in range(3):
        for c in range(4):
            a, b = board[r][c], board[r + 1][c]
            if a != 0 and b != 0:
                penalty -= abs(a - b)

    return penalty


def monotonicity(board: Board) -> float:
    """
    Reward rows/cols that consistently increase or decrease.
    This is a simple version.
    """
    score = 0.0

    # Rows
    for row in board:
        inc = 0
        dec = 0
        for i in range(3):
            if row[i] <= row[i + 1]:
                inc += row[i + 1] - row[i]
            else:
                dec += row[i] - row[i + 1]
        score -= min(inc, dec)

    # Columns
    for c in range(4):
        col = [board[r][c] for r in range(4)]
        inc = 0
        dec = 0
        for i in range(3):
            if col[i] <= col[i + 1]:
                inc += col[i + 1] - col[i]
            else:
                dec += col[i] - col[i + 1]
        score -= min(inc, dec)

    return score


def weighted_tile_sum(board: Board) -> float:
    """
    Reward large tiles more heavily.
    """
    total = 0.0
    for row in board:
        for x in row:
            if x > 0:
                total += x ** 1.2
    return total


def evaluate(board: Board) -> float:
    """
    Heuristic board evaluator.
    Higher is better.
    """
    return (
        250.0 * count_empty(board)
        + 2.0 * smoothness(board)
        + 4.0 * monotonicity(board)
        + 1.0 * weighted_tile_sum(board)
        + 500.0 * corner_max_bonus(board)
        + 1.5 * max_tile(board)
    )

class ExpectimaxAgent(BaseAgent):
    def __init__(self, name: str = "ExpectimaxAgent", max_depth: int = 3):
        super().__init__(name)
        self.max_depth = max_depth
        self.cache = {}
    
    def reset_cache(self):
        self.cache = {}
    
    def explore_future_value(self, move: str, board: Board, depth: int, is_chance: bool) -> int:
        # check if we've explored this board before at this depth
        board_key = (board_to_key(board), depth, is_chance)
        if board_key in self.cache:
            return self.cache[board_key]
        
        if depth == 0 or is_game_over(board):
            value = evaluate(board)
            self.cache[board_key] = value
            return value
        
        if is_chance:
            empties = check_empty_tiles(board)
            if not empties:
                value = self.explore_future_value(move, board, depth - 1, is_chance=False)
                self.cache[board_key] = value
                return value

            total = 0.0
            for (r, c) in empties:
                board_with_2 = place_tile(board, 2, r, c)

                p_cell = 1 / len(empties)
                total += p_cell * 0.9 * self.explore_future_value(move, board_with_2, depth - 1, is_chance=False)

            self.cache[board_key] = total
            return total

        else:
            legal_moves = []
            for move in POSSIBLE_MOVES:
                if check_move_valid(board, move):
                    legal_moves.append(move)

            if not legal_moves:
                value = evaluate(board)
                self.cache[board_key] = value
                return value

            best = float("-inf")
            for move in legal_moves:
                next_board, _ = MOVE_FUNCTIONS[move](board)
                best = max(best, self.explore_future_value(move, next_board, depth - 1, is_chance=True))

            self.cache[board_key] = best
            return best
    
    def get_move(self, board: Board) -> Tuple[str, int]:
        
        # First we check how many legal moves are possible
        legal_moves = []
        for move in POSSIBLE_MOVES:
            if check_move_valid(board, move):
                legal_moves.append(move)
        
        if not legal_moves:
            return None
        
        self.reset_cache()

        best_move = None
        current_best_value = -float("inf")

        for move in legal_moves:
            next_board, _ = MOVE_FUNCTIONS[move](board)
            projected_value = self.explore_future_value(move, next_board, self.max_depth-1, is_chance=True)

            if projected_value > current_best_value:
                best_move = move
                current_best_value = projected_value
        
        return best_move, current_best_value


        

            

