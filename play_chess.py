import chess
import numpy as np
from tensorflow.keras.models import load_model

MODEL_FILE = "Cerebrus.h5"
SEARCH_DEPTH = 3

piece_map = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_input_array(board):
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            r, c = chess.square_rank(sq), chess.square_file(sq)
            arr[r, c, piece_map[p.symbol()]] = 1.0
    return arr

model = load_model(MODEL_FILE)

def evaluate_board(board):
    inp = board_to_input_array(board)[None, ...]
    return model.predict(inp, verbose=0)[0,0]

def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing:
        max_eval = -np.inf
        for m in board.legal_moves:
            board.push(m)
            ev = minimax(board, depth-1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for m in board.legal_moves:
            board.push(m)
            ev = minimax(board, depth-1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, depth=SEARCH_DEPTH):
    best_move, best_val = None, -np.inf if board.turn == chess.WHITE else np.inf

    maximizing = (board.turn == chess.WHITE)
    for m in board.legal_moves:
        board.push(m)
        ev = minimax(board, depth-1, -np.inf, np.inf, not maximizing)
        board.pop()

        if maximizing and ev > best_val:
            best_val, best_move = ev, m
        if not maximizing and ev < best_val:
            best_val, best_move = ev, m

    return best_move

if __name__ == "__main__":
    board = chess.Board()
    while not board.is_game_over():
        print(board, "\n")
        if board.turn == chess.WHITE:
            print("ИИ думает...")
            move = get_best_move(board)
            print("ИИ ходит:", move, "\n")
            board.push(move)
        else:
            usr = input("Ваш ход (UCI): ")
            try:
                mv = chess.Move.from_uci(usr)
                if mv in board.legal_moves:
                    board.push(mv)
                else:
                    print("Недопустимый ход!")
            except:
                print("Неверный формат!")
    print("Игра завершена:", board.result())
