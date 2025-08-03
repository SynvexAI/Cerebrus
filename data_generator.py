import chess
import chess.engine
import numpy as np
import random

STOCKFISH_PATH = "stockfish.exe"

def get_stockfish_evaluation(board, depth):
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white().score(mate_score=10000)
            return score
    except chess.engine.EngineTerminatedError:
        print("Stockfish engine terminated unexpectedly.")
        return 0

def board_to_input_array(board):
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    input_array = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = chess.square_rank(square), chess.square_file(square)
            piece_channel = piece_map[piece.symbol()]
            input_array[row, col, piece_channel] = 1.0
    return input_array

def generate_dataset(num_samples, max_moves=100, depth=20):
    X, y = [], []
    for i in range(num_samples):
        board = chess.Board()
        num_moves_in_game = random.randint(1, max_moves)
        for _ in range(num_moves_in_game):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            board.push(move)

        if not board.is_game_over():
            evaluation = get_stockfish_evaluation(board, depth)
            normalized_eval = np.tanh(evaluation / 1000.0)
            X.append(board_to_input_array(board))
            y.append(normalized_eval)
        
        if (i + 1) % 100 == 0:
            print(f"Сгенерировано {i + 1}/{num_samples} позиций")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X_train, y_train = generate_dataset(10000)
    np.savez("chess_dataset.npz", X=X_train, y=y_train)
    print("Датасет сохранен в chess_dataset.npz")