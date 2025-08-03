import chess
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("Cerebrus.h5")

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

def evaluate_board(board):
    input_array = board_to_input_array(board)
    input_tensor = np.expand_dims(input_array, axis=0)
    return model.predict(input_tensor)[0][0]

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, depth):
    """
    Находит лучший ход для текущей позиции.
    """
    best_move = None
    max_eval = -np.inf
    min_eval = np.inf

    for move in board.legal_moves:
        board.push(move)
        if board.turn == chess.BLACK:
            eval = minimax(board, depth - 1, -np.inf, np.inf, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        else:
            eval = minimax(board, depth - 1, -np.inf, np.inf, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        board.pop()
    return best_move

if __name__ == "__main__":
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            print("ИИ думает...")
            ai_move = get_best_move(board, depth=3)
            print(f"ИИ выбрал ход: {ai_move}")
            board.push(ai_move)
        else:
            move_str = input("Ваш ход (в формате UCI, например, e2e4): ")
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Недопустимый ход!")
            except:
                print("Неверный формат хода!")
    print("Игра окончена. Результат: ", board.result()) 