import chess, chess.engine
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import random
import zstandard as zstd
import io

MODEL_FILE = "Cerebrus.h5"
STOCKFISH_PATH = "dataset/stockfish.exe"
SELFPLAY_GAMES = 100
EVAL_DEPTH = 12     
PLAY_DEPTH = 3                
DATASET_FILE = "dataset/chess_dataset.npz"
NEW_DATA_FILE = "dataset/selfplay_data.npz"

piece_map = { 'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11 }
def board_to_input_array(board):
    arr = np.zeros((8,8,12),dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            r,c = chess.square_rank(sq), chess.square_file(sq)
            arr[r,c,piece_map[p.symbol()]] = 1
    return arr

model = load_model(MODEL_FILE)

def evaluate_model(board):
    inp = board_to_input_array(board)[None,...]
    return float(model.predict(inp, verbose=0)[0,0])

def minimax(board, depth, alpha, beta, maximizing):
    if depth==0 or board.is_game_over():
        return evaluate_model(board)
    if maximizing:
        max_eval = -np.inf
        for m in board.legal_moves:
            board.push(m)
            ev = minimax(board, depth-1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta<=alpha: break
        return max_eval
    else:
        min_eval = np.inf
        for m in board.legal_moves:
            board.push(m)
            ev = minimax(board, depth-1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta<=alpha: break
        return min_eval

def get_model_move(board):
    best_m, best_v = None, -np.inf if board.turn==chess.WHITE else np.inf
    maximizing = (board.turn==chess.WHITE)
    for m in board.legal_moves:
        board.push(m)
        v = minimax(board, PLAY_DEPTH-1, -np.inf, np.inf, not maximizing)
        board.pop()
        if maximizing and v>best_v:
            best_v, best_m = v, m
        if not maximizing and v<best_v:
            best_v, best_m = v, m
    return best_m

positions = []
scores = []
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

for game_idx in range(SELFPLAY_GAMES):
    board = chess.Board()
    move_list = []
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = get_model_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(depth=PLAY_DEPTH))
            move = result.move
        board.push(move)
        positions.append(board.fen())
    for fen in positions[-len(move_list):]:
        b = chess.Board(fen)
        info = engine.analyse(b, chess.engine.Limit(depth=EVAL_DEPTH))
        sc = info["score"].white().score(mate_score=10000) or 0
        scores.append(np.tanh(sc/1000.))
print(f"Собрано {len(positions)} позиций из self-play.")

engine.quit()

X = np.array([board_to_input_array(chess.Board(f)) for f in positions], dtype=np.float32)
y = np.array(scores, dtype=np.float32)
np.savez(NEW_DATA_FILE, X=X, y=y)
print("Self-play данные сохранены в", NEW_DATA_FILE)
