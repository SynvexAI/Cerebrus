import chess
import chess.engine
import chess.pgn
import numpy as np
import multiprocessing as mp
from pathlib import Path
import random
import io
import zstandard as zstd
import chess.pgn

STOCKFISH_PATH = "dataset/stockfish.exe"
PGN_FILE = "dataset/lichess_db_standard_rated_2014-02.pgn.zst"
OUTPUT_FILE = "dataset/chess_dataset.npz"
EVAL_DEPTH = 12

piece_map = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def open_pgn_stream(path):
    f = open(path, 'rb')
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(f)
    return io.TextIOWrapper(reader, encoding='utf-8')

def board_to_input_array(board):
    arr = np.zeros((8, 8, 12), dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            r, c = chess.square_rank(sq), chess.square_file(sq)
            arr[r, c, piece_map[p.symbol()]] = 1.0
    return arr

def evaluate_position(fen):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as eng:
        info = eng.analyse(board, chess.engine.Limit(depth=EVAL_DEPTH))
        score = info["score"].white().score(mate_score=10000) or 0
    return score

def process_chunk(fens):
    X_chunk, y_chunk = [], []
    for fen in fens:
        score = evaluate_position(fen)
        norm = np.tanh(score / 1000.0)
        X_chunk.append(board_to_input_array(chess.Board(fen)))
        y_chunk.append(norm)
    return X_chunk, y_chunk

def extract_positions(pgn_path, max_positions):
    positions = []
    with open_pgn_stream(pgn_path) as f:
        while len(positions) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.is_game_over():
                    break
                positions.append(board.fen())
                if len(positions) >= max_positions:
                    break
    return positions

def generate_dataset(num_samples):
    print("Извлечение позиций из PGN...")
    fens = extract_positions(PGN_FILE, num_samples)
    print(f"Найдено {len(fens)} позиций, начинаем оценку...")

    cpu = mp.cpu_count()
    chunks = np.array_split(fens, cpu)
    with mp.Pool(cpu) as pool:
        results = pool.map(process_chunk, chunks)

    X, y = [], []
    for Xc, yc in results:
        X.extend(Xc)
        y.extend(yc)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"Датасет сохранён в {OUTPUT_FILE} (X: {X.shape}, y: {y.shape})")

if __name__ == "__main__":
    generate_dataset(num_samples=20000)
