import chess
import chess.engine
import chess.pgn
import numpy as np
import multiprocessing as mp
from pathlib import Path
import random
import io
import zstandard as zstd
import os

STOCKFISH_PATH = "dataset/stockfish.exe"
PGN_FILE = "dataset/lichess_db_standard_rated_2014-02.pgn.zst"
OUTPUT_DIR = "dataset/chess_batches"
EVAL_DEPTH = 20
NUM_SELFPLAY_GAMES = 1000
MAX_POSITIONS_FROM_PGN = 10000
MAX_POSITIONS_SELFPLAY = 10000
BATCH_SIZE = 1000

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


def extract_positions_from_pgn(pgn_path, max_positions):
    positions = []
    with open_pgn_stream(pgn_path) as f:
        while len(positions) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if 10 < board.fullmove_number < 60 and len(board.piece_map()) > 8:
                    positions.append(board.fen())
                if len(positions) >= max_positions:
                    break
    return positions


def generate_selfplay_positions(num_games, max_positions):
    positions = []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for _ in range(num_games):
            board = chess.Board()
            while not board.is_game_over() and len(positions) < max_positions:
                result = engine.play(board, limit=chess.engine.Limit(depth=EVAL_DEPTH))
                board.push(result.move)
                if 10 < board.fullmove_number < 60 and len(board.piece_map()) > 8:
                    positions.append(board.fen())
            if len(positions) >= max_positions:
                break
    return positions


def worker(fens):
    X_batch, y_batch = [], []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for fen in fens:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=EVAL_DEPTH))
            score = info["score"].white().score(mate_score=100000) or 0
            norm = max(min(score / 600, 1), -1)
            X_batch.append(board_to_input_array(board))
            y_batch.append(norm)
    return X_batch, y_batch


def save_batch(X, y, batch_idx):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"batch_{batch_idx}.npz")
    np.savez_compressed(filename, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.float32))
    print(f"✅ Сохранено: {filename} ({len(X)} позиций)")


def generate_dataset():
    print("📥 Извлечение позиций из PGN...")
    pgn_fens = extract_positions_from_pgn(PGN_FILE, MAX_POSITIONS_FROM_PGN)
    print(f"PGN-позиций: {len(pgn_fens)}")

    print("♟ Генерация self-play позиций...")
    selfplay_fens = generate_selfplay_positions(NUM_SELFPLAY_GAMES, MAX_POSITIONS_SELFPLAY)
    print(f"Self-play позиций: {len(selfplay_fens)}")

    fens = pgn_fens + selfplay_fens
    random.shuffle(fens)
    print(f"Всего позиций: {len(fens)}")

    cpu = mp.cpu_count()
    pool = mp.Pool(cpu)
    batch_idx = 0

    for i in range(0, len(fens), BATCH_SIZE):
        chunk = fens[i:i + BATCH_SIZE]
        subchunks = np.array_split(chunk, cpu)
        results = pool.map(worker, subchunks)

        X, y = [], []
        for Xc, yc in results:
            X.extend(Xc)
            y.extend(yc)

        save_batch(X, y, batch_idx)
        batch_idx += 1

    pool.close()
    pool.join()
    print("🎉 Генерация завершена! Все батчи сохранены в", OUTPUT_DIR)


if __name__ == "__main__":
    generate_dataset()
