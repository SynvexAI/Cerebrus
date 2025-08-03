import chess
import chess.engine
import chess.pgn
import numpy as np
import multiprocessing as mp
from pathlib import Path
import random
import io
import zstandard as zstd

STOCKFISH_PATH = "dataset/stockfish.exe"
PGN_FILE = "dataset/lichess_db_standard_rated_2014-02.pgn.zst"
OUTPUT_FILE = "dataset/chess_dataset.npz"
EVAL_DEPTH = 12
NUM_SELFPLAY_GAMES = 1000
MAX_POSITIONS_FROM_PGN = 10000
MAX_POSITIONS_SELFPLAY = 10000

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
        score = info.get("score").white().score(mate_score=100000) or 0
    return score


def process_chunk(fens):
    Xc, yc = [], []
    for fen in fens:
        score = evaluate_position(fen)
        norm = np.tanh(score / 1000.0)
        Xc.append(board_to_input_array(chess.Board(fen)))
        yc.append(norm)
    return Xc, yc


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
                if board.is_game_over():
                    break
                positions.append(board.fen())
                if len(positions) >= max_positions:
                    break
    return positions


def generate_selfplay_positions(num_games, max_positions):
    positions = []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for g in range(num_games):
            board = chess.Board()
            while not board.is_game_over() and len(positions) < max_positions:
                result = engine.play(board, limit=chess.engine.Limit(depth=EVAL_DEPTH))
                board.push(result.move)
                positions.append(board.fen())
            if len(positions) >= max_positions:
                break
    return positions


def generate_dataset(num_samples_pgn, num_games, samples_selfplay):
    print("Извлечение позиций из PGN...")
    pgn_fens = extract_positions_from_pgn(PGN_FILE, num_samples_pgn)
    print(f"PGN-позиций: {len(pgn_fens)}")

    print("Генерация позиций через self-play Stockfish vs Stockfish...")
    selfplay_fens = generate_selfplay_positions(num_games, samples_selfplay)
    print(f"Self-play позиций: {len(selfplay_fens)}")

    fens = pgn_fens + selfplay_fens
    random.shuffle(fens)
    print(f"Всего позиций для оценки: {len(fens)}")

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
    print(f"Сохранено: {OUTPUT_FILE} (X: {X.shape}, y: {y.shape})")


if __name__ == "__main__":
    generate_dataset(
        num_samples_pgn=MAX_POSITIONS_FROM_PGN,
        num_games=NUM_SELFPLAY_GAMES,
        samples_selfplay=MAX_POSITIONS_SELFPLAY
    )
