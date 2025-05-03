import pandas as pd
import chess
import chess.engine
import os
import argparse
import random

INPUT_FILE = "training_data/lichess_db_puzzle.csv"
OUTPUT_FILE = "training_data/phase1/phase1_endgame_dataset.csv"
NUM_SAMPLES = 100
STOCKFISH_PATH = "src/stockfish/stockfish-macos-m1-apple-silicon"  # Adjust to your stockfish binary path


def apply_moves(fen, moves):
    """Apply moves to starting FEN, return resulting board."""
    board = chess.Board(fen)
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


def get_stockfish_best_move(board, engine):
    """Query stockfish for best move from given board."""
    result = engine.analyse(board, chess.engine.Limit(depth=15))
    return result['pv'][0].uci() if 'pv' in result else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-stockfish", action="store_true",
                        help="Enable validation using stockfish engine.")
    args = parser.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file '{INPUT_FILE}' not found.")
        return

    print(f"✅ Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    print(f"✅ Original dataset contains {len(df)} puzzles.")

    sampled_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)

    engine = None
    if args.check_stockfish:
        if not os.path.exists(STOCKFISH_PATH):
            print(f"❌ Stockfish not found at {STOCKFISH_PATH}.")
            return
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print(f"✅ Stockfish engine loaded from {STOCKFISH_PATH}.")

    records = []
    skipped = 0
    mismatched = 0

    for idx, row in sampled_df.iterrows():
        fen_start = row['FEN']
        moves = row['Moves'].split()

        if len(moves) < 1:
            skipped += 1
            continue

        if len(moves) == 1:
            fen_before_last = fen_start
            last_move = moves[0]
        else:
            try:
                board_before_last = apply_moves(fen_start, moves[:-1])
            except Exception as e:
                print(f"⚠️ Move application error at row {idx}: {e}")
                skipped += 1
                continue
            fen_before_last = board_before_last.fen()
            last_move = moves[-1]

        # If stockfish check is enabled, validate move
        if engine:
            try:
                stockfish_move = get_stockfish_best_move(board_before_last, engine)
            except Exception as e:
                print(f"⚠️ Stockfish error at row {idx}: {e}")
                skipped += 1
                continue

            if stockfish_move != last_move:
                mismatched += 1
                continue  # Skip if stockfish doesn't agree

        prompt = f"What is the best move in this position? FEN: {fen_before_last}"
        completion = f" {last_move}"

        records.append({
            "prompt": prompt,
            "completion": completion,
            "theme": row['Themes'],
            "rating": row['Rating']
        })

    if engine:
        engine.quit()

    print(f"✅ Generated {len(records)} examples (skipped {skipped}, stockfish mismatch {mismatched}).")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved dataset to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
