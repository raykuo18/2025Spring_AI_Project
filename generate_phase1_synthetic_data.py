import chess
import random
import pandas as pd
from tqdm import tqdm
import numpy as np

# Number of examples per task
NUM_EXAMPLES = 1000  # Adjust as you wish

# === Helper: generate random realistic boards ===
def random_board(max_halfmoves=30, min_halfmoves=10):
    board = chess.Board()
    num_moves = random.randint(min_halfmoves, max_halfmoves)
    
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break  # Dead position
        move = random.choice(legal_moves)
        board.push(move)

        if board.is_game_over():
            break

    return board

# === Task A: Move Legality Prediction ===
def generate_legality_data(num_examples):
    data = []
    while len(data) < num_examples:
        board = random_board()
        fen = board.fen()
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            continue  # Skip dead boards

        if random.random() < 0.5:
            move = random.choice(legal_moves)
            label = "legal"
        else:
            illegal_move = chess.Move.from_uci("a1h8")
            label = "illegal" if illegal_move not in legal_moves else "legal"
            move = illegal_move

        prompt = f"Given chess position: {fen}. Is the move {move.uci()} legal? Answer:"
        completion = label

        data.append({"prompt": prompt, "completion": completion})

    return data

# === Task B: Simple Tactical Motifs ===
def generate_tactical_data(num_examples):
    simple_puzzles = [
        ("2r3k1/5ppp/p7/8/8/4Q3/PPP2PPP/2K5 w - - 0 1", "e3e8"),
        ("8/8/8/8/8/8/KP6/k7 w - - 0 1", "b2b4"),
        # Add more simple puzzles here
    ]
    data = []
    for _ in tqdm(range(num_examples), desc="Generating tactical puzzles"):
        fen, best_move = random.choice(simple_puzzles)
        prompt = f"Given chess position: {fen}. What is the best move? Answer:"
        completion = best_move
        data.append({"prompt": prompt, "completion": completion})
    return data

# === Task C: Special Moves Detection ===
def generate_special_move_data(num_examples):
    data = []
    for _ in tqdm(range(num_examples), desc="Generating special moves data"):
        board = random_board()
        fen = board.fen()

        special_moves = []
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            special_moves.append("castling")
        if board.ep_square:
            special_moves.append("en passant")
        if any(move.promotion for move in board.legal_moves):
            special_moves.append("promotion")

        special_label = ", ".join(special_moves) if special_moves else "none"
        prompt = f"Given chess position: {fen}. What special moves are available? Answer:"
        completion = special_label

        data.append({"prompt": prompt, "completion": completion})

    return data

# === Combine and Save All Data ===
def generate_and_save_combined():
    legality_data = generate_legality_data(NUM_EXAMPLES)
    tactical_data = generate_tactical_data(NUM_EXAMPLES)
    special_move_data = generate_special_move_data(NUM_EXAMPLES)

    combined_data = legality_data + tactical_data + special_move_data
    random.shuffle(combined_data)  # Shuffle for better mixing during training

    df = pd.DataFrame(combined_data)
    df.to_csv("training_data/phase1/phase1_synthetic_data.csv", index=False)
    print("✅ Combined dataset generated and saved as 'phase1_synthetic_data.csv'.")

if __name__ == "__main__":
    generate_and_save_combined()
