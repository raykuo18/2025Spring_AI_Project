import random
import json
import os
import chess.pgn
import requests
from stockfish import Stockfish

# ====== Configuration ======
HF_API_KEY = ""  # Get from https://huggingface.co/settings/tokens
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
PGN_PATH = "short_example.pgn"
NUM_SAMPLES = 1
USE_STOCKFISH = True
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"

# ====== Hugging Face API Setup ======
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_hf_for_move_and_explanation(fen):
    prompt = f"""
Given the following chess position in FEN: {fen}
Suggest the best next move and explain why in simple terms.
Format:
move: <move_in_uci>
explanation: <your explanation>
    """.strip()

    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.7, "max_new_tokens": 150},
        "options": {"use_cache": True}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()

        if isinstance(output, list) and "generated_text" in output[0]:
            full_output = output[0]["generated_text"]
            new_text = full_output[len(prompt):].strip()
            return new_text  # Only return new generation, not the repeated prompt
        else:
            print("Unexpected response format:", output)
            return None
    except Exception as e:
        print(f"HF API Error: {e}")
        return None

def parse_response(fen, response):
    try:
        lines = response.strip().split('\n')
        move = lines[0].split("move: ")[-1].strip()
        explanation = lines[1].split("explanation: ")[-1].strip()
        return {'fen': fen, 'hf_move': move, 'explanation': explanation}
    except Exception as e:
        print(f"Parse Error: {e}")
        return None

def get_stockfish_move(fen):
    if not USE_STOCKFISH:
        return None
    stockfish.set_fen_position(fen)
    return stockfish.get_best_move()

def generate_fen_samples(pgn_path, num_samples=100):
    games = []
    with open(pgn_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            games.append(game)
    random.shuffle(games)

    fen_positions = []
    for game in games[:num_samples]:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            if board.is_game_over(): break
            fen_positions.append(board.fen())
    return fen_positions

def main():
    if USE_STOCKFISH:
        global stockfish
        stockfish = Stockfish(path=STOCKFISH_PATH)

    fens = generate_fen_samples(PGN_PATH, NUM_SAMPLES)
    dataset = []

    for i, fen in enumerate(fens):
        print(f"[{i+1}/{len(fens)}] Querying HuggingFace for FEN...")
        response = query_hf_for_move_and_explanation(fen)
        if response:
            parsed = parse_response(fen, response)
            if parsed:
                if USE_STOCKFISH:
                    parsed['stockfish_move'] = get_stockfish_move(fen)
                dataset.append(parsed)

    os.makedirs("output", exist_ok=True)
    with open("output/augmented_dataset_2.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Saved {len(dataset)} entries to output/augmented_dataset.json")

if __name__ == "__main__":
    main()
