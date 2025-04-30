# === generate_phase2_moves_and_explanations.py ===

"""
This script generates three datasets:
1. Move Only: Given FEN, ask model for best move.
2. Explanation Only: Given FEN, ask model for an explanation.
3. Move + Explanation: Given FEN, ask model for move and explanation.

Outputs:
- move_only.jsonl
- explanation_only.jsonl
- move_plus_explanation.jsonl
"""

import chess
import chess.engine
import random
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import json

# === Settings ===
MODEL_PATH = "llm-models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
STOCKFISH_PATH = "src/stockfish/stockfish-macos-m1-apple-silicon"
NUM_EXAMPLES = 5

# === Utilities ===
def random_board(min_half=10, max_half=30):
    board = chess.Board()
    for _ in range(random.randint(min_half, max_half)):
        legal_moves = list(board.legal_moves)
        if not legal_moves: break
        board.push(random.choice(legal_moves))
        if board.is_game_over(): break
    return board

def build_prompt(fen, prompt_type):
    fen = fen.strip()
    if prompt_type == "move_only":
        return f"Given this chess board (FEN): {fen}. What is the best next move? Answer in UCI format."
    elif prompt_type == "explanation_only":
        return f"Given this chess board (FEN): {fen}. Explain what the current player should aim for strategically and tactically."
    elif prompt_type == "move_plus_explanation":
        return f"Given this chess board (FEN): {fen}\nPlease respond in the following format:\nMove: [UCI move]\nExplanation: [text explaining the move]."
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

def parse_move_from_text(text):
    try:
        lines = text.splitlines()
        for line in lines:
            if "move:" in line.lower():
                move_part = line.split(":")[1].strip()
                if len(move_part) == 4:
                    return move_part.lower()
    except:
        pass
    return None

def parse_explanation_from_text(text):
    try:
        lines = text.splitlines()
        for line in lines:
            if "explanation:" in line.lower():
                explanation_part = line.split(":", 1)[1].strip()
                return explanation_part
    except:
        pass
    return None

def evaluate_move(board, move_uci, engine):
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return "invalid"
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info["score"].white().score(mate_score=10000)
        board.pop()
        return score
    except:
        return "invalid"

# === Main ===
if __name__ == "__main__":
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=20,
        use_mlock=True,
        use_mmap=True
    )

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    move_only_path = open("training_data/phase2/move_only.jsonl", "w")
    explanation_only_path = open("training_data/phase2/explanation_only.jsonl", "w")
    move_plus_path = open("training_data/phase2/move_plus_explanation.jsonl", "w")

    for _ in tqdm(range(NUM_EXAMPLES), desc="Generating examples"):
        board = random_board()
        fen = board.fen()

        # Move Only
        prompt = build_prompt(fen, "move_only")
        response = llm(prompt, max_tokens=100, temperature=0.5)["choices"][0]["text"].strip()
        parsed_move = parse_move_from_text(response)
        move_valid = parsed_move in [m.uci() for m in board.legal_moves] if parsed_move else False
        move_score = evaluate_move(board, parsed_move, engine) if move_valid else "invalid"
        move_only_path.write(json.dumps({
            "fen": fen,
            "prompt": prompt,
            "response": response,
            "move_parsed": parsed_move,
            "move_valid": move_valid,
            "move_score": move_score
        }) + "\n")

        # Explanation Only
        prompt = build_prompt(fen, "explanation_only")
        response = llm(prompt, max_tokens=200, temperature=0.5)["choices"][0]["text"].strip()
        explanation_only_path.write(json.dumps({
            "fen": fen,
            "prompt": prompt,
            "response": response
        }) + "\n")

        # Move + Explanation
        prompt = build_prompt(fen, "move_plus_explanation")
        response = llm(prompt, max_tokens=250, temperature=0.5)["choices"][0]["text"].strip()
        parsed_move = parse_move_from_text(response)
        parsed_explanation = parse_explanation_from_text(response)
        move_valid = parsed_move in [m.uci() for m in board.legal_moves] if parsed_move else False
        move_score = evaluate_move(board, parsed_move, engine) if move_valid else "invalid"
        explanation_valid = parsed_explanation is not None
        move_plus_path.write(json.dumps({
            "fen": fen,
            "prompt": prompt,
            "response": response,
            "move_parsed": parsed_move,
            "move_valid": move_valid,
            "move_score": move_score,
            "explanation_parsed": parsed_explanation,
            "explanation_valid": explanation_valid
        }) + "\n")

    move_only_path.close()
    explanation_only_path.close()
    move_plus_path.close()

    print("✅ Datasets saved as JSONL files:")
    print("  - move_only.jsonl")
    print("  - explanation_only.jsonl")
    print("  - move_plus_explanation.jsonl")

    engine.quit()
