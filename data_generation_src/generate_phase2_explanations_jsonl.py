# === generate_phase2_moves_and_explanations.py ===

"""
This script generates three datasets:
1. Move Only: Given FEN, ask model for best move.
2. Explanation Only: Given FEN, ask model for an explanation.
3. Move + Explanation: Given FEN, ask model for move and explanation.

Also collects statistics and saves top-K Stockfish moves.

Outputs:
- move_only.jsonl
- explanation_only.jsonl
- move_plus_explanation.jsonl
- stockfish_topk.jsonl
"""

import chess
import chess.engine
import random
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import json
import re
from collections import defaultdict

# === Settings ===
MODEL_PATH = "llm-models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
STOCKFISH_PATH = "src/stockfish/stockfish-macos-m1-apple-silicon"
NUM_EXAMPLES = 5
TOP_K = 5
FILE_NAME_PREFIX = "new_prompt_and_parse_"

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
        return (
            f"You are a chess engine. Given the position below, output the best next move "
            f"in strictly 4-character UCI format, with no explanation or extra characters.\n"
            f"FEN: {fen}\n"
            f"Output Format Example: e2e4\n"
            f"Move:")
    elif prompt_type == "explanation_only":
        return f"You are a chess coach. Given this position (FEN): {fen}, explain what the current player should aim for tactically and strategically."
    elif prompt_type == "move_plus_explanation":
        return (
            f"You are a chess tutor. Given this position (FEN): {fen}, return a JSON response in this format:\n"
            f"{{\"move\": \"e2e4\", \"explanation\": \"this move controls the center...\"}}\n"
            f"Only output valid JSON."
        )
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

def parse_move_from_text(text, board):
    try:
        if text.strip().startswith("{"):
            parsed = json.loads(text)
            move = parsed.get("move", "").strip().lower()
            if len(move) == 4 and move in [m.uci() for m in board.legal_moves]:
                return move
        uci_candidates = re.findall(r'\b[a-h][1-8][a-h][1-8]\b', text.lower())
        for cand in uci_candidates:
            try:
                move = chess.Move.from_uci(cand)
                if move in board.legal_moves:
                    return cand
            except:
                continue
    except:
        pass
    return None

def parse_explanation_from_text(text):
    try:
        if text.strip().startswith("{"):
            parsed = json.loads(text)
            return parsed.get("explanation", None)
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

def get_top_k_moves(board, engine, k):
    result = []
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.5), multipv=k)
        if isinstance(info, list):
            for entry in info:
                result.append({
                    "move": entry["pv"][0].uci(),
                    "score": entry["score"].white().score(mate_score=10000)
                })
    except:
        pass
    return result

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

    move_only_path = open(f"training_data/phase2/explanations/{FILE_NAME_PREFIX}move_only.jsonl", "w")
    explanation_only_path = open(f"training_data/phase2/explanations/{FILE_NAME_PREFIX}explanation_only.jsonl", "w")
    move_plus_path = open(f"training_data/phase2/explanations/{FILE_NAME_PREFIX}move_plus_explanation.jsonl", "w")
    topk_path = open(f"training_data/phase2/explanations/{FILE_NAME_PREFIX}stockfish_topk.jsonl", "w")

    stats_move_only = defaultdict(int)
    stats_move_plus = defaultdict(int)
    valid_scores_move_only = []
    valid_scores_move_plus = []
    in_top_k_move_only = 0
    in_top_k_move_plus = 0
    move_only_count = 0
    move_plus_count = 0

    for _ in tqdm(range(NUM_EXAMPLES), desc="Generating examples"):
        board = random_board()
        fen = board.fen()
        top_k_moves = get_top_k_moves(board, engine, TOP_K)
        top_k_set = set(m["move"] for m in top_k_moves)
        topk_path.write(json.dumps({"fen": fen, "top_k_moves": top_k_moves}) + "\n")

        # Move Only
        prompt = build_prompt(fen, "move_only")
        response = llm(prompt, max_tokens=100, temperature=0.5)["choices"][0]["text"].strip()
        parsed_move = parse_move_from_text(response, board)
        is_valid = parsed_move in [m.uci() for m in board.legal_moves] if parsed_move else False
        move_score = evaluate_move(board, parsed_move, engine) if is_valid else "invalid"
        move_only_count += 1
        if parsed_move: stats_move_only["parsed"] += 1
        if is_valid:
            stats_move_only["valid"] += 1
            if isinstance(move_score, int):
                valid_scores_move_only.append(move_score)
            if parsed_move in top_k_set:
                in_top_k_move_only += 1
        move_only_path.write(json.dumps({
            "fen": fen,
            "prompt": prompt,
            "response": response,
            "move_parsed": parsed_move,
            "move_valid": is_valid,
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
        parsed_move = parse_move_from_text(response, board)
        parsed_explanation = parse_explanation_from_text(response)
        move_valid = parsed_move in [m.uci() for m in board.legal_moves] if parsed_move else False
        move_score = evaluate_move(board, parsed_move, engine) if move_valid else "invalid"
        move_plus_count += 1
        if parsed_move: stats_move_plus["parsed"] += 1
        if move_valid:
            stats_move_plus["valid"] += 1
            if isinstance(move_score, int):
                valid_scores_move_plus.append(move_score)
            if parsed_move in top_k_set:
                in_top_k_move_plus += 1
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
    topk_path.close()

    engine.quit()
    
    # Finalize and save statistics
    stats_output_move_only = {
        "total_examples": move_only_count,
        "parsed_moves": stats_move_only["parsed"],
        "valid_moves": stats_move_only["valid"],
        "parse_rate": stats_move_only["parsed"] / move_only_count if move_only_count else 0,
        "valid_rate": stats_move_only["valid"] / move_only_count if move_only_count else 0,
        "in_top_k": in_top_k_move_only,
        "in_top_k_rate": in_top_k_move_only / move_only_count if move_only_count else 0,
        "average_score": sum(valid_scores_move_only) / len(valid_scores_move_only) if valid_scores_move_only else None,
        "score_min": min(valid_scores_move_only) if valid_scores_move_only else None,
        "score_max": max(valid_scores_move_only) if valid_scores_move_only else None
    }

    stats_output_move_plus = {
        "total_examples": move_plus_count,
        "parsed_moves": stats_move_plus["parsed"],
        "valid_moves": stats_move_plus["valid"],
        "parse_rate": stats_move_plus["parsed"] / move_plus_count if move_plus_count else 0,
        "valid_rate": stats_move_plus["valid"] / move_plus_count if move_plus_count else 0,
        "in_top_k": in_top_k_move_plus,
        "in_top_k_rate": in_top_k_move_plus / move_plus_count if move_plus_count else 0,
        "average_score": sum(valid_scores_move_plus) / len(valid_scores_move_plus) if valid_scores_move_plus else None,
        "score_min": min(valid_scores_move_plus) if valid_scores_move_plus else None,
        "score_max": max(valid_scores_move_plus) if valid_scores_move_plus else None
    }

    with open(f"training_data/phase2/explanations/{FILE_NAME_PREFIX}generation_stats_move_only.json", "w") as stats_file:
        json.dump(stats_output_move_only, stats_file, indent=2)

    with open(f"training_data/phase2/explanations/{FILE_NAME_PREFIX}generation_stats_move_plus_explanation.json", "w") as stats_file:
        json.dump(stats_output_move_plus, stats_file, indent=2)

    print("✅ Datasets saved as JSONL files:")
    print(f"  - {FILE_NAME_PREFIX}move_only.jsonl")
    print(f"  - {FILE_NAME_PREFIX}explanation_only.jsonl")
    print(f"  - {FILE_NAME_PREFIX}move_plus_explanation.jsonl")
    print(f"  - {FILE_NAME_PREFIX}stockfish_topk.jsonl")
    print("\n✅ Statistics saved to JSON:")
    print(f"  - {FILE_NAME_PREFIX}generation_stats_move_only.json")
    print(f"  - {FILE_NAME_PREFIX}generation_stats_move_plus_explanation.json")
