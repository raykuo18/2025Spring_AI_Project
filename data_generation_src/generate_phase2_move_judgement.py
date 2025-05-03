# === generate_phase2_move_judgement.py ===
"""
This script generates two datasets:
1. move_judgement_random.jsonl
   - Given a FEN and a random move (valid or invalid), the model must judge the move and explain.
   - Includes: Stockfish eval, best move, top-5 avg score

2. move_judgement_topk.jsonl
   - Given a FEN and a top-K move (sampled), the model must judge and explain.
   - Includes: Stockfish score, top-5 avg score

Also collects statistics and saves JSON logs.
"""

import chess
import chess.engine
import random
import json
import re
from tqdm import tqdm
from llama_cpp import Llama

MODEL_PATH = "llm-models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
STOCKFISH_PATH = "src/stockfish/stockfish-macos-m1-apple-silicon"
NUM_EXAMPLES = 5
TOP_K = 5
SAVE_PREFIX = "training_data/phase2/judgements/move_judgement_"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=20,
    use_mlock=True,
    use_mmap=True
)

def get_top_k_moves(board, engine, k=5):
    topk = []
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.3), multipv=k)
        for entry in info:
            topk.append({
                "move": entry["pv"][0].uci(),
                "score": entry["score"].white().score(mate_score=10000)
            })
    except:
        pass
    return topk

def evaluate(board, move, engine):
    try:
        if move not in board.legal_moves:
            return "illegal"
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info["score"].white().score(mate_score=10000)
        board.pop()
        return score
    except:
        return "error"

def query_llm(prompt):
    try:
        output = llm(prompt, max_tokens=256, temperature=0.5)["choices"][0]["text"].strip()
        return output
    except:
        return ""

def generate_prompt(fen, move):
    return (
        f"You are a chess coach. Given the board position below and a candidate move, evaluate the quality of the move.\n"
        f"Respond in JSON with two fields: 'verdict' (excellent/good/dubious/blunder) and 'explanation'.\n"
        f"FEN: {fen}\nMove: {move}\n"
        f"Respond with: {{\"verdict\": ..., \"explanation\": ...}}"
    )

def parse_verdict_from_output(output):
    try:
        parsed = json.loads(output)
        return parsed.get("verdict", "unknown").lower()
    except:
        output = re.sub(r"[^a-z]", " ", output.lower()).strip()
        if "excellent" in output:
            return "excellent"
        elif "good" in output:
            return "good"
        elif "dubious" in output:
            return "dubious"
        elif "blunder" in output:
            return "blunder"
        else:
            return "parse_error"

def main():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    f_random = open(SAVE_PREFIX + "random.jsonl", "w")
    f_topk = open(SAVE_PREFIX + "topk.jsonl", "w")

    stats_random = {"total": 0, "illegal": 0, "score_sum": 0, "llm_empty": 0, "verdict_count": {}}
    stats_topk = {"total": 0, "score_sum": 0, "llm_empty": 0, "verdict_count": {}}

    for _ in tqdm(range(NUM_EXAMPLES), desc="Generating judgement examples"):
        board = chess.Board()
        for _ in range(random.randint(10, 30)):
            if board.is_game_over(): break
            board.push(random.choice(list(board.legal_moves)))

        fen = board.fen()
        legal = list(board.legal_moves)
        topk = get_top_k_moves(board, engine, TOP_K)
        best = topk[0]["move"] if topk else None
        avg_top5 = sum([x["score"] for x in topk]) / len(topk) if topk else None

        # --- Random Move ---
        rand_move = random.choice(legal + [chess.Move.null()])
        score = evaluate(board, rand_move, engine)
        prompt = generate_prompt(fen, rand_move.uci())
        output = query_llm(prompt)
        verdict = parse_verdict_from_output(output)
        stats_random["total"] += 1
        if score == "illegal": stats_random["illegal"] += 1
        if isinstance(score, int): stats_random["score_sum"] += score
        if output == "": stats_random["llm_empty"] += 1
        stats_random["verdict_count"][verdict] = stats_random["verdict_count"].get(verdict, 0) + 1
        f_random.write(json.dumps({
            "fen": fen,
            "move": rand_move.uci(),
            "prompt": prompt,
            "llm_output": output,
            "parsed_verdict": verdict,
            "stockfish_score": score,
            "stockfish_best": best,
            "top5_avg": avg_top5
        }) + "\n")

        # --- Top-K Sample ---
        topk_move = random.choice(topk)["move"] if topk else best
        move_obj = chess.Move.from_uci(topk_move)
        score = evaluate(board, move_obj, engine)
        prompt = generate_prompt(fen, topk_move)
        output = query_llm(prompt)
        verdict = parse_verdict_from_output(output)
        stats_topk["total"] += 1
        if isinstance(score, int): stats_topk["score_sum"] += score
        if output == "": stats_topk["llm_empty"] += 1
        stats_topk["verdict_count"][verdict] = stats_topk["verdict_count"].get(verdict, 0) + 1
        f_topk.write(json.dumps({
            "fen": fen,
            "move": topk_move,
            "prompt": prompt,
            "llm_output": output,
            "parsed_verdict": verdict,
            "stockfish_score": score,
            "top5_avg": avg_top5
        }) + "\n")

    f_random.close()
    f_topk.close()
    engine.quit()

    # Save statistics
    with open(SAVE_PREFIX + "random_stats.json", "w") as f:
        json.dump(stats_random, f, indent=2)
    with open(SAVE_PREFIX + "topk_stats.json", "w") as f:
        json.dump(stats_topk, f, indent=2)

    print("\n✅ Completed generation and saved statistics:")
    print(f"  - {SAVE_PREFIX}random.jsonl")
    print(f"  - {SAVE_PREFIX}topk.jsonl")
    print(f"  - {SAVE_PREFIX}random_stats.json")
    print(f"  - {SAVE_PREFIX}topk_stats.json")

if __name__ == "__main__":
    main()
