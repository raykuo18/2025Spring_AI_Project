import chess
import chess.engine
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore
import numpy as np

# --- CONFIGURATION ---
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Change as needed
BERT_MODEL = 'all-MiniLM-L6-v2'

# --- Initialize Evaluators ---
stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
bert_model = SentenceTransformer(BERT_MODEL)

# --- Model Loader ---
def load_model(model_type="tinyllama"):
    if model_type == "tinyllama":
        model = Llama(model_path="./models/tinyllama.gguf", n_ctx=4096)
    elif model_type == "capybarahermes":
        model = Llama(model_path="./models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf", n_ctx=4096)
    else:
        raise ValueError("Unknown model type.")
    
    def inference_fn(prompt):
        output = model(prompt, max_tokens=150, stop=["\n\n"])["choices"][0]["text"]
        return output.strip()

    return inference_fn

# --- Helper functions ---
def parse_response(response):
    try:
        parts = response.split("Move:")
        explanation = parts[0].replace("Reasoning:", "").strip()
        move = parts[1].strip().split()[0]
        return explanation, move
    except:
        return response, "0000"

def get_move_score(board, move_uci):
    try:
        move = chess.Move.from_uci(move_uci)
        info = stockfish.analyse(board, chess.engine.Limit(depth=10), root_moves=[move])
        return info["score"].white().score(mate_score=10000)
    except:
        return -10000

def get_top_k_moves(board, k=3):
    infos = stockfish.analyse(board, chess.engine.Limit(depth=10), multipv=k)
    return [info["pv"][0].uci() for info in infos]

def compute_perplexity(text):
    tokens = text.split()
    unique_tokens = set(tokens)
    if len(unique_tokens) == 0:
        return float('inf')
    return np.exp(len(tokens)/len(unique_tokens))

def local_llm_coherence_eval(move, explanation, inference_fn):
    prompt = f"""You are a chess assistant. Check if the explanation matches the move.

Move: {move}
Explanation: {explanation}

Question: Is the explanation consistent with the move? Answer only "Yes" or "No":
Answer:"""
    
    answer = inference_fn(prompt).lower()
    return answer.startswith("yes")

# --- Evaluate Single Position ---
def evaluate_position(fen, inference_fn):
    board = chess.Board(fen)
    prompt = f"Position: {fen}\nExplain your reasoning and best move:\nReasoning:"
    response = inference_fn(prompt)

    explanation, move = parse_response(response)
    move = move.strip()

    # Move Quality
    info = stockfish.analyse(board, chess.engine.Limit(depth=15))
    best_move = info["pv"][0].uci()
    best_score = info["score"].white().score(mate_score=10000)
    chosen_move_score = get_move_score(board, move)

    ssd = best_score - chosen_move_score
    top_moves = get_top_k_moves(board, 3)
    top3_agreement = int(move in top_moves)

    # Explanation Quality (BERTScore)
    reference_exp = f"The best move is {best_move}, improving position strategically."
    P, _, _ = bertscore([explanation], [reference_exp], lang='en')
    bert_sim = P.mean().item()

    # Fluency (Perplexity)
    perplexity = compute_perplexity(explanation)

    # Move-Explanation Coherence (Local LLM)
    coherence = local_llm_coherence_eval(move, explanation, inference_fn)

    return {
        "ssd": ssd,
        "top3_agreement": top3_agreement,
        "bert_sim": bert_sim,
        "perplexity": perplexity,
        "coherence": coherence
    }

# --- Main Evaluation ---
if __name__ == "__main__":
    positions = [
        "r1bqkbnr/pppp1ppp/2n5/4p3/1bP5/5NP1/PP1PPPBP/RNBQ1RK1 w kq - 0 4",
        "r2q1rk1/pp1bbppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
        # Add more FEN positions here
    ]

    methods = {
        "TinyLlama": "tinyllama",
        "CapybaraHermes (Simulating Mixtral)": "capybarahermes",
        "Stockfish + TinyLlama": "tinyllama",
        "Stockfish + CapybaraHermes": "capybarahermes",
    }

    for method_name, model_type in methods.items():
        print(f"\nEvaluating {method_name}")
        inference_fn = load_model(model_type)
        all_results = []

        for fen in positions:
            if "Stockfish" in method_name:
                board = chess.Board(fen)
                best_move = stockfish.play(board, chess.engine.Limit(depth=15)).move.uci()
                prompt = f"Position: {fen}\nExplain the move {best_move}:\nReasoning:"
                response = inference_fn(prompt)
                explanation = response.strip()
                move = best_move
            else:
                eval_res = evaluate_position(fen, inference_fn)
                all_results.append(eval_res)
                continue  # Already evaluated inside evaluate_position

            # Evaluate Stockfish methods
            chosen_move_score = get_move_score(board, move)
            best_score = chosen_move_score  # Move is from Stockfish, so best score
            ssd = 0
            top3_agreement = 1

            # Explanation Quality
            reference_exp = f"The best move is {move}, improving position strategically."
            P, _, _ = bertscore([explanation], [reference_exp], lang='en')
            bert_sim = P.mean().item()
            perplexity = compute_perplexity(explanation)

            coherence = local_llm_coherence_eval(move, explanation, inference_fn)

            all_results.append({
                "ssd": ssd,
                "top3_agreement": top3_agreement,
                "bert_sim": bert_sim,
                "perplexity": perplexity,
                "coherence": coherence
            })

        # Aggregate and print results
        avg_ssd = np.mean([r["ssd"] for r in all_results])
        top3_acc = np.mean([r["top3_agreement"] for r in all_results])
        avg_bert = np.mean([r["bert_sim"] for r in all_results])
        avg_perp = np.mean([r["perplexity"] for r in all_results])
        coherence_rate = np.mean([r["coherence"] for r in all_results])

        print(f"Avg Stockfish Delta (lower is better): {avg_ssd:.2f}")
        print(f"Top-3 Move Accuracy: {top3_acc:.2f}")
        print(f"Explanation Similarity (BERTScore): {avg_bert:.2f}")
        print(f"Explanation Fluency (Perplexity, lower better): {avg_perp:.2f}")
        print(f"Move-Explanation Coherence: {coherence_rate:.2f}")

    stockfish.quit()
