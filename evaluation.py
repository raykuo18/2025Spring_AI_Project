#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import chess
import chess.engine
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from bert_score import score as bert_score_calculate # BERTScore
from typing import List, Dict, Optional, Any

# --- Model Configuration Mapping (can be shared with training script) ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "add_eos_token": True,
    },
    "Gemma-2B": {
        "hf_model_name": "google/gemma-2b",
        "add_eos_token": True,
    },
    "Phi-2": {
        "hf_model_name": "microsoft/phi-2",
        "add_eos_token": True,
    }
}

def get_stockfish_analysis(board: chess.Board, engine: chess.engine.SimpleEngine, 
                           time_limit: Optional[float] = None, depth_limit: Optional[int] = None, 
                           multipv: int = 3) -> Dict[str, Any]:
    """Gets Stockfish analysis for a given board position."""
    limit = None
    if time_limit:
        limit = chess.engine.Limit(time=time_limit)
    elif depth_limit:
        limit = chess.engine.Limit(depth=depth_limit)
    else: # Default if neither is provided
        limit = chess.engine.Limit(time=0.1) # Default to a quick analysis

    try:
        info_list = engine.analyse(board, limit, multipv=multipv) # Get top N moves
        
        results = {"top_moves_uci": [], "top_moves_san": [], "scores_cp": []}
        if not info_list:
            # print(f"Warning: Stockfish returned no analysis for FEN: {board.fen()}")
            return results

        # Current position evaluation (from White's perspective) from the first PV
        current_eval_white_pov = info_list[0].get("score").white().score(mate_score=10000) if info_list[0].get("score") else None
        results["current_eval_cp_white_pov"] = current_eval_white_pov


        for info in info_list:
            if "pv" in info and info["pv"]:
                move = info["pv"][0]
                results["top_moves_uci"].append(move.uci())
                try:
                    results["top_moves_san"].append(board.san(move))
                except ValueError: # If SAN is ambiguous without full context
                    results["top_moves_san"].append(board.variation_san([move]))


                # Score of the position *after* this PV move
                # We need to make the move on a temp board and re-evaluate for SSD accurately
                # Or, use the score provided in the multipv info if it represents score *after* the move
                # The score in multipv usually is for the current board IF that move is played.
                score_obj = info.get("score")
                if score_obj:
                    # Ensure score is from White's perspective for consistency
                    score_cp_white_pov = score_obj.white().score(mate_score=10000)
                    results["scores_cp"].append(score_cp_white_pov) 
                else:
                    results["scores_cp"].append(None) # Should not happen if PV exists

        return results
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, Exception) as e:
        print(f"Stockfish analysis error for FEN {board.fen()}: {e}")
        return {"top_moves_uci": [], "top_moves_san": [], "scores_cp": []}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a base model or a LoRA-adapted model on chess tasks.")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Name of the base model.")
    parser.add_argument("--base_model_cache_dir", type=str, default="./hf_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Optional: Path to the LoRA adapter directory (e.g., a checkpoint or final_lora_adapter).")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the JSONL test file.")
    parser.add_argument("--stockfish_path", type=str, required=True, help="Path to Stockfish executable.")
    parser.add_argument("--stockfish_analysis_time", type=float, default=0.2, help="Time (seconds) for Stockfish analysis per move.")
    parser.add_argument("--top_k_agreement", type=int, nargs='+', default=[1, 3], help="List of k values for top-k agreement (e.g., 1 3).")
    parser.add_argument("--bert_score_model_type", type=str, default=None, help="BERT model for BERTScore (e.g., microsoft/deberta-xlarge-mnli). Default uses library's default.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of test samples for quick evaluation.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length for tokenizer.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA style).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for model inference.")
    parser.add_argument("--output_results_file", type=str, default="evaluation_results.json", help="JSON file to save detailed results and metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")


    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 1. Load Model and Tokenizer ---
    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]

    print(f"Loading tokenizer for {hf_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Important for batch generation
    print("Tokenizer loaded.")

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    
    print(f"Loading base model: {hf_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        cache_dir=args.base_model_cache_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization_config else torch.float16, # Use bfloat16 with QLoRA, float16 otherwise for speed
        device_map="auto"
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.lora_adapter_path:
        print(f"Loading LoRA adapter from: {args.lora_adapter_path}...")
        if not os.path.isdir(args.lora_adapter_path):
             # Try if it's a checkpoint subfolder (e.g. checkpoint-1000)
            if os.path.isdir(os.path.join(args.lora_adapter_path, "adapter_model.safetensors")) or \
               os.path.isdir(os.path.join(args.lora_adapter_path, "adapter_model.bin")):
                 # It might be a full trainer checkpoint. PeftModel.from_pretrained should handle it.
                 pass
            else:
                print(f"Error: LoRA adapter path '{args.lora_adapter_path}' not found or not a valid adapter directory.")
                sys.exit(1)
        try:
            model = PeftModel.from_pretrained(model, args.lora_adapter_path)
            print("LoRA adapter loaded successfully.")
        except Exception as e:
            print(f"Error loading LoRA adapter: {e}. Ensure the path is correct and contains adapter_config.json and adapter_model files.")
            sys.exit(1)
    
    model.eval() # Set to evaluation mode
    print("Model ready for evaluation.")

    # --- 2. Initialize Stockfish Engine ---
    stockfish_engine = None
    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        print(f"Stockfish engine initialized from: {args.stockfish_path}")
    except Exception as e:
        print(f"Error initializing Stockfish: {e}. Move evaluation metrics will be skipped.")
        # Proceed without stockfish for explanation tasks if any

    # --- 3. Load Test Dataset ---
    print(f"Loading test data from: {args.test_file}")
    raw_dataset = load_dataset("json", data_files=args.test_file, split="train") # Assumes JSONL is a "train" split
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        if args.max_eval_samples < len(raw_dataset):
            print(f"Sub-sampling test data to {args.max_eval_samples} samples.")
            raw_dataset = raw_dataset.select(range(args.max_eval_samples))
    print(f"Using {len(raw_dataset)} samples for evaluation.")

    # --- 4. Evaluation Loop ---
    results_per_sample = []
    # Metrics aggregation
    total_ssd_sum = 0
    ssd_count = 0
    top_k_correct = {k: 0 for k in args.top_k_agreement}
    move_prediction_count = 0
    bert_score_preds = []
    bert_score_refs = []
    explanation_count = 0

    # Group data by task type if needed, or process all and filter by task later
    # For now, iterate and dispatch based on task field
    
    # Prepare for batching model inference
    prompts_for_inference = [item['input'] for item in raw_dataset]
    data_items_for_processing = list(raw_dataset) # To keep original items aligned

    generated_outputs_text = []

    for i in tqdm(range(0, len(prompts_for_inference), args.batch_size), desc="Model Inference"):
        batch_prompts = prompts_for_inference[i:i + args.batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(DEVICE)
        
        max_gen_tokens = 10 # For move prediction (UCI is short)
        # Check if any prompt in batch is for explanation to adjust max_gen_tokens
        # This is a simplification; ideally, batch by task type or have adaptive max_gen_tokens
        if any("explain" in p.lower() for p in batch_prompts):
            max_gen_tokens = 150 # Longer for explanations
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False # For deterministic move prediction; use True for creative explanations
            )
        
        # Decode generated part only
        for k_idx, output_ids_tensor in enumerate(outputs):
            prompt_len = inputs["input_ids"][k_idx].ne(tokenizer.pad_token_id).sum().item()
            decoded_output = tokenizer.decode(output_ids_tensor[prompt_len:], skip_special_tokens=True)
            generated_outputs_text.append(decoded_output.strip())


    # Process results
    for idx, item in enumerate(tqdm(data_items_for_processing, desc="Calculating Metrics")):
        task_type = item.get("task", "unknown")
        input_prompt_str = item["input"] # Student prompt
        model_generated_output = generated_outputs_text[idx]
        reference_output = item.get("output") # UCI move or reference explanation
        current_result = {"task_id": item.get("task_id"), "task_type": task_type, "input_prompt": input_prompt_str, "model_output": model_generated_output, "reference_output": reference_output}

        if "predict_move" in task_type and stockfish_engine:
            move_prediction_count += 1
            # Extract FEN from input_prompt_str
            fen_match = re.search(r"\[FEN\]\s*(.*?)\s*\[SEP\]", input_prompt_str)
            if fen_match:
                fen = fen_match.group(1).strip()
                board = chess.Board(fen)
                
                # Model's predicted move (already generated)
                predicted_uci = model_generated_output 
                try:
                    model_move = board.parse_uci(predicted_uci)
                    if model_move not in board.legal_moves: # Check if UCI is legal from current FEN
                        # print(f"Warning: Model predicted illegal UCI '{predicted_uci}' for FEN '{fen}'. Skipping Stockfish eval for this move.")
                        current_result["ssd_cp"] = None
                        current_result["stockfish_top1_uci"] = None
                        current_result["stockfish_top1_eval_cp"] = None
                        for k_val in args.top_k_agreement: current_result[f"in_top_{k_val}"] = False
                        results_per_sample.append(current_result)
                        continue
                except ValueError:
                    # print(f"Warning: Model predicted invalid UCI format '{predicted_uci}'. Skipping Stockfish eval.")
                    current_result["ssd_cp"] = None; # ... (set other stockfish fields to None)
                    results_per_sample.append(current_result)
                    continue

                # Get Stockfish analysis
                sf_analysis = get_stockfish_analysis(board, stockfish_engine, time_limit=args.stockfish_analysis_time, multipv=max(args.top_k_agreement, default=3))
                
                if sf_analysis["top_moves_uci"]:
                    sf_best_move_uci = sf_analysis["top_moves_uci"][0]
                    sf_best_move_eval_cp = sf_analysis["scores_cp"][0] if sf_analysis["scores_cp"] else None
                    current_result["stockfish_top1_uci"] = sf_best_move_uci
                    current_result["stockfish_top1_eval_cp"] = sf_best_move_eval_cp

                    # Calculate SSD
                    if sf_best_move_eval_cp is not None:
                        board_after_model_move = board.copy()
                        board_after_model_move.push(model_move)
                        # Re-evaluate position after model's move with Stockfish consistently
                        info_after_model_move = stockfish_engine.analyse(board_after_model_move, chess.engine.Limit(time=args.stockfish_analysis_time))
                        eval_after_model_move_cp = info_after_model_move.get("score").white().score(mate_score=10000) if info_after_model_move.get("score") else None
                        
                        if eval_after_model_move_cp is not None:
                            # SSD: (Eval after Stockfish's best move) - (Eval after model's move)
                            # Positive means model was worse than Stockfish's best.
                            # All evals are from White's perspective.
                            # We need the score of the position *if* sf_best_move_uci was played.
                            # sf_analysis["scores_cp"][0] IS this score.
                            
                            ssd = None
                            if board.turn == chess.WHITE: # If it was White's turn to move
                                ssd = sf_best_move_eval_cp - eval_after_model_move_cp
                            else: # If it was Black's turn to move (higher white_pov_score is worse for black)
                                ssd = eval_after_model_move_cp - sf_best_move_eval_cp # Loss for black means this value is positive

                            current_result["ssd_cp"] = ssd
                            if ssd is not None: total_ssd_sum += ssd; ssd_count += 1
                        else: current_result["ssd_cp"] = None
                    else: current_result["ssd_cp"] = None

                    # Top-k agreement
                    for k_val in args.top_k_agreement:
                        is_in_top_k = predicted_uci in sf_analysis["top_moves_uci"][:k_val]
                        current_result[f"in_top_{k_val}"] = is_in_top_k
                        if is_in_top_k: top_k_correct[k_val] += 1
                else: # Stockfish failed to give moves
                    current_result["ssd_cp"] = None

            elif "explain" in task_type.lower() and reference_output: # Assuming explanation tasks have "explain"
                explanation_count += 1
                bert_score_preds.append(model_generated_output)
                bert_score_refs.append(reference_output)
                current_result["bert_score_precision"] = None # Will be filled later if batching BERTScore
                current_result["bert_score_recall"] = None
                current_result["bert_score_f1"] = None
            
            results_per_sample.append(current_result)

    # --- Aggregate and Report Metrics ---
    final_metrics = {}
    if ssd_count > 0:
        final_metrics["average_ssd_cp"] = total_ssd_sum / ssd_count
    if move_prediction_count > 0:
        for k_val in args.top_k_agreement:
            final_metrics[f"top_{k_val}_agreement_rate"] = top_k_correct[k_val] / move_prediction_count
    
    if bert_score_preds and bert_score_refs:
        print("Calculating BERTScore for explanations...")
        P, R, F1 = bert_score_calculate(bert_score_preds, bert_score_refs, lang="en", 
                                        model_type=args.bert_score_model_type, verbose=True, device=DEVICE)
        final_metrics["bert_score_precision_avg"] = P.mean().item()
        final_metrics["bert_score_recall_avg"] = R.mean().item()
        final_metrics["bert_score_f1_avg"] = F1.mean().item()
        
        # Add individual BERTScore to results_per_sample
        bert_idx = 0
        for res_item in results_per_sample:
            if "explain" in res_item["task_type"].lower() and res_item["reference_output"]:
                 if bert_idx < len(P): # Check bounds
                    res_item["bert_score_precision"] = P[bert_idx].item()
                    res_item["bert_score_recall"] = R[bert_idx].item()
                    res_item["bert_score_f1"] = F1[bert_idx].item()
                    bert_idx +=1


    print("\n--- Aggregated Evaluation Metrics ---")
    for metric, value in final_metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    # Fluency: Typically requires human evaluation or more advanced NLP models.
    # For now, we can mention it as a qualitative aspect or future work.
    print("\nFluency of explanations: To be assessed qualitatively or with advanced metrics (e.g., perplexity against a general LM).")

    # Save detailed results
    if args.output_results_file:
        print(f"Saving detailed results to: {args.output_results_file}")
        output_to_save = {"aggregated_metrics": final_metrics, "per_sample_results": results_per_sample}
        try:
            with open(args.output_results_file, "w") as f:
                json.dump(output_to_save, f, indent=4)
            print(f"Results saved to {args.output_results_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

    # --- Cleanup ---
    if stockfish_engine:
        stockfish_engine.quit()
        print("Stockfish engine quit.")
    
    # Clear model from GPU memory
    del model
    del tokenizer
    if 'stockfish_engine' in locals() and stockfish_engine: del stockfish_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Evaluation complete. Models cleared from memory.")


if __name__ == "__main__":
    main()
