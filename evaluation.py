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
from bert_score import score as bert_score_calculate
from typing import List, Dict, Optional, Any, Set
import sys
import glob
import math # For math.isclose
import re

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {"hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "add_eos_token": True,},
    "Gemma-2B": {"hf_model_name": "google/gemma-2b", "add_eos_token": True,},
    "Phi-2": {"hf_model_name": "microsoft/phi-2", "add_eos_token": True,}
}

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_stockfish_analysis(board: chess.Board, engine: chess.engine.SimpleEngine, 
                           time_limit: Optional[float] = None, depth_limit: Optional[int] = None, 
                           multipv: int = 3) -> Dict[str, Any]:
    limit = None
    if time_limit: limit = chess.engine.Limit(time=time_limit)
    elif depth_limit: limit = chess.engine.Limit(depth=depth_limit)
    else: limit = chess.engine.Limit(time=0.1) 
    results = {"top_moves_uci": [], "top_moves_san": [], "scores_cp_after_move": [], "current_eval_cp_white_pov": None}
    try:
        initial_analysis = engine.analyse(board, chess.engine.Limit(depth=5, time=0.05))
        if initial_analysis and initial_analysis.get("score"):
            results["current_eval_cp_white_pov"] = initial_analysis["score"].white().score(mate_score=10000)
        info_list = engine.analyse(board, limit, multipv=multipv)
        if not info_list: return results
        for info in info_list:
            if "pv" in info and info["pv"]:
                move = info["pv"][0]; results["top_moves_uci"].append(move.uci())
                try: results["top_moves_san"].append(board.san(move))
                except ValueError: results["top_moves_san"].append(board.variation_san([move]))
                score_obj = info.get("score")
                if score_obj: results["scores_cp_after_move"].append(score_obj.white().score(mate_score=10000))
                else: results["scores_cp_after_move"].append(None)
        return results
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, Exception) as e:
        # print(f"Stockfish analysis error for FEN {board.fen()}: {e}") # Can be very verbose
        return results


def post_process_model_output(raw_text: str, task_type: str, 
                              teacher_task_instruction: Optional[str]=None,
                              reference_output: Optional[str]=None) -> str:
    """Cleans the generated output based on the task type."""
    processed_text = raw_text.strip()

    # 1. Generic Boilerplate Removal
    common_boilerplate_patterns = [
        r"^\s*explanation:\s*\[assistant\]\s*", r"^\s*explanation:\s*", r"^\s*\[assistant\]\s*",
        r"^\s*okay, here's an explanation:\s*", r"^\s*sure, i can explain that\s*[:.]?\s*",
        r"^\s*here is the explanation:\s*", r"^\s*here's a concise explanation:\s*",
        r"^\s*the explanation is as follows:\s*"
    ]
    for bp_pattern in common_boilerplate_patterns:
        processed_text = re.sub(bp_pattern, "", processed_text, flags=re.IGNORECASE | re.DOTALL).strip()

    # 2. Remove echoed task instructions (more specific than generic boilerplate)
    # These are phrases that might come from the Mixtral prompt task description
    prompt_guidance_echoes = [
        r"\(e\.g\., central control, development, king safety, pawn structure\) OR immediate tactical ideas \(e\.g\., threats, defenses, piece activation, opening ideas\)\.",
        r"opening ideas\)\.",
        r"piece activation, opening ideas\)\.",
        r"\d+ resulted in '[^']+'. Explain concisely \(1-2 sentences\) what this event means or achieves in this specific position\.",
        # Add more specific patterns based on your Mixtral prompts for various tasks
    ]
    if teacher_task_instruction: # If the original task instruction is available
         prompt_guidance_echoes.insert(0, re.escape(teacher_task_instruction.split('\n')[0].strip()))

    for echo_pattern in prompt_guidance_echoes:
        match = re.match(rf"^\s*{echo_pattern}\s*", processed_text, re.IGNORECASE | re.DOTALL)
        if match:
            processed_text = processed_text[match.end():].strip()
            # Try removing assistant tag again if it was after the echoed prompt
            processed_text = re.sub(r"^\s*\[assistant\]\s*", "", processed_text, flags=re.IGNORECASE).strip()

    # 3. Task-specific extraction
    if task_type == "predict_move":
        uci_match = re.match(r"^\s*([a-h][1-8][a-h][1-8][qrnb]?)", processed_text)
        return uci_match.group(1) if uci_match else processed_text.split(" ")[0] if processed_text else ""
    
    elif task_type == "identify_piece":
        # Look for single piece characters (case sensitive as per reference_output "P")
        piece_match = re.search(r"\b([pnbrqkPNBRQK])\b", processed_text) # Match as a whole word
        if not piece_match and processed_text: # Fallback: first char if it's a piece
             if processed_text[0] in "pnbrqkPNBRQK": piece_match = re.match(r"([pnbrqkPNBRQK])", processed_text)
        return piece_match.group(1) if piece_match else processed_text.split(" ")[0] if processed_text else ""

    elif task_type == "identify_color":
        if re.search(r"\bwhite\b", processed_text, re.IGNORECASE): return "White"
        if re.search(r"\bblack\b", processed_text, re.IGNORECASE): return "Black"
        return processed_text.split(" ")[0] if processed_text else "Unknown" # Best guess

    elif task_type in ["is_square_attacked", "can_piece_move"]: # Yes/No answers
        if re.search(r"\byes\b", processed_text, re.IGNORECASE): return "Yes"
        if re.search(r"\bno\b", processed_text, re.IGNORECASE): return "No"
        return processed_text.split(" ")[0] if processed_text else "Unknown"

    elif task_type == "list_legal_moves":
        potential_ucis = re.findall(r"[a-h][1-8][a-h][1-8][qrnb]?", processed_text)
        return " ".join(sorted(list(set(potential_ucis))))

    # For explanation tasks, apply sentence truncation and punctuation
    if "explain" in task_type.lower():
        sentences = re.split(r'(?<=[.!?])\s+', processed_text)
        max_sentences = 4
        if sentences and sentences[0]: processed_text = " ".join(sentences[:max_sentences])
        else: processed_text = "" if not raw_text.strip() else raw_text.strip() # Fallback
        if processed_text and processed_text[-1] not in ".!?": processed_text += "."
    
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip()
    processed_text = processed_text.replace("\n", " ").strip()
    
    # Final check for any remaining leading assistant tags if missed
    if processed_text.lower().startswith("[assistant]"):
         processed_text = processed_text[len("[assistant]"):].strip()
    return processed_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate a base model or a LoRA-adapted model on chess tasks.")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Name of the base model.")
    parser.add_argument("--base_model_cache_dir", type=str, default="./hf_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Optional: Path to the LoRA adapter directory.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the JSONL test file.")
    parser.add_argument("--stockfish_path", type=str, help="Path to Stockfish executable. Required for move evaluation metrics.")
    parser.add_argument("--stockfish_analysis_time", type=float, default=0.2, help="Time (seconds) for Stockfish analysis per move.")
    parser.add_argument("--top_k_agreement", type=int, nargs='+', default=[1, 3], help="List of k for top-k agreement.")
    parser.add_argument("--bert_score_model_type", type=str, default=None, help="BERT model for BERTScore. Default: library's default.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of test samples.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length for tokenizer.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA style).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for model inference.")
    parser.add_argument("--default_max_new_tokens", type=int, default=150, help="Default max new tokens for generation.")
    parser.add_argument("--output_results_file", type=str, default="evaluation_results.json", help="JSON file to save detailed results and metrics.")
    # --- NEW ARGUMENT for numerical summary ---
    parser.add_argument("--output_numerical_summary", type=str, default=None, help="Optional text file to save a concise numerical summary of key metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    set_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]
    print(f"Loading tokenizer for {hf_model_name}..."); tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "left"
    else: tokenizer.padding_side = "left" # Ensure left padding for generation
    print("Tokenizer loaded.")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else None
    print(f"Loading base model: {hf_model_name}..."); model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quantization_config, torch_dtype=torch.bfloat16 if quantization_config else torch.float16, device_map="auto", trust_remote_code=True)
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id
    model.eval(); print("Base model loaded.")

    if args.lora_adapter_path:
        print(f"Loading LoRA adapter from: {args.lora_adapter_path}...")
        if not os.path.isdir(args.lora_adapter_path): print(f"Error: LoRA path '{args.lora_adapter_path}' not found."); sys.exit(1)
        try: model = PeftModel.from_pretrained(model, args.lora_adapter_path); print("LoRA adapter loaded."); model.eval()
        except Exception as e: print(f"Error loading LoRA adapter: {e}."); sys.exit(1)
    print("Model ready for evaluation.")

    stockfish_engine = None
    if args.stockfish_path:
        try: stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path); print(f"Stockfish initialized: {args.stockfish_path}")
        except Exception as e: print(f"Error initializing Stockfish: {e}. Move metrics skipped.")
    else: print("No Stockfish path. Move metrics skipped.")

    print(f"Loading test data from: {args.test_file}"); raw_dataset = load_dataset("json", data_files=args.test_file, split="train")
    if args.max_eval_samples and args.max_eval_samples > 0 and args.max_eval_samples < len(raw_dataset):
        print(f"Sub-sampling test data to {args.max_eval_samples} samples."); raw_dataset = raw_dataset.select(range(args.max_eval_samples))
    print(f"Using {len(raw_dataset)} samples for evaluation.")

    results_per_sample = []; total_ssd_sum = 0.0; ssd_count = 0; top_k_correct = {k: 0 for k in args.top_k_agreement}; move_prediction_count = 0
    bert_score_preds, bert_score_refs = [], []; explanation_count = 0; accuracy_tasks_counts = {}

    prompts_for_inference = [item['input'] for item in raw_dataset if 'input' in item]
    data_items_for_processing = [item for item in raw_dataset if 'input' in item]
    generated_outputs_text = []

    for i in tqdm(range(0, len(prompts_for_inference), args.batch_size), desc="Model Inference", ncols=100):
        batch_prompts = prompts_for_inference[i:i + args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(model.device)
        max_gen_tokens_for_batch = args.default_max_new_tokens
        current_batch_items = data_items_for_processing[i:i + args.batch_size]
        if all("predict_move" in item.get("task", "").lower() for item in current_batch_items) and any("predict_move" in item.get("task","").lower() for item in current_batch_items):
            max_gen_tokens_for_batch = 10 
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_gen_tokens_for_batch, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
        for k_idx, output_ids_tensor in enumerate(outputs):
            prompt_len = inputs["attention_mask"][k_idx].sum().item()
            decoded_output = tokenizer.decode(output_ids_tensor[prompt_len:], skip_special_tokens=True)
            generated_outputs_text.append(decoded_output.strip())

    for idx, item in enumerate(tqdm(data_items_for_processing, desc="Calculating Metrics", ncols=100)):
        task_type = item.get("task", "unknown").lower() # Normalize task type
        input_prompt_str = item["input"]
        model_raw_output = generated_outputs_text[idx] if idx < len(generated_outputs_text) else "GENERATION_INDEX_ERROR"
        reference_output = item.get("output")
        
        # Pass the original task instruction if it's part of the input prompt for post_process
        teacher_task_instr_for_postproc = None # Placeholder, ideally extract from item if stored from step 1
        
        model_processed_output = post_process_model_output(model_raw_output, task_type, teacher_task_instr_for_postproc, reference_output)
        
        current_result = {
            "task_id": item.get("task_id"), "task_type": task_type, "input_prompt": input_prompt_str,
            "model_raw_output": model_raw_output, # Store raw
            "model_processed_output": model_processed_output, # Store processed
            "reference_output": reference_output, "is_correct": None
        }

        if task_type == "predict_move":
            if reference_output: current_result["is_correct"] = (model_processed_output == reference_output)
            if stockfish_engine:
                move_prediction_count += 1; fen_match = re.search(r"\[FEN\]\s*(.*?)\s*\[SEP\]", input_prompt_str)
                if fen_match:
                    fen = fen_match.group(1).strip()
                    try:
                        board = chess.Board(fen); predicted_uci = model_processed_output; model_move_obj = None
                        try: model_move_obj = board.parse_uci(predicted_uci)
                        except ValueError: pass 
                        if model_move_obj and model_move_obj not in board.legal_moves: model_move_obj = None
                        
                        sf_analysis = get_stockfish_analysis(board, stockfish_engine, time_limit=args.stockfish_analysis_time, multipv=max(args.top_k_agreement))
                        if sf_analysis["top_moves_uci"]:
                            sf_best_move_uci = sf_analysis["top_moves_uci"][0]
                            sf_eval_after_sf_best_cp = sf_analysis["scores_cp_after_move"][0] if sf_analysis["scores_cp_after_move"] else None
                            current_result["stockfish_top1_uci"] = sf_best_move_uci; current_result["stockfish_top1_eval_cp"] = sf_eval_after_sf_best_cp
                            if model_move_obj:
                                board_after_model_move = board.copy(); board_after_model_move.push(model_move_obj)
                                info_after_model_move = stockfish_engine.analyse(board_after_model_move, chess.engine.Limit(time=args.stockfish_analysis_time))
                                eval_after_model_move_cp = info_after_model_move.get("score").white().score(mate_score=10000) if info_after_model_move.get("score") else None
                                if sf_eval_after_sf_best_cp is not None and eval_after_model_move_cp is not None:
                                    ssd = (sf_eval_after_sf_best_cp - eval_after_model_move_cp) if board.turn == chess.WHITE else (eval_after_model_move_cp - sf_eval_after_sf_best_cp)
                                    current_result["ssd_cp"] = ssd; total_ssd_sum += ssd; ssd_count += 1
                            for k_val in args.top_k_agreement:
                                is_in_top_k = predicted_uci in sf_analysis["top_moves_uci"][:k_val]
                                current_result[f"in_top_{k_val}"] = is_in_top_k
                                if is_in_top_k: top_k_correct[k_val] += 1
                    except Exception as e_sf: current_result["ssd_cp"] = f"SF_Error: {e_sf}"
        elif task_type in ["identify_piece", "identify_color", "is_square_attacked", "can_piece_move"]:
            if reference_output: current_result["is_correct"] = (model_processed_output.lower() == reference_output.lower())
        elif task_type == "list_legal_moves":
            if reference_output:
                ref_moves = set(reference_output.split()); pred_moves = set(model_processed_output.split())
                if len(ref_moves) > 0 or len(pred_moves) > 0:
                    precision = len(ref_moves.intersection(pred_moves)) / len(pred_moves) if len(pred_moves) > 0 else 0
                    recall = len(ref_moves.intersection(pred_moves)) / len(ref_moves) if len(ref_moves) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    current_result["jaccard_f1"] = round(f1, 4)
                    current_result["is_correct"] = math.isclose(f1, 1.0)
                else: current_result["jaccard_f1"] = 1.0 if not reference_output and not model_processed_output else 0.0; current_result["is_correct"] = (not reference_output and not model_processed_output)
        elif "explain" in task_type and reference_output:
            explanation_count += 1
            bert_score_preds.append(model_processed_output) # Use processed output for BERTScore
            bert_score_refs.append(reference_output)
        results_per_sample.append(current_result)

    final_metrics = {}
    if ssd_count > 0: final_metrics["average_ssd_cp"] = round(total_ssd_sum / ssd_count, 2)
    else: final_metrics["average_ssd_cp"] = None
    if move_prediction_count > 0:
        for k_val in args.top_k_agreement: final_metrics[f"top_{k_val}_agreement_rate"] = round(top_k_correct[k_val] / move_prediction_count, 4)
    
    # Calculate accuracy for relevant tasks
    accuracy_task_types = ["predict_move", "identify_piece", "identify_color", "is_square_attacked", "can_piece_move", "list_legal_moves"]
    for acc_task in accuracy_task_types:
        correct_count = sum(1 for r in results_per_sample if r["task_type"] == acc_task and r.get("is_correct") is True)
        total_task_samples = sum(1 for r in results_per_sample if r["task_type"] == acc_task)
        if total_task_samples > 0: final_metrics[f"{acc_task}_accuracy"] = round(correct_count / total_task_samples, 4)
        accuracy_tasks_counts[acc_task] = {"correct": correct_count, "total": total_task_samples}


    if bert_score_preds and bert_score_refs:
        print("Calculating BERTScore for explanations...");
        try:
            P, R, F1 = bert_score_calculate(bert_score_preds, bert_score_refs, lang="en", model_type=args.bert_score_model_type, verbose=False, device=DEVICE, batch_size=args.batch_size*2)
            final_metrics["bert_score_precision_avg"] = round(P.mean().item(), 4); final_metrics["bert_score_recall_avg"] = round(R.mean().item(), 4); final_metrics["bert_score_f1_avg"] = round(F1.mean().item(), 4)
            bert_idx = 0
            for res_item in results_per_sample:
                if "explain" in res_item["task_type"].lower() and res_item["reference_output"]:
                     if bert_idx < len(P): res_item["bert_score_precision"] = round(P[bert_idx].item(), 4); res_item["bert_score_recall"] = round(R[bert_idx].item(), 4); res_item["bert_score_f1"] = round(F1[bert_idx].item(), 4); bert_idx +=1
        except Exception as e_bs: print(f"Error calculating BERTScore: {e_bs}."); final_metrics["bert_score_f1_avg"] = None
    
    print("\n--- Aggregated Evaluation Metrics ---")
    if not final_metrics and not any(accuracy_tasks_counts.values()): print("No metrics calculated.")
    else:
        for metric, value in sorted(final_metrics.items()): print(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}")
    print("\nFluency of explanations: To be assessed qualitatively or with more advanced metrics.")

    if args.output_results_file:
        output_dir = os.path.dirname(args.output_results_file);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        print(f"Saving detailed results to: {args.output_results_file}")
        output_to_save = {"aggregated_metrics": final_metrics, "per_sample_results": results_per_sample, "accuracy_counts_per_task": accuracy_tasks_counts}
        try:
            with open(args.output_results_file, "w") as f: json.dump(output_to_save, f, indent=4)
            print(f"Results saved to {args.output_results_file}")
        except Exception as e: print(f"Error saving results file: {e}")

    # --- Generate Simple Numerical Results File ---
    if args.output_numerical_summary:
        numerical_summary_path = args.output_numerical_summary
        summary_dir = os.path.dirname(numerical_summary_path)
        if summary_dir and not os.path.exists(summary_dir): os.makedirs(summary_dir, exist_ok=True)
        print(f"\nSaving numerical summary to: {numerical_summary_path}")
        try:
            with open(numerical_summary_path, "w") as f_summary:
                for metric, value in sorted(final_metrics.items()):
                    f_summary.write(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}\n")
            print(f"Numerical summary saved to {numerical_summary_path}")
        except Exception as e: print(f"Error saving numerical summary: {e}")


    if stockfish_engine: stockfish_engine.quit(); print("Stockfish engine quit.")
    del model; del tokenizer;
    if 'stockfish_engine' in locals() and stockfish_engine: del stockfish_engine
    if torch.cuda.is_available(): torch.cuda.empty_cache(); print("CUDA cache emptied.")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()