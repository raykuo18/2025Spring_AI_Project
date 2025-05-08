#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import chess
import chess.engine
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from bert_score import score as bert_score_calculate
from typing import List, Dict, Optional, Any, Set
import sys
import glob
import math
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
    # ... (function unchanged from previous version) ...
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
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, Exception):
        return results

def post_process_model_output(raw_text: str, task_type: str, 
                              teacher_task_instruction: Optional[str]=None,
                              reference_output: Optional[str]=None) -> str:
    # ... (function unchanged from previous version) ...
    processed_text = raw_text.strip()
    common_boilerplate_patterns = [r"^\s*explanation:\s*\[assistant\]\s*", r"^\s*explanation:\s*", r"^\s*\[assistant\]\s*",r"^\s*okay, here's an explanation:\s*", r"^\s*sure, i can explain that\s*[:.]?\s*",r"^\s*here is the explanation:\s*", r"^\s*here's a concise explanation:\s*",r"^\s*the explanation is as follows:\s*"]
    for bp_pattern in common_boilerplate_patterns: processed_text = re.sub(bp_pattern, "", processed_text, flags=re.IGNORECASE | re.DOTALL).strip()
    prompt_guidance_echoes = [r"\(e\.g\., central control.*opening ideas\)\.", r"opening ideas\)\.", r"piece activation, opening ideas\)\.", r"\d+ resulted in '[^']+'. Explain concisely \(1-2 sentences\) what this event means or achieves in this specific position\.",]
    if teacher_task_instruction: prompt_guidance_echoes.insert(0, re.escape(teacher_task_instruction.split('\n')[0].strip()))
    for echo_pattern in prompt_guidance_echoes:
        match = re.match(rf"^\s*{echo_pattern}\s*", processed_text, re.IGNORECASE | re.DOTALL)
        if match: processed_text = processed_text[match.end():].strip(); processed_text = re.sub(r"^\s*\[assistant\]\s*", "", processed_text, flags=re.IGNORECASE).strip()
    if task_type == "predict_move":
        uci_match = re.match(r"^\s*([a-h][1-8][a-h][1-8][qrnb]?)", processed_text); return uci_match.group(1) if uci_match else processed_text.split(" ")[0] if processed_text else ""
    elif task_type == "identify_piece":
        piece_match = re.search(r"\b([pnbrqkPNBRQK])\b", processed_text)
        if not piece_match and processed_text and processed_text[0] in "pnbrqkPNBRQK": piece_match = re.match(r"([pnbrqkPNBRQK])", processed_text)
        return piece_match.group(1) if piece_match else processed_text.split(" ")[0] if processed_text else ""
    elif task_type == "identify_color":
        if re.search(r"\bwhite\b", processed_text, re.IGNORECASE): return "White"
        if re.search(r"\bblack\b", processed_text, re.IGNORECASE): return "Black"
        return processed_text.split(" ")[0] if processed_text else "Unknown"
    elif task_type in ["is_square_attacked", "can_piece_move"]:
        if re.search(r"\byes\b", processed_text, re.IGNORECASE): return "Yes"
        if re.search(r"\bno\b", processed_text, re.IGNORECASE): return "No"
        return processed_text.split(" ")[0] if processed_text else "Unknown"
    elif task_type == "list_legal_moves":
        potential_ucis = re.findall(r"[a-h][1-8][a-h][1-8][qrnb]?", processed_text); return " ".join(sorted(list(set(potential_ucis))))
    if "explain" in task_type.lower():
        sentences = re.split(r'(?<=[.!?])\s+', processed_text); max_sentences = 4 
        if sentences and sentences[0]: processed_text = " ".join(sentences[:max_sentences])
        else: processed_text = "" if not raw_text.strip() else raw_text.strip() 
        if processed_text and processed_text[-1] not in ".!?": processed_text += "."
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip(); processed_text = processed_text.replace("\n", " ").strip()
    if processed_text.lower().startswith("[assistant]"): processed_text = processed_text[len("[assistant]"):].strip()
    return processed_text


def load_processed_task_ids(output_folder_path: str) -> Set[str]: # Not used here but keep for potential future use
    # ... (function unchanged from previous version) ...
    processed_ids = set();
    if not os.path.isdir(output_folder_path): return processed_ids
    file_patterns = [os.path.join(output_folder_path, "training_data_slice_*_part_*.jsonl"), os.path.join(output_folder_path, "training_data_slice_*_all_current_run.jsonl"), os.path.join(output_folder_path, "training_data_part_*.jsonl"), os.path.join(output_folder_path, "training_data_full_current_run.jsonl"), os.path.join(output_folder_path, "training_data_full.jsonl")]
    files_to_check = []; [files_to_check.extend(glob.glob(pattern)) for pattern in file_patterns]
    if not files_to_check: return processed_ids
    unique_files = sorted(list(set(files_to_check))); print(f"Checking {len(unique_files)} existing file(s) for processed tasks...")
    for file_path in unique_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try: data = json.loads(line); processed_ids.add(data["task_id"])
                    except (json.JSONDecodeError, KeyError): pass 
        except Exception as e: print(f"Error reading {file_path} for processed IDs: {e}. Continuing...")
    print(f"Found {len(processed_ids)} task_ids in existing files."); return processed_ids

def main():
    parser = argparse.ArgumentParser(description="Evaluate a base model or a LoRA-adapted model on chess tasks.")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--base_model_cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    # <<< MODIFIED Data Arguments >>>
    parser.add_argument("--test_file", type=str, help="Path to JSONL test file for Phase 1 type tasks (predict_move, rules etc.).")
    parser.add_argument("--explanation_test_folder", type=str, default=None, help="Path to folder containing JSONL test files for Phase 2 explanation tasks.")
    parser.add_argument("--max_p1_eval_samples", type=int, default=None, help="Limit number of Phase 1 test samples.")
    parser.add_argument("--max_p2_eval_samples", type=int, default=None, help="Limit number of Phase 2 test samples.")
    # <<< END MODIFIED >>>
    parser.add_argument("--stockfish_path", type=str, help="Path to Stockfish executable.")
    parser.add_argument("--stockfish_analysis_time", type=float, default=0.2)
    parser.add_argument("--top_k_agreement", type=int, nargs='+', default=[1, 3])
    parser.add_argument("--bert_score_model_type", type=str, default=None)
    # Removed --max_eval_samples, replaced by the two above
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--default_max_new_tokens", type=int, default=150)
    parser.add_argument("--output_results_file", type=str, default="evaluation_results.json")
    parser.add_argument("--output_numerical_summary", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # Add arguments to enable/disable specific metric calculations
    parser.add_argument("--eval_move_pred", action="store_true", help="Enable evaluation for move prediction tasks (requires Stockfish).")
    parser.add_argument("--eval_rule_tasks", action="store_true", help="Enable evaluation for rule-based tasks (identify piece, color, attacks etc.).")
    parser.add_argument("--eval_explanation", action="store_true", help="Enable evaluation for explanation tasks (calculates BERTScore).")


    args = parser.parse_args()
    set_seed(args.seed)

    if not args.test_file and not args.explanation_test_folder:
        parser.error("You must provide either --test_file (for Phase 1 tasks) or --explanation_test_folder (for Phase 2 tasks).")
    if args.eval_move_pred and not args.stockfish_path:
         parser.error("--eval_move_pred requires --stockfish_path to be set.")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {DEVICE}")
    model_config_details = MODEL_CONFIGS[args.model_name]; hf_model_name = model_config_details["hf_model_name"]
    print(f"Loading tokenizer for {hf_model_name}..."); tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "left"
    else: tokenizer.padding_side = "left"
    print("Tokenizer loaded.")

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else None
    print(f"Loading base model: {hf_model_name}..."); model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quant_config, torch_dtype=torch.bfloat16 if quant_config else torch.float16, device_map="auto", trust_remote_code=True)
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id
    model.eval(); print("Base model loaded.")

    if args.lora_adapter_path:
        print(f"Loading LoRA adapter from: {args.lora_adapter_path}...")
        if not os.path.isdir(args.lora_adapter_path): print(f"Error: LoRA path '{args.lora_adapter_path}' not found."); sys.exit(1)
        try: model = PeftModel.from_pretrained(model, args.lora_adapter_path); print("LoRA adapter loaded."); model.eval()
        except Exception as e: print(f"Error loading LoRA adapter: {e}."); sys.exit(1)
    print("Model ready for evaluation.")

    stockfish_engine = None
    if args.eval_move_pred and args.stockfish_path: # Only init if needed and path exists
        try: stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path); print(f"Stockfish initialized: {args.stockfish_path}")
        except Exception as e: print(f"Error initializing Stockfish: {e}. Move metrics skipped.")
    elif args.eval_move_pred: print("Stockfish path not provided, but move evaluation requested. Skipping move metrics.")
    else: print("Move evaluation metrics (SSD, Top-K) disabled by arguments.")

    # --- Load Test Dataset(s) ---
    all_data_items_to_process = []
    if args.test_file:
        print(f"Loading Phase 1 type test data from: {args.test_file}")
        try:
            p1_dataset = load_dataset("json", data_files=args.test_file, split="train")
            # Apply P1 sample limit
            if args.max_p1_eval_samples and args.max_p1_eval_samples > 0 and args.max_p1_eval_samples < len(p1_dataset):
                print(f"Sub-sampling Phase 1 test data to {args.max_p1_eval_samples} samples.")
                p1_dataset = p1_dataset.select(range(args.max_p1_eval_samples))
            all_data_items_to_process.extend(list(p1_dataset))
            print(f"Loaded {len(p1_dataset)} samples from Phase 1 test file.")
        except Exception as e: print(f"Error loading Phase 1 test file {args.test_file}: {e}")
    
    if args.explanation_test_folder:
        print(f"Loading Phase 2 type test data from folder: {args.explanation_test_folder}")
        explanation_test_files = glob.glob(os.path.join(args.explanation_test_folder, "*.jsonl"))
        if not explanation_test_files: print(f"Warning: No *.jsonl files found in {args.explanation_test_folder}")
        else:
            try:
                p2_dataset_list = [load_dataset("json", data_files=f, split="train") for f in explanation_test_files]
                if p2_dataset_list:
                    p2_dataset_full = concatenate_datasets(p2_dataset_list)
                    # Apply P2 sample limit
                    if args.max_p2_eval_samples and args.max_p2_eval_samples > 0 and args.max_p2_eval_samples < len(p2_dataset_full):
                        print(f"Sub-sampling Phase 2 test data to {args.max_p2_eval_samples} samples.")
                        p2_dataset_full = p2_dataset_full.select(range(args.max_p2_eval_samples))
                    all_data_items_to_process.extend(list(p2_dataset_full))
                    print(f"Loaded {len(p2_dataset_full)} samples from Phase 2 explanation test folder.")
            except Exception as e: print(f"Error loading Phase 2 explanation test data from {args.explanation_test_folder}: {e}")

    if not all_data_items_to_process: print("No samples found from specified test sources. Exiting."); sys.exit(0)
    
    # Note: Max samples are now applied per source. Total might exceed individual max if both sources used.
    # If a total limit across both is needed, apply random.sample or slicing *after* combining them.
    print(f"Total samples for evaluation: {len(all_data_items_to_process)}")

    # --- Evaluation Loop ---
    results_per_sample = []; total_ssd_sum = 0.0; ssd_count = 0; top_k_correct = {k: 0 for k in args.top_k_agreement}; move_prediction_count = 0
    bert_score_preds, bert_score_refs = [], []; explanation_count = 0; accuracy_tasks_counts = {}

    prompts_for_inference = [item['input'] for item in all_data_items_to_process if 'input' in item]
    data_items_for_processing = [item for item in all_data_items_to_process if 'input' in item]
    generated_outputs_text = []

    # --- Model Inference ---
    for i in tqdm(range(0, len(prompts_for_inference), args.batch_size), desc="Model Inference", ncols=100):
        batch_prompts = prompts_for_inference[i:i + args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(model.device)
        max_gen_tokens_for_batch = args.default_max_new_tokens 
        current_batch_items = data_items_for_processing[i:i + args.batch_size]
        # Check if all tasks in batch are move prediction to use shorter length
        is_pred_task_batch = all(("predict_move" in item.get("task", "").lower() or "list_legal_moves" in item.get("task","").lower()) for item in current_batch_items)
        if is_pred_task_batch and any(("predict_move" in item.get("task","").lower() or "list_legal_moves" in item.get("task","").lower()) for item in current_batch_items):
            max_gen_tokens_for_batch = 100 # Slightly more generous for list_legal_moves
            if all("predict_move" in item.get("task","").lower() for item in current_batch_items):
                 max_gen_tokens_for_batch = 10 # Shortest for single UCI prediction
            
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_gen_tokens_for_batch, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
        
        for k_idx, output_ids_tensor in enumerate(outputs):
            prompt_len = inputs["attention_mask"][k_idx].sum().item()
            decoded_output = tokenizer.decode(output_ids_tensor[prompt_len:], skip_special_tokens=True)
            generated_outputs_text.append(decoded_output.strip())

    # --- Metrics Calculation ---
    for idx, item in enumerate(tqdm(data_items_for_processing, desc="Calculating Metrics", ncols=100)):
        task_type = item.get("task", "unknown").lower(); input_prompt_str = item["input"]
        model_raw_output = generated_outputs_text[idx] if idx < len(generated_outputs_text) else "GENERATION_INDEX_ERROR"
        reference_output = item.get("output")
        model_processed_output = post_process_model_output(model_raw_output, task_type, None, reference_output)
        current_result = {"task_id": item.get("task_id"), "task_type": task_type, "input_prompt": input_prompt_str, "model_raw_output": model_raw_output, "model_processed_output": model_processed_output, "reference_output": reference_output, "is_correct": None, "jaccard_f1": None}
        
        # --- Calculate Metrics based on Enabled Evals and Task Type ---
        # Stockfish Metrics
        if args.eval_move_pred and stockfish_engine and task_type == "predict_move":
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
                        sf_best_move_uci = sf_analysis["top_moves_uci"][0]; sf_eval_after_sf_best_cp = sf_analysis["scores_cp_after_move"][0] if sf_analysis["scores_cp_after_move"] else None
                        current_result["stockfish_top1_uci"] = sf_best_move_uci; current_result["stockfish_top1_eval_cp"] = sf_eval_after_sf_best_cp
                        if model_move_obj:
                            board_after_model_move = board.copy(); board_after_model_move.push(model_move_obj)
                            info_after_model_move = stockfish_engine.analyse(board_after_model_move, chess.engine.Limit(time=args.stockfish_analysis_time))
                            eval_after_model_move_cp = info_after_model_move.get("score").white().score(mate_score=10000) if info_after_model_move.get("score") else None
                            if sf_eval_after_sf_best_cp is not None and eval_after_model_move_cp is not None:
                                ssd = (sf_eval_after_sf_best_cp - eval_after_model_move_cp) if board.turn == chess.WHITE else (eval_after_model_move_cp - sf_eval_after_sf_best_cp)
                                current_result["ssd_cp"] = ssd; total_ssd_sum += ssd; ssd_count += 1
                        for k_val in args.top_k_agreement:
                            is_in_top_k = predicted_uci in sf_analysis["top_moves_uci"][:k_val]; current_result[f"in_top_{k_val}"] = is_in_top_k
                            if is_in_top_k: top_k_correct[k_val] += 1
                except Exception as e_sf: current_result["ssd_cp"] = f"SF_Error: {e_sf}"

        # Accuracy for Rule-Based Tasks
        if args.eval_rule_tasks:
            if task_type == "predict_move": # Accuracy vs reference move
                if reference_output: current_result["is_correct"] = (model_processed_output == reference_output)
            elif task_type in ["identify_piece", "identify_color", "is_square_attacked", "can_piece_move"]: # Case-insensitive compare for keywords
                if reference_output: current_result["is_correct"] = (model_processed_output.lower() == reference_output.lower())
            elif task_type == "list_legal_moves": # Jaccard/F1 for sets
                if reference_output:
                    ref_moves = set(reference_output.split()); pred_moves = set(model_processed_output.split())
                    if len(ref_moves) > 0 or len(pred_moves) > 0:
                        precision = len(ref_moves.intersection(pred_moves)) / len(pred_moves) if len(pred_moves) > 0 else 0
                        recall = len(ref_moves.intersection(pred_moves)) / len(ref_moves) if len(ref_moves) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        current_result["jaccard_f1"] = round(f1, 4); current_result["is_correct"] = math.isclose(f1, 1.0)
                    else: current_result["jaccard_f1"] = 1.0 if not reference_output and not model_processed_output else 0.0; current_result["is_correct"] = (not reference_output and not model_processed_output)

        # BERTScore for Explanations
        if args.eval_explanation and ("explain" in task_type or task_type.startswith("p2.")) and reference_output:
            explanation_count += 1; bert_score_preds.append(model_processed_output); bert_score_refs.append(reference_output)
            
        results_per_sample.append(current_result)


    # --- Aggregate and Report Metrics ---
    final_metrics = {}
    # Only calculate metrics if they were enabled and data existed
    if args.eval_move_pred and ssd_count > 0: final_metrics["average_ssd_cp"] = round(total_ssd_sum / ssd_count, 2)
    else: final_metrics["average_ssd_cp"] = None
    if args.eval_move_pred and move_prediction_count > 0:
        for k_val in args.top_k_agreement: final_metrics[f"top_{k_val}_agreement_rate"] = round(top_k_correct[k_val] / move_prediction_count, 4)
    
    if args.eval_rule_tasks:
        accuracy_task_types = ["predict_move", "identify_piece", "identify_color", "is_square_attacked", "can_piece_move", "list_legal_moves"]
        for acc_task in accuracy_task_types:
            correct_count = sum(1 for r in results_per_sample if r["task_type"] == acc_task and r.get("is_correct") is True)
            total_task_samples = sum(1 for r in results_per_sample if r["task_type"] == acc_task)
            if total_task_samples > 0: final_metrics[f"{acc_task}_accuracy"] = round(correct_count / total_task_samples, 4)
            accuracy_tasks_counts[acc_task] = {"correct": correct_count, "total": total_task_samples}

    if args.eval_explanation and bert_score_preds and bert_score_refs:
        print("Calculating BERTScore for explanations...");
        try:
            # Import locally if only needed here
            from bert_score import score as bert_score_calculate
            P, R, F1 = bert_score_calculate(bert_score_preds, bert_score_refs, lang="en", model_type=args.bert_score_model_type, verbose=False, device=DEVICE, batch_size=args.batch_size*2)
            final_metrics["bert_score_precision_avg"] = round(P.mean().item(), 4); final_metrics["bert_score_recall_avg"] = round(R.mean().item(), 4); final_metrics["bert_score_f1_avg"] = round(F1.mean().item(), 4)
            bert_idx = 0
            for res_item in results_per_sample:
                if ("explain" in res_item["task_type"].lower() or res_item["task_type"].startswith("p2.")) and res_item["reference_output"]:
                     if bert_idx < len(P): res_item["bert_score_precision"] = round(P[bert_idx].item(), 4); res_item["bert_score_recall"] = round(R[bert_idx].item(), 4); res_item["bert_score_f1"] = round(F1[bert_idx].item(), 4); bert_idx +=1
        except ImportError: print("BERTScore library not found. Skipping BERTScore calculation. Install with: pip install bert-score")
        except Exception as e_bs: print(f"Error calculating BERTScore: {e_bs}."); final_metrics["bert_score_f1_avg"] = None
    
    print("\n--- Aggregated Evaluation Metrics ---")
    if not final_metrics and not any(v.get('total',0) > 0 for v in accuracy_tasks_counts.values()) and not explanation_count: print("No metrics calculated (check arguments and input data).")
    else:
        for metric, value in sorted(final_metrics.items()): print(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}")
    print("\nFluency of explanations: Requires qualitative assessment.")

    # Save detailed results
    if args.output_results_file:
        output_dir = os.path.dirname(args.output_results_file);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        print(f"Saving detailed results to: {args.output_results_file}")
        output_to_save = {"args": vars(args), "aggregated_metrics": final_metrics, "accuracy_counts_per_task": accuracy_tasks_counts, "per_sample_results": results_per_sample}
        try:
            with open(args.output_results_file, "w") as f: json.dump(output_to_save, f, indent=4)
            print(f"Results saved to {args.output_results_file}")
        except Exception as e: print(f"Error saving results file: {e}")

    # Generate Simple Numerical Results File
    if args.output_numerical_summary:
        numerical_summary_path = args.output_numerical_summary; summary_dir = os.path.dirname(numerical_summary_path)
        if summary_dir and not os.path.exists(summary_dir): os.makedirs(summary_dir, exist_ok=True)
        print(f"\nSaving numerical summary to: {numerical_summary_path}")
        try:
            with open(numerical_summary_path, "w") as f_summary:
                f_summary.write(f"Evaluation Summary for Model: {args.model_name}\n")
                if args.lora_adapter_path: f_summary.write(f"LoRA Adapter: {args.lora_adapter_path}\n")
                f_summary.write(f"Test Samples (P1): {len([i for i in all_data_items_to_process if args.test_file and i.get('task','').lower() != 'explain']) if args.test_file else 'N/A'}\n")
                f_summary.write(f"Test Samples (P2): {len([i for i in all_data_items_to_process if args.explanation_test_folder and 'explain' in i.get('task','').lower()]) if args.explanation_test_folder else 'N/A'}\n")
                f_summary.write("--- Metrics ---\n")
                for metric, value in sorted(final_metrics.items()): f_summary.write(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}\n")
            print(f"Numerical summary saved to {numerical_summary_path}")
        except Exception as e: print(f"Error saving numerical summary: {e}")

    # Cleanup
    if stockfish_engine: stockfish_engine.quit(); print("Stockfish engine quit.")
    del model; del tokenizer;
    if 'stockfish_engine' in locals() and stockfish_engine: del stockfish_engine
    if torch.cuda.is_available(): torch.cuda.empty_cache(); print("CUDA cache emptied.")
    print("Evaluation complete.")

if __name__ == "__main__":
    # Import bert_score locally if needed for the main check
    try:
        import bert_score
    except ImportError:
        print("Note: bert-score library not found. Explanation metrics using BERTScore will be skipped. Install with: pip install bert-score")
    
    main()