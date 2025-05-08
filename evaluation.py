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
# Ensure bert_score is installed: pip install bert-score[torch]
try:
    from bert_score import score as bert_score_calculate
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    # print("Warning: bert-score library not found...") # Print only if needed later

from typing import List, Dict, Optional, Any, Set
import sys
import glob
import math
import re
import hashlib
from collections import defaultdict # For grouping explanation tasks

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
    # ... (function unchanged) ...
    limit = None
    if time_limit: limit = chess.engine.Limit(time=time_limit)
    elif depth_limit: limit = chess.engine.Limit(depth=depth_limit)
    else: limit = chess.engine.Limit(time=0.1) 
    results = {"top_moves_uci": [], "top_moves_san": [], "scores_cp_after_move": [], "current_eval_cp_white_pov": None}
    try:
        initial_analysis = engine.analyse(board, chess.engine.Limit(depth=5, time=0.05))
        if initial_analysis and initial_analysis.get("score"): results["current_eval_cp_white_pov"] = initial_analysis["score"].white().score(mate_score=10000)
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
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, Exception): return results

def post_process_model_output(raw_text: str, task_type: str, 
                              reference_output: Optional[str]=None) -> str:
    # Removed teacher_task_instruction as it's not available during eval typically
    # Rely on task_type and general cleanup rules.
    processed_text = raw_text.strip()
    common_boilerplate_patterns = [
        r"^\s*explanation:\s*\[assistant\]\s*", r"^\s*explanation:\s*", r"^\s*\[assistant\]\s*",
        r"^\s*okay, here's an explanation:\s*", r"^\s*sure, i can explain that\s*[:.]?\s*",
        r"^\s*here is the explanation:\s*", r"^\s*here's a concise explanation:\s*",
        r"^\s*the explanation is as follows:\s*"
    ]
    for bp_pattern in common_boilerplate_patterns: processed_text = re.sub(bp_pattern, "", processed_text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    prompt_guidance_echoes = [ # Remove prompt fragments seen in examples
        r"^\s*\(e\.g\., central control.*opening ideas\)\.\s*",
        r"^\s*opening ideas\)\.\s*", 
        r"^\s*piece activation, opening ideas\)\.\s*",
        r"^\s*\d+ resulted in '[^']+'. Explain concisely.*position\.\s*",
    ]
    for echo_pattern in prompt_guidance_echoes:
        match = re.match(echo_pattern, processed_text, re.IGNORECASE | re.DOTALL)
        if match: processed_text = processed_text[match.end():].strip(); processed_text = re.sub(r"^\s*\[assistant\]\s*", "", processed_text, flags=re.IGNORECASE).strip()
            
    # Task-specific extraction
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
    elif task_type in ["is_square_attacked", "can_piece_move", "parse_comment_mate_unavoidable"]: # Yes/No answers
        if re.search(r"\byes\b", processed_text, re.IGNORECASE): return "Yes"
        if re.search(r"\bno\b", processed_text, re.IGNORECASE): return "No"
        return processed_text.split(" ")[0] if processed_text else "Unknown"
    elif task_type == "list_legal_moves":
        potential_ucis = re.findall(r"[a-h][1-8][a-h][1-8][qrnb]?", processed_text); return " ".join(sorted(list(set(potential_ucis))))
    elif task_type == "extract_comment_best_move": # Expect SAN
        san_match = re.match(r"^\s*([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)\b", processed_text)
        return san_match.group(1) if san_match else processed_text.split(" ")[0] if processed_text else ""
    
    # Fallback / General cleaning for explanations
    # Truncate sentences only if explicitly identified as an explanation type
    is_explanation_task_flag = "explain" in task_type.lower() or task_type.startswith("p2")
    if is_explanation_task_flag:
        sentences = re.split(r'(?<=[.!?])\s+', processed_text); max_sentences = 4 
        if sentences and sentences[0]: processed_text = " ".join(sentences[:max_sentences])
        else: processed_text = "" if not raw_text.strip() else raw_text.strip() 
        if processed_text and processed_text[-1] not in ".!?": processed_text += "."
    
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip(); processed_text = processed_text.replace("\n", " ").strip()
    if processed_text.lower().startswith("[assistant]"): processed_text = processed_text[len("[assistant]"):].strip()
    return processed_text

def parse_task_subtype_from_id(task_id: str) -> str:
    """Extracts detailed task type like P2.1_General from task_id."""
    if not task_id: return "unknown"
    parts = task_id.split('_')
    # Assume format like gameX_Y_color_TaskType_Subtype
    if len(parts) >= 4 and (parts[3].lower() == "p2" or parts[3].lower().startswith("p2.")):
        return "_".join(parts[3:]) # e.g., P2.1_General, P2.3_Event_Capture
    elif len(parts) >= 4: # For Phase 1 tasks if ID follows similar pattern
         return parts[3] # e.g. predict_move
    return "unknown" # Fallback

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on chess tasks with inference caching.")
    # --- Arguments ---
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--base_model_cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    # Separate Data Sources
    parser.add_argument("--test_file", type=str, help="Path to JSONL test file for Phase 1 type tasks.")
    parser.add_argument("--explanation_test_folder", type=str, default=None, help="Path to folder with JSONL test files for Phase 2 tasks.")
    # Separate Sample Limits
    parser.add_argument("--max_p1_eval_samples", type=int, default=None, help="Limit Phase 1 test samples.")
    parser.add_argument("--max_p2_eval_samples", type=int, default=None, help="Limit Phase 2 test samples.")
    # Metrics Switches
    parser.add_argument("--eval_move_pred", action="store_true", help="Enable move prediction metrics (SSD, Top-K Acc). Requires --stockfish_path.")
    parser.add_argument("--eval_rule_tasks", action="store_true", help="Enable accuracy/F1 evaluation for rule-based tasks.")
    parser.add_argument("--eval_explanation", action="store_true", help="Enable BERTScore evaluation for explanation tasks.")
    # Other Args
    parser.add_argument("--stockfish_path", type=str, help="Path to Stockfish executable.")
    parser.add_argument("--stockfish_analysis_time", type=float, default=0.2)
    parser.add_argument("--top_k_agreement", type=int, nargs='+', default=[1, 3])
    parser.add_argument("--bert_score_model_type", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--default_max_new_tokens", type=int, default=150)
    parser.add_argument("--output_results_file", type=str, default="evaluation_results.json")
    parser.add_argument("--output_numerical_summary", type=str, default=None)
    parser.add_argument("--inference_cache_folder", type=str, default=None, help="Optional: Path to folder to save/load inference results.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    # --- Validation ---
    if not args.test_file and not args.explanation_test_folder: parser.error("Must provide --test_file or --explanation_test_folder.")
    if args.eval_move_pred and not args.stockfish_path: parser.error("--eval_move_pred requires --stockfish_path.")
    if args.eval_explanation and not BERT_SCORE_AVAILABLE: print("Error: --eval_explanation requires `bert-score`."); sys.exit(1)
    if args.inference_cache_folder: os.makedirs(args.inference_cache_folder, exist_ok=True); print(f"Using inference cache: {args.inference_cache_folder}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {DEVICE}")
    # ... (Load Tokenizer, Load Base Model, Load LoRA Adapter as before) ...
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

    # --- Initialize Stockfish ---
    stockfish_engine = None
    if args.eval_move_pred and args.stockfish_path:
        try: stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path); print(f"Stockfish initialized: {args.stockfish_path}")
        except Exception as e: print(f"Error initializing Stockfish: {e}. Move metrics skipped.")
    elif args.eval_move_pred: print("Stockfish path not provided. Move metrics skipped.")

    # --- Load Test Dataset(s) and tag source ---
    all_data_items_to_process = []
    p1_samples_loaded = 0
    p2_samples_loaded = 0
    # Load Phase 1 data
    if args.test_file:
        print(f"Loading Phase 1 test data from: {args.test_file}")
        try:
            p1_dataset = load_dataset("json", data_files=args.test_file, split="train")
            limit = args.max_p1_eval_samples if args.max_p1_eval_samples and args.max_p1_eval_samples > 0 else None
            if limit and limit < len(p1_dataset):
                print(f"Sub-sampling Phase 1 test data to {limit}.")
                p1_dataset = p1_dataset.select(range(limit))
            p1_samples_loaded = len(p1_dataset)
            for item in p1_dataset: item['data_source'] = 'P1'; all_data_items_to_process.append(item)
            print(f"Loaded {p1_samples_loaded} samples from Phase 1.")
        except Exception as e: print(f"Error loading Phase 1 test file {args.test_file}: {e}")
    
    # Load Phase 2 data
    if args.explanation_test_folder:
        print(f"Loading Phase 2 test data from folder: {args.explanation_test_folder}")
        explanation_test_files = glob.glob(os.path.join(args.explanation_test_folder, "*.jsonl"))
        if not explanation_test_files: print(f"Warning: No *.jsonl files found in {args.explanation_test_folder}")
        else:
            try:
                p2_dataset_list = [load_dataset("json", data_files=f, split="train") for f in explanation_test_files]
                if p2_dataset_list:
                    p2_dataset_full = concatenate_datasets(p2_dataset_list)
                    limit = args.max_p2_eval_samples if args.max_p2_eval_samples and args.max_p2_eval_samples > 0 else None
                    if limit and limit < len(p2_dataset_full):
                         print(f"Sub-sampling Phase 2 test data to {limit}.")
                         p2_dataset_full = p2_dataset_full.select(range(limit))
                    p2_samples_loaded = len(p2_dataset_full)
                    for item in p2_dataset_full: item['data_source'] = 'P2'; all_data_items_to_process.append(item)
                    print(f"Loaded {p2_samples_loaded} samples from Phase 2.")
            except Exception as e: print(f"Error loading Phase 2 data from {args.explanation_test_folder}: {e}")

    if not all_data_items_to_process: print("No samples loaded. Exiting."); sys.exit(0)
    random.shuffle(all_data_items_to_process)
    print(f"Total samples for evaluation: {len(all_data_items_to_process)}")

    # --- Evaluation Loop ---
    results_per_sample = []; total_ssd_sum = 0.0; ssd_count = 0; top_k_correct = {k: 0 for k in args.top_k_agreement}; move_prediction_count = 0
    # For BERTScore per subtype
    bert_data_by_subtype = defaultdict(lambda: {"preds": [], "refs": []})
    explanation_count = 0; accuracy_tasks_counts = defaultdict(lambda: {"correct": 0, "total": 0}); list_legal_metrics_agg = {"f1": 0.0, "prec": 0.0, "rec": 0.0, "count": 0}

    prompts_for_inference = [item['input'] for item in all_data_items_to_process if 'input' in item]
    data_items_for_processing = [item for item in all_data_items_to_process if 'input' in item] # Keep aligned
    generated_outputs_text = [None] * len(prompts_for_inference)
    prompts_needing_inference = []; indices_needing_inference = []; cache_hits = 0

    # --- Inference Caching Logic ---
    if args.inference_cache_folder:
        print("Checking inference cache...")
        for idx, prompt in enumerate(tqdm(prompts_for_inference, desc="Cache Check", ncols=100)):
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            cache_file_path = os.path.join(args.inference_cache_folder, f"{prompt_hash}.txt")
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, "r", encoding="utf-8") as f_cache: generated_outputs_text[idx] = f_cache.read()
                    cache_hits += 1
                except Exception as e_cache_read: print(f"Warn: Error read cache {cache_file_path}: {e_cache_read}. Re-infer."); prompts_needing_inference.append(prompt); indices_needing_inference.append(idx)
            else: prompts_needing_inference.append(prompt); indices_needing_inference.append(idx)
        print(f"Found {cache_hits} cached results. Running inference for {len(prompts_needing_inference)} prompts.")
    else:
        prompts_needing_inference = prompts_for_inference; indices_needing_inference = list(range(len(prompts_for_inference)))
        print(f"No cache folder. Running inference for all {len(prompts_for_inference)} prompts.")

    # --- Run Inference ---
    if prompts_needing_inference:
        # ... (Inference loop as before) ...
        num_inference_batches = (len(prompts_needing_inference) + args.batch_size - 1) // args.batch_size
        for i in tqdm(range(0, len(prompts_needing_inference), args.batch_size), desc="Model Inference", ncols=100, total=num_inference_batches):
            batch_prompts = prompts_needing_inference[i:i + args.batch_size]; batch_indices = indices_needing_inference[i:i + args.batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(model.device)
            max_gen_tokens_for_batch = args.default_max_new_tokens 
            current_batch_items = [data_items_for_processing[k] for k in batch_indices]
            batch_data_sources = {item.get('data_source', 'P1') for item in current_batch_items}
            is_pred_task_batch = all(item.get("task", "").lower() == "predict_move" for item in current_batch_items) if 'P1' in batch_data_sources else False
            is_list_task_in_batch = any(item.get("task","").lower() == "list_legal_moves" for item in current_batch_items) if 'P1' in batch_data_sources else False
            if is_pred_task_batch and not is_list_task_in_batch: max_gen_tokens_for_batch = 10 
            elif is_list_task_in_batch: max_gen_tokens_for_batch = 150
            with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=max_gen_tokens_for_batch, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            for k_idx, output_ids_tensor in enumerate(outputs):
                original_index = batch_indices[k_idx]; prompt_len = inputs["attention_mask"][k_idx].sum().item()
                decoded_output = tokenizer.decode(output_ids_tensor[prompt_len:], skip_special_tokens=True).strip()
                generated_outputs_text[original_index] = decoded_output
                if args.inference_cache_folder:
                    prompt_hash = hashlib.sha256(batch_prompts[k_idx].encode()).hexdigest()
                    cache_file_path = os.path.join(args.inference_cache_folder, f"{prompt_hash}.txt")
                    try:
                        with open(cache_file_path, "w", encoding="utf-8") as f_cache: f_cache.write(decoded_output)
                    except Exception as e_cache_write: print(f"Warn: Error writing cache file {cache_file_path}: {e_cache_write}")


    # --- Metrics Calculation ---
    for idx, item in enumerate(tqdm(data_items_for_processing, desc="Calculating Metrics", ncols=100)):
        task_id = item.get("task_id", f"sample_{idx}")
        input_prompt_str = item["input"]
        model_raw_output = generated_outputs_text[idx] if idx < len(generated_outputs_text) and generated_outputs_text[idx] is not None else "GENERATION_ERROR"
        reference_output = item.get("output")
        data_source = item.get("data_source", "Unknown") # P1 or P2
        task_type = item.get("task", "unknown").lower() # Original task type
        
        # --- Determine Effective Task Type & Flag ---
        is_explanation_task_flag = (data_source == 'P2')
        if is_explanation_task_flag and task_type == "unknown":
            effective_task_type = parse_task_subtype_from_id(task_id) # Infer detailed type from ID
            if effective_task_type == "unknown": effective_task_type = "explanation_generic" # Fallback
        else:
            effective_task_type = task_type # Use the type from P1 data

        model_processed_output = post_process_model_output(model_raw_output, effective_task_type, None, reference_output, is_explanation_task_flag)
        
        current_result = {"task_id": task_id, "task_type": effective_task_type, "data_source": data_source, "input_prompt": input_prompt_str, "model_raw_output": model_raw_output, "model_processed_output": model_processed_output, "reference_output": reference_output, "is_correct": None, "list_f1": None, "list_precision": None, "list_recall": None}
        
        is_p1_rule_task = (data_source == 'P1') and effective_task_type in ["predict_move", "identify_piece", "identify_color", "is_square_attacked", "can_piece_move", "list_legal_moves", "extract_comment_best_move", "extract_comment_mate_unavoidable"]

        # Stockfish Metrics (Only for P1 predict_move)
        if args.eval_move_pred and stockfish_engine and effective_task_type == "predict_move" and data_source == 'P1':
            # ... (SSD and Top-K logic - unchanged) ...
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


        # Accuracy/F1 for Rule-Based Tasks (Only for P1 source)
        elif args.eval_rule_tasks and data_source == 'P1':
            accuracy_tasks_counts[effective_task_type]["total"] += 1
            correct = False
            if effective_task_type == "predict_move":
                 if reference_output: correct = (model_processed_output == reference_output)
            elif effective_task_type in ["identify_piece", "identify_color", "is_square_attacked", "can_piece_move", "extract_comment_best_move", "extract_comment_mate_unavoidable"]:
                 if reference_output: correct = (model_processed_output.lower() == reference_output.lower())
            elif effective_task_type == "list_legal_moves":
                 if reference_output:
                    ref_moves = set(reference_output.split()); pred_moves = set(model_processed_output.split())
                    if len(ref_moves) > 0 or len(pred_moves) > 0:
                        intersect_count = len(ref_moves.intersection(pred_moves)); precision = intersect_count / len(pred_moves) if len(pred_moves) > 0 else 0; recall = intersect_count / len(ref_moves) if len(ref_moves) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        current_result["list_f1"] = round(f1, 4); current_result["list_precision"] = round(precision, 4); current_result["list_recall"] = round(recall, 4); 
                        list_legal_metrics_agg["f1"] += f1; list_legal_metrics_agg["prec"] += precision; list_legal_metrics_agg["rec"] += recall; list_legal_metrics_agg["count"] += 1
                        correct = math.isclose(f1, 1.0) # Exact match for accuracy count
                    else: # Both empty
                         current_result["list_f1"]= 1.0; current_result["list_precision"]= 1.0; current_result["list_recall"]= 1.0; correct = True; list_legal_count += 1; list_legal_f1_sum += 1.0; list_legal_precision_sum += 1.0; list_legal_recall_sum += 1.0 # Count as correct if both empty
            current_result["is_correct"] = correct
            if correct: accuracy_tasks_counts[effective_task_type]["correct"] += 1

        # BERTScore for Explanations (Only for P2 source)
        elif args.eval_explanation and data_source == 'P2' and reference_output:
            explanation_subtype = parse_task_subtype_from_id(task_id)
            bert_data_by_subtype[explanation_subtype]["preds"].append(model_processed_output)
            bert_data_by_subtype[explanation_subtype]["refs"].append(reference_output)
            explanation_count += 1 # Count total explanations evaluated
            
        results_per_sample.append(current_result)

    # --- Aggregate and Report Metrics ---
    final_metrics = {}
    # Stockfish Aggregation
    if args.eval_move_pred:
        final_metrics["average_ssd_cp"] = round(total_ssd_sum / ssd_count, 2) if ssd_count > 0 else None
        if move_prediction_count > 0:
            for k_val in args.top_k_agreement: final_metrics[f"top_{k_val}_agreement_rate"] = round(top_k_correct[k_val] / move_prediction_count, 4)
    
    # Rule Task Accuracy Aggregation
    if args.eval_rule_tasks:
        for task_name, counts in accuracy_tasks_counts.items():
            if counts["total"] > 0 and task_name != "list_legal_moves": # Calculate accuracy for simple tasks
                final_metrics[f"{task_name}_accuracy"] = round(counts["correct"] / counts["total"], 4)
        # Aggregate list_legal_moves F1/P/R
        if list_legal_metrics_agg["count"] > 0:
            count = list_legal_metrics_agg["count"]
            final_metrics["list_legal_moves_f1_avg"] = round(list_legal_metrics_agg["f1"] / count, 4)
            final_metrics["list_legal_moves_precision_avg"] = round(list_legal_metrics_agg["prec"] / count, 4)
            final_metrics["list_legal_moves_recall_avg"] = round(list_legal_metrics_agg["rec"] / count, 4)


    # BERTScore Aggregation (Overall and Per Subtype)
    if args.eval_explanation and explanation_count > 0:
        print(f"Calculating BERTScore for {explanation_count} explanation samples...")
        if BERT_SCORE_AVAILABLE:
            all_preds_combined = []; all_refs_combined = []
            # Calculate per-subtype scores
            for subtype, data in bert_data_by_subtype.items():
                if data["preds"] and data["refs"]:
                    print(f"Calculating BERTScore for subtype: {subtype} ({len(data['preds'])} samples)")
                    try:
                        P, R, F1 = bert_score_calculate(data["preds"], data["refs"], lang="en", model_type=args.bert_score_model_type, verbose=False, device=DEVICE, batch_size=args.batch_size*2)
                        final_metrics[f"bert_score_f1_avg_{subtype}"] = round(F1.mean().item(), 4)
                        final_metrics[f"bert_score_precision_avg_{subtype}"] = round(P.mean().item(), 4)
                        final_metrics[f"bert_score_recall_avg_{subtype}"] = round(R.mean().item(), 4)
                        all_preds_combined.extend(data["preds"])
                        all_refs_combined.extend(data["refs"])
                        
                        # Add individual scores back to per_sample_results (requires mapping subtype data back to original index or task_id)
                        # This part is complex; skipping adding individual scores back for now for simplicity
                        
                    except Exception as e_bs: print(f"Error calculating BERTScore for subtype {subtype}: {e_bs}.")
            
            # Calculate overall BERTScore
            if all_preds_combined and all_refs_combined:
                print("Calculating overall BERTScore...")
                try:
                    P_all, R_all, F1_all = bert_score_calculate(all_preds_combined, all_refs_combined, lang="en", model_type=args.bert_score_model_type, verbose=False, device=DEVICE, batch_size=args.batch_size*2)
                    final_metrics["bert_score_f1_avg_OVERALL"] = round(F1_all.mean().item(), 4)
                    final_metrics["bert_score_precision_avg_OVERALL"] = round(P_all.mean().item(), 4)
                    final_metrics["bert_score_recall_avg_OVERALL"] = round(R_all.mean().item(), 4)
                except Exception as e_bs_all: print(f"Error calculating overall BERTScore: {e_bs_all}.")
        else: print("BERTScore calculation skipped: library not installed.")
    
    # --- Reporting ---
    print("\n--- Aggregated Evaluation Metrics ---")
    # ... (Printing aggregated metrics, sorted) ...
    if not final_metrics and not any(v.get('total',0) > 0 for v in accuracy_tasks_counts.values()) and not explanation_count: print("No metrics calculated.")
    else:
        # Separate Phase 1 and Phase 2 metrics for clarity maybe? Or just list all sorted.
        print("-- Phase 1 Metrics (Move Pred / Rules) --")
        for metric, value in sorted(final_metrics.items()):
             if "bert_score" not in metric:
                 print(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}")
        print("-- Phase 2 Metrics (Explanation) --")
        for metric, value in sorted(final_metrics.items()):
             if "bert_score" in metric:
                 print(f"{metric.replace('bert_score_', '').replace('_avg','').replace('_',' ').title()}: {value if value is not None else 'N/A'}")
    print("\nFluency of explanations: Requires qualitative assessment.")


    # --- Saving Results ---
    if args.output_results_file:
        # ... (Save detailed JSON results as before, maybe add accuracy_tasks_counts) ...
        output_dir = os.path.dirname(args.output_results_file);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        print(f"Saving detailed results to: {args.output_results_file}")
        output_to_save = {"args": vars(args), "aggregated_metrics": final_metrics, "accuracy_counts_per_task": dict(accuracy_tasks_counts), "per_sample_results": results_per_sample}
        try:
            with open(args.output_results_file, "w") as f: json.dump(output_to_save, f, indent=4)
            print(f"Results saved to {args.output_results_file}")
        except Exception as e: print(f"Error saving results file: {e}")
    if args.output_numerical_summary:
        # ... (Save numerical summary as before, ensuring it reflects the structure of final_metrics) ...
        numerical_summary_path = args.output_numerical_summary; summary_dir = os.path.dirname(numerical_summary_path)
        if summary_dir and not os.path.exists(summary_dir): os.makedirs(summary_dir, exist_ok=True)
        print(f"\nSaving numerical summary to: {numerical_summary_path}")
        try:
            p1_count = sum(1 for item in all_data_items_to_process if item.get('data_source') == 'P1')
            p2_count = sum(1 for item in all_data_items_to_process if item.get('data_source') == 'P2')
            with open(numerical_summary_path, "w") as f_summary:
                f_summary.write(f"Evaluation Summary for Model: {args.model_name}\n"); f_summary.write(f"LoRA Adapter: {args.lora_adapter_path if args.lora_adapter_path else 'None (Base Model)'}\n")
                f_summary.write(f"Test Samples (P1 source): {p1_count}\n"); f_summary.write(f"Test Samples (P2 source): {p2_count}\n")
                f_summary.write("--- Metrics ---\n")
                # Separate P1/P2 metrics in summary?
                f_summary.write("-- Phase 1 Metrics (Move Pred / Rules) --\n")
                for metric, value in sorted(final_metrics.items()):
                    if "bert_score" not in metric: f_summary.write(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}\n")
                f_summary.write("-- Phase 2 Metrics (Explanation) --\n")
                for metric, value in sorted(final_metrics.items()):
                    if "bert_score" in metric: f_summary.write(f"{metric.replace('bert_score_', '').replace('_avg','').replace('_',' ').title()}: {value if value is not None else 'N/A'}\n")
            print(f"Numerical summary saved to {numerical_summary_path}")
        except Exception as e: print(f"Error saving numerical summary: {e}")

    # --- Cleanup ---
    if stockfish_engine: stockfish_engine.quit(); print("Stockfish engine quit.")
    del model; del tokenizer;
    if 'stockfish_engine' in locals() and stockfish_engine: del stockfish_engine
    if torch.cuda.is_available(): torch.cuda.empty_cache(); print("CUDA cache emptied.")
    print("Evaluation complete.")

if __name__ == "__main__":
    if '--eval_explanation' in sys.argv and not BERT_SCORE_AVAILABLE:
         print("\nWarning: --eval_explanation requested but bert-score library not found. BERTScore metrics will be skipped.")
         print("Install with: pip install bert-score[torch]")
    main()