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
import sys
import glob
import math
import re
import hashlib
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set

# --- Metric Libraries (with try-except for optional ones) ---
try:
    from bert_score import score as bert_score_calculate
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_SCORE_AVAILABLE = True
except ImportError:
    ROUGE_SCORE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

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
                              teacher_task_instruction: Optional[str]=None,
                              reference_output: Optional[str]=None,
                              is_explanation_task: bool = False) -> str:
    processed_text = raw_text.strip()
    common_boilerplate_patterns = [
        r"^\s*explanation:\s*\[assistant\]\s*", r"^\s*explanation:\s*", r"^\s*\[assistant\]\s*",
        r"^\s*okay, here's an explanation:\s*", r"^\s*sure, i can explain that\s*[:.]?\s*",
        r"^\s*here is the explanation:\s*", r"^\s*here's a concise explanation:\s*",
        r"^\s*the explanation is as follows:\s*"
    ]
    for bp_pattern in common_boilerplate_patterns:
        processed_text = re.sub(bp_pattern, "", processed_text, flags=re.IGNORECASE | re.DOTALL).strip()
    prompt_guidance_echoes = [
        r"\(e\.g\., central control.*opening ideas\)\.", r"opening ideas\)\.",
        r"piece activation, opening ideas\)\.", r"\d+ resulted in '[^']+'. Explain concisely.+position\.",
    ]
    for echo_pattern in prompt_guidance_echoes:
        temp_text_for_echo_check = processed_text 
        match = re.match(rf"^\s*{echo_pattern}\s*", temp_text_for_echo_check, re.IGNORECASE | re.DOTALL)
        if match:
            processed_text = temp_text_for_echo_check[match.end():].strip()
            processed_text = re.sub(r"^\s*\[assistant\]\s*", "", processed_text, flags=re.IGNORECASE).strip()
    task_type_lower = task_type.lower()
    if task_type_lower == "predict_move":
        text_to_search = processed_text
        uci_candidates = re.findall(r"\b([a-h][1-8][a-h][1-8][qrnb]?)\b", text_to_search)
        if uci_candidates: return uci_candidates[0]
        san_pattern = r"\b(?:[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?|O-O(?:-O)?)(?:[+#])?\b"
        san_candidates = re.findall(san_pattern, text_to_search)
        if san_candidates: return san_candidates[0]
        move_notation_capture = r"([a-h][1-8][a-h][1-8][qrnb]?|[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQK])?|O-O(?:-O)?)"
        patterns_with_move = [
            rf"(?:move is|next move is|plays|best move is|suggests|recommend|ответ:)\s*{move_notation_capture}\b",
            rf"{move_notation_capture}\.\s*$",
        ]
        for pattern_str in patterns_with_move:
            match = re.search(pattern_str, text_to_search, re.IGNORECASE)
            if match and match.group(1): return match.group(1)
        first_word = text_to_search.split(" ")[0]
        if first_word and (re.fullmatch(r"[a-h][1-8][a-h][1-8][qrnb]?", first_word) or \
                           re.fullmatch(san_pattern, first_word) or \
                           (len(first_word) >= 2 and len(first_word) <= 7)):
            return first_word
        return ""
    elif task_type_lower == "identify_piece":
        piece_match = re.search(r"\b([pnbrqkPNBRQK])\b", processed_text)
        if not piece_match and processed_text and processed_text[0] in "pnbrqkPNBRQK": piece_match = re.match(r"([pnbrqkPNBRQK])", processed_text)
        return piece_match.group(1) if piece_match else processed_text.split(" ")[0] if processed_text else ""
    elif task_type_lower == "identify_color":
        if re.search(r"\bwhite\b", processed_text, re.IGNORECASE): return "White"
        if re.search(r"\bblack\b", processed_text, re.IGNORECASE): return "Black"
        return processed_text.split(" ")[0] if processed_text else "Unknown"
    elif task_type_lower in ["is_square_attacked", "can_piece_move", "parse_comment_mate_unavoidable"]:
        if re.search(r"\byes\b", processed_text, re.IGNORECASE): return "Yes"
        if re.search(r"\bno\b", processed_text, re.IGNORECASE): return "No"
        return processed_text.split(" ")[0] if processed_text else "Unknown"
    elif task_type_lower == "list_legal_moves":
        potential_ucis = re.findall(r"[a-h][1-8][a-h][1-8][qrnb]?", processed_text)
        return " ".join(sorted(list(set(potential_ucis))))
    elif task_type_lower == "extract_comment_best_move":
        san_match = re.match(r"^\s*([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)\b", processed_text)
        return san_match.group(1) if san_match else processed_text.split(" ")[0] if processed_text else ""
    if is_explanation_task:
        sentences = re.split(r'(?<=[.!?])\s+', processed_text)
        max_sentences = 4
        if sentences and sentences[0]: processed_text = " ".join(sentences[:max_sentences])
        else: processed_text = ""
        if processed_text and processed_text[-1] not in ".!?": processed_text += "."
    processed_text = re.sub(r'\s\s+', ' ', processed_text).strip()
    processed_text = processed_text.replace("\n", " ").strip()
    if processed_text.lower().startswith("[assistant]"): processed_text = processed_text[len("[assistant]"):].strip()
    return processed_text

def parse_task_subtype_from_id(task_id: str) -> str:
    if not task_id: return "unknown_subtype"
    parts = task_id.split('_')
    if len(parts) >= 4 and (parts[3].lower() == "p2" or parts[3].lower().startswith("p2.")):
        return "_".join(parts[3:])
    elif len(parts) >= 4: return parts[3]
    return "unknown_subtype"

def convert_numpy_types(obj):
    """ Recursively converts NumPy types to native Python types for JSON serialization. """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (bool, np.bool_)): # Handle numpy bool
        return bool(obj)
    return obj

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on chess tasks.")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--base_model_cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--phase1_lora_path", type=str, default=None)
    parser.add_argument("--phase1_adapter_name", type=str, default="phase1_core")
    parser.add_argument("--phase2_lora_path", type=str, default=None)
    parser.add_argument("--phase2_adapter_name", type=str, default="phase2_explainer")
    parser.add_argument("--alpha_p1_weight", type=float, default=1.0)
    parser.add_argument("--beta_p2_weight", type=float, default=1.0)
    parser.add_argument("--combination_type", type=str, default="linear", choices=["linear", "svd", "cat", "ties", "dare_ties", "dare_linear", "dare_svd"])
    parser.add_argument("--test_file", type=str, help="Path to JSONL test file for Phase 1 type tasks.")
    parser.add_argument("--explanation_test_folder", type=str, default=None)
    parser.add_argument("--max_p1_eval_samples", type=int, default=None)
    parser.add_argument("--max_p2_eval_samples", type=int, default=None)
    parser.add_argument("--eval_move_pred", action="store_true")
    parser.add_argument("--eval_rule_tasks", action="store_true")
    parser.add_argument("--eval_explanation", action="store_true")
    parser.add_argument("--stockfish_path", type=str)
    parser.add_argument("--stockfish_analysis_time", type=float, default=0.2)
    parser.add_argument("--top_k_agreement", type=int, nargs='+', default=[1, 3], help="List of K values for Top-K agreement rate and SF-based 'in Top-K' accuracy.")
    parser.add_argument("--sf_reference_ks", type=int, nargs='+', default=[1], help="List of K values for Stockfish's K-th move to use as a reference for SSD and delta calculations (e.g., 1 3 5).")
    parser.add_argument("--bert_score_model_type", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--default_max_new_tokens", type=int, default=150)
    parser.add_argument("--output_results_file", type=str, default="evaluation_results.json")
    parser.add_argument("--output_numerical_summary", type=str, default=None)
    parser.add_argument("--inference_cache_folder", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    if not args.test_file and not args.explanation_test_folder: parser.error("Must provide --test_file or --explanation_test_folder.")
    if args.eval_move_pred and not args.stockfish_path: parser.error("--eval_move_pred requires --stockfish_path.")
    if args.eval_explanation:
        if not BERT_SCORE_AVAILABLE: print("Warning: --eval_explanation active but `bert-score` not found. BERTScore metrics skipped. `pip install bert-score[torch]`")
        if not ROUGE_SCORE_AVAILABLE: print("Warning: --eval_explanation active but `rouge-score` not found. ROUGE metrics skipped. `pip install rouge-score`")
        if not NLTK_AVAILABLE: print("Warning: --eval_explanation active but `nltk` not found. BLEU/Distinct-N metrics skipped. `pip install nltk`")
        if not LEVENSHTEIN_AVAILABLE: print("Warning: --eval_explanation active but `Levenshtein` not found. Edit Distance metrics skipped. `pip install python-Levenshtein`")
    if args.inference_cache_folder: os.makedirs(args.inference_cache_folder, exist_ok=True); print(f"Using inference cache: {args.inference_cache_folder}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {DEVICE}")
    model_config_details = MODEL_CONFIGS[args.model_name]; hf_model_name = model_config_details["hf_model_name"]
    print(f"Loading tokenizer for {hf_model_name}..."); tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "left"
    else: tokenizer.padding_side = "left"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if args.load_in_4bit else None
    print(f"Loading base model: {hf_model_name}..."); model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quant_config, torch_dtype=torch.bfloat16 if quant_config else torch.float16, device_map="auto", trust_remote_code=True)
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id
    model.eval(); print("Base model loaded.")
    p1_loaded_successfully = False
    if args.phase1_lora_path:
        print(f"Loading phase 1 LoRA adapter from: {args.phase1_lora_path}...")
        if not os.path.isdir(args.phase1_lora_path): print(f"Error: LoRA path '{args.phase1_lora_path}' not found."); sys.exit(1)
        try:
            if not isinstance(model, PeftModel): model = PeftModel.from_pretrained(model, args.phase1_lora_path, adapter_name=args.phase1_adapter_name)
            else: model.load_adapter(args.phase1_lora_path, adapter_name=args.phase1_adapter_name)
            p1_loaded_successfully = True; print(f"Phase 1 adapter '{args.phase1_adapter_name}' loaded.")
        except Exception as e: print(f"Error loading Phase 1 LoRA: {e}"); sys.exit(1)
    p2_loaded_successfully = False
    if args.phase2_lora_path:
        print(f"Loading phase 2 LoRA adapter from: {args.phase2_lora_path}...")
        if not os.path.isdir(args.phase2_lora_path): print(f"Error: LoRA path '{args.phase2_lora_path}' not found."); sys.exit(1)
        try:
            if not isinstance(model, PeftModel): model = PeftModel.from_pretrained(model, args.phase2_lora_path, adapter_name=args.phase2_adapter_name)
            else: model.load_adapter(args.phase2_lora_path, adapter_name=args.phase2_adapter_name)
            p2_loaded_successfully = True; print(f"Phase 2 adapter '{args.phase2_adapter_name}' loaded.")
        except Exception as e: print(f"Error loading Phase 2 LoRA: {e}"); sys.exit(1)
    if p1_loaded_successfully and p2_loaded_successfully:
        if not isinstance(model, PeftModel): print("Error: Model not PeftModel for weighted adapter."); sys.exit(1)
        combined_adapter_name = f"blend_{args.phase1_adapter_name}{args.alpha_p1_weight}_{args.phase2_adapter_name}{args.beta_p2_weight}".replace(".","_")
        print(f"Creating weighted combination: '{combined_adapter_name}'...")
        try:
            model.add_weighted_adapter(adapters=[args.phase1_adapter_name, args.phase2_adapter_name], weights=[args.alpha_p1_weight, args.beta_p2_weight], adapter_name=combined_adapter_name, combination_type=args.combination_type)
            model.set_adapter(combined_adapter_name)
            print(f"Set active adapter to combined: {combined_adapter_name}")
        except Exception as e: print(f"Error creating weighted adapter: {e}. Active adapter might be default or last loaded.");
    elif p1_loaded_successfully and not p2_loaded_successfully: model.set_adapter(args.phase1_adapter_name); print(f"Set active adapter to Phase 1: {args.phase1_adapter_name}")
    elif p2_loaded_successfully and not p1_loaded_successfully: model.set_adapter(args.phase2_adapter_name); print(f"Set active adapter to Phase 2: {args.phase2_adapter_name}")
    elif not p1_loaded_successfully and not p2_loaded_successfully: print("No LoRA adapters loaded. Using base model.")

    stockfish_engine = None
    if args.eval_move_pred and args.stockfish_path:
        try: stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path); print(f"Stockfish initialized: {args.stockfish_path}")
        except Exception as e: print(f"Error initializing Stockfish: {e}. Move metrics skipped.")
    elif args.eval_move_pred: print("Stockfish path not provided. Move metrics skipped.")
    all_data_items_to_process = []
    if args.test_file:
        try:
            p1_dataset = load_dataset("json", data_files=args.test_file, split="train")
            limit = args.max_p1_eval_samples if args.max_p1_eval_samples and args.max_p1_eval_samples > 0 else len(p1_dataset)
            p1_dataset = p1_dataset.select(range(min(limit, len(p1_dataset))))
            for item in p1_dataset: item['data_source'] = 'P1'; all_data_items_to_process.append(item)
            print(f"Loaded {len(p1_dataset)} samples from Phase 1 test file.")
        except Exception as e: print(f"Error loading P1 test file {args.test_file}: {e}")
    if args.explanation_test_folder:
        explanation_test_files = glob.glob(os.path.join(args.explanation_test_folder, "*.jsonl"))
        if explanation_test_files:
            try:
                p2_dataset_full = concatenate_datasets([load_dataset("json", data_files=f, split="train") for f in explanation_test_files])
                limit = args.max_p2_eval_samples if args.max_p2_eval_samples and args.max_p2_eval_samples > 0 else len(p2_dataset_full)
                p2_dataset_full = p2_dataset_full.select(range(min(limit, len(p2_dataset_full))))
                for item in p2_dataset_full: item['data_source'] = 'P2'; all_data_items_to_process.append(item)
                print(f"Loaded {len(p2_dataset_full)} samples from Phase 2 explanation folder.")
            except Exception as e: print(f"Error loading P2 data: {e}")
        else: print(f"Warning: No *.jsonl files found in {args.explanation_test_folder}")
    if not all_data_items_to_process: print("No samples loaded. Exiting."); sys.exit(0)
    random.shuffle(all_data_items_to_process); print(f"Total samples for evaluation: {len(all_data_items_to_process)}")

    results_per_sample = []
    ssds_by_k = defaultdict(list); ssd_counts_by_k = defaultdict(int)
    deltas_llm_vs_sf_wpov_by_k = defaultdict(list); deltas_sf_vs_gt_wpov_by_k = defaultdict(list)
    top_k_correct = {k: 0 for k in args.top_k_agreement}; move_prediction_count = 0
    llm_better_than_gt_count = 0; deltas_llm_vs_gt_cp = []; comparison_with_gt_possible_count = 0
    explanation_data_by_subtype = defaultdict(lambda: {"preds": [], "refs": []})
    accuracy_tasks_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    list_legal_metrics_agg = {"f1": 0.0, "prec": 0.0, "rec": 0.0, "count": 0}

    prompts_for_inference = [item['input'] for item in all_data_items_to_process if 'input' in item]
    data_items_for_processing = [item for item in all_data_items_to_process if 'input' in item]
    generated_outputs_text = [None] * len(prompts_for_inference)
    if args.inference_cache_folder:
        print("Checking inference cache...")
        prompts_needing_inference = []; indices_needing_inference = []; cache_hits = 0
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
    if prompts_needing_inference:
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

    # --- Metrics Calculation Loop ---
    for idx, item in enumerate(tqdm(data_items_for_processing, desc="Calculating Metrics", ncols=100)):
        task_id = item.get("task_id", f"sample_{idx}"); input_prompt_str = item["input"]
        model_raw_output = generated_outputs_text[idx] if idx < len(generated_outputs_text) and generated_outputs_text[idx] is not None else "GENERATION_ERROR"
        reference_output = item.get("output"); data_source = item.get("data_source", "Unknown")
        task_type = item.get("task", "unknown").lower()
        is_explanation_task_flag = (data_source == 'P2')
        effective_task_type = parse_task_subtype_from_id(task_id) if is_explanation_task_flag and task_type == "unknown" else task_type
        if is_explanation_task_flag and effective_task_type == "unknown_subtype": effective_task_type = "explanation_generic"
        
        model_processed_output = post_process_model_output(model_raw_output, effective_task_type, None, reference_output, is_explanation_task_flag)
        current_result = {"task_id": task_id, "task_type": effective_task_type, "data_source": data_source, "input_prompt": input_prompt_str, "model_raw_output": model_raw_output, "model_processed_output": model_processed_output, "reference_output": reference_output, 
                          "is_correct_em": None, "list_f1": None, "list_precision": None, "list_recall": None, "is_correct": None, 
                          "eval_llm_move_cp": None, "eval_after_gt_move_cp": None, "delta_llm_vs_gt_cp": None, "llm_better_than_gt": None}
        for k_ref in args.sf_reference_ks:
            current_result[f"stockfish_baseline_k{k_ref}_eval_cp"] = None; current_result[f"ssd_cp_vs_sf_top{k_ref}"] = None
            current_result[f"delta_llm_vs_sf_wpov_vs_top{k_ref}"] = None; current_result[f"delta_sf_vs_gt_wpov_vs_top{k_ref}"] = None
        for k_agree in args.top_k_agreement: current_result[f"is_correct_sf_in_top{k_agree}"] = None; current_result[f"in_top_{k_agree}"] = None

        if data_source == 'P1':
            fen_from_prompt = None; fen_match = re.search(r"\[FEN\]\s*(.*?)\s*\[SEP\]", input_prompt_str)
            if fen_match: fen_from_prompt = fen_match.group(1).strip()

            if effective_task_type == "predict_move":
                if args.eval_rule_tasks and reference_output is not None and fen_from_prompt:
                    em_model_uci = None
                    try:
                        temp_board_for_em = chess.Board(fen_from_prompt)
                        try:
                            em_move_obj_temp = temp_board_for_em.parse_uci(model_processed_output)
                            if em_move_obj_temp in temp_board_for_em.legal_moves: em_model_uci = em_move_obj_temp.uci()
                        except ValueError:
                            try:
                                em_move_obj_temp = temp_board_for_em.parse_san(model_processed_output)
                                em_model_uci = em_move_obj_temp.uci()
                            except: pass
                    except Exception: pass
                    is_em_correct = (em_model_uci == reference_output) if em_model_uci else (model_processed_output == reference_output)
                    current_result["is_correct_em"] = is_em_correct
                    accuracy_tasks_counts['predict_move_em']["total"] += 1
                    if is_em_correct: accuracy_tasks_counts['predict_move_em']["correct"] += 1

                if args.eval_move_pred and stockfish_engine and fen_from_prompt:
                    move_prediction_count += 1
                    try:
                        board_for_sf = chess.Board(fen_from_prompt)
                        llm_move_obj, llm_move_uci = None, ""
                        try:
                            llm_move_obj = board_for_sf.parse_uci(model_processed_output)
                            if llm_move_obj not in board_for_sf.legal_moves: llm_move_obj = None
                            else: llm_move_uci = llm_move_obj.uci()
                        except ValueError:
                            try:
                                llm_move_obj = board_for_sf.parse_san(model_processed_output)
                                llm_move_uci = llm_move_obj.uci()
                            except: llm_move_obj = None
                        
                        multipv_needed = 1
                        if args.top_k_agreement: multipv_needed = max(multipv_needed, max(args.top_k_agreement))
                        if args.sf_reference_ks: multipv_needed = max(multipv_needed, max(args.sf_reference_ks))
                        sf_analysis = get_stockfish_analysis(board_for_sf, stockfish_engine, time_limit=args.stockfish_analysis_time, multipv=multipv_needed)
                        
                        eval_llm_cp = None
                        if llm_move_obj:
                            board_after_llm = board_for_sf.copy(); board_after_llm.push(llm_move_obj)
                            info_llm = stockfish_engine.analyse(board_after_llm, chess.engine.Limit(time=args.stockfish_analysis_time))
                            eval_llm_cp = info_llm.get("score").white().score(mate_score=10000) if info_llm.get("score") else None
                            current_result["eval_llm_move_cp"] = eval_llm_cp
                        
                        eval_gt_cp = None
                        if reference_output:
                            gt_move_obj = None
                            try: gt_move_obj = board_for_sf.parse_uci(reference_output)
                            except: pass
                            if gt_move_obj and gt_move_obj in board_for_sf.legal_moves:
                                board_after_gt = board_for_sf.copy(); board_after_gt.push(gt_move_obj)
                                info_gt = stockfish_engine.analyse(board_after_gt, chess.engine.Limit(time=args.stockfish_analysis_time))
                                eval_gt_cp = info_gt.get("score").white().score(mate_score=10000) if info_gt.get("score") else None
                                current_result["eval_after_gt_move_cp"] = eval_gt_cp
                        
                        if sf_analysis["scores_cp_after_move"]:
                            current_result["stockfish_top1_uci"] = sf_analysis["top_moves_uci"][0] # Always store actual Top-1
                            current_result["stockfish_top1_eval_cp"] = sf_analysis["scores_cp_after_move"][0]

                            for k_ref in args.sf_reference_ks:
                                eval_sf_baseline_cp_k = None; actual_sf_moves_count = len(sf_analysis["scores_cp_after_move"])
                                valid_k_for_ref = k_ref
                                if not (1 <= k_ref <= actual_sf_moves_count): valid_k_for_ref = 1 
                                if actual_sf_moves_count > 0: eval_sf_baseline_cp_k = sf_analysis["scores_cp_after_move"][valid_k_for_ref - 1]
                                current_result[f"stockfish_baseline_k{k_ref}_eval_cp"] = eval_sf_baseline_cp_k # Use original k_ref for logging
                                if eval_sf_baseline_cp_k is not None and eval_llm_cp is not None:
                                    ssd_k = (eval_sf_baseline_cp_k - eval_llm_cp) if board_for_sf.turn == chess.WHITE else (eval_llm_cp - eval_sf_baseline_cp_k)
                                    current_result[f"ssd_cp_vs_sf_top{k_ref}"] = ssd_k; ssds_by_k[k_ref].append(ssd_k); ssd_counts_by_k[k_ref] += 1
                                    delta_llm_sf_k = eval_llm_cp - eval_sf_baseline_cp_k
                                    current_result[f"delta_llm_vs_sf_wpov_vs_top{k_ref}"] = delta_llm_sf_k; deltas_llm_vs_sf_wpov_by_k[k_ref].append(delta_llm_sf_k)
                                if eval_sf_baseline_cp_k is not None and eval_gt_cp is not None:
                                    delta_sf_gt_k = eval_sf_baseline_cp_k - eval_gt_cp
                                    current_result[f"delta_sf_vs_gt_wpov_vs_top{k_ref}"] = delta_sf_gt_k; deltas_sf_vs_gt_wpov_by_k[k_ref].append(delta_sf_gt_k)
                        
                        if eval_llm_cp is not None and eval_gt_cp is not None:
                            comparison_with_gt_possible_count += 1; delta_lg = eval_llm_cp - eval_gt_cp
                            current_result["delta_llm_vs_gt_cp"] = delta_lg; deltas_llm_vs_gt_cp.append(delta_lg)
                            llm_is_better = (eval_llm_cp > eval_gt_cp) if board_for_sf.turn == chess.WHITE else (eval_llm_cp < eval_gt_cp)
                            current_result["llm_better_than_gt"] = llm_is_better
                            if llm_is_better: llm_better_than_gt_count += 1

                        for k_agree in args.top_k_agreement: # Top-K agreement rates and "in_topK" accuracies
                            is_in_k = (llm_move_uci in sf_analysis.get("top_moves_uci", [])[:k_agree]) if llm_move_uci else False
                            current_result[f"in_top_{k_agree}"] = is_in_k
                            current_result[f"is_correct_sf_in_top{k_agree}"] = is_in_k
                            if sf_analysis.get("top_moves_uci"): # Only count if SF analysis provided moves
                                accuracy_tasks_counts[f'predict_move_sf_in_top{k_agree}']["total"] += 1
                                if is_in_k: top_k_correct[k_agree] += 1; accuracy_tasks_counts[f'predict_move_sf_in_top{k_agree}']["correct"] += 1
                    except Exception as e_sf_main: current_result["stockfish_error"] = f"SF_Main_Error: {type(e_sf_main).__name__} - {e_sf_main}"
            
            elif args.eval_rule_tasks: # Other P1 rule tasks
                accuracy_tasks_counts[effective_task_type]["total"] += 1; correct = False
                if effective_task_type in ["identify_piece", "identify_color", "is_square_attacked", "can_piece_move", "extract_comment_best_move", "parse_comment_mate_unavoidable"]:
                     if reference_output is not None: correct = (model_processed_output.lower() == str(reference_output).lower())
                elif effective_task_type == "list_legal_moves":
                     if reference_output is not None:
                        ref_moves = set(str(reference_output).split()) if reference_output else set(); pred_moves = set(model_processed_output.split()) if model_processed_output else set()
                        if not pred_moves and not ref_moves: correct = True; precision = 1.0; recall = 1.0; f1 = 1.0
                        elif pred_moves or ref_moves:
                            intersect_count = len(ref_moves.intersection(pred_moves)); precision = intersect_count / len(pred_moves) if len(pred_moves) > 0 else 0.0; recall = intersect_count / len(ref_moves) if len(ref_moves) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0; correct = math.isclose(f1, 1.0)
                        else: precision, recall, f1 = 0.0, 0.0, 0.0; correct = False
                        current_result["list_f1"]=round(f1,4); current_result["list_precision"]=round(precision,4); current_result["list_recall"]=round(recall,4)
                        list_legal_metrics_agg["f1"]+=f1; list_legal_metrics_agg["prec"]+=precision; list_legal_metrics_agg["rec"]+=recall; list_legal_metrics_agg["count"]+=1
                current_result["is_correct"] = correct
                if correct: accuracy_tasks_counts[effective_task_type]["correct"] += 1
        
        elif args.eval_explanation and data_source == 'P2' and reference_output is not None:
            subtype_for_grouping = parse_task_subtype_from_id(task_id)
            explanation_data_by_subtype[subtype_for_grouping]["preds"].append(model_processed_output)
            explanation_data_by_subtype[subtype_for_grouping]["refs"].append(reference_output)
        results_per_sample.append(current_result)

    # --- Aggregate and Report Metrics ---
    final_metrics = {}
    if args.eval_move_pred:
        for k_ref_val in args.sf_reference_ks:
            final_metrics[f"average_ssd_cp_vs_sf_top{k_ref_val}"] = round(sum(ssds_by_k[k_ref_val]) / ssd_counts_by_k[k_ref_val], 2) if ssd_counts_by_k[k_ref_val] > 0 else None
            if deltas_llm_vs_sf_wpov_by_k[k_ref_val]:
                final_metrics[f"avg_delta_llm_vs_sf_wpov_vs_top{k_ref_val}"] = round(np.mean(deltas_llm_vs_sf_wpov_by_k[k_ref_val]), 2)
                final_metrics[f"median_delta_llm_vs_sf_wpov_vs_top{k_ref_val}"] = round(np.median(deltas_llm_vs_sf_wpov_by_k[k_ref_val]), 2)
                final_metrics[f"stddev_delta_llm_vs_sf_wpov_vs_top{k_ref_val}"] = round(np.std(deltas_llm_vs_sf_wpov_by_k[k_ref_val]), 2)
            else:
                for stat in ["avg", "median", "stddev"]: final_metrics[f"{stat}_delta_llm_vs_sf_wpov_vs_top{k_ref_val}"] = None
            if deltas_sf_vs_gt_wpov_by_k[k_ref_val]:
                final_metrics[f"avg_delta_sf_vs_gt_wpov_vs_top{k_ref_val}"] = round(np.mean(deltas_sf_vs_gt_wpov_by_k[k_ref_val]), 2)
                final_metrics[f"median_delta_sf_vs_gt_wpov_vs_top{k_ref_val}"] = round(np.median(deltas_sf_vs_gt_wpov_by_k[k_ref_val]), 2)
                final_metrics[f"stddev_delta_sf_vs_gt_wpov_vs_top{k_ref_val}"] = round(np.std(deltas_sf_vs_gt_wpov_by_k[k_ref_val]), 2)
            else:
                for stat in ["avg", "median", "stddev"]: final_metrics[f"{stat}_delta_sf_vs_gt_wpov_vs_top{k_ref_val}"] = None

        if move_prediction_count > 0:
            for k_val in args.top_k_agreement: final_metrics[f"top_{k_val}_agreement_rate"] = round(top_k_correct[k_val] / move_prediction_count, 4)
        if comparison_with_gt_possible_count > 0:
            final_metrics["llm_better_than_gt_rate"] = round(llm_better_than_gt_count / comparison_with_gt_possible_count, 4)
            if deltas_llm_vs_gt_cp:
                final_metrics["avg_delta_llm_vs_gt_cp"] = round(np.mean(deltas_llm_vs_gt_cp), 2); final_metrics["median_delta_llm_vs_gt_cp"] = round(np.median(deltas_llm_vs_gt_cp), 2)
                final_metrics["stddev_delta_llm_vs_gt_cp"] = round(np.std(deltas_llm_vs_gt_cp), 2); final_metrics["min_delta_llm_vs_gt_cp"] = round(np.min(deltas_llm_vs_gt_cp), 2)
                final_metrics["max_delta_llm_vs_gt_cp"] = round(np.max(deltas_llm_vs_gt_cp), 2)
        else:
            final_metrics["llm_better_than_gt_rate"] = None
            for key in ["avg_delta_llm_vs_gt_cp", "median_delta_llm_vs_gt_cp", "stddev_delta_llm_vs_gt_cp", "min_delta_llm_vs_gt_cp", "max_delta_llm_vs_gt_cp"]: final_metrics[key] = None

    if args.eval_rule_tasks:
        for task_name, counts in accuracy_tasks_counts.items():
            if counts["total"] > 0:
                if task_name == "predict_move_em": final_metrics["predict_move_em_accuracy"] = round(counts["correct"] / counts["total"], 4)
                elif task_name.startswith("predict_move_sf_in_top"): final_metrics[f"{task_name}_accuracy"] = round(counts["correct"] / counts["total"], 4)
                elif task_name != "list_legal_moves": final_metrics[f"{task_name}_accuracy"] = round(counts["correct"] / counts["total"], 4)
        if list_legal_metrics_agg["count"] > 0:
            count = list_legal_metrics_agg["count"]
            final_metrics["list_legal_moves_f1_avg"] = round(list_legal_metrics_agg["f1"] / count, 4); final_metrics["list_legal_moves_precision_avg"] = round(list_legal_metrics_agg["prec"] / count, 4); final_metrics["list_legal_moves_recall_avg"] = round(list_legal_metrics_agg["rec"] / count, 4)

    if args.eval_explanation: # Explanation metrics aggregation remains the same
        all_preds_for_overall_metrics = []; all_refs_for_overall_metrics = []; num_total_explanation_samples = 0
        for subtype, data in explanation_data_by_subtype.items():
            subtype_preds = data["preds"]; subtype_refs = data["refs"]; num_subtype_samples = len(subtype_preds)
            if num_subtype_samples > 0:
                num_total_explanation_samples += num_subtype_samples; all_preds_for_overall_metrics.extend(subtype_preds); all_refs_for_overall_metrics.extend(subtype_refs)
                print(f"Calculating explanation metrics for subtype: {subtype} ({num_subtype_samples} samples)...")
                if BERT_SCORE_AVAILABLE:
                    try:
                        P_sub, R_sub, F1_sub = bert_score_calculate(subtype_preds, subtype_refs, lang="en", model_type=args.bert_score_model_type, verbose=False, device=DEVICE, batch_size=args.batch_size*2)
                        final_metrics[f"bert_score_f1_avg_{subtype}"] = round(F1_sub.mean().item(), 4); final_metrics[f"bert_score_precision_avg_{subtype}"] = round(P_sub.mean().item(), 4); final_metrics[f"bert_score_recall_avg_{subtype}"] = round(R_sub.mean().item(), 4)
                    except Exception as e: print(f" Error BERTScore subtype {subtype}: {e}")
                if ROUGE_SCORE_AVAILABLE:
                    try:
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True); r1_f, r2_f, rL_f = [], [], []
                        for ref, pred in zip(subtype_refs, subtype_preds): s = scorer.score(ref, pred); r1_f.append(s['rouge1'].fmeasure); r2_f.append(s['rouge2'].fmeasure); rL_f.append(s['rougeL'].fmeasure)
                        final_metrics[f"rouge_1_f1_avg_{subtype}"] = round(np.mean(r1_f), 4) if r1_f else 0.0; final_metrics[f"rouge_2_f1_avg_{subtype}"] = round(np.mean(r2_f), 4) if r2_f else 0.0; final_metrics[f"rouge_l_f1_avg_{subtype}"] = round(np.mean(rL_f), 4) if rL_f else 0.0
                    except Exception as e: print(f" Error ROUGE subtype {subtype}: {e}")
                if NLTK_AVAILABLE:
                    try:
                        tok_refs = [[r.split()] for r in subtype_refs]; tok_preds = [p.split() for p in subtype_preds]; sf = SmoothingFunction(); bleu_val_sub = None
                        try: bleu_val_sub = corpus_bleu(tok_refs, tok_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=sf.method1); final_metrics[f"bleu_4_avg_{subtype}"] = round(bleu_val_sub, 4)
                        except ValueError: bleu_val_sub = corpus_bleu(tok_refs, tok_preds, weights=(1,0,0,0), smoothing_function=sf.method1); final_metrics[f"bleu_1_avg_{subtype}"] = round(bleu_val_sub, 4)
                        except Exception: pass 
                    except Exception as e: print(f" Error BLEU subtype {subtype}: {e}")
                if LEVENSHTEIN_AVAILABLE:
                    try:
                        norm_ed_sub = [Levenshtein.distance(p, r) / max(len(p), len(r)) if max(len(p),len(r)) > 0 else 0 for p, r in zip(subtype_preds, subtype_refs)]
                        final_metrics[f"avg_norm_edit_distance_avg_{subtype}"] = round(np.mean(norm_ed_sub), 4) if norm_ed_sub else 0.0
                    except Exception as e: print(f" Error EditDist subtype {subtype}: {e}")
        if num_total_explanation_samples > 0:
            print(f"\nCalculating OVERALL explanation metrics for {num_total_explanation_samples} total samples...")
            if BERT_SCORE_AVAILABLE:
                try:
                    P_all, R_all, F1_all = bert_score_calculate(all_preds_for_overall_metrics, all_refs_for_overall_metrics, lang="en", model_type=args.bert_score_model_type, verbose=False, device=DEVICE, batch_size=args.batch_size*2)
                    final_metrics["bert_score_f1_overall"] = round(F1_all.mean().item(), 4); final_metrics["bert_score_precision_overall"] = round(P_all.mean().item(), 4); final_metrics["bert_score_recall_overall"] = round(R_all.mean().item(), 4)
                except Exception as e: print(f"Error Overall BERTScore: {e}")
            if ROUGE_SCORE_AVAILABLE: # ... same overall ROUGE ...
                try:
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True); r1_f_all, r2_f_all, rL_f_all = [], [], []
                    for ref, pred in zip(all_refs_for_overall_metrics, all_preds_for_overall_metrics): s = scorer.score(ref, pred); r1_f_all.append(s['rouge1'].fmeasure); r2_f_all.append(s['rouge2'].fmeasure); rL_f_all.append(s['rougeL'].fmeasure)
                    final_metrics["rouge_1_f1_overall"] = round(np.mean(r1_f_all), 4) if r1_f_all else 0.0; final_metrics["rouge_2_f1_overall"] = round(np.mean(r2_f_all), 4) if r2_f_all else 0.0; final_metrics["rouge_l_f1_overall"] = round(np.mean(rL_f_all), 4) if rL_f_all else 0.0
                except Exception as e: print(f"Error Overall ROUGE: {e}")
            if NLTK_AVAILABLE: # ... same overall BLEU & Distinct ...
                tok_refs_all = [[r.split()] for r in all_refs_for_overall_metrics]; tok_preds_all = [p.split() for p in all_preds_for_overall_metrics]; sf = SmoothingFunction(); bleu_val_overall = None
                try: bleu_val_overall = corpus_bleu(tok_refs_all, tok_preds_all, weights=(0.25,0.25,0.25,0.25), smoothing_function=sf.method1); final_metrics["bleu_4_overall"] = round(bleu_val_overall, 4)
                except ValueError: bleu_val_overall = corpus_bleu(tok_refs_all, tok_preds_all, weights=(1,0,0,0), smoothing_function=sf.method1); final_metrics["bleu_1_overall"] = round(bleu_val_overall, 4)
                except Exception as e_bleu_overall: final_metrics["bleu_score_overall_error"] = str(e_bleu_overall); print(f"Error Overall BLEU: {e_bleu_overall}")
                all_pred_tokens_flat = [token for pred_tokens_list in tok_preds_all for token in pred_tokens_list if token]
                if all_pred_tokens_flat:
                    final_metrics["distinct_1_overall"] = round(len(set(all_pred_tokens_flat)) / len(all_pred_tokens_flat), 4)
                    bigrams = list(ngrams(all_pred_tokens_flat, 2)); final_metrics["distinct_2_overall"] = round(len(set(bigrams)) / len(bigrams), 4) if bigrams else 0.0
                else: final_metrics["distinct_1_overall"] = 0.0; final_metrics["distinct_2_overall"] = 0.0
            if LEVENSHTEIN_AVAILABLE: # ... same overall Edit Distance ...
                try:
                    norm_ed_all = [Levenshtein.distance(p, r) / max(len(p), len(r)) if max(len(p),len(r)) > 0 else 0 for p,r in zip(all_preds_for_overall_metrics, all_refs_for_overall_metrics)]
                    final_metrics["avg_norm_edit_distance_overall"] = round(np.mean(norm_ed_all), 4) if norm_ed_all else 0.0
                except Exception as e: print(f"Error Overall EditDist: {e}")
        else: print("No explanation samples with references to calculate any explanation metrics.")

    # --- Reporting & Saving ---
    print("\n--- Aggregated Evaluation Metrics ---")
    if not final_metrics and not any(v.get('total',0) > 0 for v in accuracy_tasks_counts.values()): print("No metrics calculated.")
    else:
        print("-- Phase 1 Metrics (Move Pred / Rules / Stockfish) --")
        for metric, value in sorted(final_metrics.items()):
             is_exp_metric = any(metric.startswith(p) for p in ["bert_score_", "rouge_", "bleu_", "distinct_", "avg_norm_edit_distance_"])
             if not is_exp_metric: print(f"{metric.replace('_', ' ').title()}: {value if value is not None else 'N/A'}")
        print("\n-- Phase 2 Metrics (Explanation) --")
        for metric, value in sorted(final_metrics.items()):
             is_exp_metric = any(metric.startswith(p) for p in ["bert_score_", "rouge_", "bleu_", "distinct_", "avg_norm_edit_distance_"])
             if is_exp_metric:
                 disp_name = metric
                 for prefix in ["bert_score_", "rouge_", "bleu_", "distinct_", "avg_norm_edit_distance_"]:
                     if disp_name.startswith(prefix): disp_name = disp_name[len(prefix):]; break
                 disp_name = disp_name.replace('_avg', ' Avg').replace('_overall', ' Overall').replace('_f1', ' F1').replace('_precision', ' Prec').replace('_recall', ' Rec')
                 disp_name = disp_name.replace('_', ' ').strip().title()
                 print(f"{disp_name}: {value if value is not None else 'N/A'}")
    print("\nFluency of explanations: Requires qualitative assessment.")

    if args.output_results_file:
        output_dir = os.path.dirname(args.output_results_file);
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types before saving
        output_to_save = {
            "args": convert_numpy_types(vars(args)),
            "aggregated_metrics": convert_numpy_types(final_metrics),
            "accuracy_counts_per_task": convert_numpy_types(dict(accuracy_tasks_counts)),
            "per_sample_results": convert_numpy_types(results_per_sample)
        }
        try:
            with open(args.output_results_file, "w") as f: json.dump(output_to_save, f, indent=4)
            print(f"Detailed results saved to {args.output_results_file}")
        except Exception as e: print(f"Error saving detailed results: {e}")
    
    if args.output_numerical_summary:
        numerical_summary_path = args.output_numerical_summary; summary_dir = os.path.dirname(numerical_summary_path)
        if summary_dir and not os.path.exists(summary_dir): os.makedirs(summary_dir, exist_ok=True)
        print(f"\nSaving numerical summary to: {numerical_summary_path}")
        try:
            p1_count = sum(1 for item in all_data_items_to_process if item.get('data_source') == 'P1')
            p2_count = sum(1 for item in all_data_items_to_process if item.get('data_source') == 'P2')
            with open(numerical_summary_path, "w") as f_summary:
                f_summary.write(f"Eval Summary - Model: {args.model_name}\n")
                f_summary.write(f"P1 Adapter: {args.phase1_lora_path if args.phase1_lora_path else 'N/A'}\n")
                f_summary.write(f"P2 Adapter: {args.phase2_lora_path if args.phase2_lora_path else 'N/A'}\n")
                f_summary.write(f"Test Samples (P1 source): {p1_count}\n"); f_summary.write(f"Test Samples (P2 source): {p2_count}\n")
                if args.eval_move_pred: f_summary.write(f"SF Reference Ks for SSD/Delta: {args.sf_reference_ks}\n") # Log K list
                f_summary.write("\n--- Aggregated Metrics ---\n")
                for metric, value in sorted(final_metrics.items()):
                    f_summary.write(f"{metric}: {value if value is not None else 'N/A'}\n")
                f_summary.write("\n--- Rule-Based Task Counts (Correct/Total) ---\n")
                for task_name, counts_dict in sorted(accuracy_tasks_counts.items()):
                     if counts_dict["total"] > 0:
                        correct_count = counts_dict["correct"]; total_count = counts_dict["total"]
                        f_summary.write(f"{task_name}_counts: {correct_count}/{total_count}\n")
            print(f"Numerical summary saved to {numerical_summary_path}")
        except Exception as e: print(f"Error saving numerical summary: {e}")

    if stockfish_engine: stockfish_engine.quit(); print("Stockfish engine quit.")
    if 'model' in locals(): del model
    if 'tokenizer' in locals(): del tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache(); print("CUDA cache emptied.")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()