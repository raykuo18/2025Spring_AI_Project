#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import sys
import time
import glob
from typing import Optional
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer, # Will be subclassed
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType
)

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # target_modules for the new LoRA adapter being trained in this script
        "target_modules_phase2": ["q_proj", "v_proj"], # User request
        "add_eos_token": True,
    },
    "Gemma-2B": {
        "hf_model_name": "google/gemma-2b",
        "target_modules_phase2": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "add_eos_token": True,
    },
    "Phi-2": {
        "hf_model_name": "microsoft/phi-2",
        "target_modules_phase2": ["q_proj", "v_proj", "k_proj", "dense"],
        "add_eos_token": True,
    }
}

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --- Custom Callbacks (MemoryUsage, CheckpointMetadataLogger - same as before) ---
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, logging_steps_multiplier=1): super().__init__(); self.logging_steps_multiplier = logging_steps_multiplier
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        effective_logging_steps = args.logging_steps * self.logging_steps_multiplier
        if state.global_step > 0 and effective_logging_steps > 0 and state.global_step % effective_logging_steps == 0:
            if torch.cuda.is_available():
                trainer_instance = kwargs.get("trainer", None)
                if trainer_instance:
                    try:
                        current_device = trainer_instance.args.device
                        if current_device.type == 'cuda':
                            allocated = torch.cuda.memory_allocated(device=current_device) / (1024**3)
                            reserved = torch.cuda.memory_reserved(device=current_device) / (1024**3)
                            log_output = {"vram_allocated_gb": round(allocated, 2), "vram_reserved_gb": round(reserved, 2)}
                            trainer_instance.log(log_output)
                    except Exception as e: print(f"Warning: Could not get VRAM usage: {e}")

class CheckpointMetadataLoggerCallback(TrainerCallback):
    def __init__(self, metadata_output_file: Optional[str] = None, all_script_args: Optional[argparse.Namespace] = None):
        super().__init__(); self.metadata_output_file = metadata_output_file; self.all_script_args = all_script_args
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.metadata_output_file or not state.is_world_process_zero: return
        full_checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        train_loss, eval_loss, current_lr = None, None, None; eval_metrics_specific = {}
        trainer = kwargs.get("trainer")
        if trainer and hasattr(trainer, "optimizer") and trainer.optimizer and len(trainer.optimizer.param_groups) > 0: current_lr = trainer.optimizer.param_groups[0].get('lr')
        if state.log_history:
            for log in reversed(state.log_history):
                if train_loss is None and 'loss' in log and 'eval_loss' not in log: train_loss = log['loss']
                if current_lr is None and 'learning_rate' in log: current_lr = log['learning_rate']
                if eval_loss is None and 'eval_loss' in log:
                    eval_loss = log['eval_loss']
                    for k, v in log.items():
                        if k.startswith("eval_") and k not in ["eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]: eval_metrics_specific[k] = v
                    break 
            if train_loss is None: 
                for log in reversed(state.log_history):
                    if 'loss' in log: train_loss = log['loss']; break
        with open(self.metadata_output_file, "a", encoding="utf-8") as f:
            f.write(f"--- Checkpoint @ {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---\n"); f.write(f"Global Step: {state.global_step}\nEpoch: {state.epoch:.2f}\n")
            if current_lr is not None: f.write(f"Learning Rate: {current_lr:.3e}\n")
            if train_loss is not None: f.write(f"Training Loss (last logged): {train_loss:.4f}\n")
            if eval_loss is not None: f.write(f"Validation Loss (last eval): {eval_loss:.4f}\n")
            for name, value in eval_metrics_specific.items(): f.write(f"{name.replace('_', ' ').title()}: {value:.4f}\n")
            f.write(f"Saved Checkpoint To: {full_checkpoint_path}\n--------------------------------------------------\n\n")
        print(f"Metadata logged for checkpoint {state.global_step}.")

class CustomSafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels"); outputs = model(**inputs)
        if labels is not None and hasattr(outputs, "loss") and outputs.loss is not None: loss = outputs.loss
        elif labels is not None and hasattr(outputs, "logits"):
            logits = outputs.logits; labels_on_device = labels.to(logits.device); loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous(); shift_labels = labels_on_device[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = None; print("Warning: CustomSafeTrainer.compute_loss did not find/compute loss.")
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tune a Causal LM LoRA for explanations.")
    # Model and Data Arguments
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Name of the base model.")
    # <<< MODIFIED: phase1_adapter_path is now optional >>>
    parser.add_argument("--phase1_adapter_path", type=str, default=None, help="Optional: Path to pre-trained Phase 1 LoRA adapter to load and freeze.")
    
    parser.add_argument("--train_folder", type=str, required=True, help="Path to folder with training JSONL files (Phase 2 data).")
    parser.add_argument("--val_folder", type=str, required=True, help="Path to folder with validation JSONL files (Phase 2 data).")
    parser.add_argument("--test_folder", type=str, default=None, help="Optional: Path to folder for test JSONL files.")
    
    parser.add_argument("--base_model_cache_dir", type=str, default="./pretrained_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save training runs for Phase 2 adapters.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for this training run.")
    parser.add_argument("--output_adapter_name", type=str, default="phase2_explainer_lora", help="Name for the new LoRA adapter being trained.")

    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--tokenized_data_path", type=str, default=None, help="Base path to save/load tokenized Phase 2 datasets.")
    parser.add_argument("--overwrite_tokenized_cache", action="store_true", help="Force re-tokenization.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples.")
    parser.add_argument("--tokenize_only", action="store_true", help="If set, load data, tokenize, save, and exit.")
    
    # LoRA Arguments (for the new Phase 2 adapter)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank for explanation adapter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha for explanation adapter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout for explanation adapter.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size per GPU.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for explanation adapter.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--gradient_checkpointing_use_reentrant", type=bool, default=False, help="For gradient_checkpointing.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to.")
    parser.add_argument("--checkpoint_metadata_file_name", type=str, default="checkpoint_log_phase2.txt", help="Filename for checkpoint metadata log.")
    parser.add_argument("--vram_log_multiplier", type=int, default=1, help="Log VRAM every N * logging_steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA).")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Use Flash Attention 2.")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Create Unique Run Directory ---
    if args.run_name:
        sane_run_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in args.run_name)
        run_specific_output_dir = os.path.join(args.output_dir, sane_run_name)
    else:
        timestamp = time.strftime("%Y%m%d_%d%H%M%S")
        run_name_prefix = args.model_name.split('/')[-1].replace('-', '_')
        run_specific_output_dir = os.path.join(args.output_dir, f"{run_name_prefix}_phase2_{timestamp}")
    effective_output_dir_for_trainer = run_specific_output_dir
    if not args.tokenize_only:
        os.makedirs(run_specific_output_dir, exist_ok=True)
        print(f"All outputs for this Phase 2 run will be saved to: {run_specific_output_dir}")
        run_config_path = os.path.join(run_specific_output_dir, "run_config_phase2.json")
        try:
            with open(run_config_path, "w") as f: json.dump(vars(args), f, indent=4)
            print(f"Run configuration saved to {run_config_path}")
        except Exception as e: print(f"Warning: Could not save run configuration: {e}")
    
    # --- 1. Load Tokenizer ---
    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]; add_eos_token_to_output = model_config_details.get("add_eos_token", True)
    target_modules_phase2 = model_config_details["target_modules_phase2"]
    print(f"Loading tokenizer for {hf_model_name}..."); tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: print("Tokenizer missing pad_token, setting to eos_token."); tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Dataset Preparation ---
    tokenized_train_dataset: Optional[Dataset] = None; tokenized_eval_dataset: Optional[Dataset] = None; tokenized_test_dataset: Optional[Dataset] = None
    sanitized_model_name_for_cache = hf_model_name.split('/')[-1].replace('-', '_')
    
    datasets_to_process_config = {}
    if args.train_folder and os.path.isdir(args.train_folder): datasets_to_process_config["train"] = {"folder": args.train_folder, "max_samples": args.max_train_samples, "cache_suffix": "train_phase2"}
    if args.val_folder and os.path.isdir(args.val_folder): datasets_to_process_config["validation"] = {"folder": args.val_folder, "max_samples": args.max_val_samples, "cache_suffix": "eval_phase2"}
    if args.test_folder and os.path.isdir(args.test_folder): datasets_to_process_config["test"] = {"folder": args.test_folder, "max_samples": args.max_test_samples, "cache_suffix": "test_phase2"}
    tokenized_datasets_loaded = {}

    for split_name, config in datasets_to_process_config.items():
        folder_path = config["folder"]; max_samples_for_split = config["max_samples"]; cache_suffix_for_split = config["cache_suffix"]
        cache_path_for_split = None; raw_file_paths = glob.glob(os.path.join(folder_path, "*.jsonl"))
        if not raw_file_paths: print(f"Warning: No *.jsonl files found for {split_name} in folder {folder_path}. Skipping."); continue
        
        if args.tokenized_data_path:
            subset_info_suffix = f"_top{max_samples_for_split}" if max_samples_for_split is not None else ""
            cache_dir_for_split = os.path.join(args.tokenized_data_path, f"{sanitized_model_name_for_cache}_seq{args.max_seq_length}_{cache_suffix_for_split}{subset_info_suffix}")
            cache_path_for_split = cache_dir_for_split
            if not args.overwrite_tokenized_cache and os.path.exists(os.path.join(cache_path_for_split, "dataset_info.json")):
                print(f"Loading tokenized {split_name} from cache: {cache_path_for_split}")
                try: tokenized_datasets_loaded[split_name] = Dataset.load_from_disk(cache_path_for_split); print(f"Loaded {len(tokenized_datasets_loaded[split_name])} for {split_name}."); continue
                except Exception as e: print(f"Failed to load {split_name} cache: {e}.")        
        print(f"Preparing to tokenize {split_name} data from {len(raw_file_paths)} file(s) in {folder_path}...")
        # Load all jsonl files from the folder for this split
        raw_dataset_this_split_full = load_dataset("json", data_files=raw_file_paths, split="train") # 'train' is default name for multiple files

        if max_samples_for_split is not None and max_samples_for_split > 0:
            if max_samples_for_split < len(raw_dataset_this_split_full):
                print(f"Sub-sampling RAW {split_name} data to {max_samples_for_split} samples BEFORE tokenization.")
                raw_dataset_this_split = raw_dataset_this_split_full.select(range(max_samples_for_split))
            else: raw_dataset_this_split = raw_dataset_this_split_full
        else: raw_dataset_this_split = raw_dataset_this_split_full
        if not raw_dataset_this_split or len(raw_dataset_this_split) == 0: print(f"Warning: No data for {split_name}. Skipping."); continue

        def preprocess_function(examples): # Same as Phase 1
            inputs_text = []; prompts_text = examples["input"]; outputs_text = examples["output"]
            tokenized_prompts_for_length_calc = tokenizer(prompts_text, max_length=args.max_seq_length, truncation=True, add_special_tokens=True)
            for i in range(len(prompts_text)):
                output_str = str(outputs_text[i]) if outputs_text[i] is not None else ""; text = prompts_text[i] + output_str 
                if add_eos_token_to_output: text += tokenizer.eos_token
                inputs_text.append(text)
            model_inputs = tokenizer(inputs_text, max_length=args.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = model_inputs["input_ids"].clone()
            for i in range(len(prompts_text)):
                prompt_tokens_length = sum(tokenized_prompts_for_length_calc['attention_mask'][i]); mask_len = min(prompt_tokens_length, labels.shape[1]); labels[i, :mask_len] = -100
            model_inputs["labels"] = labels; return model_inputs
        num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        print(f"Tokenizing {split_name} set ({len(raw_dataset_this_split)} raw samples) with {num_proc} procs...")
        tokenized_split = raw_dataset_this_split.map(preprocess_function, batched=True, remove_columns=raw_dataset_this_split.column_names, num_proc=num_proc, desc=f"Tokenizing {split_name}")
        tokenized_datasets_loaded[split_name] = tokenized_split
        if cache_path_for_split: print(f"Saving tokenized {split_name} to cache: {cache_path_for_split}"); os.makedirs(cache_path_for_split, exist_ok=True); tokenized_split.save_to_disk(cache_path_for_split)
    
    tokenized_train_dataset = tokenized_datasets_loaded.get("train"); tokenized_eval_dataset = tokenized_datasets_loaded.get("validation"); tokenized_test_dataset = tokenized_datasets_loaded.get("test")
    if not tokenized_train_dataset: print("Error: Training dataset missing. Exiting."); sys.exit(1)
    print(f"Final train dataset size: {len(tokenized_train_dataset) if tokenized_train_dataset else 0}")
    # ... (rest of dataset print/check logic)

    if args.tokenize_only:
        if args.tokenized_data_path: print(f"Tokenization complete. Phase 2 Datasets saved in: {args.tokenized_data_path}")
        else: print("Tokenization complete. No --tokenized_data_path provided.")
        sys.exit(0)

    if not effective_output_dir_for_trainer: parser.error("--output_dir is effectively required if not --tokenize_only."); sys.exit(1)
    if not tokenized_train_dataset: print("Error: Training data not available. Exiting."); sys.exit(1)

    # --- Model Loading & Adapter Setup ---
    quantization_config = None
    if args.load_in_4bit: quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, bnb_4bit_use_double_quant=True)
    print(f"Loading base model: {hf_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quantization_config, torch_dtype="auto", device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2" if args.use_flash_attention_2 and quantization_config is None else "sdpa",)
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config") and hasattr(model.config, "use_cache"): model.config.use_cache = False # For training
    if args.load_in_4bit: print("Preparing base model for k-bit training (QLoRA)..."); model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # --- LoRA Adapter Handling ---
    if args.phase1_adapter_path:
        print(f"Loading and freezing Phase 1 LoRA adapter from: {args.phase1_adapter_path}")
        model = PeftModel.from_pretrained(model, args.phase1_adapter_path, adapter_name="phase1_move_predictor", is_trainable=False)
        print("Phase 1 adapter loaded and frozen.")
        print(f"Adding new LoRA adapter '{args.output_adapter_name}' for Phase 2 explanation training...")
        lora_config_phase2 = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules_phase2, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
        model.add_adapter(peft_config=lora_config_phase2, adapter_name=args.output_adapter_name)
        combined_adapter_name = "combined_adapter"
        model.add_weighted_adapter(
                adapters=["phase1_move_predictor", args.output_adapter_name],
                weights=[1.0, 1.0],
                adapter_name=combined_adapter_name,
                combination_type="linear"
            )
        print(f"Phase 2 adapter '{args.output_adapter_name}' added. Active adapters set to: {model.active_adapters}.")
    else:
        print(f"No Phase 1 adapter provided. Training new LoRA adapter '{args.output_adapter_name}' directly on base model for Phase 2 task.")
        lora_config_phase2 = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules_phase2, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_config_phase2) # Default adapter name will be 'default'
        # We will save the 'default' adapter to the directory named args.output_adapter_name
        print(f"Applied LoRA configuration for adapter '{args.output_adapter_name}' (internally 'default').")

    if args.gradient_checkpointing: # Ensure GC is enabled on the potentially PEFT model
        # For QLoRA, prepare_model_for_kbit_training should handle GC.
        # For non-QLoRA + PEFT, or to be certain for QLoRA+PEFT:
        print(f"Enabling gradient checkpointing with use_reentrant={args.gradient_checkpointing_use_reentrant} for PEFT model.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})
    
    model.print_trainable_parameters()
    if hasattr(model, "config") and hasattr(model.config, "use_cache") and model.config.use_cache: model.config.use_cache = False
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "config") and hasattr(model.base_model.model.config, "use_cache") and model.base_model.model.config.use_cache:
        try: model.base_model.model.config.use_cache = False; print("Set base_model.model.config.use_cache=False")
        except AttributeError: pass
    
    # --- Training ---
    training_args_dict = {
        "output_dir":effective_output_dir_for_trainer, # ... (rest of TrainingArguments as before) ...
        "num_train_epochs":args.num_train_epochs,
        "per_device_train_batch_size":args.per_device_train_batch_size, 
        "per_device_eval_batch_size":args.per_device_eval_batch_size if tokenized_eval_dataset else args.per_device_train_batch_size,
        "gradient_accumulation_steps":args.gradient_accumulation_steps, "learning_rate":args.learning_rate, "weight_decay":args.weight_decay, "warmup_ratio":args.warmup_ratio, "lr_scheduler_type":args.lr_scheduler_type, 
        "optim":args.optim, "logging_dir":os.path.join(effective_output_dir_for_trainer, "hf_trainer_logs_phase2"), "logging_steps":args.logging_steps,
        "evaluation_strategy":"steps" if tokenized_eval_dataset and args.eval_steps > 0 else "no", "eval_steps":args.eval_steps if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "save_strategy":"steps", "save_steps":args.save_steps, "save_total_limit":args.save_total_limit,
        "load_best_model_at_end":True if tokenized_eval_dataset and args.eval_steps > 0 else False, "metric_for_best_model":"eval_loss" if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "greater_is_better":False if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "report_to":args.report_to.split(',') if args.report_to and args.report_to != "none" else None,
        "fp16":torch.cuda.is_available() and not args.load_in_4bit and not torch.cuda.is_bf16_supported(),
        "bf16":torch.cuda.is_bf16_supported() and not args.load_in_4bit, "seed":args.seed,
    }
    if args.gradient_checkpointing: training_args_dict["gradient_checkpointing"] = True; training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
    training_args = TrainingArguments(**training_args_dict)

    callbacks_to_use = []
    memory_callback = MemoryUsageCallback(logging_steps_multiplier=args.vram_log_multiplier); callbacks_to_use.append(memory_callback)
    if args.checkpoint_metadata_file_name and not args.tokenize_only:
        metadata_log_filepath = os.path.join(effective_output_dir_for_trainer, args.checkpoint_metadata_file_name)
        if not os.path.exists(os.path.dirname(metadata_log_filepath)): os.makedirs(os.path.dirname(metadata_log_filepath), exist_ok=True)
        initial_log_mode = "w" if not os.path.exists(metadata_log_filepath) or os.path.getsize(metadata_log_filepath) == 0 else "a"
        with open(metadata_log_filepath, initial_log_mode, encoding="utf-8") as f:
            if initial_log_mode == "w" or (os.path.exists(metadata_log_filepath) and os.path.getsize(metadata_log_filepath) == 0) :
                 f.write(f"P2 Checkpoint Log - Run: {os.path.basename(effective_output_dir_for_trainer)} @ {time.strftime('%Y%m%d-%H%M%S')}\n"); [f.write(f"  {k}: {v}\n") for k, v in sorted(vars(args).items())]; f.write("--------------------------------------------------\n\n")
        metadata_logger_callback = CheckpointMetadataLoggerCallback(metadata_output_file=metadata_log_filepath, all_script_args=args ); callbacks_to_use.append(metadata_logger_callback)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Initializing CustomSafeTrainer for Phase 2..."); trainer = CustomSafeTrainer(model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_eval_dataset, tokenizer=tokenizer, data_collator=data_collator, callbacks=callbacks_to_use)
    print("Starting Phase 2 training (explanation adapter)..."); trainer.train(); print("Phase 2 training complete.")
    
    if tokenized_test_dataset:
        print("\nEvaluating on the Phase 2 test set..."); test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test_phase2")
        print(f"Phase 2 Test Set Metrics: {test_results}"); test_results_path = os.path.join(effective_output_dir_for_trainer, "test_results_phase2.json")
        with open(test_results_path, "w") as f: json.dump(test_results, f, indent=4); print(f"Test results saved: {test_results_path}")
    
    # --- Saving the trained Phase 2 adapter ---
    # The directory name uses args.output_adapter_name for clarity.
    # The adapter *inside* this directory might be named 'default' if no P1 adapter was loaded,
    # or it will be args.output_adapter_name if P1 was loaded and P2 was added with that name.
    final_adapter_dir_path = os.path.join(effective_output_dir_for_trainer, args.output_adapter_name) 
    print(f"Saving final Phase 2 LoRA adapter to directory: {final_adapter_dir_path}...")
    
    if args.phase1_adapter_path:
        # If P1 adapter was loaded, we specifically trained args.output_adapter_name
        model.save_pretrained(final_adapter_dir_path, selected_adapters=[args.output_adapter_name])
        print(f"Phase 2 LoRA adapter '{args.output_adapter_name}' saved successfully.")
    else:
        # If no P1 adapter, get_peft_model created a 'default' adapter which was trained.
        # save_pretrained on the PeftModel will save the active ('default') adapter.
        # The directory 'final_adapter_dir_path' IS ALREADY NAMED WITH args.output_adapter_name.
        model.save_pretrained(final_adapter_dir_path) 
        print(f"Phase 2 LoRA adapter (likely named 'default' internally) saved to directory '{args.output_adapter_name}'.")

    print(f"Base model was: {hf_model_name}")
    if args.phase1_adapter_path:
        print(f"Built on top of Phase 1 adapter from: {args.phase1_adapter_path}")


if __name__ == "__main__":
    main()
