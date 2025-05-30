#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import sys
import time # For timestamped run directories
import glob
from typing import Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "v_proj"], # <<< MODIFIED as per request
        "add_eos_token": True,
    },
    "Gemma-2B": {
        "hf_model_name": "google/gemma-2b",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "add_eos_token": True,
    },
    "Phi-2": {
        "hf_model_name": "microsoft/phi-2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "dense"],
        "add_eos_token": True,
    }
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Custom Callbacks ---
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, logging_steps_multiplier=1):
        super().__init__()
        self.logging_steps_multiplier = logging_steps_multiplier
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        effective_logging_steps = args.logging_steps * self.logging_steps_multiplier
        if state.global_step > 0 and effective_logging_steps > 0 and state.global_step % effective_logging_steps == 0:
            if torch.cuda.is_available():
                trainer_instance = kwargs.get("trainer", None)
                if trainer_instance:
                    current_device = trainer_instance.args.device
                    allocated = torch.cuda.memory_allocated(device=current_device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device=current_device) / (1024**3)
                    log_output = {"vram_allocated_gb": round(allocated, 2), "vram_reserved_gb": round(reserved, 2)}
                    trainer_instance.log(log_output)

class CheckpointMetadataLoggerCallback(TrainerCallback):
    def __init__(self, metadata_output_file: Optional[str] = None, all_script_args: Optional[argparse.Namespace] = None):
        super().__init__()
        self.metadata_output_file = metadata_output_file
        self.all_script_args = all_script_args
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.metadata_output_file or not state.is_world_process_zero: return
        checkpoint_dir_name = f"checkpoint-{state.global_step}"
        full_checkpoint_path = os.path.join(args.output_dir, checkpoint_dir_name)
        train_loss, eval_loss, current_lr = None, None, None
        eval_metrics_specific = {}
        trainer = kwargs.get("trainer")
        if trainer and hasattr(trainer, "optimizer") and trainer.optimizer and len(trainer.optimizer.param_groups) > 0:
            current_lr = trainer.optimizer.param_groups[0].get('lr')
        if state.log_history:
            for log in reversed(state.log_history):
                if train_loss is None and 'loss' in log and 'eval_loss' not in log: train_loss = log['loss']
                if current_lr is None and 'learning_rate' in log: current_lr = log['learning_rate']
                if eval_loss is None and 'eval_loss' in log:
                    eval_loss = log['eval_loss']
                    for k, v in log.items():
                        if k.startswith("eval_") and k not in ["eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]:
                            eval_metrics_specific[k] = v
                    break
            if train_loss is None: # Fallback if only eval logs exist so far
                for log in reversed(state.log_history):
                    if 'loss' in log: train_loss = log['loss']; break
        with open(self.metadata_output_file, "a", encoding="utf-8") as f:
            f.write(f"--- Checkpoint @ {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---\n")
            f.write(f"Global Step: {state.global_step}\nEpoch: {state.epoch:.2f}\n")
            if current_lr is not None: f.write(f"Learning Rate: {current_lr:.3e}\n")
            if train_loss is not None: f.write(f"Training Loss (last logged): {train_loss:.4f}\n")
            if eval_loss is not None: f.write(f"Validation Loss (last eval): {eval_loss:.4f}\n")
            for name, value in eval_metrics_specific.items(): f.write(f"{name.replace('_', ' ').title()}: {value:.4f}\n")
            f.write(f"Saved Checkpoint To: {full_checkpoint_path}\n")
            f.write("--------------------------------------------------\n\n")
        print(f"Metadata logged for checkpoint {state.global_step}.")

class CustomSafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif labels is not None and hasattr(outputs, "logits"):
            logits = outputs.logits; labels_on_device = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous(); shift_labels = labels_on_device[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = None
            if not (hasattr(outputs, "loss") and outputs.loss is not None): print("Warning: CustomSafeTrainer.compute_loss: Model did not return loss or prerequisites not met.")
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Fine-tune or only tokenize for Causal LM with LoRA.")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Name of the base model.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation JSONL.")
    parser.add_argument("--test_file", type=str, default=None, help="Optional: Path to test JSONL for final evaluation.")
    parser.add_argument("--base_model_cache_dir", type=str, default="./pretrained_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save training runs.")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for this training run. If None, a timestamped name is generated.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--tokenized_data_path", type=str, default=None, help="Base path to save/load tokenized datasets.")
    parser.add_argument("--overwrite_tokenized_cache", action="store_true", help="Force re-tokenization.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples.")
    parser.add_argument("--tokenize_only", action="store_true", help="If set, load data, tokenize, save, and exit.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size per GPU.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--gradient_checkpointing_use_reentrant", type=bool, default=False, help="For gradient_checkpointing (False recommended).")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Limit total checkpoints (plus best).")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to (e.g. tensorboard, wandb, none).")
    # <<< MODIFIED: checkpoint_metadata_file_name default value >>>
    parser.add_argument("--checkpoint_metadata_file_name", type=str, default="checkpoint_log.txt", 
                        help="Filename for the checkpoint metadata log within the run directory.")
    parser.add_argument("--vram_log_multiplier", type=int, default=1, help="Log VRAM every N * logging_steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA).")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Use Flash Attention 2.")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Create Unique Run Directory ---
    if args.run_name:
        # Sanitize run_name to be a valid directory name
        sane_run_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in args.run_name)
        run_specific_output_dir = os.path.join(args.output_dir, sane_run_name)
    else:
        # <<< MODIFIED: Timestamp format for auto-generated run_name >>>
        timestamp = time.strftime("%d%H%M%S")
        run_specific_output_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{timestamp}")
    
    effective_output_dir_for_trainer = run_specific_output_dir # This will be used by TrainingArguments
    
    if not args.tokenize_only:
        if os.path.exists(run_specific_output_dir) and os.listdir(run_specific_output_dir) and not any("resume_from_checkpoint" in arg for arg in sys.argv):
             print(f"Warning: Output directory {run_specific_output_dir} already exists and is not empty. May overwrite or resume.")
        os.makedirs(run_specific_output_dir, exist_ok=True)
        print(f"All outputs for this run will be saved to: {run_specific_output_dir}")

        run_config_path = os.path.join(run_specific_output_dir, "run_config.json")
        try:
            with open(run_config_path, "w") as f: json.dump(vars(args), f, indent=4)
            print(f"Run configuration saved to {run_config_path}")
        except Exception as e: print(f"Warning: Could not save run configuration: {e}")
    
    # --- 1. Load Tokenizer ---
    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]
    add_eos_token_to_output = model_config_details.get("add_eos_token", True)
    print(f"Loading tokenizer for {hf_model_name}..."); tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: print("Tokenizer missing pad_token, setting to eos_token."); tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Dataset Preparation ---
    tokenized_train_dataset: Optional[Dataset] = None; tokenized_eval_dataset: Optional[Dataset] = None; tokenized_test_dataset: Optional[Dataset] = None
    sanitized_model_name_for_cache = hf_model_name.split('/')[-1].replace('-', '_') # Use for cache path consistency
    datasets_to_process_config = {"train": {"file": args.train_file, "max_samples": args.max_train_samples, "cache_suffix": "train"},"validation": {"file": args.val_file, "max_samples": args.max_val_samples, "cache_suffix": "eval"},}
    if args.test_file: datasets_to_process_config["test"] = {"file": args.test_file, "max_samples": args.max_test_samples, "cache_suffix": "test"}
    tokenized_datasets_loaded = {}
    for split_name, config in datasets_to_process_config.items():
        raw_file_path = config["file"]; max_samples_for_split = config["max_samples"]; cache_suffix_for_split = config["cache_suffix"]
        cache_path_for_split = None
        if args.tokenized_data_path:
            subset_info_suffix = f"_top{max_samples_for_split}" if max_samples_for_split is not None else ""
            cache_dir_for_split = os.path.join(args.tokenized_data_path, f"{sanitized_model_name_for_cache}_seq{args.max_seq_length}_{cache_suffix_for_split}{subset_info_suffix}")
            cache_path_for_split = cache_dir_for_split
            if not args.overwrite_tokenized_cache and os.path.exists(os.path.join(cache_path_for_split, "dataset_info.json")):
                print(f"Loading tokenized {split_name} from cache: {cache_path_for_split}")
                try: tokenized_datasets_loaded[split_name] = Dataset.load_from_disk(cache_path_for_split); print(f"Loaded {len(tokenized_datasets_loaded[split_name])} for {split_name}."); continue
                except Exception as e: print(f"Failed to load {split_name} cache: {e}.")        
        print(f"Preparing to tokenize {split_name} from {raw_file_path}...")
        current_data_file_dict_for_load = {split_name: raw_file_path}; slicing_instruction = split_name
        if max_samples_for_split is not None and max_samples_for_split > 0: slicing_instruction = f"{split_name}[:{max_samples_for_split}]"; print(f"Will load max {max_samples_for_split} for {split_name} using split='{slicing_instruction}'.")
        try: raw_dataset_this_split = load_dataset("json", data_files=current_data_file_dict_for_load, split=slicing_instruction)
        except Exception as e: print(f"Error loading raw {split_name}: {e}. Skipping."); continue
        def preprocess_function(examples):
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
        print(f"Tokenizing {split_name} ({len(raw_dataset_this_split)} raw) with {num_proc} procs...")
        tokenized_split = raw_dataset_this_split.map(preprocess_function, batched=True, remove_columns=raw_dataset_this_split.column_names, num_proc=num_proc, desc=f"Tokenizing {split_name}")
        tokenized_datasets_loaded[split_name] = tokenized_split
        if cache_path_for_split: print(f"Saving tokenized {split_name} to cache: {cache_path_for_split}"); os.makedirs(cache_path_for_split, exist_ok=True); tokenized_split.save_to_disk(cache_path_for_split)
    tokenized_train_dataset = tokenized_datasets_loaded.get("train"); tokenized_eval_dataset = tokenized_datasets_loaded.get("validation"); tokenized_test_dataset = tokenized_datasets_loaded.get("test")
    if not tokenized_train_dataset: print("Error: Training dataset missing. Exiting."); sys.exit(1)
    if args.val_file and not tokenized_eval_dataset: print("Warning: Validation dataset specified but could not be prepared.")
    if args.test_file and not tokenized_test_dataset: print("Warning: Test dataset specified but could not be prepared.")
    print(f"Final train dataset size: {len(tokenized_train_dataset) if tokenized_train_dataset else 0}")
    print(f"Final validation dataset size: {len(tokenized_eval_dataset) if tokenized_eval_dataset else 0}")
    if tokenized_test_dataset: print(f"Final test dataset size: {len(tokenized_test_dataset)}")

    if args.tokenize_only:
        if args.tokenized_data_path: print(f"Tokenization complete. Datasets saved in: {args.tokenized_data_path}")
        else: print("Tokenization complete. No --tokenized_data_path, data not saved by script.")
        sys.exit(0)

    # effective_output_dir_for_trainer was already set
    if not effective_output_dir_for_trainer: parser.error("--output_dir (interpreted as run base) is effectively required if not --tokenize_only."); sys.exit(1)
    if not tokenized_train_dataset: print("Error: Training data not available for training. Exiting."); sys.exit(1)

    target_modules = model_config_details["target_modules"]
    quantization_config = None
    if args.load_in_4bit: quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, bnb_4bit_use_double_quant=True)
    print(f"Loading base model for training: {hf_model_name} into {effective_output_dir_for_trainer} (run specific)...")
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quantization_config, torch_dtype="auto", device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2" if args.use_flash_attention_2 and quantization_config is None else "sdpa",)
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config") and hasattr(model.config, "use_cache"): model.config.use_cache = False
    if args.load_in_4bit: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing and not args.load_in_4bit: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})
    print("Setting up LoRA configuration..."); lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config); model.print_trainable_parameters()
    if hasattr(model, "config") and hasattr(model.config, "use_cache") and model.config.use_cache: model.config.use_cache = False
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "config") and hasattr(model.base_model.model.config, "use_cache") and model.base_model.model.config.use_cache:
        try: model.base_model.model.config.use_cache = False; print("Set model.base_model.model.config.use_cache to False")
        except AttributeError: pass
    
    training_args_dict = {
        "output_dir": effective_output_dir_for_trainer, # Use the unique run directory
        "num_train_epochs":args.num_train_epochs,
        "per_device_train_batch_size":args.per_device_train_batch_size, 
        "per_device_eval_batch_size":args.per_device_eval_batch_size if tokenized_eval_dataset else args.per_device_train_batch_size,
        "gradient_accumulation_steps":args.gradient_accumulation_steps, "learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay, "warmup_ratio":args.warmup_ratio, "lr_scheduler_type":args.lr_scheduler_type, 
        "optim":args.optim, "logging_dir":os.path.join(effective_output_dir_for_trainer, "hf_trainer_logs"),
        "logging_steps":args.logging_steps,
        "evaluation_strategy":"steps" if tokenized_eval_dataset and args.eval_steps > 0 else "no", 
        "eval_steps":args.eval_steps if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "save_strategy":"steps", "save_steps":args.save_steps, "save_total_limit":args.save_total_limit,
        "load_best_model_at_end":True if tokenized_eval_dataset and args.eval_steps > 0 else False,
        "metric_for_best_model":"eval_loss" if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "greater_is_better":False if tokenized_eval_dataset and args.eval_steps > 0 else None,
        "report_to":args.report_to.split(',') if args.report_to and args.report_to != "none" else None,
        "fp16":torch.cuda.is_available() and not args.load_in_4bit and not torch.cuda.is_bf16_supported(),
        "bf16":torch.cuda.is_bf16_supported() and not args.load_in_4bit, "seed":args.seed,
    }
    if args.gradient_checkpointing:
        training_args_dict["gradient_checkpointing"] = True
        training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
    training_args = TrainingArguments(**training_args_dict)

    callbacks_to_use = []
    memory_callback = MemoryUsageCallback(logging_steps_multiplier=args.vram_log_multiplier)
    callbacks_to_use.append(memory_callback)
    if args.checkpoint_metadata_file_name and not args.tokenize_only: # Only init if training
        metadata_log_filepath = os.path.join(effective_output_dir_for_trainer, args.checkpoint_metadata_file_name)
        # Write initial header for this run to the metadata file
        if not os.path.exists(os.path.dirname(metadata_log_filepath)): os.makedirs(os.path.dirname(metadata_log_filepath), exist_ok=True)
        initial_log_mode = "w" if not os.path.exists(metadata_log_filepath) else "a" # 'w' if new, 'a' if resuming run and file exists
        with open(metadata_log_filepath, initial_log_mode, encoding="utf-8") as f:
            if initial_log_mode == "w" or os.path.getsize(metadata_log_filepath) == 0 :
                 f.write(f"Checkpoint Metadata Log - Run: {os.path.basename(effective_output_dir_for_trainer)} - Started at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
                 f.write("Initial Training Arguments (from script args):\n")
                 for arg_name, value in sorted(vars(args).items()): f.write(f"  {arg_name}: {value}\n")
                 f.write("--------------------------------------------------\n\n")
        metadata_logger_callback = CheckpointMetadataLoggerCallback(metadata_output_file=metadata_log_filepath, all_script_args=args )
        callbacks_to_use.append(metadata_logger_callback)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print("Initializing CustomSafeTrainer..."); trainer = CustomSafeTrainer(model=model, args=training_args, train_dataset=tokenized_train_dataset, eval_dataset=tokenized_eval_dataset, tokenizer=tokenizer, data_collator=data_collator, callbacks=callbacks_to_use)
    print("Starting training..."); trainer.train(); print("Training complete.")
    if tokenized_test_dataset:
        print("\nEvaluating on the test set..."); test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test")
        print(f"Test Set Metrics: {test_results}"); test_results_path = os.path.join(effective_output_dir_for_trainer, "test_results.json")
        with open(test_results_path, "w") as f: json.dump(test_results, f, indent=4)
        print(f"Test results saved to {test_results_path}")
    final_adapter_path = os.path.join(effective_output_dir_for_trainer, "final_lora_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}..."); model.save_pretrained(final_adapter_path)
    print("LoRA adapter saved successfully."); print(f"Base model was: {hf_model_name}")

if __name__ == "__main__":
    main()