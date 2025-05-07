#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
import sys
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback, # Added for custom callback
    TrainerState,    # Added
    TrainerControl   # Added
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# tqdm is implicitly used by datasets.map and Trainer

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
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

# --- Custom Callback for VRAM Logging ---
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, logging_steps_multiplier=1):
        super().__init__()
        # This multiplier allows logging VRAM less frequently than main logs if desired
        # e.g., if logging_steps is 10, and multiplier is 5, VRAM logs every 50 steps.
        self.logging_steps_multiplier = logging_steps_multiplier 

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log VRAM usage at the same frequency as other logs by default
        effective_logging_steps = args.logging_steps * self.logging_steps_multiplier
        if state.global_step > 0 and effective_logging_steps > 0 and state.global_step % effective_logging_steps == 0:
            if torch.cuda.is_available():
                trainer_instance = kwargs.get("trainer", None) # Trainer instance is passed in kwargs
                if trainer_instance:
                    allocated = torch.cuda.memory_allocated(device=trainer_instance.args.device) / (1024**3)  # GB
                    reserved = torch.cuda.memory_reserved(device=trainer_instance.args.device) / (1024**3)    # GB
                    # max_allocated = torch.cuda.max_memory_allocated(device=trainer_instance.args.device) / (1024**3) # Peak
                    
                    log_output = {
                        "vram_allocated_gb": round(allocated, 2),
                        "vram_reserved_gb": round(reserved, 2),
                        # "vram_max_allocated_gb": round(max_allocated, 2)
                    }
                    trainer_instance.log(log_output) # Use trainer's log method to send to reporters
                else: # Fallback print if trainer instance isn't available (should be)
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    print(f"Step {state.global_step}: VRAM Usage - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB (trainer not found in callback kwargs)")


# --- Custom Trainer (to handle potential device mismatches if they re-occur) ---
# (Keeping the CustomSafeTrainer as it's good practice for device_map="auto")
class CustomSafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif labels is not None and hasattr(outputs, "logits"):
            logits = outputs.logits
            labels_on_device = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_on_device[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = None
            if not (hasattr(outputs, "loss") and outputs.loss is not None):
                 print("Warning: CustomSafeTrainer.compute_loss: Model did not return loss and manual computation prerequisites not fully met.")
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune or only tokenize for Causal LM with LoRA.")
    # Model and Data Arguments
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(), help="Name of the base model.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation JSONL.")
    parser.add_argument("--test_file", type=str, default=None, help="Optional: Path to test JSONL for final evaluation.")
    parser.add_argument("--base_model_cache_dir", type=str, default="./pretrained_cache", help="Cache dir for Hugging Face models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir for LoRA adapter & training artifacts (if training).")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length (affects speed and memory).")

    # Tokenized Data Caching & Sub-sampling Arguments
    parser.add_argument("--tokenized_data_path", type=str, default=None, help="Base path to save/load tokenized datasets.")
    parser.add_argument("--overwrite_tokenized_cache", action="store_true", help="Force re-tokenization.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples to load/tokenize (for faster testing).")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples to load/tokenize.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples to load/tokenize.")
    parser.add_argument("--tokenize_only", action="store_true", help="If set, load data, tokenize, save tokenized data, and exit.")
    
    # LoRA Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs (1-3 often enough for LoRA).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Train batch size per GPU (increase as VRAM allows).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation (effective_batch_size = N_GPU * per_device_bs * grad_accum).")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer (paged_adamw_8bit for QLoRA).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing (saves VRAM, slows steps).")
    parser.add_argument("--gradient_checkpointing_use_reentrant", type=bool, default=False, help="For gradient_checkpointing (False recommended).")

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics (loss, VRAM etc.) every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to (e.g. tensorboard, wandb, none).")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (QLoRA).")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Use Flash Attention 2 (if available and not using 4-bit).")
    parser.add_argument("--vram_log_multiplier", type=int, default=1, help="Log VRAM every N * logging_steps. Default 1 (same as other logs).")


    args = parser.parse_args()
    set_seed(args.seed)

    # --- 1. Load Tokenizer ---
    model_config_details = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config_details["hf_model_name"]
    add_eos_token_to_output = model_config_details.get("add_eos_token", True)

    print(f"Loading tokenizer for {hf_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.base_model_cache_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer missing pad_token, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Dataset Preparation ---
    # ... (Dataset loading, tokenizing, caching logic from your previous script, ensuring preprocess_function is the optimized one) ...
    tokenized_train_dataset: Optional[Dataset] = None
    tokenized_eval_dataset: Optional[Dataset] = None
    tokenized_test_dataset: Optional[Dataset] = None
    sanitized_model_name = hf_model_name.split('/')[-1].replace('-', '_')
    datasets_to_process_config = {
        "train": {"file": args.train_file, "max_samples": args.max_train_samples, "cache_suffix": "train"},
        "validation": {"file": args.val_file, "max_samples": args.max_val_samples, "cache_suffix": "eval"},
    }
    if args.test_file: datasets_to_process_config["test"] = {"file": args.test_file, "max_samples": args.max_test_samples, "cache_suffix": "test"}
    tokenized_datasets_loaded = {}
    for split_name, config in datasets_to_process_config.items():
        raw_file_path = config["file"]; max_samples_for_split = config["max_samples"]; cache_suffix_for_split = config["cache_suffix"]
        cache_path_for_split = None
        if args.tokenized_data_path:
            subset_info_suffix = f"_top{max_samples_for_split}" if max_samples_for_split is not None else ""
            cache_dir_for_split = os.path.join(args.tokenized_data_path, f"{sanitized_model_name}_seq{args.max_seq_length}_{cache_suffix_for_split}{subset_info_suffix}")
            cache_path_for_split = cache_dir_for_split
            if not args.overwrite_tokenized_cache and os.path.exists(os.path.join(cache_path_for_split, "dataset_info.json")):
                print(f"Loading tokenized {split_name} data from cache: {cache_path_for_split}")
                try: tokenized_datasets_loaded[split_name] = Dataset.load_from_disk(cache_path_for_split); print(f"Loaded {len(tokenized_datasets_loaded[split_name])} samples for {split_name} from cache."); continue
                except Exception as e: print(f"Failed to load {split_name} cache: {e}. Will re-tokenize.")
        print(f"Preparing to tokenize {split_name} data from {raw_file_path}...")
        current_data_file_dict_for_load = {split_name: raw_file_path}; slicing_instruction = split_name
        if max_samples_for_split is not None and max_samples_for_split > 0:
            slicing_instruction = f"{split_name}[:{max_samples_for_split}]"
            print(f"Will load a maximum of {max_samples_for_split} raw samples for {split_name} set using split='{slicing_instruction}'.")
        try: raw_dataset_this_split = load_dataset("json", data_files=current_data_file_dict_for_load, split=slicing_instruction)
        except Exception as e: print(f"Error loading raw {split_name} data: {e}. Skipping this split."); continue

        # Using the optimized preprocess_function from previous iterations
        def preprocess_function(examples):
            inputs_text = []
            prompts_text = examples["input"]
            outputs_text = examples["output"]

            # Tokenize all prompts in the batch once to get their lengths efficiently
            # Ensure this tokenization matches how they appear in the final combined input
            # (e.g., regarding special tokens like BOS if added by the main tokenizer for the sequence)
            tokenized_prompts_for_length_calc = tokenizer(
                prompts_text,
                max_length=args.max_seq_length, # Apply truncation to prompt if it's too long
                truncation=True,
                add_special_tokens=True # To match how the prompt part appears in the combined sequence
            )

            for i in range(len(prompts_text)):
                output_str = str(outputs_text[i]) if outputs_text[i] is not None else ""
                text = prompts_text[i] + output_str 
                if add_eos_token_to_output: text += tokenizer.eos_token
                inputs_text.append(text)

            model_inputs = tokenizer(inputs_text, max_length=args.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            labels = model_inputs["input_ids"].clone()

            for i in range(len(prompts_text)):
                # Use the attention mask from the prompt-only tokenization to get true length
                prompt_tokens_length = sum(tokenized_prompts_for_length_calc['attention_mask'][i]).item()
                mask_len = min(prompt_tokens_length, labels.shape[1])
                labels[i, :mask_len] = -100
            
            model_inputs["labels"] = labels
            return model_inputs

        num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        print(f"Tokenizing {split_name} set ({len(raw_dataset_this_split)} raw samples) with {num_proc} processes...")
        tokenized_split = raw_dataset_this_split.map(preprocess_function, batched=True, remove_columns=raw_dataset_this_split.column_names, num_proc=num_proc, desc=f"Tokenizing {split_name} set")
        tokenized_datasets_loaded[split_name] = tokenized_split
        if cache_path_for_split: print(f"Saving tokenized {split_name} data to cache: {cache_path_for_split}"); os.makedirs(cache_path_for_split, exist_ok=True); tokenized_split.save_to_disk(cache_path_for_split)
    
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

    if not args.output_dir: parser.error("--output_dir required if not --tokenize_only.")
    if not tokenized_train_dataset: print("Error: Training data not available. Exiting."); sys.exit(1)

    # --- Model loading for training ---
    target_modules = model_config_details["target_modules"]
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                             bnb_4bit_use_double_quant=True)
    print(f"Loading base model for training: {hf_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, cache_dir=args.base_model_cache_dir, quantization_config=quantization_config,
        torch_dtype="auto", device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 and quantization_config is None else "sdpa",
    )
    if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.eos_token_id
    
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        print(f"Setting model.config.use_cache to False for training. Original: {model.config.use_cache}")
        model.config.use_cache = False # Disable for training/gradient checkpointing

    if args.load_in_4bit: 
        print("Preparing model for k-bit training (QLoRA)...")
        # Pass gradient_checkpointing explicitly to prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    
    # Explicitly enable gradient checkpointing on the model if requested,
    # and set use_reentrant. This is important if QLoRA's prepare didn't fully set it,
    # or if not using QLoRA.
    if args.gradient_checkpointing:
        print(f"Enabling gradient checkpointing with use_reentrant={args.gradient_checkpointing_use_reentrant}")
        # For PEFT models, gradient checkpointing needs to be enabled on the base model
        # before wrapping or by enabling it on the PeftModel that then delegates.
        # `get_peft_model` might handle some of this, but explicit can be clearer.
        # If model is already a PeftModel, this might need adjustment.
        # Let's assume `prepare_model_for_kbit_training` handles it for QLoRA.
        # For non-QLoRA, we enable it on the base model before PEFT.
        if not args.load_in_4bit: # If not QLoRA
             model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant})


    print("Setting up LoRA configuration..."); 
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config); 
    model.print_trainable_parameters()

    # Ensure use_cache is False after PEFT wrapping as well, as get_peft_model might restore it from base config.
    if hasattr(model, "config") and hasattr(model.config, "use_cache") and model.config.use_cache:
        print(f"Re-setting model.config.use_cache to False after PEFT model wrapping.")
        model.config.use_cache = False
    # For PEFT models, the underlying model's config might need to be set.
    # Try to set it on the peft_config of the base model if it exists
    try:
        if model.base_model.model.config.use_cache: # Common path for PEFT wrapped models
            model.base_model.model.config.use_cache = False
            print("Set model.base_model.model.config.use_cache to False")
    except AttributeError:
        pass # If the model structure is different

    print("Setting up Training Arguments..."); 
    training_args_dict = {
        "output_dir":args.output_dir, "num_train_epochs":args.num_train_epochs,
        "per_device_train_batch_size":args.per_device_train_batch_size, 
        "per_device_eval_batch_size":args.per_device_eval_batch_size if tokenized_eval_dataset else args.per_device_train_batch_size,
        "gradient_accumulation_steps":args.gradient_accumulation_steps, "learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay, "warmup_ratio":args.warmup_ratio, "lr_scheduler_type":args.lr_scheduler_type, 
        "optim":args.optim, "logging_dir":os.path.join(args.output_dir, "logs"), "logging_steps":args.logging_steps,
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
    # Explicitly pass gradient checkpointing arguments if enabled
    if args.gradient_checkpointing:
        training_args_dict["gradient_checkpointing"] = True
        training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
    
    training_args = TrainingArguments(**training_args_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # --- Instantiate Custom Trainer with MemoryUsageCallback ---
    memory_callback = MemoryUsageCallback(logging_steps_multiplier=args.vram_log_multiplier)
    print("Initializing CustomSafeTrainer with MemoryUsageCallback...")
    trainer = CustomSafeTrainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset, 
        eval_dataset=tokenized_eval_dataset, 
        tokenizer=tokenizer, 
        data_collator=data_collator,
        callbacks=[memory_callback] # Add the custom callback
    )
    
    print("Starting training..."); trainer.train(); print("Training complete.")
    
    if tokenized_test_dataset:
        print("\nEvaluating on the test set..."); test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test")
        print(f"Test Set Metrics: {test_results}")
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f: json.dump(test_results, f, indent=4)
        print(f"Test results saved to {test_results_path}")
    
    final_adapter_path = os.path.join(args.output_dir, "final_lora_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}..."); model.save_pretrained(final_adapter_path)
    print("LoRA adapter saved successfully."); print(f"Base model was: {hf_model_name}")

    # Note on torch.compile: For PyTorch 2.0+, you could experiment with:
    # if hasattr(torch, 'compile') and not args.load_in_4bit:
    #     print("Attempting to compile model with torch.compile()...")
    #     model = torch.compile(model) # May speed up training/inference but has limitations

if __name__ == "__main__":
    main()