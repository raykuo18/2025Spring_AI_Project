#!/usr/bin/env python3

import argparse
import os
import json
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig # If you want to load base model in 4-bit for QLoRA
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# --- Model Configuration Mapping ---
MODEL_CONFIGS = {
    "TinyLLaMA": {
        "hf_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], # Common for Llama
        "add_eos_token": True, # Typically for chat models
    },
    "Gemma-2B": {
        "hf_model_name": "google/gemma-2b", # Use "google/gemma-2b-it" for instruct version
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], # Common for Gemma
        "add_eos_token": True,
    },
    "Phi-2": {
        "hf_model_name": "microsoft/phi-2",
        "target_modules": ["q_proj", "v_proj", "k_proj", "dense"], # More common for Phi-2 architecture
        "add_eos_token": True,
    }
}

def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Causal LM with LoRA for Phase 1 tasks.")

    # Model and Data Arguments
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_CONFIGS.keys(),
                        help="Name of the base model to fine-tune.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data JSONL file.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation data JSONL file.")
    parser.add_argument("--base_model_cache_dir", type=str, default="./pretrained_cache",
                        help="Directory to cache Hugging Face models.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned LoRA adapter and training artifacts.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for tokenization.")

    # LoRA Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (rank).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate.") # Common for LoRA
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer to use (e.g., adamw_torch, paged_adamw_8bit for QLoRA).")


    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to (e.g., tensorboard, wandb).")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit for QLoRA-style training.")
    parser.add_argument("--use_flash_attention_2", action="store_true", help="Use Flash Attention 2 if available (requires compatible hardware/libraries).")


    args = parser.parse_args()
    set_seed(args.seed)

    # --- 1. Load Model and Tokenizer ---
    model_config = MODEL_CONFIGS[args.model_name]
    hf_model_name = model_config["hf_model_name"]
    target_modules = model_config["target_modules"]
    add_eos_token_to_output = model_config.get("add_eos_token", True)


    print(f"Loading tokenizer for {hf_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        cache_dir=args.base_model_cache_dir,
        use_fast=True,
        trust_remote_code=True # Some models like Phi-2 might require this
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        print("Tokenizer missing pad_token, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure model config is also updated if pad_token_id was not set
        # model.config.pad_token_id = tokenizer.pad_token_id (do this after model load)

    # Set padding side for decoder-only models, typically left for training to enable batching
    # However, Trainer often handles this. Let's ensure pad_token_id is set.
    tokenizer.padding_side = "right" # Default for HF trainer, will be handled by data collator with attention mask

    quantization_config = None
    if args.load_in_4bit:
        print("Loading model in 4-bit (QLoRA style)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {hf_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        cache_dir=args.base_model_cache_dir,
        quantization_config=quantization_config, # None if not args.load_in_4bit
        torch_dtype="auto", # Automatically select best dtype (e.g. bfloat16 if available)
        device_map="auto", # Distribute model across GPUs if available
        trust_remote_code=True, # For models like Phi-2
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 and quantization_config is None else "sdpa", # Flash Attn 2 for full precision
    )

    if tokenizer.pad_token_id is None: # Set after model load to sync config
        model.config.pad_token_id = tokenizer.eos_token_id


    if args.load_in_4bit:
        print("Preparing model for k-bit training (QLoRA)...")
        model = prepare_model_for_kbit_training(model)


    # --- 2. Dataset Preparation ---
    print("Loading and tokenizing datasets...")
    data_files = {'train': args.train_file, 'validation': args.val_file}
    raw_datasets = load_dataset("json", data_files=data_files)

    def preprocess_function(examples):
        inputs = []
        labels_list = []

        for i in range(len(examples["input"])):
            # Concatenate prompt and label for Causal LM fine-tuning
            # The model learns to predict the 'output' part.
            # Add EOS token to the end of the label if it's a chat/instruct model
            # The prompt itself (example["input"]) typically ends with [/INST]
            text = examples["input"][i] + examples["output"][i]
            if add_eos_token_to_output: # Ensure EOS marks end of generation
                 text += tokenizer.eos_token
            inputs.append(text)

        # Tokenize the full combined texts
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_seq_length,
            padding="max_length", # Pad to max_seq_length
            truncation=True,
            return_tensors="pt"
        )

        # Create labels: copy input_ids, then mask prompt tokens
        labels = model_inputs["input_ids"].clone()

        # Mask prompt tokens by finding length of tokenized input prompt
        for i in range(len(examples["input"])):
            prompt_only_tokens = tokenizer(
                examples["input"][i],
                max_length=args.max_seq_length, # Should not truncate prompt ideally
                truncation=True,
                add_special_tokens=False # Avoid adding BOS/EOS to prompt only for length calculation
            ).input_ids
            prompt_len = len(prompt_only_tokens)
            
            # For models that add BOS token automatically (like Llama), tokenizer(prompt) might include it.
            # We need to be careful with length calculation.
            # A robust way is to find where the label (examples["output"][i]) starts in the combined text.
            # If tokenizer adds BOS to combined but not to prompt_only, prompt_len might be off by 1.

            # Assuming prompt ends with [/INST] and output starts right after:
            # Tokenize prompt part (which ends with [/INST])
            tokenized_prompt_part = tokenizer(examples["input"][i], add_special_tokens=False) # No BOS/EOS
            len_prompt_tokens = len(tokenized_prompt_part['input_ids'])

            # If tokenizer adds BOS to the full input, account for it
            if tokenizer.bos_token_id and model_inputs["input_ids"][i, 0] == tokenizer.bos_token_id:
                 # We want to mask from after BOS up to end of prompt
                 labels[i, :len_prompt_tokens + 1] = -100 # Mask BOS + prompt
            else:
                 labels[i, :len_prompt_tokens] = -100 # Mask prompt only

            # Ensure that if output was truncated, the corresponding label isn't also -100 for the part that *was* part of output.
            # This is tricky if both input and output are very long and get truncated.
            # The above logic assumes prompt fits. If prompt is truncated, masking can be inaccurate.
            # For simplicity, we assume prompt fits. A more robust way is needed if prompts are very long.

        model_inputs["labels"] = labels
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names, # Remove original text columns
        num_proc=os.cpu_count() // 2 if os.cpu_count() > 1 else 1, # Parallelize mapping
        desc="Tokenizing datasets"
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    if len(train_dataset) > 0:
        print("Example of tokenized input:", train_dataset[0])


    # --- 3. LoRA Configuration ---
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules, # Specific to model architecture
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    # --- 4. Training ---
    print("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # load_best_model_at_end=True, # Requires metric_for_best_model
        # metric_for_best_model="loss", # Or "eval_loss"
        report_to=args.report_to.split(',') if args.report_to else ["tensorboard"],
        fp16=torch.cuda.is_available() and not args.load_in_4bit and not torch.cuda.is_bf16_supported(), # Use fp16 if available and not QLoRA/bf16
        bf16=torch.cuda.is_bf16_supported() and not args.load_in_4bit, # Use bf16 if available and not QLoRA
        seed=args.seed,
        # ddp_find_unused_parameters=False, # If using DDP and encountering issues
    )

    # Data collator - for Causal LM, it usually handles padding inputs and labels
    # Since we are pre-padding and masking labels, a default collator might be okay,
    # or DataCollatorForLanguageModeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete.")

    # --- 5. Save Final LoRA Adapter ---
    final_adapter_path = os.path.join(args.output_dir, "final_adapter")
    print(f"Saving final LoRA adapter to {final_adapter_path}...")
    model.save_pretrained(final_adapter_path)
    # tokenizer.save_pretrained(final_adapter_path) # Tokenizer usually saved with base, but good practice if changes were made

    print("LoRA adapter saved successfully.")
    print(f"Base model was: {hf_model_name}")
    print(f"To use the adapter: Load the base model ('{hf_model_name}') and then load the adapter from '{final_adapter_path}'.")


if __name__ == "__main__":
    main()
