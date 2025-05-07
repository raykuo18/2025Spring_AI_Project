#!/bin/bash

# Check if GPU count argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_gpus>"
  exit 1
fi

NUM_GPUS=$1

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=/home/skuo/out_phase1_training_%j.txt
#SBATCH --error=/home/skuo/err_phase1_training_%j.txt
#SBATCH --time=2-00:00
#SBATCH --mem=70000
#SBATCH --gres=gpu:${NUM_GPUS}

# Test CUDA compiler (not needed by deep learning people, we just use the python libraries)
# /cm/shared/apps/cuda11.1/toolkit/11.1.1/bin/nvcc -o saxpy /home/<netid>/cuda_c_code/saxpy.cu && ./saxpy

# Test nvidia-smi
nvidia-smi

# Test Python conda environment
# source activate your_conda_env
# /home/<netid>/miniconda3/envs/your_conda_env/bin/python /home/<netid>/run.py
cd /home/skuo/2025Spring_AI_Project

# Example for TinyLLaMA
python phase1_continue_training.py \
    --model_name "TinyLLaMA" \
    --resume_lora_adapter_path training_output/tinyllama_phase1/TinyLLaMA_06225337/final_lora_adapter \
    --train_file training_data/phase1/train.jsonl \
    --val_file training_data/phase1/val.jsonl \
    --tokenized_data_path training_data/phase1 \
    --output_dir ./training_output/tinyllama_phase1_continued \
    --base_model_cache_dir ./hf_cache \
    --max_seq_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --optim "paged_adamw_8bit" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --save_total_limit 5 \
    --report_to "tensorboard" \
    --seed 43 \
    --load_in_4bit \
    --max_train_samples 60000 \
    --max_val_samples 500 \
    --max_test_samples 500 \
    --gradient_checkpointing 
    # --gradient_checkpointing_use_reentrant False # This is already the default in the script's argparse
    # --vram_log_multiplier 1 # Optional: To control VRAM logging frequency relative to logging_steps. Default is 1.
    # --use_flash_attention_2 # Only if NOT using --load_in_4bit and your setup supports it
EOT
