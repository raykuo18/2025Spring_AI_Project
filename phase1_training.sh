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
python phase1_training.py \
    --model_name "TinyLLaMA" \
    --train_file training_data/phase1/train.jsonl \
    --val_file training_data/phase1/val.jsonl \
    --test_file training_data/phase1/test.jsonl \
    --tokenized_data_path training_data/phase1 \
    --output_dir ./training_output/tinyllama_phase1 \
    --base_model_cache_dir ./hf_cache \
    --max_seq_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
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
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --seed 42 \
    --load_in_4bit 
    # --use_flash_attention_2 # Uncomment if your setup supports it AND you are NOT using --load_in_4bit (usually one or the other)
    # --max_train_samples 10000 # Optional: If you want to train on a subset of your tokenized train data
    # --max_eval_samples 1000   # Optional: If you want to evaluate on a subset of your tokenized val data
EOT
