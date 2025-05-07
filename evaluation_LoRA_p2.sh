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
#SBATCH --output=/home/skuo/out_LoRA_p2_%j.txt
#SBATCH --error=/home/skuo/err_LoRA_p2_%j.txt
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

python evaluation.py \
    --model_name "TinyLLaMA" \
    --lora_adapter_path training_output/tinyllama_phase1/TinyLLaMA_07070558/checkpoint-1000 \
    --test_file training_data/phase2_corrected/test/data_slice_200_to_999_part_100_to_199.jsonl \
    --stockfish_path ../stockfish/stockfish-ubuntu-x86-64 \
    --output_results_file ./evaluation_results/tinyllama_phase1_LoRA_p2_eval.json \
    --base_model_cache_dir ./hf_cache \
    --stockfish_analysis_time 0.5 \
    --top_k_agreement 1 3 \
    --bert_score_model_type "microsoft/deberta-xlarge-mnli" \
    --max_eval_samples 100 \
    --max_seq_length 1024 \
    --batch_size 8 \
    --load_in_4bit \
    --seed 42
EOT