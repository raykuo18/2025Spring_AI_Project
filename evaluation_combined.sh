#!/bin/bash

# Check if GPU count argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_gpus>"
  exit 1
fi

NUM_GPUS=$1
VERSION=$2
ALPHA=$3
BETA=$4
P1_SAMPLES=$5
P2_SAMPLES=$6

# if [ "$VERSION" = "v1" ]; then
#     PATH1="checkpoints/phase1_conti_checkpoint-10000"
#     PATH2="checkpoints/phase2_from_1000"
# elif [ "$VERSION" = "v2" ]; then
#     PATH1="checkpoints/phase2_from_conti_final-800/phase1_move_predictor/"
#     PATH2="checkpoints/phase2_from_conti_final-800/phase2_explainer_lora/"
# elif [ "$VERSION" = "v3" ]; then # our pipeline
#     PATH1="checkpoints/phase1_conti_final/"
#     PATH2="checkpoints/phase2_corrected-final/phase2_explainer_lora/"
# elif [ "$VERSION" = "v4" ]; then # seperate LoRA
#     PATH1="checkpoints/phase1_conti_final/"
#     PATH2="checkpoints/LoRA2_only_explainer_lora_v2/"
# else
#     echo "Unknown version: $VERSION"
# fi

PATH1=training_output/tinyllama_phase2_from_LoRA_conti_final_corrected/TinyLLaMA_phase2_20250509_09023426/checkpoint-2340/phase1_move_predictor
PATH2=training_output/tinyllama_phase2_from_LoRA_conti_final_corrected/TinyLLaMA_phase2_20250509_09023426/checkpoint-2340/phase2_explainer_lora

echo "PATH1=${PATH1}"
echo "PATH2=${PATH2}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=/home/skuo/out_evaluation_combined_%j.txt
#SBATCH --error=/home/skuo/err_evaluation_combined_%j.txt
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

python evaluation_combined.py \
    --model_name "TinyLLaMA" \
    --base_model_cache_dir ./hf_cache \
    --phase1_lora_path ${PATH1} \
    --phase1_adapter_name "p1_move_predictor" \
    --phase2_lora_path ${PATH2} \
    --phase2_adapter_name "p2_explainer" \
    --alpha_p1_weight ${ALPHA} \
    --beta_p2_weight ${BETA} \
    --combination_type "linear" \
    --test_file training_data/phase1/test.jsonl \
    --explanation_test_folder training_data/phase2_corrected/test \
    --eval_move_pred \
    --eval_rule_tasks \
    --eval_explanation \
    --stockfish_path ../stockfish-11-linux/Linux/stockfish_20011801_x64 \
    --bert_score_model_type "microsoft/deberta-xlarge-mnli" \
    --stockfish_analysis_time 0.3 \
    --top_k_agreement 1 3 5 10 20 30 40 50\
    --sf_reference_ks 1 3 5 10 30 50 \
    --max_p1_eval_samples ${P1_SAMPLES} \
    --max_p2_eval_samples ${P2_SAMPLES} \
    --output_results_file ./evaluation_results/combined_${VERSION}_${ALPHA}_${BETA}/result.json \
    --output_numerical_summary ./evaluation_results/combined_${VERSION}_${ALPHA}_${BETA}/summary.txt \
    --inference_cache_folder ./evaluation_results/inference_cache_combined_${VERSION}_${ALPHA}_${BETA} \
    --max_seq_length 1024 \
    --batch_size 64 \
    --seed 42 \
    --default_max_new_tokens 150 \
    --load_in_4bit
EOT