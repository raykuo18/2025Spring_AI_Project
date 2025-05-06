#!/bin/bash

# Check if GPU count argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_gpus>"
  exit 1
fi

# Command to submit:
# $ ./generate_phase2_explanations_submit.sh {GPUs} {PHASE} {START} {NUM} {SAVE_EVERY}

NUM_GPUS=$1
PHASE=$2

START=$3
NUM=$4
EVERY=$5

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=/home/skuo/test_out_${PHASE}_${START}_%j.txt
#SBATCH --error=/home/skuo/test_err_${PHASE}_${START}_%j.txt
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
cd /home/skuo/2025Spring_AI_Project/training_data

python generate_phase2_explanations.py \
    --input-prompts-file phase2/prompts_${PHASE}.jsonl \
    --output-training-folder phase2/${PHASE}/ \
    --model-name-or-path "../models/Mixtral" \
    --slice-start-index ${START} \
    --slice-num-samples ${NUM} \
    --checkpoint-every ${EVERY} \
    --batch-size 4 \
    --max-new-tokens 120
EOT
