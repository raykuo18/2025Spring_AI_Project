#!/usr/bin/env bash

# Prior to job sumbission, run:
# 
# source activate cse537 # Do this on CLI prior to job submission
# 
# Install huggingface-cli
# pip install huggingface_hub
# 
# huggingface-cli login # Input Access Token

# Submit job command: ./test-py_cluster.sh

# Old code: 
# 
# scripts_dir=\$(dirname \$(realpath \${0}))

sbatch <<EOT
#!/usr/bin/env bash
#SBATCH --job-name=train-resource
#SBATCH --output=/home/adbraimah/cse537/2025Spring_AI_Project/output-experiment-%j.txt
#SBATCH --error=/home/adbraimah/cse537/2025Spring_AI_Project/error-experiment-%j.txt
#SBATCH --time=6-00:00
#SBATCH --mem=70000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adbraimah@cs.stonybrook.edu

scripts_dir=/home/adbraimah/cse537/2025Spring_AI_Project
cd \${scripts_dir}

# Run python script
python /home/adbraimah/cse537/2025Spring_AI_Project/check_training_resource.py
EOT