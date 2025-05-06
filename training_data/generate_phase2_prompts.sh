python generate_phase2_prompts.py \
    --input-file processed_data/broadcasts_v2_test.json \
    --output-file phase2/prompts_test.jsonl \
    --history-len 10 \
    --seed 42 \
    --p2-general-sample-rate 0.1

python generate_phase2_prompts.py \
    --input-file processed_data/broadcasts_v2_train.json \
    --output-file phase2/prompts_train.jsonl \
    --history-len 10 \
    --seed 42 \
    --p2-general-sample-rate 0.1

python generate_phase2_prompts.py \
    --input-file processed_data/broadcasts_v2_val.json \
    --output-file phase2/prompts_val.jsonl \
    --history-len 10 \
    --seed 42 \
    --p2-general-sample-rate 0.1