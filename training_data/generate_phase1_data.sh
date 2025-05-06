python generate_phase1_data.py \
    --input-file processed_data/broadcasts_v2_test.json \
    --output-file phase1/test.jsonl \
    --predict-move-sample-rate 0.3 \
    --legal-move-sample-rate 0.1 \
    --basic-rule-sample-rate 0.05 \
    --hide-choices-prob 0.25 \
    --history-len 12

python generate_phase1_data.py \
    --input-file processed_data/broadcasts_v2_val.json \
    --output-file phase1/val.jsonl \
    --predict-move-sample-rate 0.3 \
    --legal-move-sample-rate 0.1 \
    --basic-rule-sample-rate 0.05 \
    --hide-choices-prob 0.25 \
    --history-len 12

python generate_phase1_data.py \
    --input-file processed_data/broadcasts_v2_train.json \
    --output-file phase1/train.jsonl \
    --predict-move-sample-rate 0.3 \
    --legal-move-sample-rate 0.1 \
    --basic-rule-sample-rate 0.05 \
    --hide-choices-prob 0.25 \
    --history-len 12