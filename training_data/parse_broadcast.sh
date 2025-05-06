python parse_broadcast.py \
    --output-file result.json \
    --stockfish-path ../src/stockfish/stockfish-macos-m1-apple-silicon \
    --cache-dir cache \
    --only-standard \
    --only-complete \
    --skip-unfinished \
    --mismatch-threshold 30 \
    --analysis-time 0.1 \
    lichess_database/broadcasts/lichess_db_broadcast_2025-01.pgn