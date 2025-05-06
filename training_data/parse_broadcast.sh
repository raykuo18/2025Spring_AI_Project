python parse_broadcast_parallel.py \
    --output-file broadcasts_v2.json \
    --stockfish-path ../src/stockfish/stockfish-macos-m1-apple-silicon_ \
    --cache-dir cache \
    --only-standard \
    --only-complete \
    --skip-unfinished \
    --mismatch-threshold 30 \
    --analysis-time 0.1 \
    lichess_database/broadcasts/lichess_db_broadcast_2025-01.pgn \
    lichess_database/broadcasts/lichess_db_broadcast_2025-02.pgn \
    lichess_database/broadcasts/lichess_db_broadcast_2025-03.pgn \
    lichess_database/broadcasts/lichess_db_broadcast_2025-04.pgn