#!/usr/bin/env python3

import json
import re
import os
import hashlib
import argparse
from typing import List, Dict, Optional, Tuple
import chess
import chess.engine
from tqdm import tqdm
import time # For unique file names
import concurrent.futures
import tempfile # For temporary cache files
import gc # For garbage collection, maybe helpful

# --- Global variable for worker engine (managed per process) ---
# This is one way to reuse the engine within a worker process across multiple tasks
worker_engine_instance = None

# --- Engine Initialization for Worker ---
def initialize_worker_engine(engine_path):
    """Initializes the engine for a worker process."""
    global worker_engine_instance
    if worker_engine_instance is None and engine_path and os.path.exists(engine_path):
        try:
            worker_engine_instance = chess.engine.SimpleEngine.popen_uci(engine_path)
            # Optional: Configure engine? e.g., worker_engine_instance.configure({"Threads": 1})
            # print(f"Worker {os.getpid()} initialized engine.") # Debug
        except Exception as e:
            print(f"Worker {os.getpid()} failed to initialize engine: {e}")
            worker_engine_instance = None # Ensure it's None if failed
    # else: # Debug
        # print(f"Worker {os.getpid()} using existing engine or no path provided.")

# --- Engine Cleanup for Worker ---
def close_worker_engine():
    """Closes the engine for a worker process."""
    global worker_engine_instance
    if worker_engine_instance is not None:
        try:
            worker_engine_instance.quit()
            # print(f"Worker {os.getpid()} closed engine.") # Debug
        except chess.engine.EngineTerminatedError:
            pass # Already closed
        except Exception as e:
            print(f"Worker {os.getpid()} error closing engine: {e}")
        worker_engine_instance = None

# --- Metadata and Annotation Parsing (No changes needed) ---
def parse_pgn_metadata(headers: List[str]) -> Dict:
    meta = {}
    for line in headers:
        match = re.match(r'\[\s*(\w+)\s*"(.*)"\s*\]', line)
        if match: key, val = match.groups(); meta[key] = val
    return meta

def parse_eval(eval_str: str) -> Optional[float]:
    try:
        eval_str = eval_str.strip()
        if eval_str.startswith("#"):
            mate_in = int(eval_str[1:])
            if mate_in == 0: return 10000 if eval_str == "#0" else -10000
            return (10000 - abs(mate_in)) * (1 if mate_in > 0 else -1)
        return float(eval_str.lstrip('+')) * 100
    except (ValueError, TypeError): return None

def extract_move_annotations(text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    eval_val, clk_val, comment_val = None, None, None
    eval_match = re.search(r'\[%eval\s+([#\d\.\+-]+(?:\/\d+)?)\s*\]', text)
    if eval_match: eval_val = parse_eval(eval_match.group(1))
    clk_match = re.search(r'\[%clk\s+([\d:]{4,})\s*\]', text)
    if clk_match: clk_val = clk_match.group(1)
    text_no_annotations = re.sub(r'\[%(eval|clk)\s+[^\]]*?\]', '', text)
    comments = re.findall(r'\{(.*?)\}', text_no_annotations, re.DOTALL)
    cleaned_comments = [c.strip() for c in comments if c.strip()]
    if cleaned_comments:
        comment_val = ' '.join(cleaned_comments)
        # print(f"Found Comment: {comment_val}") # Keep if desired
    return eval_val, clk_val, comment_val

# --- Move Parsing with UCI (No changes needed) ---
def parse_pgn_moves(pgn_text: str, board: chess.Board) -> List[Dict]:
    moves_data = []
    pgn_text = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', pgn_text).strip()
    pgn_text = re.sub(r'\([^)]*\)', '', pgn_text)
    pgn_text = re.sub(r'\s+', ' ', pgn_text)
    san_regex = re.compile(r'([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)([?!+#]*)')
    move_num_regex = re.compile(r'(\d+)\s*\.{1,3}')
    current_pos = 0
    current_pair: Optional[Dict] = None
    while current_pos < len(pgn_text):
        search_start = current_pos
        while search_start < len(pgn_text) and pgn_text[search_start] in (' ', '{', '['):
            if pgn_text[search_start] == '{': match = re.match(r'\{.*?\}', pgn_text[search_start:]); search_start += match.end() if match else 1
            elif pgn_text[search_start] == '[': match = re.match(r'\[.*?\]', pgn_text[search_start:]); search_start += match.end() if match else 1
            else: search_start += 1
        current_pos = search_start
        if current_pos >= len(pgn_text): break
        num_match = move_num_regex.match(pgn_text, current_pos)
        san_match = san_regex.match(pgn_text, current_pos)
        move_number_found = 0
        if num_match:
            move_number_found = int(num_match.group(1))
            potential_san_after_num = san_regex.match(pgn_text, num_match.end())
            if potential_san_after_num: san_match = potential_san_after_num; current_pos = num_match.end()
            else: current_pos = num_match.end(); san_match = None
        if san_match:
            san_for_parsing, nags_checks = san_match.groups()
            san_original = san_for_parsing + nags_checks
            current_pos = san_match.end()
            if not re.search('[a-hO]', san_for_parsing): continue
            current_move_number = move_number_found if move_number_found > 0 else board.fullmove_number
            is_black_turn_to_move = (board.turn == chess.BLACK)
            move_uci, valid_move = None, False
            try: move = board.parse_san(san_for_parsing); move_uci = move.uci(); board.push(move); valid_move = True
            except ValueError as e: print(f"Warning: Invalid SAN '{san_for_parsing}' ('{san_original}') for move ~{current_move_number} on {board.fen()}. Error: {e}. Skipping."); continue
            annotation_text = ""
            annot_start = current_pos
            while annot_start < len(pgn_text) and pgn_text[annot_start] in (' ', '{', '['):
                 if pgn_text[annot_start] == '{': match = re.match(r'\{.*?\}', pgn_text[annot_start:]); annotation_text += match.group(0) + " "; annot_start += match.end() if match else 1
                 elif pgn_text[annot_start] == '[': match = re.match(r'\[.*?\]', pgn_text[annot_start:]); annotation_text += match.group(0) + " "; annot_start += match.end() if match else 1
                 else: annot_start += 1
            current_pos = annot_start
            eval_cp, clock, comment = extract_move_annotations(annotation_text.strip())
            move_info = {'san': san_original, 'uci': move_uci, 'eval_cp': eval_cp, 'clock': clock, 'comment': comment}
            if not is_black_turn_to_move:
                 if current_pair and current_pair['white_move'] is None and current_pair['black_move'] is not None: print(f"Warning: Finalizing Black-only pair {current_pair['move_number']}"); moves_data.append(current_pair); current_pair = None
                 current_pair = {"move_number": current_move_number, "white_move": move_info, "black_move": None}
            else:
                 if current_pair and current_pair['move_number'] == current_move_number and current_pair['white_move'] is not None: current_pair['black_move'] = move_info; moves_data.append(current_pair); current_pair = None
                 else: print(f"Warning: Black move {current_move_number}...{san_original} found without preceding white move."); moves_data.append({"move_number": current_move_number, "white_move": None, "black_move": move_info}); current_pair = None
        elif not num_match:
             if current_pos < len(pgn_text): current_pos += 1
             else: break
    if current_pair and current_pair['white_move'] and not current_pair['black_move']: moves_data.append(current_pair)
    return moves_data

# --- Filtering and Validation Funcs (No changes needed) ---
def filter_game(meta: Dict, args) -> bool:
    if args.skip_unfinished and meta.get("Result") == "*": return False
    if args.only_standard and meta.get("Variant", "Standard").lower() != "standard": return False
    result = meta.get("Result")
    if args.only_complete and result not in ["1-0", "0-1", "1/2-1/2"]: return False
    try: white_elo = int(meta.get("WhiteElo", "0"))
    except ValueError: white_elo = 0
    try: black_elo = int(meta.get("BlackElo", "0"))
    except ValueError: black_elo = 0
    if args.min_elo is not None and (white_elo < args.min_elo or black_elo < args.min_elo): return False
    if args.max_elo is not None and (white_elo > args.max_elo or black_elo > args.max_elo): return False
    return True

def stockfish_eval(engine, board: chess.Board, analysis_time: float) -> Optional[float]:
    if not engine: return None
    try:
        info = engine.analyse(board, chess.engine.Limit(time=analysis_time, depth=15))
        if "score" in info:
            score = info["score"].white()
            if score.is_mate(): mate_plies = score.mate(); return (10000 - abs(mate_plies)) * (1 if mate_plies > 0 else -1)
            else: return score.score(mate_score=10000)
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, Exception) as e: print(f"Stockfish analysis failed for FEN {board.fen()}: {e}"); return None
    return None

def compute_move_key(fen: str, move_number: int, is_black: bool) -> str:
    turn_indicator = "b" if is_black else "w"; key_str = f"{fen}|{move_number}|{turn_indicator}"; return hashlib.md5(key_str.encode()).hexdigest()

def validate_eval(existing_cp: Optional[float], computed_cp: Optional[float], threshold: float) -> bool:
    if existing_cp is None or computed_cp is None: return True
    if abs(existing_cp) > 9000 and abs(computed_cp) > 9000: return existing_cp == computed_cp
    return abs(existing_cp - computed_cp) <= threshold

# --- JSON Output Creation (No changes needed) ---
def create_json_entry(meta: Dict, moves: List[Dict], starting_fen: str) -> Dict:
    try: white_elo = int(meta.get("WhiteElo", "0"))
    except ValueError: white_elo = None
    try: black_elo = int(meta.get("BlackElo", "0"))
    except ValueError: black_elo = None
    return {"game_metadata": {"fen": starting_fen, "result": meta.get("Result"), "white_elo": white_elo, "black_elo": black_elo, "eco": meta.get("ECO"), "variant": meta.get("Variant", "Standard"), "game_url": meta.get("GameURL") or meta.get("Site"), "pgn": moves}, "synthetic_data": [], "distilled_data": []}

# --- <<< NEW: Worker Function for Processing a Single Game >>> ---
def process_single_game_worker(task_data: Dict) -> Optional[Tuple[Dict, Dict, Dict]]:
    """
    Processes a single raw game string. Designed to be run in a worker process.
    Initializes engine if needed for this process. Manages local cache.
    Returns tuple: (processed_game_json, stats_for_game, final_local_cache) or None if filtered/error.
    """
    global worker_engine_instance # Use the engine instance associated with this worker process

    raw_game = task_data['raw_game']
    args = task_data['args']
    local_cache = task_data['cache_copy'] # Start with a copy of the initial cache

    # --- Basic parsing and filtering ---
    raw_game = raw_game.strip()
    if not raw_game or not raw_game.startswith("[Event"): return None

    parts = raw_game.split('\n\n', 1)
    header_text = parts[0]
    moves_text = parts[1].strip() if len(parts) > 1 else ""
    headers = header_text.splitlines()
    meta = parse_pgn_metadata(headers)

    # Initialize stats for this single game
    game_stats = { k: 0 for k in ["games_processed", "games_filtered_out", "moves_total", "invalid_san_skipped", "moves_with_pgn_eval", "moves_validated", "moves_mismatch", "moves_recomputed", "moves_missing_eval"] }

    if not filter_game(meta, args):
        game_stats["games_filtered_out"] = 1
        # Return stats indicating filter out, but no game JSON and the unmodified local cache
        return None, game_stats, local_cache

    # --- Setup board ---
    board = chess.Board()
    fen = meta.get("FEN")
    variant = meta.get("Variant", "Standard").lower()
    if fen:
        try: board.set_fen(fen)
        except ValueError: print(f"Warning: Invalid FEN '{fen}'. Using standard start."); board.reset()
    elif variant != "standard": print(f"Warning: Variant '{variant}' without FEN. Using standard start.")

    starting_fen = board.fen()
    board_for_parsing = board.copy()
    board_for_eval = board.copy() if worker_engine_instance else None

    # --- Parse Moves ---
    parsed_moves = parse_pgn_moves(moves_text, board_for_parsing)

    # --- Stockfish Validation/Computation ---
    if worker_engine_instance:
        current_eval_board = board_for_eval
        for move_pair in parsed_moves:
            move_num = move_pair['move_number']
            # --- Process White ---
            if move_pair['white_move']:
                w_move_data = move_pair['white_move']
                if w_move_data['uci'] is None: game_stats["invalid_san_skipped"] += 1; continue
                board_fen_before_move = current_eval_board.fen()
                fen_key = compute_move_key(board_fen_before_move, move_num, is_black=False)
                cached_eval = local_cache.get(fen_key) # Use local_cache
                computed_eval = None; pgn_eval = w_move_data['eval_cp']
                if pgn_eval is not None: game_stats["moves_with_pgn_eval"] += 1
                needs_computation = (pgn_eval is None) or (args.overwrite_eval) or (cached_eval is None)
                if needs_computation:
                    if cached_eval is not None: computed_eval = cached_eval
                    else:
                        computed_eval = stockfish_eval(worker_engine_instance, current_eval_board, args.analysis_time)
                        if computed_eval is not None: local_cache[fen_key] = computed_eval; game_stats["moves_recomputed"] += 1 # Update local_cache
                else: computed_eval = pgn_eval
                if pgn_eval is not None and computed_eval is not None:
                     game_stats["moves_validated"] += 1
                     if not validate_eval(pgn_eval, computed_eval, args.mismatch_threshold):
                         game_stats["moves_mismatch"] += 1
                         # Reduced print frequency for parallel runs, maybe log to worker file instead?
                         # print(f"pid {os.getpid()} ⚠️ Mismatch W {move_num} ({w_move_data['san']}): PGN {pgn_eval:.0f} vs SF {computed_eval:.0f}")
                         if args.abort_on_mismatch: raise RuntimeError(f"Aborted due to mismatch pid {os.getpid()}")
                         if args.overwrite_eval: w_move_data['eval_cp'] = computed_eval
                     elif args.overwrite_eval: w_move_data['eval_cp'] = computed_eval
                elif pgn_eval is None and computed_eval is not None: w_move_data['eval_cp'] = computed_eval; game_stats["moves_missing_eval"] += 1
                try: current_eval_board.push_uci(w_move_data['uci'])
                except ValueError: break # Error in game, stop eval

            # --- Process Black ---
            if move_pair['black_move']:
                b_move_data = move_pair['black_move']
                if b_move_data['uci'] is None: game_stats["invalid_san_skipped"] += 1; continue
                board_fen_before_move = current_eval_board.fen()
                fen_key = compute_move_key(board_fen_before_move, move_num, is_black=True)
                cached_eval = local_cache.get(fen_key) # Use local_cache
                computed_eval = None; pgn_eval = b_move_data['eval_cp']
                if pgn_eval is not None: game_stats["moves_with_pgn_eval"] += 1
                needs_computation = (pgn_eval is None) or (args.overwrite_eval) or (cached_eval is None)
                if needs_computation:
                    if cached_eval is not None: computed_eval = cached_eval
                    else:
                        computed_eval = stockfish_eval(worker_engine_instance, current_eval_board, args.analysis_time)
                        if computed_eval is not None: local_cache[fen_key] = computed_eval; game_stats["moves_recomputed"] += 1 # Update local_cache
                else: computed_eval = pgn_eval
                if pgn_eval is not None and computed_eval is not None:
                     game_stats["moves_validated"] += 1
                     if not validate_eval(pgn_eval, computed_eval, args.mismatch_threshold):
                         game_stats["moves_mismatch"] += 1
                         # print(f"pid {os.getpid()} ⚠️ Mismatch B {move_num} ({b_move_data['san']}): PGN {pgn_eval:.0f} vs SF {computed_eval:.0f}")
                         if args.abort_on_mismatch: raise RuntimeError(f"Aborted due to mismatch pid {os.getpid()}")
                         if args.overwrite_eval: b_move_data['eval_cp'] = computed_eval
                     elif args.overwrite_eval: b_move_data['eval_cp'] = computed_eval
                elif pgn_eval is None and computed_eval is not None: b_move_data['eval_cp'] = computed_eval; game_stats["moves_missing_eval"] += 1
                try: current_eval_board.push_uci(b_move_data['uci'])
                except ValueError: break

    # --- Create JSON entry ---
    json_entry = create_json_entry(meta, parsed_moves, starting_fen)
    game_stats["games_processed"] = 1
    for mv_pair in parsed_moves:
        if mv_pair.get('white_move'): game_stats['moves_total'] += 1
        if mv_pair.get('black_move'): game_stats['moves_total'] += 1

    # Return JSON, stats for this game, and the potentially updated local cache for merging
    return json_entry, game_stats, local_cache

# --- <<< NEW: Cache Merging Function >>> ---
def merge_caches(main_cache, worker_cache):
    """Merges updates from worker_cache into main_cache."""
    # Simple merge: worker updates overwrite main cache entries
    # More sophisticated merging could be added if needed (e.g., based on timestamp or analysis time if stored)
    main_cache.update(worker_cache)


# --- <<< REFACTORED Main Processing Logic >>> ---
def main(args):
    # --- Cache Loading (as before) ---
    cache_file = None
    main_cache = {} # Renamed for clarity
    if args.cache_dir:
        # ... (cache directory creation logic as before) ...
        if args.cache_dir:
             cache_file = os.path.join(args.cache_dir, "eval_cache.json")
             if os.path.exists(cache_file):
                 try:
                     with open(cache_file, "r") as f: main_cache = json.load(f)
                     print(f"Loaded {len(main_cache)} cached evaluations from {cache_file}")
                 except Exception as e: print(f"Warning: Error loading cache file {cache_file}: {e}. Starting fresh."); main_cache = {}

    # --- Prepare Game Data ---
    print("Reading PGN files and splitting into games...")
    all_raw_games = []
    total_games_found = 0
    for file_path in args.input_files:
        if not os.path.exists(file_path): print(f"Warning: Input file not found: {file_path}. Skipping."); continue
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f: content = f.read()
            # Split into games
            games_in_file = re.split(r'\n\s*\n(?=\[Event)', content.strip())
            if games_in_file and games_in_file[0]: # Ensure non-empty and valid first game
                 all_raw_games.extend(games_in_file)
                 total_games_found += len(games_in_file)
            elif content.strip().startswith("[Event "): # Handle file with only one game
                 all_raw_games.append(content.strip())
                 total_games_found += 1

        except Exception as e: print(f"Warning: Could not read/split file {file_path}: {e}")

    if not all_raw_games: print("Error: No games found in any input file."); return
    print(f"Found {total_games_found} games to process across all files.")

    # --- Initialize Worker Pool ---
    num_workers = args.num_workers if args.num_workers else os.cpu_count()
    print(f"Using {num_workers} worker processes.")

    # Prepare task data for each game
    tasks = []
    # Create a copy of the cache for each task to avoid race conditions if passed directly
    # Passing large dicts repeatedly can be slow, but safer than direct sharing without Manager
    initial_cache_copy = main_cache.copy() # Make one copy here
    for raw_game in all_raw_games:
         task_data = {
              'raw_game': raw_game,
              'args': args, # Pass command line args
              'cache_copy': initial_cache_copy.copy() # Give each task its own copy
              # Engine path passed separately to initializer
         }
         tasks.append(task_data)
    del initial_cache_copy # Free memory of the intermediate copy

    all_processed_games = []
    total_stats = { k: 0 for k in ["games_processed", "games_filtered_out", "moves_total", "invalid_san_skipped", "moves_with_pgn_eval", "moves_validated", "moves_mismatch", "moves_recomputed", "moves_missing_eval"] }
    updated_caches_from_workers = [] # List to store final cache state from each task

    # Determine engine path for initializer
    engine_path = args.stockfish_path if args.stockfish_path and os.path.exists(args.stockfish_path) else None

    # --- Run Tasks in Parallel ---
    print("Starting parallel processing...")
    # Use try/finally to ensure engine cleanup even on errors
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=initialize_worker_engine, initargs=(engine_path,)) as executor:
            # Use tqdm with as_completed for progress updates as tasks finish
            futures = [executor.submit(process_single_game_worker, task) for task in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), unit="game", desc="Processing"):
                try:
                    result = future.result()
                    if result:
                        game_json, game_stats, worker_cache_state = result
                        if game_json: # Check if game wasn't filtered out
                            all_processed_games.append(game_json)
                        # Aggregate stats
                        for key, value in game_stats.items():
                            total_stats[key] = total_stats.get(key, 0) + value
                        # Store the returned cache state for later merging
                        updated_caches_from_workers.append(worker_cache_state)

                    # Explicitly delete task data and result to free memory sooner
                    del result
                    gc.collect() # Suggest garbage collection

                except Exception as exc:
                    print(f'\n--- Worker Error: Game processing generated an exception: {exc} ---')
                    # Optional: Log the specific task data that failed
                    # traceback.print_exc() # More detailed traceback
    finally:
         # Ensure worker engines are closed (though ProcessPoolExecutor shutdown should handle this)
         # Calling it explicitly might be redundant but safer if executor shutdown is unclean
         # This won't actually run in the worker processes from here. Cleanup needs initializer/finalizer pattern or happens on exit.
         # Relying on ProcessPoolExecutor shutdown is standard.
         pass

    # --- Merge Caches ---
    print(f"\nMerging cache data from {len(updated_caches_from_workers)} results...")
    # Start with the initially loaded cache
    final_cache = main_cache
    for worker_cache in updated_caches_from_workers:
         merge_caches(final_cache, worker_cache)
    print(f"Final cache size: {len(final_cache)} entries.")
    del updated_caches_from_workers # Free memory
    gc.collect()

    # --- Cache Saving ---
    if cache_file and args.stockfish_path: # Check engine path again
         try:
             print(f"Saving {len(final_cache)} evaluations to cache: {cache_file}")
             with open(cache_file, "w") as f: json.dump(final_cache, f) # Save the final merged cache
         except Exception as e: print(f"Error saving final cache file {cache_file}: {e}")

    # --- Write Output ---
    output_file = args.output_file
    if not all_processed_games: print("\nNo games processed successfully. No output file written."); return
    print(f"\nWriting {len(all_processed_games)} processed games to {output_file}...")
    # ... (output writing logic as before using all_processed_games) ...
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        if args.jsonl:
            with open(output_file, "w", encoding="utf-8") as f:
                for game in all_processed_games: f.write(json.dumps(game) + "\n")
        else:
            with open(output_file, "w", encoding="utf-8") as f: json.dump(all_processed_games, f, indent=2)
        print(f"✅ Successfully saved output to {output_file}")
    except Exception as e: print(f"❌ Error writing output file {output_file}: {e}")


    # --- Print Aggregated Stats ---
    print("\n--- Aggregated Statistics ---")
    # ... (stats printing logic as before using total_stats) ...
    if total_stats:
        stat_order = ["games_processed", "games_filtered_out", "moves_total", "invalid_san_skipped", "moves_with_pgn_eval", "moves_validated", "moves_mismatch", "moves_recomputed", "moves_missing_eval"]
        # Add the count of games found across all files
        total_stats["total_games_found_across_files"] = total_games_found
        stat_order.insert(0, "total_games_found_across_files")

        for key in stat_order:
             if key in total_stats:
                 key_formatted = key.replace('_', ' ').title()
                 print(f"{key_formatted}: {total_stats[key]}")
        for key, value in total_stats.items(): # Print any extras
            if key not in stat_order: print(f"{key.replace('_', ' ').title()}: {value}")
    else: print("No statistics aggregated.");
    print("-" * 27)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse PGN files to JSON (in parallel) with UCI, annotations, and optional Stockfish validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    # --- Input/Output Args (as before) ---
    parser.add_argument("input_files", nargs="+", help="One or more input PGN files.")
    parser.add_argument("-o", "--output-file", required=True, help="Output file path for JSON or JSONL.")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSON Lines.")

    # --- Filtering Args (as before) ---
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument("--skip-unfinished", action="store_true", help="Skip games with Result '*'.")
    filter_group.add_argument("--only-standard", action="store_true", help="Keep only Variant 'Standard'.")
    filter_group.add_argument("--only-complete", action="store_true", help="Keep only 1-0, 0-1, 1/2-1/2 results.")
    filter_group.add_argument("--min-elo", type=int, default=None, metavar='ELO', help="Min Elo for both players.")
    filter_group.add_argument("--max-elo", type=int, default=None, metavar='ELO', help="Max Elo for both players.")

    # --- Stockfish Args (as before) ---
    sf_group = parser.add_argument_group('Stockfish Validation/Computation')
    sf_group.add_argument("--stockfish-path", default=None, metavar='PATH', help="Path to Stockfish executable.")
    sf_group.add_argument("--analysis-time", type=float, default=0.1, metavar='SEC', help="Stockfish time per move.")
    sf_group.add_argument("--mismatch-threshold", type=float, default=50.0, metavar='CP', help="Eval mismatch tolerance (centipawns).")
    sf_group.add_argument("--overwrite-eval", action="store_true", help="Replace PGN eval with Stockfish eval.")
    sf_group.add_argument("--abort-on-mismatch", action="store_true", help="Stop on first eval mismatch.")
    sf_group.add_argument("--cache-dir", default=None, metavar='DIR', help="Directory for eval cache file 'eval_cache.json'.")

    # --- <<< NEW Parallelism Argument >>> ---
    parser.add_argument("--num-workers", type=int, default=None, metavar='CORES',
                        help="Number of worker processes to use. Defaults to system CPU count.")

    args = parser.parse_args()

    # --- Input validation (as before) ---
    valid_inputs = True
    # ... (existing validation for input_files, output_file, stockfish_path) ...
    if not args.input_files: print("Error: No input files provided."); valid_inputs = False
    if not args.output_file: print("Error: Output file path required."); valid_inputs = False
    if args.stockfish_path and not os.path.isfile(args.stockfish_path): print(f"Error: Stockfish path not found: {args.stockfish_path}. Disabling SF."); args.stockfish_path = None

    if valid_inputs:
        # Run the main parallel processing function
        main(args)
    else:
        print("\nExiting due to input errors.")