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

# --- Metadata and Annotation Parsing ---

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
        # print(f"Found Comment: {comment_val}") # Kept from user request
    return eval_val, clk_val, comment_val

# --- Move Parsing with UCI ---

def parse_pgn_moves(pgn_text: str, board: chess.Board) -> Optional[List[Dict]]: # Return Optional List
    moves_data = []
    pgn_text = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', pgn_text).strip()
    pgn_text = re.sub(r'\([^)]*\)', '', pgn_text) # Remove variations (simple)
    pgn_text = re.sub(r'\s+', ' ', pgn_text)     # Normalize whitespace

    san_regex = re.compile(
        r'([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)'
        r'([?!+#]*)'
    )
    move_num_regex = re.compile(r'(\d+)\s*\.{1,3}')

    current_pos = 0
    current_pair: Optional[Dict] = None

    while current_pos < len(pgn_text):
        search_start = current_pos
        # --- Robust pre-skipping for comments/annotations ---
        while search_start < len(pgn_text):
            char = pgn_text[search_start]
            if char == ' ':
                search_start += 1
            elif char == '{':
                try:
                    brace_depth = 1
                    end_comment = search_start + 1
                    while end_comment < len(pgn_text) and brace_depth > 0:
                        if pgn_text[end_comment] == '{': brace_depth += 1
                        elif pgn_text[end_comment] == '}': brace_depth -= 1
                        end_comment += 1
                    search_start = end_comment if brace_depth == 0 else search_start + 1
                except IndexError: search_start = len(pgn_text)
            elif char == '[':
                try:
                    end_annot = pgn_text.find(']', search_start + 1)
                    search_start = end_annot + 1 if end_annot != -1 else search_start + 1
                except IndexError: search_start = len(pgn_text)
            else:
                break
        current_pos = search_start
        if current_pos >= len(pgn_text): break
        # --- End robust pre-skipping ---


        num_match = move_num_regex.match(pgn_text, current_pos)
        san_match = san_regex.match(pgn_text, current_pos) # Initial attempt
        move_number_found = 0

        if num_match:
            move_number_found = int(num_match.group(1))
            potential_san_after_num = san_regex.match(pgn_text, num_match.end())
            if potential_san_after_num:
                 san_match = potential_san_after_num
                 current_pos = num_match.end() # Update current_pos for where san_match starts
            else:
                 current_pos = num_match.end() # Consumed number token
                 san_match = None # No SAN immediately after number

        if san_match:
            san_for_parsing = san_match.group(1)
            nags_checks = san_match.group(2)
            san_original = san_for_parsing + nags_checks

            # <<< --- FIXED LINE --- >>>
            # Use the end position of the match that found the SAN
            pos_after_san = san_match.end()
            # <<< --- END FIX --- >>>

            if not re.search('[a-hO]', san_for_parsing):
                 current_pos = pos_after_san # Ensure we move past this non-move
                 continue

            current_move_number = move_number_found if move_number_found > 0 else board.fullmove_number
            is_black_turn_to_move = (board.turn == chess.BLACK)
            move_uci = None

            try:
                move = board.parse_san(san_for_parsing)
                move_uci = move.uci()
                board.push(move)
            except ValueError as e:
                print(f"FATAL PARSE ERROR: Game may be corrupt. Invalid SAN '{san_for_parsing}' ('{san_original}') for move ~{current_move_number} on {board.fen()}. Error: {e}. Skipping this game.")
                return None # Signal to skip this game

            # Extract annotations AFTER the move
            annotation_text = ""
            annot_start = pos_after_san # Start looking for annotations right after the SAN
            temp_current_pos_for_annot = annot_start
            while temp_current_pos_for_annot < len(pgn_text):
                char = pgn_text[temp_current_pos_for_annot]
                if char == ' ': temp_current_pos_for_annot +=1; continue
                next_num_match = move_num_regex.match(pgn_text, temp_current_pos_for_annot)
                next_san_match = san_regex.match(pgn_text, temp_current_pos_for_annot)
                if next_num_match or next_san_match: break # Next move starts

                if char == '{':
                    # Use re.DOTALL for multi-line comments here
                    match = re.match(r'\{.*?\}', pgn_text[temp_current_pos_for_annot:], re.DOTALL)
                    if match: annotation_text += match.group(0) + " "; temp_current_pos_for_annot += match.end()
                    else: break # Malformed
                elif char == '[':
                    match = re.match(r'\[.*?\]', pgn_text[temp_current_pos_for_annot:])
                    if match: annotation_text += match.group(0) + " "; temp_current_pos_for_annot += match.end()
                    else: break
                else: break
            current_pos = temp_current_pos_for_annot # Advance main cursor past consumed annotations

            eval_cp, clock, comment = extract_move_annotations(annotation_text.strip())
            move_info = {'san': san_original, 'uci': move_uci, 'eval_cp': eval_cp, 'clock': clock, 'comment': comment}

            if not is_black_turn_to_move: # White's move
                 if current_pair and current_pair['white_move'] is None and current_pair['black_move'] is not None:
                      moves_data.append(current_pair); current_pair = None
                 current_pair = {"move_number": current_move_number, "white_move": move_info, "black_move": None}
            else: # Black's move
                 if current_pair and current_pair['move_number'] == current_move_number and current_pair['white_move'] is not None:
                      current_pair['black_move'] = move_info; moves_data.append(current_pair); current_pair = None
                 else:
                      moves_data.append({"move_number": current_move_number, "white_move": None, "black_move": move_info}); current_pair = None

        elif not num_match: # No SAN and no number_token found at current_pos
             if current_pos < len(pgn_text): current_pos += 1
             else: break

    if current_pair and current_pair['white_move'] and not current_pair['black_move']:
        moves_data.append(current_pair)
    return moves_data

# --- Filtering and Stockfish Validation (Unchanged) ---
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

def create_json_entry(meta: Dict, moves: List[Dict], starting_fen: str) -> Dict:
    try: white_elo = int(meta.get("WhiteElo", "0"))
    except ValueError: white_elo = None
    try: black_elo = int(meta.get("BlackElo", "0"))
    except ValueError: black_elo = None
    return {"game_metadata": {"fen": starting_fen, "result": meta.get("Result"), "white_elo": white_elo, "black_elo": black_elo, "eco": meta.get("ECO"), "variant": meta.get("Variant", "Standard"), "game_url": meta.get("GameURL") or meta.get("Site"), "pgn": moves}, "synthetic_data": [], "distilled_data": []}

# --- Main Processing Logic (Includes skip game logic) ---
def process_file(file_path: str, args, engine, cache: Dict, pbar: tqdm) -> Tuple[List[Dict], Dict]:
    try:
        with open(file_path, "r", encoding="utf-8", newline=None) as f: content = f.read()
    except Exception as e: print(f"Error reading file {file_path}: {e}"); return [], {}

    games_raw = re.split(r'\n\s*\n(?=\[Event)', content.strip())
    results = []
    stats = {
        "games_found_in_file": len(games_raw) if games_raw and games_raw[0] else 0,
        "games_processed": 0, "games_filtered_out": 0, "games_skipped_parsing_error": 0,
        "moves_total": 0, "invalid_san_skipped": 0,
        "moves_with_pgn_eval": 0, "moves_validated": 0,
        "moves_mismatch": 0, "moves_recomputed": 0, "moves_missing_eval": 0,
    }

    for raw_game in games_raw:
        pbar.update(1)
        raw_game = raw_game.strip()
        if not raw_game or not raw_game.startswith("[Event"): continue

        parts = raw_game.split('\n\n', 1)
        header_text = parts[0]; moves_text = parts[1].strip() if len(parts) > 1 else ""
        headers = header_text.splitlines(); meta = parse_pgn_metadata(headers)

        if not filter_game(meta, args): stats["games_filtered_out"] += 1; continue

        board = chess.Board()
        fen = meta.get("FEN"); variant = meta.get("Variant", "Standard").lower()
        if fen:
            try: board.set_fen(fen)
            except ValueError: print(f"Warning: Invalid FEN '{fen}'. Using standard start."); board.reset()
        elif variant == "chess960": board.chess960 = True; print(f"Info: Game is Chess960 (Variant tag). Flag: {board.chess960}")
        elif variant != "standard": print(f"Warning: Variant '{variant}' without FEN. Using standard start.")
        if board.chess960 and variant != "chess960": print(f"Info: Game is Chess960 (FEN detected). Flag: {board.chess960}")

        starting_fen = board.fen()
        board_for_parsing = board.copy()
        board_for_eval = board.copy() if engine else None

        parsed_moves = parse_pgn_moves(moves_text, board_for_parsing)
        if parsed_moves is None: # Check if game parsing failed
            stats["games_skipped_parsing_error"] += 1
            continue # Skip to the next game

        # --- Stockfish Validation (only if parsed_moves is valid) ---
        if engine:
            current_eval_board = board_for_eval
            for move_pair in parsed_moves:
                move_num = move_pair['move_number']
                # --- Process White ---
                if move_pair['white_move']:
                    w_move_data = move_pair['white_move']
                    if w_move_data['uci'] is None: stats["invalid_san_skipped"] += 1; continue
                    board_fen_before_move = current_eval_board.fen()
                    fen_key = compute_move_key(board_fen_before_move, move_num, is_black=False)
                    cached_eval = cache.get(fen_key); computed_eval = None; pgn_eval = w_move_data['eval_cp']
                    if pgn_eval is not None: stats["moves_with_pgn_eval"] += 1
                    needs_computation = (pgn_eval is None) or (args.overwrite_eval) or (cached_eval is None)
                    if needs_computation:
                        if cached_eval is not None: computed_eval = cached_eval
                        else:
                            computed_eval = stockfish_eval(engine, current_eval_board, args.analysis_time)
                            if computed_eval is not None: cache[fen_key] = computed_eval; stats["moves_recomputed"] += 1
                    else: computed_eval = pgn_eval
                    if pgn_eval is not None and computed_eval is not None:
                         stats["moves_validated"] += 1
                         if not validate_eval(pgn_eval, computed_eval, args.mismatch_threshold):
                             stats["moves_mismatch"] += 1
                             print(f"⚠️ Mismatch W Move {move_num} ({w_move_data['san']}): PGN {pgn_eval:.0f} vs SF {computed_eval:.0f} (FEN: {board_fen_before_move})")
                             if args.abort_on_mismatch: raise RuntimeError(f"Aborted W Move {move_num}.")
                             if args.overwrite_eval: w_move_data['eval_cp'] = computed_eval
                         elif args.overwrite_eval: w_move_data['eval_cp'] = computed_eval
                    elif pgn_eval is None and computed_eval is not None: w_move_data['eval_cp'] = computed_eval; stats["moves_missing_eval"] += 1
                    try: current_eval_board.push_uci(w_move_data['uci'])
                    except ValueError: print(f"EvalBoard Error W UCI {w_move_data['uci']}"); break
                # --- Process Black ---
                if move_pair['black_move']:
                    b_move_data = move_pair['black_move']
                    if b_move_data['uci'] is None: stats["invalid_san_skipped"] += 1; continue
                    board_fen_before_move = current_eval_board.fen()
                    fen_key = compute_move_key(board_fen_before_move, move_num, is_black=True)
                    cached_eval = cache.get(fen_key); computed_eval = None; pgn_eval = b_move_data['eval_cp']
                    if pgn_eval is not None: stats["moves_with_pgn_eval"] += 1
                    needs_computation = (pgn_eval is None) or (args.overwrite_eval) or (cached_eval is None)
                    if needs_computation:
                        if cached_eval is not None: computed_eval = cached_eval
                        else:
                            computed_eval = stockfish_eval(engine, current_eval_board, args.analysis_time)
                            if computed_eval is not None: cache[fen_key] = computed_eval; stats["moves_recomputed"] += 1
                    else: computed_eval = pgn_eval
                    if pgn_eval is not None and computed_eval is not None:
                         stats["moves_validated"] += 1
                         if not validate_eval(pgn_eval, computed_eval, args.mismatch_threshold):
                             stats["moves_mismatch"] += 1
                             print(f"⚠️ Mismatch B Move {move_num} ({b_move_data['san']}): PGN {pgn_eval:.0f} vs SF {computed_eval:.0f} (FEN: {board_fen_before_move})")
                             if args.abort_on_mismatch: raise RuntimeError(f"Aborted B Move {move_num}.")
                             if args.overwrite_eval: b_move_data['eval_cp'] = computed_eval
                         elif args.overwrite_eval: b_move_data['eval_cp'] = computed_eval
                    elif pgn_eval is None and computed_eval is not None: b_move_data['eval_cp'] = computed_eval; stats["moves_missing_eval"] += 1
                    try: current_eval_board.push_uci(b_move_data['uci'])
                    except ValueError: print(f"EvalBoard Error B UCI {b_move_data['uci']}"); break

        json_entry = create_json_entry(meta, parsed_moves, starting_fen)
        results.append(json_entry)
        stats["games_processed"] += 1
        for mv_pair in parsed_moves:
            if mv_pair.get('white_move'): stats['moves_total'] += 1
            if mv_pair.get('black_move'): stats['moves_total'] += 1

    # Print summary statistics for this file
    print("-" * 20)
    print(f"File: {os.path.basename(file_path)}")
    print(f"Games Found in File: {stats['games_found_in_file']}")
    print(f"Games Filtered Out: {stats['games_filtered_out']}")
    print(f"Games Skipped due to Parsing Error: {stats['games_skipped_parsing_error']}") # Added
    print(f"Games Processed: {stats['games_processed']}")
    print(f"Total Moves Parsed (in processed games): {stats['moves_total']}")
    print(f"Invalid SAN Moves Skipped (during Stockfish pass): {stats['invalid_san_skipped']}") # Clarified this stat
    if engine:
        print(f"Moves with PGN Eval: {stats['moves_with_pgn_eval']}")
        print(f"Moves Validated w/ Stockfish: {stats['moves_validated']}")
        print(f"Eval Mismatches: {stats['moves_mismatch']}")
        print(f"Moves Recomputed/Cached: {stats['moves_recomputed']}")
        print(f"Moves with Eval Added: {stats['moves_missing_eval']}")
    print("-" * 20)

    return results, stats

def main(args):
    # --- Cache and Engine Setup ---
    cache_file = None; cache = {}
    if args.cache_dir:
        if not os.path.exists(args.cache_dir):
            try: os.makedirs(args.cache_dir) ; print(f"Created cache: {args.cache_dir}")
            except OSError as e: print(f"Error creating cache dir {args.cache_dir}: {e}. Cache disabled."); args.cache_dir = None
        if args.cache_dir:
             cache_file = os.path.join(args.cache_dir, "eval_cache.json")
             if os.path.exists(cache_file):
                 try:
                     with open(cache_file, "r") as f: cache = json.load(f)
                     print(f"Loaded {len(cache)} evals from {cache_file}")
                 except Exception as e: print(f"Warn: Error loading cache {cache_file}: {e}. Starting fresh."); cache = {}
    engine = None
    if args.stockfish_path:
        if not os.path.exists(args.stockfish_path): print(f"Error: Stockfish path not found: {args.stockfish_path}. Validation disabled."); args.stockfish_path = None
        else:
             try: engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path); print(f"Stockfish engine: {args.stockfish_path}")
             except Exception as e: print(f"Error initializing Stockfish: {e}. Validation disabled."); engine = None

    # --- Pre-calculate total games ---
    print("Pre-calculating total games...")
    total_games = 0
    for file_path in args.input_files:
        if not os.path.exists(file_path): print(f"Warn: Input file not found (pre-scan): {file_path}."); continue
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                count = sum(1 for line in f if line.startswith('[Event '))
                total_games += count
        except Exception as e: print(f"Warn: Could not pre-read {file_path} to count games: {e}")
    if total_games > 0: print(f"Found approx {total_games} games.")
    else: print("Warn: No games detected or files empty.")

    # --- Process Files with tqdm ---
    all_games = []; total_stats = {}
    with tqdm(total=total_games, unit="game", desc="Processing Games", disable=(total_games == 0)) as pbar:
        for file_path in args.input_files:
            if not os.path.exists(file_path): continue
            pbar.set_description(f"Processing {os.path.basename(file_path)}")
            try:
                 games, file_stats = process_file(file_path, args, engine, cache, pbar) # Pass pbar
                 all_games.extend(games)
                 for key, value in file_stats.items(): total_stats[key] = total_stats.get(key, 0) + value
            except Exception as e:
                 pbar.set_description(f"ERROR in {os.path.basename(file_path)}")
                 print(f"\n--- CRITICAL ERROR processing {file_path}: {e} ---")
                 import traceback; traceback.print_exc()
                 print("Attempting to continue...")

    # --- Engine Shutdown ---
    if engine:
        try: engine.quit(); print("Stockfish engine shut down.")
        except chess.engine.EngineTerminatedError: pass

    # --- Cache Saving ---
    if cache_file and args.stockfish_path and engine is not None:
         try:
             with open(cache_file, "w") as f: json.dump(cache, f)
             print(f"Saved {len(cache)} evals to cache: {cache_file}")
         except Exception as e: print(f"Error saving cache {cache_file}: {e}")

    # --- Write Output ---
    output_file = args.output_file
    if not all_games: print("\nNo games processed successfully. No output file written."); return # Changed message slightly
    print(f"\nWriting {len(all_games)} processed games to {output_file}...")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        if args.jsonl:
            with open(output_file, "w", encoding="utf-8") as f:
                for game in all_games: f.write(json.dumps(game) + "\n")
        else:
            with open(output_file, "w", encoding="utf-8") as f: json.dump(all_games, f, indent=2)
        print(f"✅ Successfully saved output to {output_file}")
    except Exception as e: print(f"❌ Error writing {output_file}: {e}")

    # --- Print Aggregated Stats ---
    print("\n--- Aggregated Statistics ---")
    if total_stats:
        stat_order = [
            "total_games_found_across_files", "games_filtered_out", "games_skipped_parsing_error", # Added skipped count
            "games_processed", "moves_total", "invalid_san_skipped",
            "moves_with_pgn_eval", "moves_validated", "moves_mismatch",
            "moves_recomputed", "moves_missing_eval"
            ]
        total_stats["total_games_found_across_files"] = total_games # Add this for display
        for key in stat_order:
             if key in total_stats: print(f"{key.replace('_', ' ').title()}: {total_stats[key]}")
        for key, value in total_stats.items(): # Print any extras
            if key not in stat_order: print(f"{key.replace('_', ' ').title()}: {value}")
    else: print("No stats aggregated.");
    print("-" * 27)

# --- Argparse Setup (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PGN files to JSON...", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_files", nargs="+", help="Input PGN files.")
    parser.add_argument("-o", "--output-file", required=True, help="Output JSON/JSONL file.")
    parser.add_argument("--jsonl", action="store_true", help="Output JSON Lines.")
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument("--skip-unfinished", action="store_true", help="Skip games with Result '*'.")
    filter_group.add_argument("--only-standard", action="store_true", help="Keep only Variant 'Standard'.")
    filter_group.add_argument("--only-complete", action="store_true", help="Keep only 1-0, 0-1, 1/2-1/2 results.")
    filter_group.add_argument("--min-elo", type=int, default=None, metavar='ELO', help="Min Elo for players.")
    filter_group.add_argument("--max-elo", type=int, default=None, metavar='ELO', help="Max Elo for players.")
    sf_group = parser.add_argument_group('Stockfish Validation/Computation')
    sf_group.add_argument("--stockfish-path", default=None, metavar='PATH', help="Path to Stockfish.")
    sf_group.add_argument("--analysis-time", type=float, default=0.1, metavar='SEC', help="Stockfish time per move.")
    sf_group.add_argument("--mismatch-threshold", type=float, default=50.0, metavar='CP', help="Eval mismatch tolerance.")
    sf_group.add_argument("--overwrite-eval", action="store_true", help="Replace PGN eval with Stockfish.")
    sf_group.add_argument("--abort-on-mismatch", action="store_true", help="Stop on first eval mismatch.")
    sf_group.add_argument("--cache-dir", default=None, metavar='DIR', help="Directory for 'eval_cache.json'.")
    args = parser.parse_args()
    valid_inputs = True
    if not args.input_files: print("Error: No input files."); valid_inputs = False
    if not args.output_file: print("Error: Output file required."); valid_inputs = False
    if args.stockfish_path and not os.path.isfile(args.stockfish_path): print(f"Error: Stockfish not found: {args.stockfish_path}. Disabling SF."); args.stockfish_path = None
    if valid_inputs: main(args)
    else: print("\nExiting due to input errors.")