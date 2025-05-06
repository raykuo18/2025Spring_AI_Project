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
    """
    Parses PGN header lines into a metadata dictionary.
    Uses standard PGN tags as keys.
    """
    meta = {}
    for line in headers:
        match = re.match(r'\[\s*(\w+)\s*"(.*)"\s*\]', line)
        if match:
            key, val = match.groups()
            meta[key] = val # Keep original PGN tag casing (e.g., WhiteElo)
    return meta

def parse_eval(eval_str: str) -> Optional[float]:
    """
    Parses Stockfish evaluation string to centipawn or mate score.
    Mates are encoded as +/- (10000 - plies_to_mate).
    Returns None if parsing fails.
    """
    try:
        eval_str = eval_str.strip()
        if eval_str.startswith("#"):
            mate_in = int(eval_str[1:])
            # Use a large number to represent mate, adjusted by plies
            # Positive for White mating, negative for Black mating
            # Ensure mate in 0 is handled (though unlikely in analysis)
            if mate_in == 0: return 10000 if eval_str == "#0" else -10000 # Or some other convention
            return (10000 - abs(mate_in)) * (1 if mate_in > 0 else -1)
        # Handle potential '+' sign for positive evals
        return float(eval_str.lstrip('+')) * 100 # Convert to centipawns
    except (ValueError, TypeError):
        # print(f"Debug: Could not parse eval string: '{eval_str}'") # Optional debug
        return None

def extract_move_annotations(text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Extracts eval, clock, and comment from annotation text."""
    eval_val = None
    clk_val = None
    comment_val = None

    # Extract [%eval ...] - handle pawn or centipawn values
    eval_match = re.search(r'\[%eval\s+([#\d\.\+-]+(?:\/\d+)?)\s*\]', text) # Added optional / for fractions
    if eval_match:
        eval_val = parse_eval(eval_match.group(1))

    # Extract [%clk ...]
    clk_match = re.search(r'\[%clk\s+([\d:]{4,})\s*\]', text) # Basic validation for H:MM:SS or MM:SS
    if clk_match:
        clk_val = clk_match.group(1)

    # Extract comments {...}, removing eval and clk annotations first to avoid duplication
    # Note: This regex replacement is basic. Nested annotations might need more robust parsing.
    text_no_annotations = re.sub(r'\[%(eval|clk)\s+[^\]]*?\]', '', text)
    comments = re.findall(r'\{(.*?)\}', text_no_annotations, re.DOTALL) # Use DOTALL for multi-line comments

    # Filter out empty strings and join non-empty comments
    cleaned_comments = [c.strip() for c in comments if c.strip()]
    if cleaned_comments:
        comment_val = ' '.join(cleaned_comments)
        print(f"Found Comment: {comment_val}")

    return eval_val, clk_val, comment_val

# --- Move Parsing with UCI ---

def parse_pgn_moves(pgn_text: str, board: chess.Board) -> List[Dict]:
    """
    Parses PGN move text into structured list of moves including UCI.
    Requires a chess.Board object initialized to the game's starting position.
    Modifies the board state as it parses. Iteratively finds moves.
    """
    moves_data = []
    # Clean PGN text: remove result, variations, normalize whitespace maybe?
    pgn_text = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', pgn_text).strip()
    pgn_text = re.sub(r'\([^)]*\)', '', pgn_text) # Remove variations
    pgn_text = re.sub(r'\s+', ' ', pgn_text) # Normalize whitespace to single spaces

    # Regex to find potential SAN moves (group 1) and their NAGs/checks (group 2)
    san_regex = re.compile(
        r'([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)' # Core SAN
        r'([?!+#]*)'                                                    # NAGs/Checks/Mates
    )
    # Regex to find move number tokens
    move_num_regex = re.compile(r'(\d+)\s*\.{1,3}')

    current_pos = 0
    last_move_number = 0
    current_pair: Optional[Dict] = None

    while current_pos < len(pgn_text):
        # Skip whitespace and comments/annotations before looking for move/number
        # Find the start of the next meaningful token (move number or SAN)
        search_start = current_pos
        while search_start < len(pgn_text) and pgn_text[search_start] in (' ', '{', '['):
            if pgn_text[search_start] == '{': # Skip comment
                comment_match = re.match(r'\{.*?\}', pgn_text[search_start:])
                if comment_match:
                    search_start += comment_match.end()
                else: search_start += 1 # Avoid infinite loop on malformed comment
            elif pgn_text[search_start] == '[': # Skip annotation like [%clk...]
                 annot_match = re.match(r'\[.*?\]', pgn_text[search_start:])
                 if annot_match:
                      search_start += annot_match.end()
                 else: search_start += 1
            else: # Whitespace
                 search_start += 1
        current_pos = search_start
        if current_pos >= len(pgn_text): break # Reached end

        # Check if the next token is a move number
        num_match = move_num_regex.match(pgn_text, current_pos)
        san_match = san_regex.match(pgn_text, current_pos)

        move_number_found = 0
        if num_match:
            move_number_found = int(num_match.group(1))
            # Decide if the SAN match immediately following the number is more likely
            # This helps distinguish "1. e4" from just "1."
            potential_san_after_num = san_regex.match(pgn_text, num_match.end())
            if potential_san_after_num:
                 # It's likely "N. SAN", treat SAN as the primary match
                 san_match = potential_san_after_num
                 current_pos = num_match.end() # Start search for annotations after SAN
            else:
                 # Just a number token, consume it and continue search
                 current_pos = num_match.end()
                 san_match = None # Don't process SAN yet if only number found


        # Process the SAN move if found
        if san_match:
            san_for_parsing = san_match.group(1)
            nags_checks = san_match.group(2)
            san_original = san_for_parsing + nags_checks
            # Move current_pos past the matched SAN
            current_pos = san_match.end()

            # Check if it looks like a real move
            if not re.search('[a-hO]', san_for_parsing):
                 # print(f"Debug: Skipping '{san_original}' as it doesn't look like SAN.")
                 continue # Skip this iteration

            # Determine the correct move number
            # If a number token was just found, use it. Otherwise, infer from board state.
            current_move_number = move_number_found if move_number_found > 0 else board.fullmove_number

            # Check whose turn it is BEFORE parsing/pushing
            is_black_turn_to_move = (board.turn == chess.BLACK)

            move_uci = None
            valid_move = False
            try:
                move = board.parse_san(san_for_parsing)
                move_uci = move.uci()
                board.push(move) # Advance board state ONLY AFTER successful parse
                valid_move = True
            except ValueError as e:
                print(f"Warning: Invalid SAN '{san_for_parsing}' ('{san_original}') for move number ~{current_move_number} on {board.fen()}. Error: {e}. Skipping.")
                # If move is invalid, don't add it, try finding next token
                continue

            # Extract annotations AFTER the move
            annotation_text = ""
            annot_start = current_pos
            while annot_start < len(pgn_text) and pgn_text[annot_start] in (' ', '{', '['):
                 if pgn_text[annot_start] == '{':
                      match = re.match(r'\{.*?\}', pgn_text[annot_start:])
                      if match:
                           annotation_text += match.group(0) + " "
                           annot_start += match.end()
                      else: break # Malformed
                 elif pgn_text[annot_start] == '[':
                      match = re.match(r'\[.*?\]', pgn_text[annot_start:])
                      if match:
                           annotation_text += match.group(0) + " "
                           annot_start += match.end()
                      else: break
                 else: # Whitespace
                      annot_start += 1
            current_pos = annot_start # Consume annotations
            eval_cp, clock, comment = extract_move_annotations(annotation_text.strip())

            move_info = {
                'san': san_original,
                'uci': move_uci,
                'eval_cp': eval_cp,
                'clock': clock,
                'comment': comment
            }

            # --- Add move_info to the correct pair ---
            if not is_black_turn_to_move: # White's move
                 # Finalize previous pair if it exists and was for black only
                 if current_pair and current_pair['white_move'] is None and current_pair['black_move'] is not None:
                      print(f"Warning: Finalizing Black-only move pair {current_pair['move_number']} before White's {current_move_number}")
                      moves_data.append(current_pair)
                      current_pair = None

                 # Start a new pair for this white move
                 current_pair = {
                     "move_number": current_move_number,
                     "white_move": move_info,
                     "black_move": None
                 }
            else: # Black's move
                 if current_pair and current_pair['move_number'] == current_move_number and current_pair['white_move'] is not None:
                      # Add black move to the existing pair initiated by white
                      current_pair['black_move'] = move_info
                      moves_data.append(current_pair) # Pair complete, add it
                      current_pair = None # Reset for next pair
                 else:
                      # Black move found without a preceding white move for this number
                      # (Could be PGN starting N... or error)
                      print(f"Warning: Black move {current_move_number}...{san_original} found without preceding white move.")
                      # Create a black-only entry for this move number
                      black_only_pair = {
                           "move_number": current_move_number,
                           "white_move": None,
                           "black_move": move_info
                      }
                      moves_data.append(black_only_pair)
                      current_pair = None # Reset

        elif not num_match:
             # Didn't find a number or a SAN move, might be unexpected text or end
             # print(f"Debug: No move number or SAN found at position {current_pos}. Text: '{pgn_text[current_pos:current_pos+30]}...'")
             # Advance position cautiously to avoid infinite loops
             if current_pos < len(pgn_text):
                 current_pos += 1
             else: break # Should already be handled by outer while

    # --- After loop, add the last pair if it was White-only ---
    if current_pair and current_pair['white_move'] and not current_pair['black_move']:
        moves_data.append(current_pair)

    return moves_data

# --- Filtering and Stockfish Validation ---

def filter_game(meta: Dict, args) -> bool:
    """
    Returns True if game passes all filtering criteria.
    Uses direct key access based on parse_pgn_metadata output.
    """
    if args.skip_unfinished and meta.get("Result") == "*":
        return False
    if args.only_standard and meta.get("Variant", "Standard").lower() != "standard":
        return False
    # Result could be missing, handle None case
    result = meta.get("Result")
    if args.only_complete and result not in ["1-0", "0-1", "1/2-1/2"]:
        return False

    # Safely convert ELOs, handle missing or non-numeric values gracefully
    try:
        white_elo = int(meta.get("WhiteElo", "0"))
    except ValueError:
        white_elo = 0 # Or handle as None/skip if required
    try:
        black_elo = int(meta.get("BlackElo", "0"))
    except ValueError:
        black_elo = 0 # Or handle as None/skip if required

    if args.min_elo is not None and (white_elo < args.min_elo or black_elo < args.min_elo):
        return False
    if args.max_elo is not None and (white_elo > args.max_elo or black_elo > args.max_elo):
        return False
    return True

def stockfish_eval(engine, board: chess.Board, analysis_time: float) -> Optional[float]:
    """
    Computes Stockfish evaluation for a board.
    Returns centipawn score (positive = White advantage).
    Returns None if analysis fails.
    Mate scores are converted to the +/- (10000 - plies) format.
    """
    if not engine: return None
    try:
        # Increased depth limit slightly, time is usually the main constraint
        info = engine.analyse(board, chess.engine.Limit(time=analysis_time, depth=15))
        if "score" in info:
            score = info["score"].white() # Get score from White's perspective
            if score.is_mate():
                mate_plies = score.mate()
                # Convert mate score to our large centipawn representation
                return (10000 - abs(mate_plies)) * (1 if mate_plies > 0 else -1)
            else:
                # Ensure score is returned in centipawns
                cp = score.score(mate_score=10000) # Use mate_score consistent with above
                return cp
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, Exception) as e:
        print(f"Stockfish analysis failed for FEN {board.fen()}: {e}")
        return None
    return None # Default return if no score found

def compute_move_key(fen: str, move_number: int, is_black: bool) -> str:
    """
    Creates a unique key for caching based on FEN, move number, and turn.
    """
    turn_indicator = "b" if is_black else "w"
    key_str = f"{fen}|{move_number}|{turn_indicator}"
    return hashlib.md5(key_str.encode()).hexdigest()

def validate_eval(existing_cp: Optional[float], computed_cp: Optional[float], threshold: float) -> bool:
    """
    Checks whether existing eval matches computed within threshold.
    Handles None values (no PGN eval or failed computation).
    Returns True if they match or if comparison isn't possible/needed.
    Returns False only if both exist and differ significantly.
    """
    if existing_cp is None or computed_cp is None:
        return True # Cannot compare, or no existing eval to validate

    # Handle mate scores (large absolute values) - exact match preferred
    if abs(existing_cp) > 9000 and abs(computed_cp) > 9000:
         # Allow small ply difference? For now, require exact match for mates
         return existing_cp == computed_cp

    # Non-mate scores - compare within threshold
    return abs(existing_cp - computed_cp) <= threshold

# --- JSON Output Creation ---

def create_json_entry(meta: Dict, moves: List[Dict], starting_fen: str) -> Dict:
    """
    Creates a structured JSON entry for a game.
    """
    try:
        white_elo = int(meta.get("WhiteElo", "0"))
    except ValueError:
        white_elo = None # Use None for non-integer ELOs
    try:
        black_elo = int(meta.get("BlackElo", "0"))
    except ValueError:
        black_elo = None # Use None for non-integer ELOs

    return {
        "game_metadata": {
            "fen": starting_fen,
            "result": meta.get("Result"),
            "white_elo": white_elo,
            "black_elo": black_elo,
            "eco": meta.get("ECO"),
            "variant": meta.get("Variant", "Standard"),
            "game_url": meta.get("GameURL") or meta.get("Site"), # Try GameURL first, then Site
            "pgn": moves
        },
        "synthetic_data": [],
        "distilled_data": []
    }

# --- Main Processing Logic ---

def process_file(file_path: str, args, engine, cache: Dict) -> Tuple[List[Dict], Dict]:
    """
    Processes a PGN file into validated JSON entries.
    Optionally uses Stockfish engine for validation/computation and a cache.
    Collects unique comments found.
    Returns the list of processed game JSON objects and updated statistics.
    """
    try:
        # Read with flexible newline handling
        with open(file_path, "r", encoding="utf-8", newline=None) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], {}

    # Split into games: look for [Event...] possibly preceded by empty lines
    games_raw = re.split(r'\n\s*\n(?=\[Event)', content.strip())
    results = []
    stats = {
        "games_found": len(games_raw),
        "games_processed": 0,
        "games_filtered_out": 0,
        "moves_total": 0,
        "moves_with_pgn_eval": 0,
        "moves_validated": 0,
        "moves_mismatch": 0,
        "moves_recomputed": 0,
        "moves_missing_eval": 0,
        "invalid_san_skipped": 0,
    }

    game_counter = 0
    for raw_game in games_raw:
        game_counter += 1
        raw_game = raw_game.strip()
        if not raw_game or not raw_game.startswith("[Event"):
            # print(f"Debug: Skipping empty or non-game block starting with: {raw_game[:50]}")
            continue

        # Split headers and movetext
        parts = raw_game.split('\n\n', 1)
        header_text = parts[0]
        moves_text = parts[1].strip() if len(parts) > 1 else ""

        headers = header_text.splitlines()
        meta = parse_pgn_metadata(headers)

        # Apply Filters
        if not filter_game(meta, args):
            stats["games_filtered_out"] += 1
            continue

        # Setup board based on FEN or standard start
        board = chess.Board() # Standard starting position default
        fen = meta.get("FEN")
        variant = meta.get("Variant", "Standard").lower()

        if fen:
            try:
                board.set_fen(fen)
            except ValueError:
                print(f"Warning: Invalid FEN '{fen}' in game {game_counter}. Using standard start.")
                board.reset()
        elif variant != "standard":
             # If variant is not standard but no FEN, it's ambiguous (e.g., Chess960)
             # We proceed with standard, but user should be aware.
             print(f"Warning: Variant '{variant}' specified without FEN in game {game_counter}. Using standard start position.")

        starting_fen = board.fen()
        board_for_parsing = board.copy() # Use a copy for parsing moves
        board_for_eval = board.copy() # Use a copy for eval validation/computation

        # Parse moves first (gets SAN, UCI, existing annotations)
        parsed_moves = parse_pgn_moves(moves_text, board_for_parsing)

        # --- Optional Stockfish Validation/Computation Pass ---
        if engine:
            current_eval_board = board_for_eval # Use the clean board copy
            for move_pair in parsed_moves:
                move_num = move_pair['move_number']

                # Process White's move
                if move_pair['white_move']:
                    w_move_data = move_pair['white_move']
                    stats["moves_total"] += 1
                    if w_move_data['uci'] is None: # Check if parsing failed earlier
                         stats["invalid_san_skipped"] += 1
                         # Cannot validate or push this move if UCI is missing
                         print(f"Skipping validation for invalid White move {move_num}. {w_move_data['san']}")
                         continue # Skip validation for this move

                    board_fen_before_move = current_eval_board.fen()
                    fen_key = compute_move_key(board_fen_before_move, move_num, is_black=False)
                    cached_eval = cache.get(fen_key)
                    computed_eval = None

                    pgn_eval = w_move_data['eval_cp']
                    if pgn_eval is not None:
                        stats["moves_with_pgn_eval"] += 1

                    # Decide whether to compute/recompute
                    needs_computation = (pgn_eval is None) or (args.overwrite_eval) or (cached_eval is None)

                    if needs_computation:
                        if cached_eval is not None:
                            computed_eval = cached_eval
                        else:
                            computed_eval = stockfish_eval(engine, current_eval_board, args.analysis_time)
                            if computed_eval is not None:
                                cache[fen_key] = computed_eval
                                stats["moves_recomputed"] += 1
                    else: # Use PGN eval if not overwriting and no computation needed yet
                         computed_eval = pgn_eval # For validation comparison if needed

                    # Validate if PGN eval exists and we have a computed/cached value
                    if pgn_eval is not None and computed_eval is not None:
                         stats["moves_validated"] += 1
                         if not validate_eval(pgn_eval, computed_eval, args.mismatch_threshold):
                             stats["moves_mismatch"] += 1
                             print(f"⚠️ Mismatch W Move {move_num} ({w_move_data['san']}): PGN {pgn_eval:.0f} vs SF {computed_eval:.0f} (FEN: {board_fen_before_move})")
                             if args.abort_on_mismatch:
                                 raise RuntimeError(f"Aborted due to mismatch at white move {move_num}.")
                             if args.overwrite_eval:
                                 w_move_data['eval_cp'] = computed_eval # Overwrite
                         elif args.overwrite_eval: # Overwrite even if matches (forces SF eval)
                              w_move_data['eval_cp'] = computed_eval

                    elif pgn_eval is None and computed_eval is not None:
                         # Fill in missing eval if computed
                         w_move_data['eval_cp'] = computed_eval
                         stats["moves_missing_eval"] += 1

                    # Push white's move on the evaluation board
                    try:
                       current_eval_board.push_uci(w_move_data['uci'])
                    except ValueError:
                        # This should not happen if parse_pgn_moves worked, but safety check
                        print(f"Error: Could not push valid UCI {w_move_data['uci']} on eval board. Skipping rest of game eval.")
                        break # Stop evaluating this game


                # Process Black's move
                if move_pair['black_move']:
                    b_move_data = move_pair['black_move']
                    stats["moves_total"] += 1
                    if b_move_data['uci'] is None:
                        stats["invalid_san_skipped"] += 1
                        print(f"Skipping validation for invalid Black move {move_num}... {b_move_data['san']}")
                        continue

                    board_fen_before_move = current_eval_board.fen()
                    fen_key = compute_move_key(board_fen_before_move, move_num, is_black=True)
                    cached_eval = cache.get(fen_key)
                    computed_eval = None

                    pgn_eval = b_move_data['eval_cp']
                    if pgn_eval is not None:
                        stats["moves_with_pgn_eval"] += 1

                    needs_computation = (pgn_eval is None) or (args.overwrite_eval) or (cached_eval is None)

                    if needs_computation:
                        if cached_eval is not None:
                            computed_eval = cached_eval
                        else:
                            computed_eval = stockfish_eval(engine, current_eval_board, args.analysis_time)
                            if computed_eval is not None:
                                cache[fen_key] = computed_eval
                                stats["moves_recomputed"] += 1
                    else:
                         computed_eval = pgn_eval

                    if pgn_eval is not None and computed_eval is not None:
                         stats["moves_validated"] += 1
                         if not validate_eval(pgn_eval, computed_eval, args.mismatch_threshold):
                             stats["moves_mismatch"] += 1
                             print(f"⚠️ Mismatch B Move {move_num} ({b_move_data['san']}): PGN {pgn_eval:.0f} vs SF {computed_eval:.0f} (FEN: {board_fen_before_move})")
                             if args.abort_on_mismatch:
                                 raise RuntimeError(f"Aborted due to mismatch at black move {move_num}.")
                             if args.overwrite_eval:
                                 b_move_data['eval_cp'] = computed_eval
                         elif args.overwrite_eval:
                              b_move_data['eval_cp'] = computed_eval

                    elif pgn_eval is None and computed_eval is not None:
                         b_move_data['eval_cp'] = computed_eval
                         stats["moves_missing_eval"] += 1

                    # Push black's move on the evaluation board
                    try:
                        current_eval_board.push_uci(b_move_data['uci'])
                    except ValueError:
                        print(f"Error: Could not push valid UCI {b_move_data['uci']} on eval board. Skipping rest of game eval.")
                        break
        # --- End Stockfish Validation Pass ---

        # Create the final JSON entry for this game
        json_entry = create_json_entry(meta, parsed_moves, starting_fen)
        results.append(json_entry)
        stats["games_processed"] += 1

    # Print summary statistics
    print("-" * 20)
    print(f"File: {file_path}")
    print(f"Games Found: {stats['games_found']}")
    print(f"Games Filtered Out: {stats['games_filtered_out']}")
    print(f"Games Processed: {stats['games_processed']}")
    print(f"Total Moves Parsed: {stats['moves_total']}")
    print(f"Invalid SAN Moves Skipped: {stats['invalid_san_skipped']}")
    if engine:
        print(f"Moves with PGN Eval: {stats['moves_with_pgn_eval']}")
        print(f"Moves Validated w/ Stockfish: {stats['moves_validated']}")
        print(f"Eval Mismatches: {stats['moves_mismatch']}")
        print(f"Moves Recomputed/Cached: {stats['moves_recomputed']}")
        print(f"Moves with Eval Added: {stats['moves_missing_eval']}")
    print("-" * 20)


    return results, stats

def main(args):
    cache_file = None
    cache = {}
    # Setup Cache
    if args.cache_dir:
        if not os.path.exists(args.cache_dir):
            try:
                os.makedirs(args.cache_dir)
                print(f"Created cache directory: {args.cache_dir}")
            except OSError as e:
                print(f"Error creating cache directory {args.cache_dir}: {e}. Cache disabled.")
                args.cache_dir = None # Disable cache if dir creation fails
        if args.cache_dir:
             cache_file = os.path.join(args.cache_dir, "eval_cache.json")
             if os.path.exists(cache_file):
                 try:
                     with open(cache_file, "r") as f:
                         cache = json.load(f)
                     print(f"Loaded {len(cache)} cached evaluations from {cache_file}")
                 except json.JSONDecodeError:
                      print(f"Warning: Cache file {cache_file} is corrupted. Starting with empty cache.")
                      cache = {}
                 except Exception as e:
                      print(f"Error loading cache file {cache_file}: {e}. Starting with empty cache.")
                      cache = {}


    # Setup Stockfish Engine if path provided
    engine = None
    if args.stockfish_path:
        if not os.path.exists(args.stockfish_path):
             print(f"Error: Stockfish path not found: {args.stockfish_path}")
             print("Stockfish validation/computation will be disabled.")
        else:
             try:
                 engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
                 # Optional: Configure engine parameters (like Threads or Hash) if needed
                 # engine.configure({"Threads": 4})
                 print(f"Stockfish engine initialized: {args.stockfish_path}")
             except Exception as e:
                 print(f"Error initializing Stockfish engine: {e}.")
                 print("Stockfish validation/computation will be disabled.")
                 engine = None # Ensure engine is None if init fails

    # Process Files
    all_games = []
    total_stats = {} # Aggregate stats across files if needed

    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"Warning: Input file not found: {file_path}. Skipping.")
            continue

        print(f"\nProcessing file: {file_path}...")
        try:
             games, file_stats = process_file(file_path, args, engine, cache)
             all_games.extend(games)
             # Aggregate stats (simple sum for counts)
             for key, value in file_stats.items():
                 total_stats[key] = total_stats.get(key, 0) + value
        except Exception as e:
             print(f"\n--- CRITICAL ERROR processing {file_path} ---")
             print(f"Error: {e}")
             import traceback
             traceback.print_exc()
             print("Attempting to continue with next file...")
             # Optionally break here if one file error should stop everything
             # break


    # Shutdown engine
    if engine:
        try:
            engine.quit()
            print("Stockfish engine shut down.")
        except chess.engine.EngineTerminatedError:
             pass # Ignore if already terminated


    # Save Cache
    if cache_file and (args.stockfish_path and engine is not None): # Only save if cache was used
         try:
             with open(cache_file, "w") as f:
                 json.dump(cache, f)
             print(f"Saved {len(cache)} evaluations to cache: {cache_file}")
         except Exception as e:
             print(f"Error saving cache file {cache_file}: {e}")

    # Write Output
    output_file = args.output_file
    if not all_games:
         print("\nNo games were successfully processed or passed filters. No output file written.")
         return # Exit if no games to write

    print(f"\nWriting {len(all_games)} processed games to {output_file}...")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir) # Create output directory if it doesn't exist

        if args.jsonl:
            with open(output_file, "w", encoding="utf-8") as f:
                for game in all_games:
                    f.write(json.dumps(game) + "\n")
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_games, f, indent=2)
        print(f"✅ Successfully saved output to {output_file}")

    except Exception as e:
         print(f"❌ Error writing output file {output_file}: {e}")

    # Print final aggregated stats
    print("\n--- Aggregated Statistics ---")
    if total_stats:
        for key, value in total_stats.items():
             # Simple formatting improvement
             key_formatted = key.replace('_', ' ').title()
             print(f"{key_formatted}: {value}")
    else:
        print("No statistics aggregated (likely no files processed).")
    print("-" * 27)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse PGN files to JSON with UCI, annotations, and optional Stockfish validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
        )

    # Input/Output Arguments
    parser.add_argument("input_files", nargs="+",
                        help="One or more input PGN files.")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Output file path for JSON or JSONL.")
    parser.add_argument("--jsonl", action="store_true",
                        help="Output as JSON Lines (one JSON object per line) instead of a single JSON array.")

    # Filtering Arguments
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument("--skip-unfinished", action="store_true",
                              help="Skip games where the Result is '*'.")
    filter_group.add_argument("--only-standard", action="store_true",
                              help="Keep only games where the Variant header is 'Standard' (case-insensitive).")
    filter_group.add_argument("--only-complete", action="store_true",
                              help="Keep only games with conclusive results (1-0, 0-1, 1/2-1/2).")
    filter_group.add_argument("--min-elo", type=int, default=None, metavar='ELO',
                              help="Skip games if either player's Elo (WhiteElo/BlackElo) is below this value.")
    filter_group.add_argument("--max-elo", type=int, default=None, metavar='ELO',
                              help="Skip games if either player's Elo is above this value.")

    # Stockfish Validation Arguments
    sf_group = parser.add_argument_group('Stockfish Validation/Computation')
    sf_group.add_argument("--stockfish-path", default=None, metavar='PATH',
                           help="Path to the Stockfish executable binary. If provided, enables validation/computation.")
    sf_group.add_argument("--analysis-time", type=float, default=0.1, metavar='SEC',
                           help="Stockfish thinking time per move in seconds for validation/computation.")
    sf_group.add_argument("--mismatch-threshold", type=float, default=50.0, metavar='CP',
                           help="Acceptable centipawn difference between PGN eval and Stockfish eval.")
    sf_group.add_argument("--overwrite-eval", action="store_true",
                           help="Always replace PGN [%eval] with Stockfish's computed evaluation.")
    sf_group.add_argument("--abort-on-mismatch", action="store_true",
                           help="Stop processing immediately if an eval mismatch exceeding the threshold is found.")
    sf_group.add_argument("--cache-dir", default=None, metavar='DIR',
                           help="Directory to store/load Stockfish evaluation cache ('eval_cache.json'). Improves speed on reruns.")


    args = parser.parse_args()

    # Input validation
    valid_inputs = True
    if not args.input_files:
        print("Error: No input files provided.")
        valid_inputs = False
    if not args.output_file:
        print("Error: Output file path is required (--output-file).")
        valid_inputs = False
    if args.stockfish_path and not os.path.isfile(args.stockfish_path):
         print(f"Error: Stockfish path specified, but file not found: {args.stockfish_path}")
         print("Stockfish features will be disabled.")
         args.stockfish_path = None # Disable if path invalid

    if valid_inputs:
        main(args)
    else:
        print("\nExiting due to input errors.")
        # parser.print_help() # Optionally show help on error