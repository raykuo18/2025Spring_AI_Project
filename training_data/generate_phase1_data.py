#!/usr/bin/env python3

import json
import re
import argparse
import os
import chess # Requires python-chess
from tqdm import tqdm
import random # Needed for sampling and hiding choices

# Helper to get recent PGN history string
def get_recent_pgn_history(pgn_moves_list: list, current_half_move_index: int, num_half_moves=10) -> str:
    """ Creates a string of the last N half-moves in SAN format. """
    history = []
    # Calculate the starting index for the slice of moves we need
    # current_half_move_index is 0 for the first half-move (1. White)
    # Need to fetch moves from pgn_moves_list based on full move numbers and color
    
    start_half_move_idx = max(0, current_half_move_index - num_half_moves)
    
    for i in range(start_half_move_idx, current_half_move_index): # Up to, but not including, the current move
        move_pair_idx = i // 2
        is_black_move = (i % 2 == 1)

        if move_pair_idx >= len(pgn_moves_list): break 

        move_pair = pgn_moves_list[move_pair_idx]
        move_num = move_pair.get("move_number")

        if not is_black_move: # White's move
            # Only add move number if it's the first in the pair for this number
            if move_num is not None:
                 history.append(f"{move_num}.")
            if move_pair.get('white_move') and move_pair['white_move'].get('san'):
                 history.append(move_pair['white_move']['san'])
            else: break 
        else: # Black's move
            if move_pair.get('black_move') and move_pair['black_move'].get('san'):
                 history.append(move_pair['black_move']['san'])
            else: break
                 
    return " ".join(history)

def generate_training_instance(task_type, instruction, fen, pgn_history, move_uci, label):
    """Formats a training instance."""
    input_prompt = f"[INST] {instruction} [SEP] [FEN] {fen} [SEP] [PGN] {pgn_history}"
    # Add MOVE token only if relevant (e.g., for explaining a specific move)
    # For Phase 1 tasks based on FEN/PGN state, MOVE is often redundant with instruction
    # Let's keep it simple and omit it for Phase 1 rules for now. Can add back if needed.
    # if move_uci:
    #     input_prompt += f" [SEP] [MOVE] {move_uci}"
    input_prompt += " [/INST]"
    # Ensure label is a string
    label_str = str(label) if label is not None else ""
    return {"task": task_type, "input": input_prompt, "output": label_str}

def extract_best_move_from_comment(comment):
    """Uses regex to find 'X was best.' pattern."""
    if not comment: return None
    pattern = r'\b([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)\b\s+was best\.'
    match = re.search(pattern, comment, re.IGNORECASE)
    return match.group(1) if match else None

def detect_format(filepath):
    """Tries to detect if file is JSONL or JSON array."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip();
                if not line: continue
                if line.startswith('{'): return True
                if line.startswith('['): return False
                return False # Default
    except Exception: return False
    return False

# --- Main Data Generation ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 rule-based training data (query-label pairs) from processed game JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-file", required=True,
                        help="Path to the input JSON or JSONL file containing processed game data.")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Path to the output JSONL file for Phase 1 training data.")
    parser.add_argument("--history-len", type=int, default=10, metavar='N',
                        help="Number of preceding half-moves to include in PGN history context.")
    parser.add_argument("--seed", type=int, default=42, metavar='SEED',
                        help="Random seed for sampling.")
    # Sampling Rates
    parser.add_argument("--predict-move-sample-rate", type=float, default=0.25, metavar='RATE',
                        help="Sample rate (0.0 to 1.0) for generating 'predict_move' tasks.")
    parser.add_argument("--legal-move-sample-rate", type=float, default=0.1, metavar='RATE',
                        help="Sample rate (0.0 to 1.0) for generating 'list_legal_moves' tasks.")
    parser.add_argument("--basic-rule-sample-rate", type=float, default=0.05, metavar='RATE',
                        help="Sample rate (0.0 to 1.0) for generating basic rule tasks (piece ID, color, attack, legality check).")
    # Multiple Choice Hiding
    parser.add_argument("--hide-choices-prob", type=float, default=0.2, metavar='PROB',
                        help="Probability (0.0 to 1.0) of hiding explicit choices in multiple-choice style questions.")

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.input_file): parser.error(f"Input file not found: {args.input_file}")
    for rate in [args.predict_move_sample_rate, args.legal_move_sample_rate, args.basic_rule_sample_rate]:
        if not (0.0 <= rate <= 1.0): parser.error("Sample rates must be between 0.0 and 1.0")
    if not (0.0 <= args.hide_choices_prob <= 1.0): parser.error("Hide choices probability must be between 0.0 and 1.0")

    random.seed(args.seed) # Set seed for reproducibility

    is_jsonl = detect_format(args.input_file)
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

    game_count = 0
    total_pairs_generated = 0
    task_counts = {}

    print(f"Reading games from {args.input_file} and generating Phase 1 data...")

    try:
        with open(args.input_file, "r", encoding="utf-8") as f_in, \
             open(args.output_file, "w", encoding="utf-8") as f_out:

            games_iterator = []
            if is_jsonl: lines = f_in.readlines(); games_iterator = tqdm(lines, unit="game", desc="Processing Games")
            else:
                 try: all_games_data = json.load(f_in); games_iterator = tqdm(all_games_data, unit="game", desc="Processing Games")
                 except Exception as e: print(f"Error: Invalid JSON format: {e}"); return

            for game_data_item in games_iterator:
                # ... (Load game_obj from JSONL line or JSON list item as before) ...
                game_obj = None
                if is_jsonl:
                    line = game_data_item.strip()
                    if not line: continue
                    try: game_obj = json.loads(line)
                    except json.JSONDecodeError: continue
                else: game_obj = game_data_item
                if not game_obj or "game_metadata" not in game_obj: continue

                game_metadata = game_obj["game_metadata"]
                starting_fen = game_metadata.get("fen"); pgn_moves = game_metadata.get("pgn"); variant = game_metadata.get("variant", "Standard").lower()
                if not starting_fen or not pgn_moves: continue

                game_count += 1
                sim_board = chess.Board(starting_fen)
                if variant == "chess960": sim_board.chess960 = True

                half_move_index = 0 # 0-based index for half-moves processed

                try:
                    for move_pair_index, move_pair in enumerate(pgn_moves):
                        # --- Process White's Move ---
                        if move_pair.get('white_move'):
                            move_info = move_pair['white_move']; uci = move_info.get('uci'); san = move_info.get('san', '???')
                            if uci:
                                fen_before = sim_board.fen()
                                current_pgn_history = get_recent_pgn_history(pgn_moves, half_move_index, args.history_len)

                                # --- Generate Tasks Based on Rules ---
                                # P1.1: Predict Move Played (Sampled)
                                if random.random() < args.predict_move_sample_rate:
                                    task_type = "predict_move"; instruction = "Predict the next move."
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, None, uci)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.2: Identify Outcome
                                outcome = move_info.get("outcome")
                                if outcome:
                                    task_type = "identify_outcome"; choices = " Choose: Check, Checkmate, Capture, None."
                                    instruction = f"What is the outcome of the move {san}?" + (choices if random.random() >= args.hide_choices_prob else "")
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, outcome)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.3: Identify Special Move
                                special = move_info.get("special_move")
                                if special:
                                    task_type = "identify_special_move"; choices = " Choose: Kingside Castling, Queenside Castling, Promotion, None."
                                    simple_special = "Promotion" if special.startswith("Promotion") else special
                                    instruction = f"What type of special move is {san}?" + (choices if random.random() >= args.hide_choices_prob else "")
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, simple_special)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.4: Identify Quality Annotation
                                quality = move_info.get("quality")
                                if quality:
                                    task_type = "identify_quality"; choices = " Choose: Brilliant, Good, Interesting, Dubious, Mistake, Blunder, None."
                                    instruction = f"What is the quality annotation for {san}?" + (choices if random.random() >= args.hide_choices_prob else "")
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, quality)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.5: Parse Comment - Best Move
                                comment = move_info.get("comment"); best_move_san = extract_best_move_from_comment(comment)
                                if best_move_san:
                                     task_type = "parse_comment_best_move"; instruction = f"The comment suggests an alternative to {san}. What move was suggested as best?"
                                     instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, best_move_san)
                                     f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.6: Parse Comment - Checkmate Unavoidable
                                if comment and "checkmate is now unavoidable" in comment.lower():
                                     task_type = "parse_comment_mate_unavoidable"; instruction = f"According to the comment for {san}, is checkmate unavoidable? Answer Yes or No."
                                     instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, "Yes")
                                     f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # --- Basic Rule Tasks (Sampled) ---
                                if random.random() < args.basic_rule_sample_rate:
                                    # P1.7: List Legal Moves (already sampled via basic_rule_sample_rate)
                                    task_type_lm = "list_legal_moves"
                                    legal_moves = sorted([m.uci() for m in sim_board.legal_moves])
                                    if legal_moves:
                                        instruction_lm = "List all legal moves from the current position."
                                        instance_lm = generate_training_instance(task_type_lm, instruction_lm, fen_before, current_pgn_history, None, " ".join(legal_moves))
                                        f_out.write(json.dumps(instance_lm) + "\n"); total_pairs_generated += 1; task_counts[task_type_lm] = task_counts.get(task_type_lm, 0) + 1
                                    
                                    # P1.8: Identify Piece on Square
                                    occupied_squares = [sq for sq in chess.SQUARES if sim_board.piece_at(sq)]
                                    if occupied_squares:
                                        task_type_p = "identify_piece"
                                        sq_index = random.choice(occupied_squares)
                                        sq_name = chess.square_name(sq_index)
                                        piece = sim_board.piece_at(sq_index)
                                        piece_symbol = piece.symbol().upper() # Use uppercase K, Q, R, B, N, P
                                        choices_p = " Choose: K, Q, R, B, N, P, None."
                                        instruction_p = f"What piece is on square {sq_name}?" + (choices_p if random.random() >= args.hide_choices_prob else "")
                                        instance_p = generate_training_instance(task_type_p, instruction_p, fen_before, current_pgn_history, None, piece_symbol)
                                        f_out.write(json.dumps(instance_p) + "\n"); total_pairs_generated += 1; task_counts[task_type_p] = task_counts.get(task_type_p, 0) + 1

                                    # P1.9: Identify Color on Square
                                    if occupied_squares:
                                        task_type_c = "identify_color"
                                        # Can reuse sq_index and sq_name from P1.8
                                        color = "White" if sim_board.color_at(sq_index) == chess.WHITE else "Black"
                                        choices_c = " Choose: White, Black, None."
                                        instruction_c = f"What is the color of the piece on square {sq_name}?" + (choices_c if random.random() >= args.hide_choices_prob else "")
                                        instance_c = generate_training_instance(task_type_c, instruction_c, fen_before, current_pgn_history, None, color)
                                        f_out.write(json.dumps(instance_c) + "\n"); total_pairs_generated += 1; task_counts[task_type_c] = task_counts.get(task_type_c, 0) + 1
                                    
                                    # P1.10: Is Square Attacked?
                                    task_type_a = "is_square_attacked"
                                    sq_index_a = random.choice(chess.SQUARES) # Pick any square
                                    sq_name_a = chess.square_name(sq_index_a)
                                    attacking_color = random.choice([chess.WHITE, chess.BLACK])
                                    attacker_name = "White" if attacking_color == chess.WHITE else "Black"
                                    is_attacked = sim_board.is_attacked_by(attacking_color, sq_index_a)
                                    label_a = "Yes" if is_attacked else "No"
                                    instruction_a = f"Is square {sq_name_a} attacked by {attacker_name}? Answer Yes or No."
                                    instance_a = generate_training_instance(task_type_a, instruction_a, fen_before, current_pgn_history, None, label_a)
                                    f_out.write(json.dumps(instance_a) + "\n"); total_pairs_generated += 1; task_counts[task_type_a] = task_counts.get(task_type_a, 0) + 1

                                    # P1.11: Can Piece Move To Square?
                                    if occupied_squares:
                                        task_type_l = "can_piece_move"
                                        from_sq_index = random.choice(occupied_squares)
                                        from_sq_name = chess.square_name(from_sq_index)
                                        piece_l = sim_board.piece_at(from_sq_index)
                                        piece_name_l = piece_l.symbol() # Use symbol like 'P' or 'n'
                                        to_sq_index = random.choice(chess.SQUARES)
                                        to_sq_name = chess.square_name(to_sq_index)
                                        
                                        # Check legality (handle promotion possibility simply)
                                        is_legal = False
                                        try:
                                            move_l = chess.Move(from_sq_index, to_sq_index)
                                            if piece_l.piece_type == chess.PAWN and chess.square_rank(to_sq_index) in [0, 7]:
                                                 move_l.promotion = chess.QUEEN # Assume queen promo for check
                                            if move_l in sim_board.legal_moves:
                                                is_legal = True
                                        except Exception: # Handle potential errors in move generation itself
                                            pass 
                                            
                                        label_l = "Yes" if is_legal else "No"
                                        instruction_l = f"Can the {piece_name_l} on {from_sq_name} legally move to {to_sq_name}? Answer Yes or No."
                                        instance_l = generate_training_instance(task_type_l, instruction_l, fen_before, current_pgn_history, None, label_l)
                                        f_out.write(json.dumps(instance_l) + "\n"); total_pairs_generated += 1; task_counts[task_type_l] = task_counts.get(task_type_l, 0) + 1


                                # Finally, push the actual move for the next iteration's FEN
                                sim_board.push_uci(uci)
                                half_move_index += 1

                        # --- Process Black's Move (Apply same rules and sampling) ---
                        if move_pair.get('black_move'):
                            move_info = move_pair['black_move']; uci = move_info.get('uci'); san = move_info.get('san', '???')
                            if uci:
                                fen_before = sim_board.fen()
                                current_pgn_history = get_recent_pgn_history(pgn_moves, half_move_index, args.history_len)

                                # P1.1: Predict Move (Sampled)
                                if random.random() < args.predict_move_sample_rate:
                                    task_type = "predict_move"; instruction = "Predict the next move."
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, None, uci)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.2: Outcome
                                outcome = move_info.get("outcome")
                                if outcome:
                                    task_type = "identify_outcome"; choices = " Choose: Check, Checkmate, Capture, None."
                                    instruction = f"What is the outcome of the move {san}?" + (choices if random.random() >= args.hide_choices_prob else "")
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, outcome)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.3: Special Move
                                special = move_info.get("special_move")
                                if special:
                                    task_type = "identify_special_move"; choices = " Choose: Kingside Castling, Queenside Castling, Promotion, None."
                                    simple_special = "Promotion" if special.startswith("Promotion") else special
                                    instruction = f"What type of special move is {san}?" + (choices if random.random() >= args.hide_choices_prob else "")
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, simple_special)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.4: Quality
                                quality = move_info.get("quality")
                                if quality:
                                    task_type = "identify_quality"; choices = " Choose: Brilliant, Good, Interesting, Dubious, Mistake, Blunder, None."
                                    instruction = f"What is the quality annotation for {san}?" + (choices if random.random() >= args.hide_choices_prob else "")
                                    instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, quality)
                                    f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.5: Best Move Comment
                                comment = move_info.get("comment"); best_move_san = extract_best_move_from_comment(comment)
                                if best_move_san:
                                     task_type = "parse_comment_best_move"; instruction = f"The comment suggests an alternative to {san}. What move was suggested as best?"
                                     instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, best_move_san)
                                     f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # P1.6: Checkmate Unavoidable Comment
                                if comment and "checkmate is now unavoidable" in comment.lower():
                                     task_type = "parse_comment_mate_unavoidable"; instruction = f"According to the comment for {san}, is checkmate unavoidable? Answer Yes or No."
                                     instance = generate_training_instance(task_type, instruction, fen_before, current_pgn_history, uci, "Yes")
                                     f_out.write(json.dumps(instance) + "\n"); total_pairs_generated += 1; task_counts[task_type] = task_counts.get(task_type, 0) + 1

                                # --- Basic Rule Tasks (Sampled) ---
                                if random.random() < args.basic_rule_sample_rate:
                                    # P1.7: List Legal Moves
                                    task_type_lm = "list_legal_moves"
                                    legal_moves = sorted([m.uci() for m in sim_board.legal_moves])
                                    if legal_moves:
                                        instruction_lm = "List all legal moves from the current position."
                                        instance_lm = generate_training_instance(task_type_lm, instruction_lm, fen_before, current_pgn_history, None, " ".join(legal_moves))
                                        f_out.write(json.dumps(instance_lm) + "\n"); total_pairs_generated += 1; task_counts[task_type_lm] = task_counts.get(task_type_lm, 0) + 1

                                    # P1.8: Identify Piece on Square
                                    occupied_squares = [sq for sq in chess.SQUARES if sim_board.piece_at(sq)]
                                    if occupied_squares:
                                        task_type_p = "identify_piece"; sq_index = random.choice(occupied_squares); sq_name = chess.square_name(sq_index)
                                        piece = sim_board.piece_at(sq_index); piece_symbol = piece.symbol().upper()
                                        choices_p = " Choose: K, Q, R, B, N, P, None."
                                        instruction_p = f"What piece is on square {sq_name}?" + (choices_p if random.random() >= args.hide_choices_prob else "")
                                        instance_p = generate_training_instance(task_type_p, instruction_p, fen_before, current_pgn_history, None, piece_symbol)
                                        f_out.write(json.dumps(instance_p) + "\n"); total_pairs_generated += 1; task_counts[task_type_p] = task_counts.get(task_type_p, 0) + 1

                                    # P1.9: Identify Color on Square
                                    if occupied_squares:
                                        task_type_c = "identify_color"; # Reuse sq_index, sq_name
                                        color = "White" if sim_board.color_at(sq_index) == chess.WHITE else "Black"
                                        choices_c = " Choose: White, Black, None."
                                        instruction_c = f"What is the color of the piece on square {sq_name}?" + (choices_c if random.random() >= args.hide_choices_prob else "")
                                        instance_c = generate_training_instance(task_type_c, instruction_c, fen_before, current_pgn_history, None, color)
                                        f_out.write(json.dumps(instance_c) + "\n"); total_pairs_generated += 1; task_counts[task_type_c] = task_counts.get(task_type_c, 0) + 1

                                    # P1.10: Is Square Attacked?
                                    task_type_a = "is_square_attacked"; sq_index_a = random.choice(chess.SQUARES); sq_name_a = chess.square_name(sq_index_a)
                                    attacking_color = random.choice([chess.WHITE, chess.BLACK]); attacker_name = "White" if attacking_color == chess.WHITE else "Black"
                                    is_attacked = sim_board.is_attacked_by(attacking_color, sq_index_a); label_a = "Yes" if is_attacked else "No"
                                    instruction_a = f"Is square {sq_name_a} attacked by {attacker_name}? Answer Yes or No."
                                    instance_a = generate_training_instance(task_type_a, instruction_a, fen_before, current_pgn_history, None, label_a)
                                    f_out.write(json.dumps(instance_a) + "\n"); total_pairs_generated += 1; task_counts[task_type_a] = task_counts.get(task_type_a, 0) + 1
                                    
                                    # P1.11: Can Piece Move To Square?
                                    if occupied_squares:
                                        task_type_l = "can_piece_move"; from_sq_index = random.choice(occupied_squares); from_sq_name = chess.square_name(from_sq_index)
                                        piece_l = sim_board.piece_at(from_sq_index); piece_name_l = piece_l.symbol(); to_sq_index = random.choice(chess.SQUARES); to_sq_name = chess.square_name(to_sq_index)
                                        is_legal = False
                                        try:
                                            move_l = chess.Move(from_sq_index, to_sq_index)
                                            if piece_l.piece_type == chess.PAWN and chess.square_rank(to_sq_index) in [0, 7]: move_l.promotion = chess.QUEEN
                                            if move_l in sim_board.legal_moves: is_legal = True
                                        except Exception: pass
                                        label_l = "Yes" if is_legal else "No"
                                        instruction_l = f"Can the {piece_name_l} on {from_sq_name} legally move to {to_sq_name}? Answer Yes or No."
                                        instance_l = generate_training_instance(task_type_l, instruction_l, fen_before, current_pgn_history, None, label_l)
                                        f_out.write(json.dumps(instance_l) + "\n"); total_pairs_generated += 1; task_counts[task_type_l] = task_counts.get(task_type_l, 0) + 1

                                # Push actual move
                                sim_board.push_uci(uci)
                                half_move_index += 1

                except Exception as sim_error:
                    print(f"Error during board simulation for game (FEN: {starting_fen}): {sim_error}. Skipping rest of game.")
                    continue # Skip to next game

    except Exception as e:
        print(f"Error reading input file or writing output file: {e}")
        return

    print(f"\nProcessed {game_count} games.")
    print(f"Generated {total_pairs_generated} total Phase 1 training instances.")
    print("Instances per task type:")
    for task, count in sorted(task_counts.items()):
         print(f"- {task}: {count}")
    print(f"✅ Phase 1 training data saved to {args.output_file}")

if __name__ == "__main__":
    main()
