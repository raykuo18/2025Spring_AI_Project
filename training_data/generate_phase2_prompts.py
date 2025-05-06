#!/usr/bin/env python3

import json
import re
import argparse
import os
import chess # Requires python-chess
from tqdm import tqdm
import random
from typing import List, Dict, Optional, Tuple # Ensure Optional is here

# --- Helper Functions (reused or adapted) ---
def get_recent_pgn_history(pgn_moves_list: list, current_half_move_index: int, num_half_moves=10) -> str:
    """ Creates a string of the last N half-moves in SAN format, up to (but not including) the current move. """
    history = []
    start_half_move_idx = max(0, current_half_move_index - num_half_moves)
    
    for i in range(start_half_move_idx, current_half_move_index):
        move_pair_idx = i // 2
        is_black_move_in_history = (i % 2 == 1)

        if move_pair_idx >= len(pgn_moves_list): break 

        move_pair = pgn_moves_list[move_pair_idx]
        move_num = move_pair.get("move_number")

        current_move_text = ""
        if not is_black_move_in_history: # White's move
            if move_pair.get('white_move') and move_pair['white_move'].get('san'):
                current_move_text = f"{move_num}. {move_pair['white_move']['san']}"
            else: break 
        else: # Black's move
            if move_pair.get('black_move') and move_pair['black_move'].get('san'):
                # Add "..." only if it's black's first move shown in this snippet of history
                # and white's move for that number wasn't shown.
                prefix = f"{move_num}... " if (i == start_half_move_idx and not move_pair.get('white_move')) else ""
                current_move_text = f"{prefix}{move_pair['black_move']['san']}"
            else: break
        history.append(current_move_text)
                 
    return " ".join(history)


def extract_best_move_from_comment(comment: Optional[str]) -> Optional[str]:
    if not comment: return None
    pattern = r'\b([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[PNBRQK])?|O-O(?:-O)?)\b\s+was best\.'
    match = re.search(pattern, comment, re.IGNORECASE)
    return match.group(1) if match else None

def detect_format(filepath: str) -> bool:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip();
                if not line: continue
                if line.startswith('{'): return True # is_jsonl
                if line.startswith('['): return False # is_json_array
                return False 
    except Exception: return False
    return False

# --- Mixtral Prompt Formatting ---
MIXLTRAL_SYSTEM_PROMPT_COACH = "[SYSTEM] You are an expert chess coach and commentator. Your explanations are clear, concise, insightful, and tailored for an intermediate player. Focus on the key tactical and strategic ideas. Provide only the explanation text, without any conversational filler, greetings, or self-correction. Aim for 2-4 sentences and approximately 40-80 words, unless the specific task asks for something different. {FEW_SHOT_EXAMPLES_HERE}"
FEW_SHOT_PLACEHOLDER = "\n\n[FEW_SHOT_EXAMPLES_WOULD_GO_HERE_FOR_ACTUAL_API_CALL]\nFor example:\nContext:\nBoard State (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\nRecent Game History (PGN): 1. e4\nMove Played: c7c5 (c5)\nTask: Provide a concise explanation (2-3 sentences, approx. 40-70 words) for the move c5. Focus on the primary strategic goals OR immediate tactical ideas.\nExplanation:\n[ASSISTANT] Black responds with the Sicilian Defense, immediately challenging White's central e4-pawn and aiming for an imbalanced game. This move opens lines for Black's queen and aims to control the d4-square.\n---\n" # Add more diverse examples in practice

def format_teacher_prompt(task_instruction: str, metadata: dict, few_shot_placeholder: str = FEW_SHOT_PLACEHOLDER) -> str:
    system_prompt = MIXLTRAL_SYSTEM_PROMPT_COACH.format(FEW_SHOT_EXAMPLES_HERE=few_shot_placeholder)
    
    context_lines = [
        f"Board State (FEN) before the move: {metadata['fen_before_move']}",
        f"Recent Game History (PGN): {metadata['pgn_history_str']}",
        f"Move Played: {metadata['uci_move_played']} ({metadata['san_move']})"
    ]
    if metadata.get('quality_from_pgn'):
        context_lines.append(f"Source PGN Quality Annotation: {metadata['quality_from_pgn']}")
    if metadata.get('comment_from_pgn'):
        context_lines.append(f"Source PGN Comment: {metadata['comment_from_pgn']}")
    if metadata.get('event_from_pgn'):
        context_lines.append(f"Move Event: {metadata['event_from_pgn']}")
    if metadata.get('best_move_from_comment'):
        context_lines.append(f"Alternative Best Move suggested in comment: {metadata['best_move_from_comment']}")

    context_block = "\n".join(context_lines)
    return f"{system_prompt}\n[USER]\nContext:\n{context_block}\n\nTask: {task_instruction}\n\nExplanation:\n[ASSISTANT]"

def format_student_prompt(instruction: str, fen: str, pgn_history: str, uci_move: str) -> str:
    return f"[INST] {instruction} [SEP] [FEN] {fen} [SEP] [PGN] {pgn_history} [SEP] [MOVE] {uci_move} [/INST]"

# --- Main Data Generation ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for Phase 2 (Explanation Distillation) from processed game JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-file", required=True,
                        help="Path to the input JSON or JSONL file containing processed game data from main parser.")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Path to the output JSONL file to save prompts for Mixtral and TinyLLaMA.")
    parser.add_argument("--history-len", type=int, default=10, metavar='N',
                        help="Number of preceding half-moves to include in PGN history context.")
    parser.add_argument("--seed", type=int, default=42, metavar='SEED',
                        help="Random seed for sampling.")
    parser.add_argument("--p2-general-sample-rate", type=float, default=0.05, metavar='RATE',
                        help="Sample rate (0.0-1.0) for general move explanations (Rule P2.1).")
    # Add more sampling rates if needed for other rules, or apply them to all instances found.

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
         parser.error(f"Input file not found: {args.input_file}")

    random.seed(args.seed)

    is_jsonl = detect_format(args.input_file)
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    game_id_counter = 0
    total_prompts_generated = 0
    task_counts = {}

    print(f"Reading games from {args.input_file} and generating Phase 2 prompts...")

    try:
        with open(args.input_file, "r", encoding="utf-8") as f_in, \
             open(args.output_file, "w", encoding="utf-8") as f_out:

            games_iterator = []
            if is_jsonl: lines = f_in.readlines(); games_iterator = tqdm(lines, unit="game", desc="Processing Games")
            else:
                 try: all_games_data = json.load(f_in); games_iterator = tqdm(all_games_data, unit="game", desc="Processing Games")
                 except Exception as e: print(f"Error: Invalid JSON format: {e}"); return

            for game_data_item in games_iterator:
                game_obj = None
                if is_jsonl:
                    line = game_data_item.strip();
                    if not line: continue
                    try: game_obj = json.loads(line)
                    except json.JSONDecodeError: continue
                else: game_obj = game_data_item
                if not game_obj or "game_metadata" not in game_obj: continue

                game_metadata = game_obj["game_metadata"]
                starting_fen = game_metadata.get("fen"); pgn_moves = game_metadata.get("pgn"); variant = game_metadata.get("variant", "Standard").lower()
                if not starting_fen or not pgn_moves: continue

                game_id_counter += 1
                current_game_id = f"game{game_id_counter}"

                sim_board = chess.Board(starting_fen)
                if variant == "chess960": sim_board.chess960 = True

                half_move_index = 0 # 0-based index for half-moves processed

                for move_pair_index, move_pair in enumerate(pgn_moves):
                    for color_turn in ["white", "black"]:
                        move_key = f"{color_turn}_move"
                        if not move_pair.get(move_key): continue

                        move_info = move_pair[move_key]
                        uci = move_info.get('uci')
                        san = move_info.get('san', '???')
                        if not uci: continue

                        fen_before = sim_board.fen()
                        pgn_history_str = get_recent_pgn_history(pgn_moves, half_move_index, args.history_len)
                        
                        # --- Prepare metadata for teacher prompt ---
                        teacher_metadata = {
                            "fen_before_move": fen_before,
                            "pgn_history_str": pgn_history_str,
                            "uci_move_played": uci,
                            "san_move": san,
                            "quality_from_pgn": move_info.get("quality"),
                            "comment_from_pgn": move_info.get("comment"),
                            "event_from_pgn": move_info.get("outcome") or move_info.get("special_move") # Combine
                        }
                        
                        # --- Apply Rules to Generate Tasks ---
                        prompts_for_this_move = []

                        # Rule P2.1: General Move Explanation (Sampled)
                        if random.random() < args.p2_general_sample_rate:
                            task_id_str = f"{current_game_id}_{half_move_index}_{color_turn[0]}_P2.1_General"
                            student_instr = f"Explain the main purpose and potential consequences of the move {san}."
                            teacher_task_instr = f"Provide a concise explanation (2-3 sentences, approx. 40-70 words) for the move {san}. Focus on the primary strategic goals (e.g., central control, development, king safety, pawn structure) OR immediate tactical ideas (e.g., threats, defenses, piece activation, opening ideas)."
                            student_prompt = format_student_prompt(student_instr, fen_before, pgn_history_str, uci)
                            teacher_prompt = format_teacher_prompt(teacher_task_instr, teacher_metadata)
                            prompts_for_this_move.append({
                                "task_id": task_id_str, "student_prompt_input": student_prompt,
                                "teacher_prompt_full": teacher_prompt, "metadata_for_teacher_context": teacher_metadata,
                                "teacher_task_instruction": teacher_task_instr
                            })
                            task_counts["P2.1_General"] = task_counts.get("P2.1_General", 0) + 1

                        # Rule P2.2a: Explain Quality Annotation
                        quality = move_info.get("quality")
                        if quality:
                            task_id_str = f"{current_game_id}_{half_move_index}_{color_turn[0]}_P2.2a_Quality"
                            student_instr = f"Explain why the move {san} is described as a {quality}."
                            teacher_task_instr = f"The move {san} was annotated in the source PGN as '{quality}'. Explain concisely (2-3 sentences) the specific chess reasons that justify this annotation. What makes it a {quality}?"
                            student_prompt = format_student_prompt(student_instr, fen_before, pgn_history_str, uci)
                            # Update metadata for teacher with specific quality for this prompt
                            current_teacher_metadata = teacher_metadata.copy()
                            current_teacher_metadata["quality_from_pgn_for_task"] = quality # Explicitly pass
                            teacher_prompt = format_teacher_prompt(teacher_task_instr, current_teacher_metadata)
                            prompts_for_this_move.append({
                                "task_id": task_id_str, "student_prompt_input": student_prompt,
                                "teacher_prompt_full": teacher_prompt, "metadata_for_teacher_context": current_teacher_metadata,
                                "teacher_task_instruction": teacher_task_instr
                            })
                            task_counts["P2.2a_Quality"] = task_counts.get("P2.2a_Quality", 0) + 1


                        # Rule P2.2b: Explain "X was best" Comment
                        comment = move_info.get("comment")
                        best_move_san_from_comment = extract_best_move_from_comment(comment)
                        if best_move_san_from_comment:
                            task_id_str = f"{current_game_id}_{half_move_index}_{color_turn[0]}_P2.2b_BestMoveComment"
                            student_instr = f"The PGN comment for {san} says \"{best_move_san_from_comment} was best.\" Explain why {best_move_san_from_comment} might be better than {san}."
                            teacher_task_instr = f"The PGN comment for the played move {san} suggests \"{best_move_san_from_comment} was best.\" Briefly compare these two moves (2-4 sentences), explaining the main advantage of {best_move_san_from_comment} or the drawback of {san}."
                            student_prompt = format_student_prompt(student_instr, fen_before, pgn_history_str, uci)
                            current_teacher_metadata = teacher_metadata.copy()
                            current_teacher_metadata["best_move_from_comment"] = best_move_san_from_comment
                            teacher_prompt = format_teacher_prompt(teacher_task_instr, current_teacher_metadata)
                            prompts_for_this_move.append({
                                "task_id": task_id_str, "student_prompt_input": student_prompt,
                                "teacher_prompt_full": teacher_prompt, "metadata_for_teacher_context": current_teacher_metadata,
                                "teacher_task_instruction": teacher_task_instr
                            })
                            task_counts["P2.2b_BestMoveComment"] = task_counts.get("P2.2b_BestMoveComment", 0) + 1
                        
                        # Rule P2.2c: Elaborate on Other Informative Comments (if not P2.2b)
                        elif comment and not best_move_san_from_comment and len(comment) > 10 : # Heuristic for "informative"
                            task_id_str = f"{current_game_id}_{half_move_index}_{color_turn[0]}_P2.2c_GeneralComment"
                            student_instr = f"The PGN comment for {san} is: \"{comment}\". What does this comment mean in this chess context?"
                            teacher_task_instr = f"The PGN comment for the move {san} is: \"{comment}\". Explain concisely (2-3 sentences) the chess reasoning or meaning behind this comment in the context of the position."
                            student_prompt = format_student_prompt(student_instr, fen_before, pgn_history_str, uci)
                            teacher_prompt = format_teacher_prompt(teacher_task_instr, teacher_metadata) # teacher_metadata already has the comment
                            prompts_for_this_move.append({
                                "task_id": task_id_str, "student_prompt_input": student_prompt,
                                "teacher_prompt_full": teacher_prompt, "metadata_for_teacher_context": teacher_metadata,
                                "teacher_task_instruction": teacher_task_instr
                            })
                            task_counts["P2.2c_GeneralComment"] = task_counts.get("P2.2c_GeneralComment", 0) + 1


                        # Rule P2.3: Explain Specific Objective Events
                        event = move_info.get("outcome") or move_info.get("special_move")
                        if event and random.random() < 0.25: # Sample these to avoid too many
                            task_id_str = f"{current_game_id}_{half_move_index}_{color_turn[0]}_P2.3_Event_{event.replace(' ', '')}"
                            student_instr_event = f"The move {san} resulted in '{event}'. Explain this event."
                            if event == "Check": student_instr_event = f"Describe the check delivered by the move {san}."
                            elif event == "Checkmate": student_instr_event = f"Explain how {san} leads to checkmate."
                            elif event == "Capture": student_instr_event = f"Explain the capture made by {san}."
                            elif "Promotion" in event: student_instr_event = f"What is the significance of {event.lower()} by {san}?"
                            elif "Castling" in event: student_instr_event = f"Why castle ({san}) here?"
                            
                            teacher_task_instr = f"The move {san} resulted in '{event}'. Explain concisely (1-2 sentences) what this event means or achieves in this specific position."
                            student_prompt = format_student_prompt(student_instr_event, fen_before, pgn_history_str, uci)
                            teacher_prompt = format_teacher_prompt(teacher_task_instr, teacher_metadata)
                            prompts_for_this_move.append({
                                "task_id": task_id_str, "student_prompt_input": student_prompt,
                                "teacher_prompt_full": teacher_prompt, "metadata_for_teacher_context": teacher_metadata,
                                "teacher_task_instruction": teacher_task_instr
                            })
                            task_counts[f"P2.3_Event_{event.replace(' ', '')}"] = task_counts.get(f"P2.3_Event_{event.replace(' ', '')}", 0) + 1

                        # Write all generated prompts for this move to the output file
                        for prompt_data in prompts_for_this_move:
                            f_out.write(json.dumps(prompt_data) + "\n")
                            total_prompts_generated +=1
                        
                        # Push the actual move for the next iteration's FEN
                        sim_board.push_uci(uci)
                        half_move_index += 1
                
                # After processing both white and black for the move_pair
                # (or just white if black_move was None)
            
            # game_iterator.set_description(f"Processed game {current_game_id}")

    except Exception as e:
        print(f"Error reading input file or writing output file: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nProcessed {game_id_counter} games.")
    print(f"Generated {total_prompts_generated} total prompt sets for Phase 2.")
    print("Prompts per task type:")
    for task, count in sorted(task_counts.items()):
         print(f"- {task}: {count}")
    print(f"✅ Phase 2 prompt generation data saved to {args.output_file}")

if __name__ == "__main__":
    main()
