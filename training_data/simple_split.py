#!/usr/bin/env python3

import json
import random
import argparse
import os
import math
from tqdm import tqdm # Optional: for progress reading large JSONL

def write_output(filename: str, games: list, is_jsonl: bool):
    """Writes a list of games to a JSON or JSONL file."""
    if not games:
        print(f"Info: No games to write for {filename}.")
        return
    print(f"Writing {len(games)} games to {filename}...")
    try:
        # Ensure the directory exists before writing
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir) # Create directory if needed
        with open(filename, "w", encoding="utf-8") as f:
            if is_jsonl:
                for game in games:
                    f.write(json.dumps(game, separators=(',', ':')) + "\n")
            else:
                json.dump(games, f, indent=2)
        print(f"✅ Successfully saved output to {filename}")
    except Exception as e:
         print(f"❌ Error writing output file {filename}: {e}")

def detect_format(filepath):
    """Tries to detect if file is JSONL (starts with {) or JSON array (starts with [)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('{'):
                    print("Detected JSON Lines (JSONL) format.")
                    return True # is_jsonl = True
                if line.startswith('['):
                    print("Detected JSON array format.")
                    return False # is_jsonl = False
                print("Warning: Could not reliably detect format. Assuming JSON array.")
                return False
    except Exception as e:
        print(f"Error detecting format for {filepath}: {e}. Assuming JSON array.")
        return False
    print("Warning: File empty or format unclear. Assuming JSON array.")
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Split an existing JSON or JSONL file into train, validation, and test sets (80/10/10 split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("input_file",
                        help="Path to the input JSON or JSONL file containing the list of games.")
    # --- NEW: Output Folder Argument ---
    parser.add_argument("-o", "--output-folder", default=None, metavar='DIR',
                        help="Directory to save the output split files. Defaults to the input file's directory.")
    # --- End New ---
    parser.add_argument("--seed", type=int, default=42, metavar='SEED',
                        help="Random seed for shuffling before splitting.")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
         parser.error(f"Input file not found: {args.input_file}")

    # --- Detect Format ---
    is_jsonl = detect_format(args.input_file)

    # --- Read Input File ---
    all_games = []
    print(f"Reading games from {args.input_file}...")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            if is_jsonl:
                 line_count = 0
                 try: # Try counting lines for progress bar
                    line_count = sum(1 for line in f)
                    f.seek(0)
                 except Exception: f.seek(0)

                 for line in tqdm(f, total=line_count if line_count else None, unit="game", desc="Reading JSONL"):
                     line = line.strip()
                     if not line: continue
                     try:
                         game = json.loads(line)
                         all_games.append(game)
                     except json.JSONDecodeError as line_err:
                          print(f"Warning: Skipping invalid JSON line: {line_err} - Line: {line[:100]}...")
            else: # Regular JSON array
                 all_games = json.load(f)
                 if not isinstance(all_games, list):
                      raise TypeError("Input JSON is not a list.")
    except Exception as e:
        print(f"Error reading input file {args.input_file}: {e}")
        return

    if not all_games:
        print("No games loaded from input file.")
        return

    print(f"Loaded {len(all_games)} games.")

    # --- Shuffle ---
    print(f"Shuffling games using random seed: {args.seed}...")
    random.seed(args.seed)
    random.shuffle(all_games)

    # --- Calculate Splits (Default 80/10/10) ---
    total_n = len(all_games)
    train_ratio, val_ratio = 0.8, 0.1 # Hardcoded default ratios
    n_train = int(train_ratio * total_n)
    n_val = int(val_ratio * total_n)
    n_test = total_n - n_train - n_val

    print(f"Splitting into Train: {n_train}, Validation: {n_val}, Test: {n_test} games.")

    # --- Slice ---
    train_end_idx = n_train
    val_end_idx = n_train + n_val
    train_games = all_games[:train_end_idx]
    val_games = all_games[train_end_idx:val_end_idx]
    test_games = all_games[val_end_idx:]

    # --- Determine output directory and filenames ---
    # Get directory of input file
    input_dir = os.path.dirname(args.input_file)
    # Use specified output folder or default to input folder
    output_dir = args.output_folder if args.output_folder is not None else input_dir
    # Get base name and extension from input file
    base_name_with_ext = os.path.basename(args.input_file)
    base, ext = os.path.splitext(base_name_with_ext)

    # Construct full paths
    train_filename = os.path.join(output_dir, f"{base}_train{ext}")
    val_filename = os.path.join(output_dir, f"{base}_val{ext}")
    test_filename = os.path.join(output_dir, f"{base}_test{ext}")

    # --- Ensure output directory exists ---
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # --- Write Output Files ---
    write_output(train_filename, train_games, is_jsonl)
    write_output(val_filename, val_games, is_jsonl)
    write_output(test_filename, test_games, is_jsonl)

    print("\nSplitting complete.")

if __name__ == "__main__":
    main()