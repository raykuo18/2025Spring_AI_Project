#!/usr/bin/env python3

import re
import argparse
import os

def find_comments_in_text(text):
    """Finds all substrings within curly braces using DOTALL."""
    # re.DOTALL makes '.' match newline characters, handling multi-line comments
    return re.findall(r'\{(.*?)\}', text, re.DOTALL)

def is_valid_pgn_comment(comment_text):
    """
    Checks if a potential comment is valid.
    Filters out empty/whitespace comments and common bracketed annotations
    that might accidentally appear within braces in non-standard files.
    """
    comment = comment_text.strip()
    if not comment:
        return False
    # Exclude annotations like [%eval ...] or [%clk ...]
    if comment.startswith('[%eval') or comment.startswith('[%clk'):
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Extract all non-empty comments {...} from PGN file(s) and save them to a text file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_pgn_files", nargs='+',
                        help="Path(s) to the input PGN file(s).")
    parser.add_argument("-o", "--output-file", required=True,
                        help="Path to the output text file to save comments.")
    args = parser.parse_args()

    all_comments = [] # Use a list to store all occurrences, including duplicates

    print(f"Processing PGN files: {args.input_pgn_files}")
    for pgn_file_path in args.input_pgn_files:
        if not os.path.exists(pgn_file_path):
            print(f"Warning: Input file not found: {pgn_file_path}. Skipping.")
            continue

        print(f"Reading comments from {os.path.basename(pgn_file_path)}...")
        try:
            with open(pgn_file_path, "r", encoding="utf-8", errors='ignore') as f:
                # Read the entire file content for simplicity
                content = f.read()

            # Find all potential comments in the content
            potential_comments = find_comments_in_text(content)
            file_comment_count = 0
            for comment_text in potential_comments:
                if is_valid_pgn_comment(comment_text):
                    all_comments.append(comment_text.strip()) # Add the valid, stripped comment
                    file_comment_count += 1
            print(f" Found {file_comment_count} valid comment instances.")

        except Exception as e:
            print(f"Error processing file {pgn_file_path}: {e}")

    if not all_comments:
        print("\nNo valid comments found in any input file.")
        return

    print(f"\nFound a total of {len(all_comments)} comment instances (including duplicates).")
    print(f"Writing comments to {args.output_file}...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_file, "w", encoding="utf-8") as f_out:
            for comment in all_comments:
                # Write each comment on a new line
                f_out.write(comment + "\n")
        print(f"✅ Successfully saved comments to {args.output_file}")

    except Exception as e:
        print(f"❌ Error writing comments file {args.output_file}: {e}")

if __name__ == "__main__":
    main()