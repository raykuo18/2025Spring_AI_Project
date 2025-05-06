import json
import re
from typing import List, Dict, Optional

def parse_pgn_metadata(headers: List[str]) -> Dict:
    """
    Parse PGN header lines into metadata dictionary.
    """
    meta = {}
    for line in headers:
        if line.startswith("["):
            key, val = re.match(r'\[(\w+)\s+"(.*)"\]', line).groups()
            meta[key.lower()] = val
    return meta

def parse_eval(eval_str: str) -> float:
    """
    Parses eval string to numeric value (centipawn or mate).
    """
    if eval_str.startswith("#"):
        return float(eval_str.replace("#", "")) * 1000
    return float(eval_str)

def parse_pgn_moves(pgn_text: str) -> List[Dict]:
    """
    Parses PGN move text into structured list.
    """
    move_pattern = re.compile(
        r'(\d+)\.\s+([^\s{]+)\s+\{\s+\[%eval\s+([^\]]+)\]\s+\[%clk\s+([^\]]+)\]\s+\}'
        r'\s+([^\s{]+)\s+\{\s+\[%eval\s+([^\]]+)\]\s+\[%clk\s+([^\]]+)\]\s+\}'
    )
    moves = []
    for m in move_pattern.finditer(pgn_text):
        moves.append({
            "move_number": int(m.group(1)),
            "white_move": {"san": m.group(2), "eval_cp": parse_eval(m.group(3)), "clock": m.group(4)},
            "black_move": {"san": m.group(5), "eval_cp": parse_eval(m.group(6)), "clock": m.group(7)}
        })
    return moves

def filter_game(meta: Dict, args) -> bool:
    """
    Returns True if game passes all filter criteria.
    """
    if args.skip_unfinished and meta.get("result") == "*":
        return False
    if args.only_standard and meta.get("variant", "").lower() != "standard":
        return False
    if args.only_complete and meta.get("result") not in ["1-0", "0-1", "1/2-1/2"]:
        return False
    white_elo = int(meta.get("whiteelo", "0"))
    black_elo = int(meta.get("blackelo", "0"))
    if args.min_elo and (white_elo < args.min_elo or black_elo < args.min_elo):
        return False
    if args.max_elo and (white_elo > args.max_elo or black_elo > args.max_elo):
        return False
    return True

def create_json_entry(meta: Dict, moves: List[Dict], fen: str, turn: str, castling: str, ep: str) -> Dict:
    """
    Formats dataset JSON entry.
    """
    return {
        "game_metadata": {
            "fen": fen,
            "turn": turn,
            "castling": castling,
            "ep": ep,
            "pgn": moves,
            "result": meta.get("result"),
            "white_elo": int(meta.get("whiteelo", "0")),
            "black_elo": int(meta.get("blackelo", "0")),
            "eco": meta.get("eco"),
            "variant": meta.get("variant"),
            "game_url": meta.get("gameurl")
        },
        "synthetic_data": [],
        "distilled_data": []
    }

def process_file(file_path: str, args) -> List[Dict]:
    """
    Parses a PGN file (multi-game) and returns list of JSON entries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split games by double newline between headers
    games_raw = re.split(r'\n(?=\[Event)', content)
    results = []

    for raw_game in games_raw:
        lines = [line.strip() for line in raw_game.splitlines() if line.strip()]
        headers = [line for line in lines if line.startswith("[")]
        moves_text = "\n".join(line for line in lines if not line.startswith("["))

        if not headers:
            continue  # skip empty chunks

        metadata = parse_pgn_metadata(headers)
        if not filter_game(metadata, args):
            continue

        moves = parse_pgn_moves(moves_text)

        fen = metadata.get("fen", "<starting_fen>")
        turn = "white" if len(moves) % 2 == 0 else "black"
        castling = "KQkq"
        ep = "-"

        json_entry = create_json_entry(metadata, moves, fen, turn, castling, ep)
        results.append(json_entry)

    return results

def main(input_files: List[str], output_file: str, args):
    """
    Main function to process multiple files and export JSON.
    """
    all_games = []
    for file in input_files:
        print(f"Processing {file}...")
        games = process_file(file, args)
        all_games.extend(games)
        print(f"  → {len(games)} valid games added.")

    if args.jsonl:
        with open(output_file, "w") as f:
            for game in all_games:
                f.write(json.dumps(game) + "\n")
    else:
        with open(output_file, "w") as f:
            json.dump(all_games, f, indent=2)

    print(f"\n✅ Saved {len(all_games)} games to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse multiple PGN files into JSON dataset.")
    parser.add_argument("--input-files", nargs="+", required=True, help="List of input PGN files")
    parser.add_argument("--output-file", required=True, help="Output JSON file")
    parser.add_argument("--skip-unfinished", action="store_true", help="Skip games with Result == '*'")
    parser.add_argument("--only-standard", action="store_true", help="Keep only Variant == 'Standard'")
    parser.add_argument("--only-complete", action="store_true", help="Keep only complete results")
    parser.add_argument("--min-elo", type=int, default=None, help="Skip games if either Elo < min")
    parser.add_argument("--max-elo", type=int, default=None, help="Skip games if either Elo > max")
    parser.add_argument("--jsonl", action="store_true", help="Write output as JSONL (default: JSON array)")

    args = parser.parse_args()

    main(args.input_files, args.output_file, args)
