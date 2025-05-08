import chess.engine

# Replace this with the path to your Stockfish binary
stockfish_path = "../stockfish-11-linux/Linux/stockfish_20011801_x64"

def test_stockfish(stockfish_path):
    try:
        # Initialize the engine
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Create a new board
            board = chess.Board()
            print("Initial position:\n", board)
            
            # Ask Stockfish for the best move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            
            print(f"Stockfish suggests the move: {result.move}")
            
            return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Run the test
if test_stockfish(stockfish_path):
    print("Stockfish is working!")
else:
    print("Stockfish test failed.")
