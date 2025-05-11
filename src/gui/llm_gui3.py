#!/usr/bin/env python3
"""Minimal viable example of a chess GUI using PyQt5 and python-chess.

To run this example:

./gui.py
"""
import sys
import chess
import chess.engine
from llama_cpp import Llama  # pip install llama-cpp-python
import random
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTextEdit,
)
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtSvg import QSvgRenderer  # for rendering SVGs

from typing import Any, Dict, List, Optional
from PyQt5.QtGui import QPaintEvent, QMouseEvent, QCloseEvent
from math import atan2, cos, sin, pi

# NOTE:
# To import the module, use the following code below (for some example called run_chess_gui.py):
# # run_chess_gui.py
#
# import sys
# from PyQt5.QtWidgets import QApplication
# from llm_gui3 import MainWindow  # adjust the module path as needed
#
# def main():
#     # 1. Create the Qt application
#     app = QApplication(sys.argv)
#
#     # 2. Path to your quantized LLM model
#     local_LLM = "/path/to/your/mistral-7b-q4_0.gguf"
#
#     # 3. Instantiate and show the main window
#     window = MainWindow(local_LLM=local_LLM)
#     window.show()
#
#     # 4. Enter the Qt main loop
#     sys.exit(app.exec_())
#
# if __name__ == "__main__":
#     main()

# TODO:
# - Count number of legal/illegal moves
#
# - Inputs: user moves (model replace user), board state, stockfish (internal engine) moves
# - Outputs: latest board state (after both user and stockfish moves)
#
# - WARNING: Model training on synthetic may lead to model collapse

# directory containing your SVGs
SVG_DIR: str = "images/pieces-basic-svg"  # directory containing piece SVGs
SQUARE_SIZE: int = 60  # size of each square in pixels

# map from chess piece letter to file-base name
PIECE_NAME: Dict[str, str] = {
    "p": "pawn",
    "r": "rook",
    "n": "knight",
    "b": "bishop",
    "q": "queen",
    "k": "king",
}

# this will hold QPixmaps keyed by piece.symbol()
PIECE_IMAGES: Dict[str, str] = {}


class ChessBoardWidget(QWidget):
    """A widget that renders the chess board and pieces, handles user clicks, and draws move arrows.

    Attributes:
        board: The chess.Board instance representing the game state.
        selected_square: The currently selected square index or None.
        arrows: List of chess.Move objects to draw arrows for.
    """

    board: chess.Board
    selected_square: Optional[int]
    arrows: List[chess.Move]

    def __init__(self, board: chess.Board, *args: Any, **kwargs: Any) -> None:
        """Initialize the chess board widget, load piece images, and set up initial state.

        Args:
            board: The chess.Board instance to display.
        """
        super().__init__(*args, **kwargs)
        self.board = board
        self.selected_square = None

        # list of moves to draw as arrows
        self.arrows = []

        # Load and render each SVG once
        for sym in ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]:
            piece_type = sym.lower()
            color = "w" if sym.isupper() else "b"
            fname = f"{SVG_DIR}/{PIECE_NAME[piece_type]}-{color}.svg"

            # render SVG into a pixmap
            renderer = QSvgRenderer(fname)
            pixmap = QPixmap(SQUARE_SIZE, SQUARE_SIZE)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()

            PIECE_IMAGES[sym] = pixmap

        self.setFixedSize(8 * SQUARE_SIZE, 8 * SQUARE_SIZE)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Render the chessboard, pieces, selection highlight, and move arrows.

        Args:
            event: The paint event triggering the redraw.
        """
        qp = QPainter(self)
        colors = [Qt.white, Qt.gray]
        for rank in range(8):
            for file in range(8):
                qp.fillRect(
                    file * SQUARE_SIZE,
                    rank * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                    colors[(rank + file) % 2],
                )
                piece = self.board.piece_at(chess.square(file, 7 - rank))
                if piece:
                    qp.drawPixmap(
                        file * SQUARE_SIZE,
                        rank * SQUARE_SIZE,
                        PIECE_IMAGES[piece.symbol()],
                    )
                # draw highlight border on selected square
                if self.selected_square is not None:
                    sel_file = chess.square_file(self.selected_square)
                    sel_rank = 7 - chess.square_rank(self.selected_square)
                    if file == sel_file and rank == sel_rank:
                        pen = QPen(QColor(255, 0, 0), 3)  # red border, width=3
                        qp.setPen(pen)
                        qp.setBrush(Qt.NoBrush)
                        qp.drawRect(
                            file * SQUARE_SIZE,
                            rank * SQUARE_SIZE,
                            SQUARE_SIZE,
                            SQUARE_SIZE,
                        )
                        qp.setPen(Qt.NoPen)

        # draw arrows for each move
        pen = QPen(QColor(255, 0, 0), 2)  # red arrows
        qp.setPen(pen)
        for mv in self.arrows:
            # compute coordinates
            fx = chess.square_file(mv.from_square) * SQUARE_SIZE + SQUARE_SIZE / 2
            fy = (7 - chess.square_rank(mv.from_square)) * SQUARE_SIZE + SQUARE_SIZE / 2
            tx = chess.square_file(mv.to_square) * SQUARE_SIZE + SQUARE_SIZE / 2
            ty = (7 - chess.square_rank(mv.to_square)) * SQUARE_SIZE + SQUARE_SIZE / 2
            # draw main line
            qp.drawLine(int(fx), int(fy), int(tx), int(ty))
            # draw arrowhead
            angle = atan2(ty - fy, tx - fx)
            arrow_size = 10
            # two lines for head
            x1 = tx - arrow_size * cos(angle - pi / 6)
            y1 = ty - arrow_size * sin(angle - pi / 6)
            x2 = tx - arrow_size * cos(angle + pi / 6)
            y2 = ty - arrow_size * sin(angle + pi / 6)
            qp.drawLine(int(tx), int(ty), int(x1), int(y1))
            qp.drawLine(int(tx), int(ty), int(x2), int(y2))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse clicks to select and move pieces.

        Args:
            event: The mouse event containing click coordinates.
        """
        file = event.x() // SQUARE_SIZE
        rank = 7 - (event.y() // SQUARE_SIZE)
        sq = chess.square(file, rank)
        if self.selected_square is None:
            if self.board.piece_at(sq):
                self.selected_square = sq
                self.update()
        else:
            move = chess.Move(self.selected_square, sq)
            if move in self.board.legal_moves:
                self.window().make_user_move(move)
            self.selected_square = None
            self.update()

    def set_arrows(self, moves: List[chess.Move]) -> None:
        """Update the list of move arrows to draw and request a repaint.

        Args:
            moves: List of chess.Move objects to visualize as arrows.
        """
        self.arrows = moves
        self.update()


class MainWindow(QMainWindow):
    """Main application window integrating the chess board widget and engine controls."""

    def __init__(self, local_LLM) -> None:
        """Set up the main window, create the engine, board widget, and UI controls."""
        super().__init__()
        self.setWindowTitle("Stockfish vs You")
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")

        # # initialize local quantized LLM
        # self.llm = Llama(
        #     model_path="/Users/adebayobraimah/Desktop/projects/2025Spring_AI_Project/src/models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
        #     n_threads=8,
        # )

        # initialize local quantized LLM
        self.llm = Llama(
            model_path=local_LLM,
            n_threads=8,
        )

        central = QWidget()
        grid = QGridLayout(central)
        # ensure board column expands and move list has fixed width
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        # Rank labels panel (left side)
        rank_panel = QWidget()
        rank_layout = QVBoxLayout(rank_panel)
        for lbl in "87654321":
            w = QLabel(lbl)
            w.setAlignment(Qt.AlignCenter)
            rank_layout.addWidget(w)
        grid.addWidget(rank_panel, 0, 0)

        # chess board
        self.board_widget = ChessBoardWidget(self.board, parent=self)
        grid.addWidget(self.board_widget, 0, 1)

        # File labels panel (bottom)
        file_panel = QWidget()
        file_layout = QHBoxLayout(file_panel)
        for lbl in "abcdefgh":
            w = QLabel(lbl)
            w.setAlignment(Qt.AlignCenter)
            file_layout.addWidget(w)
        grid.addWidget(file_panel, 1, 1)

        # New Game button below board (row 3)
        new_btn = QPushButton("New Game")
        new_btn.clicked.connect(self.new_game)

        # Play button: LLM vs Stockfish (row 2)
        play_btn = QPushButton("Play LLM vs Stockfish")
        # play_btn = QPushButton("LLM vs Stockfish")
        play_btn.clicked.connect(self.play_llm_vs_stockfish)

        # Make buttons equal width
        play_btn.setFixedWidth(180)
        new_btn.setFixedWidth(180)

        # Create a panel to center both buttons under the board
        button_panel = QWidget()
        btn_layout = QHBoxLayout(button_panel)
        btn_layout.addWidget(play_btn)
        btn_layout.addWidget(new_btn)
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignCenter)
        grid.addWidget(button_panel, 2, 1)

        # Move log panel on the right
        self.move_list = QTextEdit()
        self.move_list.setReadOnly(True)
        self.move_list.setFixedWidth(200)
        grid.addWidget(self.move_list, 0, 2, 3, 1)  # span rows 0–2 in column 2

        self.setCentralWidget(central)

    def new_game(self) -> None:
        """Reset the board to the initial position."""
        self.move_list.clear()
        self.board.reset()
        self.board_widget.set_arrows([])  # clear any existing arrows
        self.board_widget.update()

    def make_user_move(self, move: chess.Move) -> None:
        """Execute the player's move, update the board, display arrows, and get the engine's response.

        Args:
            move: The player's chess.Move object.
        """
        from PyQt5.QtWidgets import QApplication

        # Before pushing, get piece and squares for descriptive move
        piece = self.board.piece_at(move.from_square)
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        self.move_list.append(
            f"Player (White): Moved {chess.piece_name(piece.piece_type)} from {from_sq} to {to_sq}."
        )
        # Special move detection for player
        if self.board.is_castling(move):
            self.move_list.append("Player (White): Castled.")
        elif move.promotion:
            self.move_list.append(
                f"Player (White): Pawn promoted to {chess.piece_name(move.promotion)} at {to_sq}."
            )
        elif self.board.is_en_passant(move):
            self.move_list.append("Player (White): En passant capture.")
        QApplication.processEvents()
        # player's move
        self.board.push(move)
        # show arrow for player's move
        self.board_widget.set_arrows([move])
        # Explanation after move (FEN post-move)
        explanation = self.explain_move(move, "Player (White)")
        self.move_list.append(f"Explanation (White): {explanation}")
        QApplication.processEvents()
        # engine's reply
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        eng = result.move
        piece_e = self.board.piece_at(eng.from_square)
        from_sq_e = chess.square_name(eng.from_square)
        to_sq_e = chess.square_name(eng.to_square)
        self.board.push(eng)
        self.move_list.append(
            f"Stockfish (Black): Moved {chess.piece_name(piece_e.piece_type)} from {from_sq_e} to {to_sq_e}."
        )
        # Explanation after engine move
        explanation = self.explain_move(eng, "Stockfish (Black)")
        self.move_list.append(f"Explanation (Black): {explanation}")
        QApplication.processEvents()
        # Special move detection for engine
        if self.board.is_castling(eng):
            self.move_list.append("Stockfish (Black): Castled.")
        elif eng.promotion:
            self.move_list.append(
                f"Stockfish (Black): Pawn promoted to {chess.piece_name(eng.promotion)} at {to_sq_e}."
            )
        elif self.board.is_en_passant(eng):
            self.move_list.append("Stockfish (Black): En passant capture.")
        # Check/checkmate detection
        if self.board.is_checkmate():
            self.move_list.append("Checkmate! Game over.")
        elif self.board.is_check():
            self.move_list.append("Check!")
        QApplication.processEvents()
        # show arrows for both moves
        self.board_widget.set_arrows([move, eng])

    def make_llm_move(self) -> chess.Move:
        """Ask the LLM for its next move in UCI format."""
        fen = self.board.fen()
        prompt = (
            "You are a chess engine. "
            f"Position (FEN): {fen}\n"
            "Respond with your next move in UCI format (e.g. e2e4), and nothing else."
        )
        resp = self.llm(prompt=prompt, max_tokens=8, stop=["\n"])
        move_str = resp["choices"][0]["text"].strip()
        try:
            mv = chess.Move.from_uci(move_str)
            if mv in self.board.legal_moves:
                return mv
        except:
            pass
        # fallback random move
        # TODO: inference again from LLM
        return random.choice(list(self.board.legal_moves))

    def explain_move(self, move: chess.Move, player: str) -> str:
        """Use the LLM to explain why a given move is a good choice."""
        fen = self.board.fen()
        prompt = (
            f"You are a chess coach. Given the position FEN: {fen} "
            f"and the move {move.uci()} by {player}, explain in concise terms why this move is good."
        )
        resp = self.llm(prompt=prompt, max_tokens=64, stop=["\n"])
        return resp["choices"][0]["text"].strip()

    def play_llm_vs_stockfish(self) -> None:
        """Alternate moves using a QTimer to keep UI responsive."""
        # initialize game
        self.move_list.clear()
        self.board.reset()
        self.board_widget.set_arrows([])
        self.board_widget.update()
        # start the iterative move loop
        QTimer.singleShot(0, self._play_next)

    def _play_next(self) -> None:
        """Play one LLM and one Stockfish move, then schedule next."""
        if self.board.is_game_over():
            print("Game over:", self.board.result())
            return
        # LLM move
        llm_mv = self.make_llm_move()
        piece = self.board.piece_at(llm_mv.from_square)
        from_sq = chess.square_name(llm_mv.from_square)
        to_sq = chess.square_name(llm_mv.to_square)
        # Special move detection for LLM
        if self.board.is_castling(llm_mv):
            self.move_list.append("LLM (White): Castled.")
        elif llm_mv.promotion:
            self.move_list.append(
                f"LLM (White): Pawn promoted to {chess.piece_name(llm_mv.promotion)} at {to_sq}."
            )
        elif self.board.is_en_passant(llm_mv):
            self.move_list.append("LLM (White): En passant capture.")
        self.board.push(llm_mv)
        self.move_list.append(
            f"LLM (White): Moved {chess.piece_name(piece.piece_type)} from {from_sq} to {to_sq}"
        )
        # Explanation after LLM move
        explanation = self.explain_move(llm_mv, "LLM (White)")
        self.move_list.append(f"Explanation (White): {explanation}")
        self.board_widget.set_arrows([llm_mv])
        # Stockfish move
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        eng_mv = result.move
        piece_e = self.board.piece_at(eng_mv.from_square)
        from_e = chess.square_name(eng_mv.from_square)
        to_e = chess.square_name(eng_mv.to_square)
        # Special move detection for engine
        if self.board.is_castling(eng_mv):
            self.move_list.append("Stockfish (Black): Castled.")
        elif eng_mv.promotion:
            self.move_list.append(
                f"Stockfish (Black): Pawn promoted to {chess.piece_name(eng_mv.promotion)} at {to_e}."
            )
        elif self.board.is_en_passant(eng_mv):
            self.move_list.append("Stockfish (Black): En passant capture.")
        self.board.push(eng_mv)
        self.move_list.append(
            f"Stockfish (Black): Moved {chess.piece_name(piece_e.piece_type)} from {from_e} to {to_e}"
        )
        # Explanation after Stockfish move
        explanation = self.explain_move(eng_mv, "Stockfish (Black)")
        self.move_list.append(f"Explanation (Black): {explanation}")
        self.board_widget.set_arrows([llm_mv, eng_mv])
        # Check/checkmate detection
        if self.board.is_checkmate():
            self.move_list.append("Checkmate! Game over.")
        elif self.board.is_check():
            self.move_list.append("Check!")
        # schedule next pair after a short delay
        QTimer.singleShot(200, self._play_next)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Quit the engine and handle window close event cleanup.

        Args:
            event: The close event.
        """
        self.engine.quit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Path to your local LLM model
    local_LLM: str = (
        "/Users/adebayobraimah/Desktop/projects/2025Spring_AI_Project/src/models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
    )
    mw = MainWindow(local_LLM=local_LLM)
    mw.show()
    sys.exit(app.exec_())
