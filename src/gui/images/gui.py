#!/usr/bin/env python3
"""Minimal viable example of a chess GUI using PyQt5 and python-chess.

To run this example:

./gui.py
"""
import sys
import chess
import chess.engine
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtSvg import QSvgRenderer  # for rendering SVGs

from typing import Any, Dict, List, Optional
from PyQt5.QtGui import QPaintEvent, QMouseEvent, QCloseEvent

import math
from math import atan2, cos, sin, pi

# TODO:
# - When clicking new game, reset arrows
# - Use LLM to play a game, take inputs as UCI
# - Board state: FEN
# - start game from some state
# - take UCI input to make a move
# - take FEN stae input to start from some state
#
# - Inputs: user moves (model replace user), board state, stockfish (internal engine) moves
# - Outputs: latest board state (after both user and stockfish moves)
#
# - add option to change boards
# - add option to change to black/white
# - add window to show moves played
# - add option to show explainer for each move

# directory containing your SVGs
SVG_DIR: str = "pieces-basic-svg"  # directory containing piece SVGs
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

    def __init__(self) -> None:
        """Set up the main window, create the engine, board widget, and UI controls."""
        super().__init__()
        self.setWindowTitle("Stockfish vs You")
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")

        central = QWidget()
        layout = QVBoxLayout(central)
        self.board_widget = ChessBoardWidget(self.board, parent=self)
        layout.addWidget(self.board_widget)

        new_btn = QPushButton("New Game")
        new_btn.clicked.connect(self.new_game)
        layout.addWidget(new_btn)

        self.setCentralWidget(central)

    def new_game(self) -> None:
        """Reset the board to the initial position."""
        self.board.reset()
        self.board_widget.update()

    def make_user_move(self, move: chess.Move) -> None:
        """Execute the player's move, update the board, display arrows, and get the engine's response.

        Args:
            move: The player's chess.Move object.
        """
        # player's move
        self.board.push(move)
        # show arrow for player's move
        self.board_widget.set_arrows([move])
        # engine's reply
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        self.board.push(result.move)
        # show arrows for both moves
        self.board_widget.set_arrows([move, result.move])

    def closeEvent(self, event: QCloseEvent) -> None:
        """Quit the engine and handle window close event cleanup.

        Args:
            event: The close event.
        """
        self.engine.quit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
