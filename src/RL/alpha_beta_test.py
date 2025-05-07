#!/usr/bin/env python3
"""
alpha_beta_rl.py
================
RL *meta-controller* that learns to set α (tactical weight) and β (strategic
weight) in order to maximise a composite reward on chess-tutor outputs.

❶  Environment  – Two-dim continuous action: (α, β) in [0, 10]²
❷  Reward       – R = w_m * f_move  +  w_e * f_expl
❸  Agent        – PPO with a small MLP policy

Run:

    python alpha_beta_rl.py train   --steps  20000
    python alpha_beta_rl.py infer   --fen   "<FEN>"

Dependencies
------------
pip install stable-baselines3[extra] python-chess stockfish transformers \
            peft bitsandbytes accelerate
"""
from __future__ import annotations

import argparse, random, os, math
import numpy as np
import chess, chess.engine

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# --------------------------------------------------------------------------- #
#  0.  Import your α–β generator (from quick_test.py or chess_tutor.py)
# --------------------------------------------------------------------------- #
from quick_test import get_device, load_base, add_blank_lora, gen as blended_generate  #

DEVICE = get_device()
STOCKFISH_PATH = "stockfish"  # adjust if needed
CP_TO_SCORE = lambda cp: 1 / (1 + math.exp(cp / 60))  # sigmoid-ish


# --------------------------------------------------------------------------- #
#  1.  Helper: move quality using Stockfish
# --------------------------------------------------------------------------- #
def centipawn_loss(fen: str, tutor_move: str, depth: int = 12) -> float:
    """
    +ve if the tutor move is worse than SF's top choice, else small/zero.
    """
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as sf:
        info_best = sf.analyse(board, chess.engine.Limit(depth=depth))
        best_score = info_best["score"].white().score(mate_score=10000)

        board.push(chess.Move.from_uci(tutor_move))
        info_play = sf.analyse(board, chess.engine.Limit(depth=depth))
        play_score = info_play["score"].white().score(mate_score=10000)

    if best_score is None or play_score is None:  # handle mates
        return 1000
    return best_score - play_score  # cp Δ  (≥0)


# --------------------------------------------------------------------------- #
#  2.  Helper: explanation quality  (stub – replace with LLM evaluator)
# --------------------------------------------------------------------------- #
def explanation_quality(prompt: str, explanation: str) -> float:
    """
    Returns a floating reward in [0, 1].  You should replace this with
    GCC-Eval style LLM grading or human labels.  Here we reward brevity +
    presence of chess terms as a cheap proxy.
    """
    keywords = ["file", "rank", "fork", "pin", "king", "threat", "plan"]
    k_hits = sum(kw in explanation.lower() for kw in keywords)
    brevity = max(0, 1 - len(explanation) / 200)  # shorter is better
    return 0.4 * (k_hits / len(keywords)) + 0.6 * brevity


# --------------------------------------------------------------------------- #
#  3.  RL Environment
# --------------------------------------------------------------------------- #
class AlphaBetaEnv(gym.Env):
    """
    Observation: none (we treat tasks i.i.d.), so we expose a dummy 1-D obs.
    Action:      Box(2) – raw α and β ∈ [0, 10]
    Reward:      weighted average of move + explanation metrics
    Episode:     one step.
    """

    def __init__(self, positions: list[str], w_move=0.6, w_expl=0.4):
        super().__init__()
        self.positions = positions
        self.w_move = w_move
        self.w_expl = w_expl
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Load LLM + two random adapters once (inference-only)
        self.model, self.tok = load_base()
        self.tac = add_blank_lora(self.model, "tac")
        self.str = add_blank_lora(self.model, "str")

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self.fen = random.choice(self.positions)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        alpha, beta = action
        prompt = self.fen
        text = blended_generate(
            prompt,
            self.model,
            self.tok,
            self.tac,
            self.str,
            alpha,
            beta,
            max_new_tokens=64,
        )
        # naive parsing: first UCI that appears
        move = None
        board = chess.Board(self.fen)
        for token in text.split():
            token = token.lower()
            try:
                candidate = token[:4]
                move_obj = chess.Move.from_uci(candidate)
                if move_obj in board.legal_moves:
                    move = candidate
                    break
            except:
                continue  # Skip invalid move formats

        if move is None:
            move_loss = 300  # punish illegal or missing move
        else:
            move_loss = centipawn_loss(self.fen, move)

        expl_score = explanation_quality(prompt, text)
        reward = self.w_move * (1 - CP_TO_SCORE(move_loss)) + self.w_expl * expl_score

        terminated = True
        return np.array([0.0], dtype=np.float32), reward, terminated, False, {}


# --------------------------------------------------------------------------- #
#  4.  Main CLI
# --------------------------------------------------------------------------- #
def load_start_positions(n=50) -> list[str]:
    """
    Cheap starting positions – pull random starting moves from lichess database
    or hard-code a few.  Here we use python-chess to sample legal moves
    in the first 4 plies.
    """
    board = chess.Board()
    fens = []
    while len(fens) < n:
        board.reset()
        for _ in range(random.randint(1, 4)):
            board.push(random.choice(list(board.legal_moves)))
        fens.append(board.fen())
    return fens


def main():
    argp = argparse.ArgumentParser()
    sub = argp.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=20000)
    i = sub.add_parser("infer")
    i.add_argument("--fen", required=True)
    args = argp.parse_args()

    if args.cmd == "train":
        env = AlphaBetaEnv(load_start_positions())
        check_env(env, warn=True)
        agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=256,
            batch_size=128,
            learning_rate=3e-4,
            gamma=0.99,
            device=DEVICE,
        )
        agent.learn(total_timesteps=args.steps)
        agent.save("ab_controller.zip")
        print("✅ RL controller saved → ab_controller.zip")

    else:  # inference
        assert os.path.exists("ab_controller.zip"), "train first!"
        env = AlphaBetaEnv(load_start_positions())  # dummy env
        agent = PPO.load("ab_controller.zip", env=env, device=DEVICE)
        # one prediction on the user FEN
        obs, _ = env.reset()
        env.fen = args.fen
        alpha_beta, _ = agent.predict(obs, deterministic=True)
        alpha, beta = map(float, alpha_beta)
        print(f"Controller chose α={alpha:.2f}, β={beta:.2f}")

        # produce explanation
        text = blended_generate(
            args.fen,
            env.model,
            env.tok,
            env.tac,
            env.str,
            alpha,
            beta,
            max_new_tokens=128,
        )
        print("\n---- Tutor output ----\n")
        print(text)


if __name__ == "__main__":
    main()
