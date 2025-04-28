# file: test_infer.py
from llama_cpp import Llama

llm = Llama(
    model_path="~/llm-models/nous-hermes/nous-hermes-2-mistral-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6  # Tune this depending on your CPU load
)

prompt = """[FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3]
Q: What is the best move for White? Explain your reasoning like a tutor.
A:"""

output = llm(prompt, max_tokens=200, temperature=0.7)
print(output["choices"][0]["text"])
