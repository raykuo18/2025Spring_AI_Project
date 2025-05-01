from llama_cpp import Llama

# llm = Llama(model_path="llm-models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf", n_ctx=512, n_threads=4, use_mlock=True)
# llm = Llama(model_path="llm-models/openchat_3.5.Q4_K_M.gguf", n_ctx=512, n_threads=4, use_mlock=True)

# print(llm("What is a fork in chess?", max_tokens=50))

llm = Llama(
    model_path="llm-models/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=20,  # try 20 for 7B Q4_K_M model
    use_mlock=True,   # keep model in memory
    use_mmap=True,     # use memory-mapped access
    verbose=False
)

print(llm("What is a fork in chess?", max_tokens=50)['choices'][0]['text'])

llm = Llama(
    model_path="llm-models/openchat_3.5.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=20,  # try 20 for 7B Q4_K_M model
    use_mlock=True,   # keep model in memory
    use_mmap=True,     # use memory-mapped access
    verbose=False
)

print(llm("What is a fork in chess?", max_tokens=50)['choices'][0]['text'])
