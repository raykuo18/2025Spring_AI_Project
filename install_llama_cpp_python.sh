CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir

# Nous Hermes 2 – Mistral 7B Q4_K_M
mkdir -p ~/llm-models/nous-hermes
cd ~/llm-models/nous-hermes

wget --header="Authorization: Bearer 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" \
  https://huggingface.co/TheBloke/Nous-Hermes-2-Mistral-7B-GGUF/resolve/main/nous-hermes-2-mistral-7b.Q4_K_M.gguf
