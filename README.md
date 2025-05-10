# 2025Spring_AI_Project

# Demo
-------

To run the demo, begin by setting up a new python env (follow the instructions in .dev/dev-README.md).

After setting up the environment, install all of the required packages:

```bash
pip install -r requirements.txt

# Note: llama-cpp-python for M1+ MacOS is best if installed from here:
# https://llama-cpp-python.readthedocs.io/en/latest/install/macos/
```

Then type:

```bash
git fetch --all
git checkout gui
git pull origin gui
```

Now download the model used for the demo, by navigating to:

```bash
cd src/models
```

And executing the code from HuggingFace to download a quantized Mixtral model to run locally.

Once downloaded, navigate to:
```bash
cd src/gui
```

And type:
```bash
chmod 755 llm_gui3.py
./llm_gui3.py
```

A screen should pop-up after some text appears on your console. Click 'Play LLM vs Stockfish' to watch the LLM play chess -- otherwise click 'New Game' for the human experience.
