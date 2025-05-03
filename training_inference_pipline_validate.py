'''
Step 1: Load base model → save Hugging Face format
'''
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    cache_dir="pretrained_cache"
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    cache_dir="pretrained_cache"
)

model.save_pretrained("./base_model/model")
tokenizer.save_pretrained("./base_model/tokenizer")

'''
Step 2: Create & train LoRA adapter → save Hugging Face PEFT format
'''
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# (OPTIONAL TRAINING HERE)
# from transformers import Trainer, TrainingArguments
# trainer = Trainer(model=model, args=..., train_dataset=...)
# trainer.train()

# ✅ save LoRA adapter
model.save_pretrained("./lora_adapter")

''''
Step 3: Convert base model → GGUF (locally)
Clone the llama.cpp repo:
$ cd llama.cpp
$ python3 convert_hf_to_gguf.py --model-dir ./base_model --outfile ./base_model.gguf
'''

'''
Step 4; Convert LoRA adapter → GGUF (web app)
As of today → only supported by Hugging Face Space:
👉 https://huggingface.co/spaces/ggml-org/gguf-my-lora
'''

'''
Step 5: Qantization Phase
In llama.cpp repo:
./quantize base_model.gguf base_model.q4_k_m.gguf q4_K_M
'''