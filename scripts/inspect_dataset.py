
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path("data/examples/format_example.jsonl")

# 1. Load our curated example.
with DATA_PATH.open("r", encoding="utf-8") as file:
    example = json.loads(file.readline())

messages = example["messages"]

# 2. Load the tokenizer belonging to our base model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Apply Qwen's native chat template.
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

# 4. Convert the formatted conversation to token IDs.
encoded = tokenizer(
    formatted_text,
    add_special_tokens=False,
)

input_ids = encoded["input_ids"]

print("\n--- FORMATTED CHAT ---")
print(formatted_text)

print("\n--- TOKEN INFORMATION ---")
print("Number of tokens:", len(input_ids))
print("First 30 token IDs:", input_ids[:30])

print("\n--- TOKEN-BY-TOKEN INSPECTION ---")
for token_id in input_ids[:40]:
    token = tokenizer.decode([token_id])
    print(f"{token_id:>7} -> {token!r}")

print("\n--- MESSAGE BOUNDARIES ---")
for special_token in [
    "<|im_start|>",
    "<|im_sep|>",
    "<|im_end|>",
]:
    token_id = tokenizer.convert_tokens_to_ids(special_token)
    print(f"{special_token}: {token_id}")
