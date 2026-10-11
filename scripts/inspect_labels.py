import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path("data/examples/format_example.jsonl")

SYSTEM_PROMPT = (
    "You are PaperMind, an assistant that analyzes research papers "
    "accurately. Do not invent information absent from the source."
)

with DATA_PATH.open("r", encoding="utf-8") as file:
    example = json.loads(file.readline())

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *example["messages"],
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Full conversation, including the assistant's answer.
full_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
)["input_ids"]

# Context only, ending with the assistant generation prefix.
prompt_ids = tokenizer.apply_chat_template(
    messages[:-1],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)["input_ids"]

# Never construct a mask unless the prefix matches exactly.
if full_ids[:len(prompt_ids)] != prompt_ids:
    raise ValueError("Prompt/full token sequences do not align.")

# Ignore context tokens in the loss; train on the answer tokens.
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

supervised_ids = [x for x in labels if x != -100]

assert len(full_ids) == len(labels)
assert len(supervised_ids) > 0

print("\n--- SEQUENCE STATISTICS ---")
print("Total tokens:", len(full_ids))
print("Context tokens:", len(prompt_ids))
print("Supervised tokens:", len(supervised_ids))
print("Labels length:", len(labels))

print("\n--- SUPERVISED TARGET ---")
print(tokenizer.decode(supervised_ids))

print("\n--- LABEL MASK PREVIEW ---")
for i in list(range(min(8, len(full_ids)))) + list(
    range(max(8, len(prompt_ids) - 3), min(len(full_ids), len(prompt_ids) + 8))
):
    token_text = tokenizer.decode([full_ids[i]])
    label_text = "-100 (ignored)" if labels[i] == -100 else repr(token_text)
    print(f"{i:>4} | input={token_text!r:<20} | label={label_text}")

print("\nAssistant-only label construction passed.")
