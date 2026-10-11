
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PAPER_EXCERPT = """
This fictional paper introduces LiteAttn, a method that reduces the
computational cost of attention in long-sequence classification.
The authors evaluate it on three text classification datasets and
compare it with a standard Transformer baseline.

LiteAttn reduces measured inference latency by 25% on the largest
dataset, while its accuracy is 1.2 percentage points lower than the
baseline. Experiments use a single hardware configuration. The paper
does not report results for generative tasks or sequences longer than
the maximum length used in its experiments.
"""

TESTS = [
    {
        "name": "concept_explanation",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain self-attention in a Transformer. "
                    "Describe Q, K, and V and how attention weights "
                    "are used. Be technically precise."
                ),
            }
        ],
    },
    {
        "name": "contribution_extraction",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Based only on this excerpt, identify the paper's "
                    f"main contribution and distinguish it from its "
                    f"reported results.\n\n{PAPER_EXCERPT}"
                ),
            }
        ],
    },
    {
        "name": "methodology",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Explain the evaluation methodology described in "
                    f"this excerpt. State what is known and what is "
                    f"not specified.\n\n{PAPER_EXCERPT}"
                ),
            }
        ],
    },
    {
        "name": "experimental_interpretation",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Interpret the reported latency and accuracy "
                    f"results. What trade-off do they suggest? Do not "
                    f"claim statistical significance.\n\n{PAPER_EXCERPT}"
                ),
            }
        ],
    },
    {
        "name": "limitations",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Identify three limitations supported by this "
                    f"excerpt. Separate explicit limitations from "
                    f"reasonable concerns, and do not invent facts."
                    f"\n\n{PAPER_EXCERPT}"
                ),
            }
        ],
    },
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    results = []

    for test in TESTS:
        prompt = tokenizer.apply_chat_template(
            test["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
            )

        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        result = {
            "test": test["name"],
            "model": MODEL_NAME,
            "answer": answer,
        }
        results.append(result)

        print(f"\n{'=' * 65}")
        print(f"TEST: {test['name']}")
        print(f"{'=' * 65}")
        print(answer)

    output_path = Path("outputs/baseline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved baseline results to {output_path}")


if __name__ == "__main__":
    main()
