
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "train.jsonl"

PARQUET_FILES = {
    "train": (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/"
        "qasper/train/0000.parquet"
    ),
    "validation": (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/"
        "qasper/validation/0000.parquet"
    ),
    "test": (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/"
        "qasper/test/0000.parquet"
    ),
}


def build_context(paper):
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""

    return f"Paper title: {title}\n\nAbstract: {abstract}".strip()


def select_answer(answers):
    for annotation in answers:
        if annotation.get("unanswerable", False):
            continue

        answer = (annotation.get("free_form_answer") or "").strip()
        if answer:
            return answer

    return None


def main():
    dataset = load_dataset("parquet", data_files=PARQUET_FILES)
    train = dataset["train"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for paper in train:
            context = build_context(paper)
            if not context:
                continue

            questions = paper["qas"]["question"]
            all_answers = paper["qas"]["answers"]

            for question, answer_group in zip(questions, all_answers):
                question = (question or "").strip()
                answer = select_answer(answer_group.get("answer", []))

                if not question or not answer:
                    skipped += 1
                    continue

                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer questions about research papers "
                                "using only the supplied context. Do not "
                                "invent facts. If the context is insufficient, "
                                "say so."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"{context}\n\nQuestion: {question}"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    ]
                }

                file.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

    print(f"Training examples written: {written}")
    print(f"Questions skipped: {skipped}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
