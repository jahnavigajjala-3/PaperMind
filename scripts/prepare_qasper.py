
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "train.jsonl"

TRAIN_PARQUET_URL = (
    "https://huggingface.co/datasets/allenai/qasper/"
    "resolve/refs%2Fconvert%2Fparquet/"
    "qasper/train/0000.parquet"
)

SYSTEM_PROMPT = (
    "Answer questions about research papers using only the supplied context. "
    "Do not invent facts. If the context is insufficient, say so."
)


def build_context(paper, evidence):
    """Build a question-specific context using the paper and its evidence."""
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()

    evidence_text = "\n\n".join(evidence)

    parts = []

    if title:
        parts.append(f"Paper title: {title}")

    if abstract:
        parts.append(f"Abstract: {abstract}")

    if evidence_text:
        parts.append(f"Relevant evidence from the paper:\n{evidence_text}")

    return "\n\n".join(parts).strip()


def select_answer(annotations):
    """Select an answer and its evidence from the same annotation."""
    for annotation in annotations:
        if annotation.get("unanswerable", False):
            continue

        answer = (annotation.get("free_form_answer") or "").strip()

        evidence = annotation.get("evidence") or []

        # Use highlighted evidence if the main evidence field is empty.
        if not evidence:
            evidence = annotation.get("highlighted_evidence") or []

        evidence = list(
            dict.fromkeys(
                passage.strip()
                for passage in evidence
                if isinstance(passage, str) and passage.strip()
            )
        )

        if answer and evidence:
            return answer, evidence

    return None, []


def main():
    # Load only the training split; validation and test stay separate.
    dataset = load_dataset(
        "parquet",
        data_files={"train": TRAIN_PARQUET_URL},
    )

    train = dataset["train"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for paper in train:
            questions = paper["qas"]["question"]
            all_answers = paper["qas"]["answers"]

            for question, answer_group in zip(questions, all_answers):
                question = (question or "").strip()

                answer, evidence = select_answer(
                    answer_group.get("answer", [])
                )

                if not question or not answer or not evidence:
                    skipped += 1
                    continue

                context = build_context(paper, evidence)

                if not context:
                    skipped += 1
                    continue

                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
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

                file.write(
                    json.dumps(example, ensure_ascii=False) + "\n"
                )
                written += 1

    print(f"Training examples written: {written}")
    print(f"Questions skipped: {skipped}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
