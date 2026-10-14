
import json
import re
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


def normalize_text(text):
    """Normalize whitespace for conservative evidence matching."""
    return re.sub(r"\s+", " ", text or "").strip()


def extract_paragraphs(paper):
    """Return full-text paragraphs paired with their section headings."""
    full_text = paper.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraph_groups = full_text.get("paragraphs") or []

    paragraphs = []

    for index, group in enumerate(paragraph_groups):
        section = (
            section_names[index]
            if index < len(section_names)
            else "Unknown section"
        )

        if isinstance(group, str):
            group = [group]

        for paragraph in group or []:
            if isinstance(paragraph, str) and paragraph.strip():
                paragraphs.append((section or "Unknown section", paragraph))

    return paragraphs


def find_evidence_paragraphs(evidence, paragraphs):
    """Match evidence to full-text paragraphs using normalized substrings."""
    matches = []
    seen = set()

    for passage in evidence:
        normalized_evidence = normalize_text(passage)

        if not normalized_evidence:
            continue

        for section, paragraph in paragraphs:
            normalized_paragraph = normalize_text(paragraph)

            if normalized_evidence in normalized_paragraph:
                key = (section, normalized_paragraph)

                if key not in seen:
                    matches.append((section, paragraph))
                    seen.add(key)

                # One matching paragraph is sufficient for this passage.
                break

    return matches


def build_context(paper, matched_paragraphs):
    """Build context from metadata and matched full-text paragraphs."""
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()

    parts = []

    if title:
        parts.append(f"Paper title: {title}")

    if abstract:
        parts.append(f"Abstract: {abstract}")

    evidence_blocks = [
        f"Section: {section}\n{paragraph}"
        for section, paragraph in matched_paragraphs
    ]

    if evidence_blocks:
        parts.append(
            "Relevant passages from the paper:\n"
            + "\n\n".join(evidence_blocks)
        )

    return "\n\n".join(parts).strip()


def select_answer(annotations):
    """Return an answer and evidence from the same annotation."""
    for annotation in annotations:
        if annotation.get("unanswerable", False):
            continue

        answer = (annotation.get("free_form_answer") or "").strip()
        evidence = annotation.get("evidence") or []

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
            paragraphs = extract_paragraphs(paper)
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

                matched_paragraphs = find_evidence_paragraphs(
                    evidence, paragraphs
                )

                # Do not train on examples with unmatched evidence.
                if not matched_paragraphs:
                    skipped += 1
                    continue

                context = build_context(paper, matched_paragraphs)

                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"{context}\n\nQuestion: {question}",
                        },
                        {"role": "assistant", "content": answer},
                    ]
                }

                file.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

    print(f"Training examples written: {written}")
    print(f"Questions skipped: {skipped}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
