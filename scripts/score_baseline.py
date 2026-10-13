
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "outputs" / "baseline.json"
REFERENCE_PATH = ROOT / "data" / "evaluation" / "reference_answers.json"
RUBRIC_PATH = ROOT / "data" / "evaluation" / "rubric.json"
OUTPUT_PATH = ROOT / "outputs" / "baseline_scores.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results, model):
    total = sum(item["score"] for item in results.values())
    maximum = 2 * len(results)

    output = {
        "model": model,
        "evaluation_type": "human_review",
        "number_of_questions": len(results),
        "total_score": total,
        "max_score": maximum,
        "percentage": round(total / maximum * 100, 2) if maximum else 0,
        "results": list(results.values()),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    baseline = load_json(BASELINE_PATH)
    references = load_json(REFERENCE_PATH)["reference_answers"]
    criteria = load_json(RUBRIC_PATH)["criteria"]

    results = {}

    # Recover previously saved scores, if available.
    if OUTPUT_PATH.exists():
        previous = load_json(OUTPUT_PATH)
        results = {
            item["test"]: item
            for item in previous.get("results", [])
        }
        print(f"Resuming: {len(results)} question(s) already scored.")

    for record in baseline:
        test_id = record["test"]

        if test_id in results:
            print(f"Skipping {test_id}: already scored.")
            continue

        if test_id not in references or test_id not in criteria:
            raise ValueError(f"Missing reference or rubric for {test_id}")

        print("\n" + "=" * 80)
        print(f"TEST: {test_id}")
        print(f"MODEL: {record['model']}")
        print("=" * 80)
        print("\nMODEL ANSWER:\n")
        print(record["answer"])

        print("\nEXPECTED POINTS:")
        for point in references[test_id]["expected_points"]:
            print(f"- {point}")

        print("\nRUBRIC CHECKS:")
        for check in criteria[test_id]["checks"]:
            print(f"- {check}")

        while True:
            try:
                score = int(input("\nScore this answer (0, 1, or 2): "))
                if score in (0, 1, 2):
                    break
                print("Enter 0, 1, or 2.")
            except ValueError:
                print("Enter a valid integer.")

        justification = input("Why did you assign this score? ").strip()
        while not justification:
            print("A short justification is required.")
            justification = input("Why did you assign this score? ").strip()

        results[test_id] = {
            "test": test_id,
            "model": record["model"],
            "score": score,
            "max_score": 2,
            "justification": justification,
            "generated_tokens": record.get("generated_tokens"),
            "eos_generated": record.get("eos_generated"),
        }

        # Save immediately so Ctrl+C won't lose completed questions.
        save_results(results, record["model"])
        print(f"\nSaved progress: {len(results)}/{len(baseline)} questions.")

    save_results(results, baseline[0]["model"] if baseline else None)
    print("\nEvaluation complete.")
    print(f"Saved to: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
