
from datasets import load_dataset
from pprint import pprint


def main():
    dataset = load_dataset(
        "parquet",
        data_files={
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
        },
    )

    paper = dataset["train"][0]

    print("TITLE:")
    print(paper["title"])

    print("\nABSTRACT:")
    print(paper["abstract"])

    print("\nFULL TEXT SECTION COUNT:")
    print(len(paper["full_text"]["paragraphs"]))

    print("\nFIRST QUESTION:")
    print(paper["qas"]["question"][0])

    print("\nANSWERS FOR FIRST QUESTION:")
    pprint(paper["qas"]["answers"][0])

    print("\nTOTAL TRAIN RECORDS:", len(dataset["train"]))


if __name__ == "__main__":
    main()
