import csv
from pathlib import Path


DATASET_ROOT = Path("wikipedia-biography-dataset")
SPLITS = ("train", "test", "valid")


def load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_rows(split_dir: Path, split: str) -> list[dict[str, str]]:
    box_lines = load_lines(split_dir / f"{split}.box")
    title_lines = load_lines(split_dir / f"{split}.title")
    nb_lines = load_lines(split_dir / f"{split}.nb")
    sent_lines = load_lines(split_dir / f"{split}.sent")

    if not (len(box_lines) == len(title_lines) == len(nb_lines)):
        raise ValueError(
            f"Mismatched line counts in {split}: "
            f"box={len(box_lines)}, title={len(title_lines)}, nb={len(nb_lines)}"
        )

    rows: list[dict[str, str]] = []
    sent_pointer = 0

    for idx, (box_text, title, nb_text) in enumerate(zip(box_lines, title_lines, nb_lines), start=1):
        try:
            nb = int(nb_text.strip())
        except ValueError as err:
            raise ValueError(f"Invalid nb value at {split} record {idx}: {nb_text!r}") from err

        record_sentences = sent_lines[sent_pointer : sent_pointer + nb]
        if len(record_sentences) != nb:
            raise ValueError(
                f"Not enough sentences for {split} record {idx}: expected {nb}, got {len(record_sentences)}"
            )
        sent_pointer += nb

        rows.append(
            {
                "box": box_text,
                "sent": " ".join(s.strip() for s in record_sentences if s.strip()),
                "nb": str(nb),
                "title": title,
            }
        )

    return rows


def write_split_csv(split: str) -> None:
    split_dir = DATASET_ROOT / split
    output_path = split_dir / f"{split}_stacked.csv"

    rows = build_rows(split_dir, split)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["box", "sent", "nb", "title"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {output_path}")


def main() -> None:
    for split in SPLITS:
        write_split_csv(split)


if __name__ == "__main__":
    main()
