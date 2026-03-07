import re
import pandas as pd

OPENIE_FILE = "output.txt.oie"
DATASET_FILE = "../cleaned_dataset/train.csv"
OUTPUT_FILE = "./outputs/clean_triples.csv"

CONFIDENCE_THRESHOLD = 0.4

RELATION_MAP = {
    "born": "birth_date",
    "was born": "birth_date",
    "born in": "birth_place",
    "is": "occupation",
    "was": "occupation",
    "played for": "team",
    "plays for": "team",
    "is best known": "known_for",
    "was named": "award",
    "was a member of": "organization",
    "is a member of": "organization"
}

PRONOUNS = {"he", "she", "his", "her"}

df = pd.read_csv(DATASET_FILE)
titles = df["title"].tolist()

triples = []

current_title_index = -1

with open(OPENIE_FILE, "r") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # Detect sentence line (no score)
        if not re.match(r"^\d+\.\d+:", line):
            current_title_index += 1
            continue

        # Extract confidence
        score = float(line.split(":")[0])

        if score < CONFIDENCE_THRESHOLD:
            continue

        # Extract triple
        triple_text = re.search(r"\((.*)\)", line)
        if not triple_text:
            continue

        parts = triple_text.group(1).split(";")

        if len(parts) != 3:
            continue

        subject = parts[0].strip()
        relation = parts[1].strip()
        obj = parts[2].strip()

        if subject.lower() in PRONOUNS:
            subject = titles[current_title_index]

        subject = re.sub(r"\(.*?\)", "", subject).strip()

        subject = titles[current_title_index]

        relation_lower = relation.lower()

        normalized_relation = RELATION_MAP.get(relation_lower, relation_lower)

        obj = obj.split(" who ")[0]
        obj = obj.split(" that ")[0]
        obj = obj.strip()

        triples.append({
            "subject": subject,
            "relation": normalized_relation,
            "object": obj
        })

triples_df = pd.DataFrame(triples).drop_duplicates()

triples_df.to_csv(OUTPUT_FILE, index=False)

print("Saved", len(triples_df), "clean triples to", OUTPUT_FILE)