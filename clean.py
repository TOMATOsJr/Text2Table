import pandas as pd
import re
import unicodedata

def clean_sentence(text):

    if pd.isna(text):
        return text

    # 1 Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2 Replace WikiBio tokens
    replacements = {
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # 3 Remove citations
    text = re.sub(r"\[[^\]]*\]", "", text)

    # 4 Normalize punctuation spacing
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # 5 Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_csv(input_file, output_file):

    df = pd.read_csv(input_file)
    df["sent"] = df["sent"].apply(clean_sentence)
    df.to_csv(output_file, index=False)

clean_csv("dataset/train_stacked.csv", "cleaned_dataset/train_stacked.csv")
clean_csv("dataset/valid_stacked.csv", "cleaned_dataset/valid_stacked.csv")
clean_csv("dataset/test_stacked.csv", "cleaned_dataset/test_stacked.csv")