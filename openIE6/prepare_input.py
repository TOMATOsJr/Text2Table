import pandas as pd

df = pd.read_csv("../cleaned_dataset/train.csv")
df = df.head(2000)

sentences = df["sent"].tolist()

with open("openie_input.txt", "w") as f:
    for s in sentences:
        f.write(s.strip() + "\n")

print("Wrote", len(sentences), "sentences to openie_input.txt")