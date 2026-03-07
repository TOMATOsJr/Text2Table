# Dataset extraction
1) Extracted 
Converted dataset into csv format for easier use.
# Dataset Preprocessing
## Dataset cleaning
1) Repaced wikibio tokens
replacements = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lsb-": "[",
    "-rsb-": "]",
    "-lcb-": "{",
    "-rcb-": "}",
}
2) NFKC normalization of unicodes
3) Remove citations
4) Remove punctuation spacing
5) Remove extra whitespaces

# Entity Extraction
Using 3 models currently:
1) openIE6  
Instructions to run
```bash
python3 run.py \
--mode predict \
--inp input.txt \
--out output.txt \
--save models/oie_model \
--gpus 0
```
Then run
```bash
python3 clean_output.py
```
after changing the paths correspondingly.
2) spacy NER + REBEL
3) 