---
marp: true
theme: default
paginate: true
math: katex
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 23px;
    padding: 30px 40px;
  }
  section.lead {
    background: #1a365d;
    color: #ffffff;
  }
  section.lead h1 {
    color: #ffffff;
    font-size: 2em;
  }
  section.lead h3 {
    color: #cbd5e0;
    font-size: 0.85em;
  }
  section.lead a {
    color: #ffffff;
    text-decoration: none;
  }
  section.lead code {
    background: #2d4a7a;
    color: #ffffff;
  }
  h1 {
    color: #1a365d;
    border-bottom: 2px solid #2b6cb0;
    padding-bottom: 0.15em;
    margin-bottom: 0.3em;
    font-size: 1.5em;
  }
  h2 {
    color: #2b6cb0;
    font-size: 1.15em;
  }
  table {
    font-size: 18px;
  }
  blockquote {
    border-left: 4px solid #2b6cb0;
    background: #ebf4ff;
    padding: 0.3em 0.8em;
    font-size: 0.85em;
    margin: 0.3em 0;
  }
  code {
    background: #edf2f7;
    padding: 0.1em 0.3em;
    border-radius: 3px;
    font-size: 0.85em;
  }
  pre {
    font-size: 0.75em;
  }
  ul, ol {
    margin: 0.2em 0;
  }
  li {
    margin: 0.1em 0;
  }
  p {
    margin: 0.3em 0;
  }
  .placeholder {
    background: #fff3cd;
    border: 1px dashed #d69e2e;
    padding: 0.3em;
    border-radius: 5px;
    text-align: center;
    font-style: italic;
    color: #744210;
    font-size: 0.85em;
  }
  footer {
    font-size: 14px;
    color: #718096;
  }
  strong {
    color: #1a365d;
  }
---

<!-- _class: lead -->

# Text to Structured Data using Entity Relationship Triples and Knowledge Graph

&nbsp;

Ananya Kasavajhala (2023113025)
Jayanth Raju (2023115016)
Tulasi Rushwik (2023111011)
Viswas Reddy (2023101008)

*Introduction to NLP - Prof. Manish Shrivastava*

---

# Problem Statement

**Task:** Given a biography paragraph, generate a structured table (Wikipedia infobox)

> *"Leonard Shenoff Randle (born February 12, 1949) is a former Major League Baseball player. He was the first-round pick of the Washington Senators..."*

| Attribute | Value |
|-----------|-------|
| name | Leonard Shenoff Randle |
| birth_date | February 12, 1949 |
| occupation | baseball player |
| team | Washington Senators |

We construct an intermediate **knowledge graph** and then flatten it into a table.

---

# Motivation

- Wikipedia has **millions of articles** but many lack structured infoboxes
- Structured data is critical for search engines, QA systems and knowledge bases
- Manually building infoboxes does not scale to millions of entries
- Existing relation extraction methods often need a **predefined schema**
  - We wanted to compare generative (seq2seq) vs. discriminative (span + classify) approaches
- An intermediate knowledge graph gives us a **checkpoint to inspect and fix errors** before the final table
  - The pipeline is modular. Each stage can be improved independently

---

# Related Work

| Work | Contribution |
|------|--------------|
| **REBEL** (Huguet Cabot & Navigli, 2021) | End-to-end relation extraction via seq2seq; 200+ Wikidata relations |
| **GLiNER** (Zaratiana et al., 2023) | Generalist NER with flexible entity labels; no retraining needed |
| **LUKE** (Yamada et al., 2020) | Entity-aware transformer; learns entity embeddings alongside tokens |
| **Sentence-BERT** (Reimers & Gurevych, 2019) | Dense sentence embeddings; we use it for relation canonicalization |
| **WikiBio** (Lebret et al., 2016) | 728K biography text-table pairs; standard benchmark |
| **Text-to-Table** (Wu et al., 2022) | Transformer-based text-to-table; similar evaluation methodology |

---

# Dataset: WikiBio

- **728,321** biographies from English Wikipedia (enwiki-20150901 dump)
- Pre-split: **~582K** train / **~72K** test / **~72K** valid

| File | Contents |
|------|----------|
| `.sent` | Biography paragraph (free text) |
| `.box` | Infobox attributes (tab-separated key-value) |
| `.title` | Wikipedia article title |
| `.nb` | Number of sentences per paragraph |

- Test set used for evaluation: **36,415** unique records, **140,249** gold attribute-value triples
- Common attributes: `birth_date`, `name`, `nationality`, `occupation`, `birth_place`, `position`

---

# Pipeline Overview

**Input:** Biography paragraph &emsp; **Output:** Structured attribute-value table

| Stage | Description |
|-------|-------------|
| **1. Preprocessing** | Clean WikiBio tokens, sentence segmentation |
| **2. Triple Extraction** | Extract (subject, relation, object) triples |
| **3. Knowledge Graph** | Group sentence-level triples per person |
| **4. KG to Table** | Normalize relations, deduplicate, assemble table |

Three approaches for **Stage 2** (triple extraction):

| Approach | Method |
|----------|--------|
| **A** | REBEL-only (base + fine-tuned) |
| **B** | Hybrid: GLiNER entity grounding + REBEL relations |
| **C** | BERT span extraction + LUKE relation classification + SBERT canonicalization |

---

# Text Preprocessing

WikiBio has dataset-specific tokens that need cleanup:

| WikiBio Token | Replacement |
|---------------|-------------|
| `-lrb-` / `-rrb-` | `(` / `)` |
| `-lsb-` / `-rsb-` | `[` / `]` |
| `-lcb-` / `-rcb-` | `{` / `}` |

**Steps applied:**
1. Replace WikiBio bracket tokens
2. Unicode NFKC normalization
3. Remove citation markers like `[1]`, `[citation needed]`, etc.
4. Fix punctuation spacing, strip extra whitespace
5. Sentence segmentation via spaCy (`en_core_web_trf`)

**Before:** `randle -lrb- born february 12 , 1949 -rrb- is a former ...`
**After:** `Randle (born February 12, 1949) is a former ...`

---

# Approach A: REBEL-Only

**REBEL** (`Babelscape/rebel-large`): Seq2seq model that generates triples directly

**Input:** biography sentence, **Output:** `<triplet> subject <subj> object <obj> relation`

**Design choices:**
- **Pronoun coreference**: `he`/`she` replaced with the person's title
- **Concatenated relation splitting**: REBEL can output merged relations like `"date of birth place of birth"`; split via greedy left-to-right match against seed vocabulary
- **Buffered processing**: adjacent sentences processed together for cross-sentence context
- Also **fine-tuned** REBEL on WikiBio infobox data converted into `<triplet>` format

---

# REBEL Fine-tuning Details

**Base model:** `Babelscape/rebel-large`
**Training data:** WikiBio infoboxes parsed into REBEL `<triplet>` labels

| Parameter | Value |
|-----------|-------|
| GPUs | 4 (Distributed Data Parallel) |
| Effective batch size | 64 (2 × 4 GPUs × 8 accumulation) |
| Learning rate | 3e-5 |
| Max epochs | 10 |
| Early stopping | patience 5 on eval loss |
| Source / Target max length | 512 / 256 tokens |
| FP16 | Yes |

Custom preprocessing script parsed indexed attributes, column vectors, sub-records and removed filler tokens from WikiBio TSV infoboxes.

---

# Approach B: Hybrid - GLiNER + REBEL

**Idea:** Use GLiNER2 for entity grounding, REBEL for relation extraction

**GLiNER2** (`fastino/gliner2-base-v1`) identifies entities with custom labels:
`person`, `occupation`, `education`, `birth_date`, `birth_place`, `nationality`, `organization`, `award`, `notable_work`

**Entity resolution step** (important addition):
- Match REBEL's short object outputs against GLiNER's full entity spans
- Example: REBEL outputs `"baseball"`, GLiNER detected `"major league baseball"` - resolved to the longer form
- Extra occupation triples inferred from GLiNER entity labels

**Scale on test set:**
44,144 sentences → 20,744 unique records → **124,784** triples after entity resolution

---

# Approach C: BERT + LUKE + Sentence-BERT

Discriminative pipeline with **three models**, each handling a different subtask:

| Step | Model | Task | Key Details |
|------|-------|------|-------------|
| **1. Span Extraction** | `bert-base-uncased` | BIO token classification (`O`, `B-ATTR`, `I-ATTR`) | 3 epochs, lr 3e-5, batch 16 |
| **2. Relation Classification** | `studio-ousia/luke-base` | Classify each span's relation type | 4 epochs, lr 2e-5, batch 16 |
| **3. Canonicalization** | `all-MiniLM-L6-v2` | Cluster & normalize relation labels | Agglom. clustering, cosine threshold 0.25 |

- **Step 1** identifies __where__ attribute values appear in the sentence
- **Step 2** input: `title: {title} [SEP] span: {span} [SEP] sentence: {sent}`
- **Step 3** groups similar relation strings (e.g. `"born in"` & `"place of birth"`), picks the most frequent as canonical

---

# BERT + LUKE: Training Data Preparation

Converting WikiBio infoboxes into supervised training data:

| Component | Approach |
|-----------|----------|
| **Value alignment** | Match each infobox value to biography text (exact substring first, fuzzy token F1 ≥ 0.6 as fallback) |
| **BIO tagging** | `B-ATTR` for first matched token, `I-ATTR` for continuation; sub-word tokens use `word_ids()`, continuation sub-words get `-100` |
| **Relation labels** | Each aligned span paired with its infobox attribute name |
| **Noise filtering** | Excluded: `image`, `caption`, `signature`, `logo`, `website`, `url` |
| **Sub-field merging** | `birth_date_1`, `birth_date_2` → `birth_date` |

---

# BERT + LUKE: Inference Pipeline

At inference, the three models are chained sequentially:

**sentence + title** → BERT Span Model → *list of spans* → LUKE Relation Model (per span) → *relation label* → SBERT Canonical Map → **output triples**

- Sentences where BERT predicts only `O` tags - no triples emitted
- Each extracted span is independently classified by LUKE
- Canonical relation map is pre-computed once and applied as a lookup at inference
- Final output per sentence: list of `(title, span_text, relation)` triples

---

# Knowledge Graph to Table Conversion

**KG Construction:** Group sentence-level triples by person title; deduplicate; store as `{title → {relation → [values]}}`

**KG to Table pipeline:**

| Phase | What it does |
|-------|-------------|
| **Anchor Detection** | Match subject to biography person via alias set |
| **Relation Normalization** | 80+ mapping rules; disambiguate date vs. place via entity types |
| **Deduplication** | Case-insensitive + substring absorption |
| **Filtering** | Drop noise: `instance_of`, `subclass_of`, `part_of`, etc. |
| **Assembly** | Order: name → birth → death → personal → career → other |

Example dedup: `"baseball"` absorbed by `"major league baseball"`

---

# Evaluation Methodology

| Metric | Description |
|--------|-------------|
| **chrF** | Character n-gram F-score (n = 1–6, $\beta$ = 2.0, recall-weighted) |
| **Triple P / R / F1** | Fuzzy matching: similarity = 0.35 × relation_sim + 0.65 × value_sim |

**Fuzzy triple matching details:**
- Value similarity = max(SequenceMatcher ratio, token F1)
- A predicted triple counts as matched if similarity ≥ **0.78**
- Greedy one-to-one alignment between predicted and gold triples

**Gold standard:** Constructed using WikiBio test set with **36,415** records, **140,249** triples

---

# Results: BERT + LUKE Pipeline

| Metric | Raw Relations | Canonicalized |
|--------|:------------:|:-------------:|
| **chrF** | **62.25** | 60.38 |
| **Precision** | **0.755** | 0.748 |
| **Recall** | **0.604** | 0.598 |
| **F1** | **0.671** | 0.664 |
| Matched Triples | 84,689 | 83,839 |
| Predicted Triples | 112,129 | 112,129 |
| Gold Triples | 140,249 | 140,249 |

- **Coverage:** 34,854 / 36,415 test records (**95.7%**)
- Raw outperforms canonical slightly. The Sentence-BERT clustering (threshold 0.25) over-merges some distinct relation types that happen to be semantically close

---

# Results: Cross-Approach Comparison

| Approach | chrF | Precision | Recall | F1 |
|----------|:----:|:---------:|:------:|:--:|
| REBEL-only (finetuned) | 53.37 | 0.38 | 0.50 | 0.44 |
| REBEL-only (canonical) | 52.39 | 0.38 | 0.51 | 0.44 |
| GLiNER + REBEL (hybrid) | 54.77 | 0.38 | 0.55 | 0.45 |
| **BERT + LUKE (raw)** | **62.25** | **0.76** | **0.60** | **0.67** |
| BERT + LUKE (canonical) | 60.38 | 0.75 | 0.60 | 0.66 |

The BERT+LUKE pipeline achieves the best performance, producing the most accurate and schema-consistent triples due to its structured, multi-stage design. The REBEL model, being generative, captures a wider range of relations but suffers from noise and inconsistency in entity representations. The hybrid approach (GLiNER + REBEL) improves entity grounding compared to pure REBEL but still underperforms the pipeline in precision and overall stability. Overall, there is a clear trade-off: structured pipelines give higher precision and consistency, while generative methods provide broader coverage but less reliability.

---

# Qualitative Examples - Good Results

**Input:** *"Leonard Shenoff Randle (born February 12, 1949) is a former Major League Baseball player..."*

| Attribute | BERT+LUKE Predicted | Gold |
|-----------|-----------|:----:|
| birth_date | february 12, 1949 | ✓ |
| debutteam | washington senators | ✓ |

**Input:** *"Philippe Adnot (born 25 August 1945 in Rhèges) is a member of the Senate of France..."*

| Attribute | BERT+LUKE Predicted | Gold |
|-----------|-----------|:----:|
| birth_date | 25 august 1945 | ✓ |
| constituency | aube | ✓ |
| occupation | farmer | ✓ |


---

# Failure Cases & Error Analysis

**1. Long input degeneration** (Finetuned REBEL - Jack Reynolds):
> Output: `...1919 1919 1919 1919 1919 1919...` - repeating year tokens

**2. Attribute confusion** (Dillon Sheppard):

| Attribute | Value | Problem |
|-----------|-------|---------|
| birth_date | 27 | Only day number captured |
| sport | 27 february 1979 | Full date in wrong field |

**3. Entity mix-up** (Teoctist Arăpașu):
- `birth_place` → `"toader arăpașu"` - actually his **birth name**
- `birth_date` → `"february 7, 1915 – july 30, 2007"` - birth+death merged

---

# Challenges & Limitations

| Challenge | Details |
|-----------|---------|
| **WikiBio tokenization** | Special tokens (`-lrb-`, `-rrb-`) needed custom cleanup |
| **Multi-sentence coreference** | Pronoun-to-title heuristic misses family member references |
| **Multi-valued attributes** | Years, clubs, stats as lists are hard to linearize |
| **No intermediate ground truth** | Cannot evaluate the KG directly, only the final table |
| **Evaluation gap** | chrF and fuzzy F1 don't capture wrong attribute assignment |
| **Long input degeneration** | Finetuned REBEL repeats tokens on long biographies |
| **Entity resolution noise** | Hybrid pipeline can produce `"new york new york"` dup objects |

---

# Conclusion & Future Work

**Achievements:**
- Modular **paragraph → knowledge graph → table** system for biographies
- Three extraction approaches: REBEL-only, GLiNER + REBEL, BERT + LUKE + SBERT
- BERT + LUKE: chrF **62.25**, triple F1 **0.671** on WikiBio test set (34,854 records)
- REBEL: chrf **53.37**, triple F1 **0.438**
- Hybrid: chrf **56.90**, triple F1 **0.47**


**Future Work:**
- Better coreference resolution (beyond pronoun-to-title heuristic)
- Cross-domain generalization beyond biographies
- End-to-end joint training (text to table without separate stages)
- Improved handling of multi-valued attributes (years, stats)

---

<!-- _class: lead -->

# References

- Huguet Cabot & Navigli (2021) - *REBEL: Relation Extraction By End-to-end Language generation*
- Zaratiana et al. (2023) - *GLiNER: Generalist Model for Named Entity Recognition*
- Yamada et al. (2020) - *LUKE: Deep Contextualized Entity Representations*
- Reimers & Gurevych (2019) - *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- Lebret et al. (2016) - *Neural Text Generation from Structured Data (WikiBio)*
- Wu et al. (2022) - *Text-to-Table: A New Way of Information Extraction*

Fine-tuned REBEL model: *TOMATOsJr/Rebel_Finetuned_on_Triples* (HuggingFace)
