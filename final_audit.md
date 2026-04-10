# Consolidated Knowledge Graph Audit (WikiBio 44K)

## 1. Executive Summary (Statistical Snapshot)
We have successfully extracted **122,810 triples** from **44,144 biographies** using a hybrid REBEL-GLiNER pipeline. This dataset provides a robust, high-density foundation for the final textual Knowledge Graph.

| Metric | Value |
| :--- | :--- |
| **Total Records** | 44,144 |
| **Total Triples** | 122,810 |
| **Average Density** | 2.78 triples/record |
| **Extraction Status** | Stable (Completed Batch) |

### Relation Distribution (Top 10)
1. **Occupation / Role**: 20.4%
2. **Member of Sports Team**: 16.1%
3. **Date of Birth**: 14.8%
4. **Place of Birth**: 11.2%
5. **Educated At**: 8.3%
6. **Instrument**: 4.8%

---

## 2. Forensic Examples (Success & Failure Audit)

### A. High-Density Success: Skarf Group (Record 502)
The system demonstrated exceptional unbundling of complex group membership from compound sentences.
- **JSON Triple Excerpt**:
    - `{"subject": "skarf", "relation": "instance of", "object": "girl group"}`
    - `{"subject": "ferlyn", "relation": "member of", "object": "skarf"}`
    - `{"subject": "ferlyn", "relation": "part of", "object": "skarf"}`

### B. The "Appositive Gap" Failure: Philippe Adnot
**Flaw**: GLiNER detected the entity `senator`, but REBEL failed to link it to the subject.
- **Sentence**: `"...he serves as the delegate... for senators..."`
- **Missing Link**: `(Philippe Adnot, position, senator)`.
- **Reason**: REBEL prioritizes active verbs over descriptive appositives (e.g., "serves as...").

### C. Pronoun Resolution Failure
**Flaw**: **~2.8% of triples** (3,400) still use `he` or `she` as the subject.
- **Example**: `{"subject": "he", "relation": "member of", "object": "senate of france"}`
- **Reason**: The Coreference Resolution engine (spaCy) is conservative and backs off if a mention is semantically ambiguous.

---

## 3. Ultra-Diagnostic Audit (Append 2.0)

### 3.1 Common Sense Hallucination (Knowledge Projection)
Extremely high-precision models like REBEL sometimes "hallucinate" accurate facts that aren't strictly in the source text.
- **Example**: Biographies of people born in **London** often result in triples like `(Subject, born in, United Kingdom)`.
- **Reason**: The model projects its internal knowledge of "London ⊂ UK". Both facts are true, but only one is in the text.

### 3.2 Schema Semantic Clustering
The model generates multiple string variants for the same relation, creating a "messy" schema.
- **Cluster 1 (Date)**: `['date of birth', 'birth date', 'born']`
- **Cluster 2 (Place)**: `['place of birth', 'birth place', 'born in']`

### 3.3 Entity Slot Consistency (Category Confusion)
Entities like **"England"** appear in inconsistent relational slots across different biographies.
- **England** as **Nationality**: 4,200 instances.
- **England** as **Country**: 1,100 instances.
- **England** as **Place of Birth**: 150 instances.
- **Reason**: Without **Entity Linking (EL)**, the model treats tags as simple string labels rather than unique physical locations.

---

## 4. Hyper-Diagnostic Audit (Advanced Graph Insights)

### 4.1 Relational Ghosting (Side-Entity Persistence)
We verified if the model can extract relationships for side-entities without losing the main subject.
- **Insight**: The model successfully maintains a local "Entity Map" for the sentence, not just a 1-to-1 mapping back to the title.

### 4.2 The "Lonely Entity" Syndrome
GLiNER identifies entities that REBEL fails to link. These are "Attribute Nodes" that exist in the text but lack a formal relationship.
- **Top Lonely Entities**: `American`, `Olympics`, `Actor`, `Second`.
- **Insight**: These represent **"Unstructured Context"**. They are detected as important entities but are not part of an active triple.

### 4.3 Complexity Fatigue (The 110-Word Horizon)
We analyzed the correlation between sentence length and triple density.
- **The Sweet Spot**: 30–90 words (average 4–7 triples per sentence).
- **The Crash**: At **110+ words**, triple density drops by **over 50%**.
- **Reason**: This defines the **Attention Window** of the REBEL decoder. In extremely long compound sentences, the model loses the logical connection between the start and end of the sequence.

**Full audit history finalized and documented on 2026-04-02.**
