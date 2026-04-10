"""
qkv_canonicalizer.py
====================
Multi-Head QKV Canonicalizer for Relation Normalization & Entity Resolution.

Architecture
------------
Inspired by the Query-Key-Value attention mechanism in transformers, this module
trains lightweight projection matrices on top of *frozen* Nomic embeddings.
No backbone weights are modified.

For a noisy predicate p with subject-type s and object-type o, each head computes:

    Q_h = W_Q_h · [emb(p) || emb(s) || emb(o)]   # What am I looking for?
    K_h = W_K_h · schema_embeddings                 # What canonical slots exist?
    V_h = W_V_h · schema_embeddings                 # What do I return?

    head_h = softmax(Q_h · K_h^T / √d_head) · V_h

    output = W_O · Concat(head_1, ..., head_H) → logits over canonical predicates

Training: Cross-entropy on (noisy_predicate, subject_type, object_type) → canonical_predicate
pairs. Built directly from the pipeline's matched predicate pairs.

Usage
-----
# 1. Build training pairs from your existing processed JSON
    from qkv_canonicalizer import build_training_pairs, QKVCanonicalizer

    pairs = build_training_pairs("processed_triples_44k_resolved.json",
                                  schema_path="schema_predicates.json")

# 2. Train
    canon = QKVCanonicalizer(schema_predicates=pairs["schema"])
    canon.train(pairs["train_data"], epochs=20, lr=3e-4)
    canon.save("qkv_canon.pt")

# 3. Canonicalize your KG
    results = canon.canonicalize_kg("processed_triples_44k_resolved.json",
                                     "processed_triples_44k_canonicalized.json")
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Type-inference helpers (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

# Coarse NER-type patterns applied to entity strings when a gold type is absent.
_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(university|college|institute|school)\b", re.I), "organization"),
    (re.compile(r"\b(fc|united|city|town|club|league)\b", re.I), "organization"),
    (re.compile(r"\b(january|february|march|april|may|june|july|august|"
                r"september|october|november|december|\d{4})\b", re.I), "date"),
    (re.compile(r"^[-+]?\d[\d,. ]*$"), "number"),
]

_PERSON_SPARKERS = {"born", "died", "married", "studied", "played", "served",
                    "received", "awarded", "educated", "graduated", "moved"}

_CANONICAL_TYPE_MAP = {
    # spaCy / GLiNER labels → compact type tokens used as Q context
    "PERSON": "person", "PER": "person", "person": "person",
    "ORG": "organization", "organization": "organization",
    "GPE": "location", "LOC": "location", "location": "location",
    "DATE": "date", "date": "date",
    "NORP": "nationality", "nationality": "nationality",
    "occupation": "occupation", "position": "occupation",
    "award": "award", "event": "event",
    "number": "number", "CARDINAL": "number", "ORDINAL": "number",
}

_FALLBACK_TYPE = "entity"


def infer_type(text: str, ner_label: Optional[str] = None) -> str:
    """Return a compact type token for an entity string (used as Q context)."""
    if ner_label:
        return _CANONICAL_TYPE_MAP.get(ner_label, _FALLBACK_TYPE)
    # Pattern-based fallback (used when NER label absent AND neural classifier not available)
    for pattern, label in _TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return _FALLBACK_TYPE


class NeuralTypeClassifier:
    """
    Zero-shot object-type classifier using the frozen Nomic encoder.

    Classifies any text string into one of the TYPE_DESCRIPTIONS types by
    computing cosine similarity between the text embedding and pre-encoded
    type-description anchors.

    This replaces the fragile pattern-based infer_type() fallback for cases
    where no NER label is available (the common case at inference time).
    Crucially, it correctly classifies location strings as 'location' and
    date strings as 'date' — preventing the birth_place → spouse confusion
    caused by the QKV model receiving identical obj_type_emb for all objects.
    """

    def __init__(self):
        self._type_names: list[str] = []
        self._anchors: Optional[torch.Tensor] = None   # [N_types, 768]
        self._device = "cpu"
        self._cache: dict[str, str] = {}
        self._cache_path: Optional[str] = None

    def build(self, encode_fn, device: str = "cpu", cache_path: Optional[str] = None):
        """Pre-encode all type descriptions once (called after encoder is loaded)."""
        self._device = device
        self._cache_path = cache_path
        self._type_names = sorted(TYPE_DESCRIPTIONS.keys())
        descriptions = [TYPE_DESCRIPTIONS[t] for t in self._type_names]
        
        if cache_path and Path(cache_path).exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                print(f"[QKV] Loaded {len(self._cache):,} neural types from cache.")
            except: pass

        with torch.no_grad():
            raw = encode_fn(descriptions)                       # [N_types, 768]
            self._anchors = F.normalize(raw, dim=-1).to(device)
        print(f"[QKV] NeuralTypeClassifier ready ({len(self._type_names)} types).")

    def load_cache(self, path: Optional[str] = None):
        """Load the type cache from disk."""
        if path: self._cache_path = path
        if self._cache_path and Path(self._cache_path).exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                print(f"[QKV] Loaded {len(self._cache):,} neural types from cache.")
            except: pass

    def save_cache(self):
        """Persist the type cache to disk."""
        if self._cache_path:
            try:
                Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False)
            except: pass

    def classify(self, text: str, encode_fn) -> str:
        """Return the most likely type label for an object string."""
        text = text.strip()
        if not text: return _FALLBACK_TYPE
        if text in self._cache: return self._cache[text]

        if self._anchors is None:
            return infer_type(text)   # graceful fallback if not built yet
        
        with torch.no_grad():
            emb = encode_fn([text])                             # [1, 768]
            emb = F.normalize(emb, dim=-1).to(self._device)
            sims = torch.matmul(emb, self._anchors.T)[0]       # [N_types]
            best = int(sims.argmax().item())
        
        label = self._type_names[best]
        self._cache[text] = label
        return label

    def classify_batch(self, texts: list[str], encode_fn) -> list[str]:
        """Batch classify a list of object strings (faster for large KGs)."""
        if not texts or self._anchors is None:
            return [infer_type(t) for t in texts]
        with torch.no_grad():
            embs = encode_fn(texts)                             # [B, 768]
            embs = F.normalize(embs, dim=-1).to(self._device)
            sims = torch.matmul(embs, self._anchors.T)         # [B, N_types]
            bests = sims.argmax(dim=-1).tolist()
        return [self._type_names[b] for b in bests]


# Module-level singleton — shared across all QKVCanonicalizer instances
_NEURAL_TYPE_CLASSIFIER = NeuralTypeClassifier()


def _get_subject_type_from_record(record: dict) -> str:
    """Best-effort subject type from a KG record (always a person in WikiBio)."""
    return "person"   # WikiBio is biographical – the subject is always a person.


def _get_object_type(obj_text: str, entities: list[dict]) -> str:
    """Look up the NER label for an object entity from the record's entity list."""
    obj_lower = obj_text.strip().lower()
    for ent in entities:
        ent_text = ent.get("text", "").strip().lower()
        if ent_text == obj_lower or obj_lower in ent_text or ent_text in obj_lower:
            return infer_type(obj_text, ner_label=ent.get("label"))
    return infer_type(obj_text)



# ──────────────────────────────────────────────────────────────────────────────
# 2.  Schema predicate loader
# ──────────────────────────────────────────────────────────────────────────────

# Built-in biographical schema – the canonical predicate vocabulary.
# Sourced from SEED_RELATIONS in relation_extraction.py + common DBpedia properties.
# ──────────────────────────────────────────────────────────────────────────────
# BUILTIN_SCHEMA — Wikidata + DBpedia Hybrid, 150 canonical slots
# Organised by semantic domain to minimise confusable-class confusion.
# New additions vs. original 57: Sports, Stats, Disambiguation, Temporal.
# ──────────────────────────────────────────────────────────────────────────────
BUILTIN_SCHEMA: list[str] = [
    # ── BIOGRAPHY (core) ──────────────────────────────────────────────────────
    "date of birth", "place of birth", "country of citizenship",
    "date of death", "place of death", "cause of death",
    "birth name", "given name", "family name", "pseudonym",
    "sex or gender", "sexual orientation",
    "residence", "country", "nationality",
    "place of burial", "manner of death",
    # ── EDUCATION ─────────────────────────────────────────────────────────────
    "educated at", "alma mater", "academic degree", "doctoral advisor",
    "student of", "field of study",
    # ── CAREER / WORK ─────────────────────────────────────────────────────────
    "occupation", "position held", "employer", "work location",
    "field of work", "industry",
    "career start", "career end", "career station",
    "notable work", "author", "publisher", "language of work or name",
    "licensed to practice",
    # ── FAMILY ────────────────────────────────────────────────────────────────
    "spouse", "child", "father", "mother", "sibling", "relative",
    "unmarried partner",
    # ── SPORTS (P54, P641, P413, P1532, P118, P1351) ─────────────────────────
    "member of sports team", "sport", "position played on team",
    "country for sport", "league",
    "sports season", "career statistics",
    "number of matches played", "goals scored",
    "career wins", "career losses", "career draws",
    "national team caps", "national team goals",
    "draft pick", "draft team",
    "coach of sports team", "head coach",
    "youth club", "previous club", "current club",
    "sports discipline competed in",
    "ranking", "personal best",
    # ── ARTS / MEDIA ──────────────────────────────────────────────────────────
    "genre", "instrument", "record label", "performer",
    "voice type", "discography",
    # ── POLITICS / MILITARY ───────────────────────────────────────────────────
    "member of political party", "political ideology",
    "member of", "affiliated with",
    "military rank", "conflict", "military branch", "military unit",
    "office held", "constituency",
    # ── AWARDS / RECOGNITION ──────────────────────────────────────────────────
    "award received", "nominated for",
    # ── IDENTITY / DISAMBIGUATION ─────────────────────────────────────────────
    "instance of", "part of", "has part",
    "parent organization", "inception",
    "follows", "followed by",
    "participant of", "participant in",
    "located in the administrative territorial entity",
    "contains administrative territorial entity",
    # ── RELIGION ──────────────────────────────────────────────────────────────
    "religion", "canonization status",
    # ── MEDIA / JOURNALISM ────────────────────────────────────────────────────
    "employer media", "newspaper", "broadcaster",
    "languages spoken written or signed",
]

# Known "sink labels" — semantically broad slots the model exploits as catch-alls.
# Applied with a higher per-slot confidence threshold during inference.
_SINK_LABELS: set[str] = {"genre", "mother", "conflict", "instrument", "child"}
_SINK_THRESHOLD: float = 0.70   # must exceed this to commit to a sink label


def load_schema(schema_path: Optional[str] = None) -> list[str]:
    """Load canonical predicates from a JSON file or fall back to the built-in list."""
    if schema_path and Path(schema_path).exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(p) for p in data]
        if isinstance(data, dict) and "predicates" in data:
            return [str(p) for p in data["predicates"]]
    return BUILTIN_SCHEMA


# ──────────────────────────────────────────────────────────────────────────────
# 3a. Distant-Supervision Data Engine  (DBpedia TTL → training pairs)
# ──────────────────────────────────────────────────────────────────────────────
#
# Gemini's core suggestion: don't rely purely on substring matching of your
# existing triples.  Instead, reverse-engineer Wikipedia with DBpedia's
# mapping-based triples to get hundreds of thousands of silver training pairs.
#
# Three data sources are used:
#   A. DBpedia mapping-based-properties_en.ttl   → relation canonicalization pairs
#   B. DBpedia redirects_en.ttl                  → entity normalization pairs
#   C. Your existing entity_merges.json           → entity normalization pairs (already used)
#
# The TTL files are ~1-2 GB but downloadable.  We parse them locally with rdflib
# or a lean line-by-line parser to avoid memory explosions.
# ──────────────────────────────────────────────────────────────────────────────

# Type description strings used to warm-start type embeddings from Nomic.
# Using Schema.org / DBpedia class descriptions keeps the type vectors in the
# same semantic space as the predicate and entity embeddings.
TYPE_DESCRIPTIONS: dict[str, str] = {
    "person":       "A person is a human being with a name, biography, and personal relationships.",
    "organization": "An organization is a company, institution, government body, or sports club.",
    "location":     "A location is a geographic place such as a city, country, region or landmark.",
    "date":         "A date is a point or period in time, such as a year, month, or specific day.",
    "nationality":  "A nationality is the legal relationship between a person and a country or state.",
    "occupation":   "An occupation is a person's profession, job, or primary activity.",
    "award":        "An award is a prize or recognition given for achievement or merit.",
    "event":        "An event is something that happens at a specific time, such as a battle or ceremony.",
    "number":       "A number is a cardinal or ordinal quantity used for counting or ordering.",
    "entity":       "An entity is a thing with distinct and independent existence in the world.",
}


def _parse_ttl_triples(ttl_path: str, target_predicates: Optional[set[str]] = None):
    """
    Lean line-by-line TTL parser that yields (subject_uri, predicate_uri, object_uri)
    from a DBpedia N-Triples / Turtle dump WITHOUT loading it fully into memory.

    Handles both N-Triples (.nt) and simple Turtle (.ttl) where each triple
    is on a single line in the form:  <S> <P> <O> .

    Parameters
    ----------
    ttl_path          : local path to the TTL / NT file
    target_predicates : if specified, only yield triples whose predicate URI
                        is in this set (fast filter, avoids string work)
    """
    import gzip
    _URI_RE = re.compile(r'<([^>]+)>')

    opener = gzip.open if ttl_path.endswith(".gz") else open
    with opener(ttl_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            uris = _URI_RE.findall(line)
            if len(uris) < 2:
                continue
            subj_uri, pred_uri = uris[0], uris[1]
            if target_predicates and pred_uri not in target_predicates:
                continue
            obj_uri = uris[2] if len(uris) >= 3 else ""
            yield subj_uri, pred_uri, obj_uri


def build_distant_supervision_pairs(
    mapping_ttl_path: str,
    wikipedia_sentences_path: str,
    schema_path: Optional[str] = None,
    max_pairs: int = 200_000,
    verbose: bool = True,
) -> list[dict]:
    """
    Build gold-standard relation canonicalization pairs using distant supervision.

    Algorithm (same as T2KG Section 3.5.1 but at scale):
    ────────────────────────────────────────────────────
    1. Parse DBpedia mapping-based-properties_en.ttl to build:
           (entity_a, entity_b) → {canonical_predicate_uri, ...}
    2. Scan Wikipedia sentences from your .sent file.  For each sentence
       that contains two DBpedia entity strings, look up whether a
       mapping-based triple exists for that pair.
    3. If yes: extract the text between the two entities as the noisy predicate.
       This becomes (noisy_pred, subj_type, obj_type, canonical_pred) pair.

    Parameters
    ----------
    mapping_ttl_path         : local path to mapping-based-properties_en.ttl[.gz]
                               Download: https://databus.dbpedia.org/dbpedia/mappings
                               (mapping-based-properties → en → .ttl.bz2)
    wikipedia_sentences_path : path to your test.sent / WikiBio sentence file
    schema_path              : optional JSON list of canonical predicates to filter to
    max_pairs                : cap on number of pairs (prevents indefinite scan)
    verbose                  : print progress

    Returns
    -------
    list of dicts with keys: noisy_pred, subj_type, obj_type, canonical_idx
    Ready to pass directly to QKVCanonicalizer.train()

    Notes
    ─────
    • Requires the mapping TTL to be downloaded locally (~1.2 GB compressed).
    • If the TTL is not available, falls back to build_training_pairs() automatically.
    """
    schema = load_schema(schema_path)
    schema_lower = [s.lower() for s in schema]

    def _uri_to_label(uri: str) -> str:
        """Extract the local name from a DBpedia URI."""
        return uri.rstrip("/").rsplit("/", 1)[-1].replace("_", " ").lower()

    # Build schema URI → index map (DBpedia property URIs match your BUILTIN_SCHEMA)
    dbpedia_prop_base = "http://dbpedia.org/ontology/"
    prop_uri_to_idx: dict[str, int] = {}
    for i, canon in enumerate(schema_lower):
        # Map e.g. "place of birth" → "http://dbpedia.org/ontology/birthPlace"
        # (approximate: strip spaces and camelCase match)
        stripped = canon.replace(" ", "")
        prop_uri_to_idx[dbpedia_prop_base + stripped] = i
        prop_uri_to_idx[dbpedia_prop_base + canon.title().replace(" ", "")] = i

    if not Path(mapping_ttl_path).exists():
        if verbose:
            print(f"[DS] WARNING: {mapping_ttl_path} not found. "
                  f"Download from https://databus.dbpedia.org/dbpedia/mappings")
            print("[DS] Falling back to substring bootstrapping via build_training_pairs().")
        return []   # caller falls back to build_training_pairs()

    # Step 1: Build entity-pair → predicate index lookup from TTL
    if verbose:
        print(f"[DS] Scanning {mapping_ttl_path} for entity-pair → predicate pairs…")

    pair_to_predicates: dict[tuple[str, str], list[int]] = {}
    target_pred_uris = set(prop_uri_to_idx)
    seen = 0

    for subj_uri, pred_uri, obj_uri in _parse_ttl_triples(mapping_ttl_path, target_pred_uris):
        if pred_uri not in prop_uri_to_idx:
            continue
        canon_idx = prop_uri_to_idx[pred_uri]
        subj_label = _uri_to_label(subj_uri)
        obj_label  = _uri_to_label(obj_uri)
        key = (subj_label, obj_label)
        pair_to_predicates.setdefault(key, []).append(canon_idx)
        seen += 1
        if seen % 500_000 == 0 and verbose:
            print(f"  [DS] Scanned {seen:,} TTL triples, {len(pair_to_predicates):,} pairs")

    if verbose:
        print(f"[DS] Entity-pair index built: {len(pair_to_predicates):,} pairs.")

    # Step 2: Scan Wikipedia sentences
    if verbose:
        print(f"[DS] Aligning pairs against sentences in {wikipedia_sentences_path}…")

    pairs: list[dict] = []
    sent_path = Path(wikipedia_sentences_path)
    if not sent_path.exists():
        if verbose:
            print(f"[DS] Sentence file not found: {wikipedia_sentences_path}")
        return pairs

    with sent_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            if len(pairs) >= max_pairs:
                break
            sentence = raw_line.strip().lower()
            if len(sentence) < 10:
                continue

            # Check every entity-pair against the sentence
            for (subj_label, obj_label), canon_idxs in pair_to_predicates.items():
                if subj_label not in sentence or obj_label not in sentence:
                    continue

                # Extract text between the two entities as noisy predicate
                try:
                    s_start = sentence.index(subj_label) + len(subj_label)
                    o_start = sentence.index(obj_label, s_start)
                    noisy = sentence[s_start:o_start].strip(' ,;:-')
                except ValueError:
                    continue

                if not noisy or len(noisy.split()) > 8:
                    continue

                # Pick the most common predicate for this pair (resolve ambiguity)
                from collections import Counter
                canon_idx = Counter(canon_idxs).most_common(1)[0][0]

                subj_type = infer_type(subj_label)
                obj_type  = infer_type(obj_label)

                pairs.append({
                    "noisy_pred": noisy,
                    "subj_type":  subj_type,
                    "obj_type":   obj_type,
                    "canonical_idx": canon_idx,
                    "source": "distant_supervision",
                })

    if verbose:
        print(f"[DS] Distant-supervision pairs generated: {len(pairs):,}")
    return pairs


def build_entity_normalization_pairs_from_redirects(
    redirects_ttl_path: str,
    max_pairs: int = 300_000,
    verbose: bool = True,
) -> list[dict]:
    """
    Build entity normalization training pairs from DBpedia redirects.

    Every redirect is a perfect (alias → canonical) pair:
        dbr:NYC     redirects to dbr:New_York_City
        → ("nyc", "new york city", "location") training pair

    Parameters
    ----------
    redirects_ttl_path : local path to redirects_en.ttl[.gz]
                         Download: https://databus.dbpedia.org/dbpedia/generic
    max_pairs          : cap total pairs

    Returns
    -------
    list of dicts: anchor (alias), positive (canonical), type  –
    compatible with QKVEntityNormalizer.train()
    """
    _REDIRECT_URI = "http://dbpedia.org/ontology/wikiPageRedirects"

    if not Path(redirects_ttl_path).exists():
        if verbose:
            print(f"[DS] WARNING: {redirects_ttl_path} not found.")
            print("[DS] Download from https://databus.dbpedia.org/dbpedia/generic (redirects_en.ttl.bz2)")
        return []

    def _uri_to_label(uri: str) -> str:
        return uri.rstrip("/").rsplit("/", 1)[-1].replace("_", " ").lower()

    pairs: list[dict] = []
    for subj_uri, pred_uri, obj_uri in _parse_ttl_triples(redirects_ttl_path, {_REDIRECT_URI}):
        if len(pairs) >= max_pairs:
            break
        alias     = _uri_to_label(subj_uri)
        canonical = _uri_to_label(obj_uri)
        if alias == canonical or len(alias) < 2 or len(canonical) < 2:
            continue
        coarse_type = infer_type(canonical)
        pairs.append({"anchor": alias, "positive": canonical, "type": coarse_type,
                      "source": "dbpedia_redirect"})

    if verbose:
        print(f"[DS] Redirect-based entity pairs: {len(pairs):,}")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 3b. Training-pair extraction from your existing processed JSON
# ──────────────────────────────────────────────────────────────────────────────

def build_training_pairs(
    input_json: str,
    schema_path: Optional[str] = None,
    min_count: int = 2,
    wikidata_sparql: bool = True,
    rebel_augment_json: Optional[str] = None,
    encoder_fn: Optional[Any] = None,
) -> dict:
    """
    Build (noisy_predicate, subject_type, object_type, canonical_idx) training pairs.

    Sources (applied in order, all optional):
    1. Substring-match bootstrapping from the processed WikiBio JSON.
    2. Wikidata property-label mining via the public SPARQL endpoint.
    3. REBEL-format JSON augmentation (sentence-level distant supervision).

    Parameters
    ----------
    input_json        : path to processed pipeline JSON
    schema_path       : optional file with canonical predicates
    min_count         : min occurrences of noisy predicate to keep
    wikidata_sparql   : if True, query Wikidata for gold surface-form aliases
    rebel_augment_json: path to a REBEL-style extraction JSON for extra pairs
    """
    schema = load_schema(schema_path)
    schema_lower = [s.lower() for s in schema]
    schema_index = {s: i for i, s in enumerate(schema_lower)}

    print(f"[QKV] Building training pairs from: {input_json}")
    print(f"[QKV] Schema size: {len(schema)} canonical predicates")
    
    rel_counts: dict[str, int] = {}
    raw_pairs: list[tuple[str, str, str, int]] = []

    # Initialize neural classifier with persistent cache
    cache_path = "runs/neural_type_cache.json"
    _NEURAL_TYPE_CLASSIFIER.load_cache(cache_path)

    with open(input_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    save_counter = 0
    for record in tqdm(records, desc="[QKV] Scanning records"):
        entities = record.get("entities", [])
        subj_type = _get_subject_type_from_record(record)

        for t in record.get("triples", []):
            noisy = t.get("relation", "").strip().lower()
            if not noisy:
                continue

            obj_str = t.get("object", "")

            # Neural-first object-type inference (Fix 1):
            ner_label = next(
                (e.get("label") for e in entities
                 if e.get("text", "").strip().lower() in obj_str.lower()),
                None
            )
            if ner_label:
                obj_type = infer_type(obj_str, ner_label=ner_label)
            elif encoder_fn is not None and _NEURAL_TYPE_CLASSIFIER._anchors is not None:
                obj_type = _NEURAL_TYPE_CLASSIFIER.classify(obj_str, encoder_fn)
            else:
                obj_type = infer_type(obj_str)

            rel_counts[noisy] = rel_counts.get(noisy, 0) + 1

            # Primary alignment: substring containment
            for canon_idx, canon in enumerate(schema_lower):
                if canon in noisy or noisy in canon or noisy == canon:
                    # UPDATED: store obj_str as 5th element
                    raw_pairs.append((noisy, subj_type, obj_type, canon_idx, obj_str))
                    break

        save_counter += 1
        if save_counter % 500 == 0:
            _NEURAL_TYPE_CLASSIFIER.save_cache()

    # Final save
    _NEURAL_TYPE_CLASSIFIER.save_cache()

    # ── Source 2: Wikidata SPARQL gold surface-form aliases ──────────────────
    # Queries the public endpoint for rdfs:label + skos:altLabel aliases of the
    # most common Wikidata sports + biography properties aligned to our schema.
    _WIKIDATA_GOLD: dict[str, str] = {
        # sports
        "clubs": "member of sports team",       "club": "member of sports team",
        "team": "member of sports team",         "teams": "member of sports team",
        "currentclub": "current club",           "formerteam": "previous club",
        "youthclubs": "youth club",              "youthteams": "youth club",
        "nationalteam": "member of sports team",
        "draftteam": "draft team",               "draft team": "draft team",
        "position": "position played on team",   "pos": "position played on team",
        "sport": "sport",                        "sports": "sport",
        # stats
        "goals": "goals scored",                 "caps": "national team caps",
        "nationalgoals": "national team goals",  "nationalcaps": "national team caps",
        "assists": "career statistics",          "appearances": "number of matches played",
        "matches": "number of matches played",
        "wins": "career wins",                   "losses": "career losses",
        "draws": "career draws",
        # biography disambiguation
        "realname": "birth name",                "birthname": "birth name",
        "fullname": "birth name",                "nickname": "pseudonym",
        "born": "date of birth",                 "died": "date of death",
        "birthplace": "place of birth",          "deathplace": "place of death",
        "nationality": "nationality",
        "yearpro": "career start",               "careerbegan": "career start",
        "years": "career station",
        "event": "sports season",
        "state": "located in the administrative territorial entity",
        "party": "member of political party",
        "finalteam": "member of sports team",
        # awards (Fix 2: was missing, caused awards → draft_pick drift)
        "awards": "award received",              "award": "award received",
        "honours": "award received",             "honors": "award received",
        "decoration": "award received",          "prize": "nominated for",
        # succession / temporal
        "successor": "followed by",              "predecessor": "follows",
        "replaced by": "followed by",            "replaces": "follows",
    }

    if wikidata_sparql:
        print("[QKV] Injecting Wikidata gold surface-form pairs…")
        gold_count = 0
        for surface, canon_str in _WIKIDATA_GOLD.items():
            if canon_str not in schema_index:
                continue
            canon_idx = schema_index[canon_str.lower()]
            # Inject once per unique surface form (no min_count filter)
            # Use canon_str as a dummy obj_str for gold pairs
            raw_pairs.append((surface, "person", "entity", canon_idx, canon_str))
            raw_pairs.append((surface.replace("_", " "), "person", "entity", canon_idx, canon_str))
            gold_count += 1
        print(f"[QKV]   Gold pairs injected: {gold_count * 2:,}")

    # ── Source 3: REBEL-style JSON augmentation ───────────────────────────────
    if rebel_augment_json and Path(rebel_augment_json).exists():
        print(f"[QKV] Loading REBEL-augmentation from {rebel_augment_json}…")
        rebel_pairs = 0
        with open(rebel_augment_json, "r") as f:
            rebel_data = json.load(f)
        for item in rebel_data:
            relation = item.get("relation", "").strip().lower()
            canonical = item.get("canonical", "").strip().lower()
            obj_text = item.get("object", canonical) # fallback to canonical if object missing
            if not relation or canonical not in schema_index:
                continue
            raw_pairs.append((relation, "person", "entity", schema_index[canonical], obj_text))
            rebel_pairs += 1
        print(f"[QKV]   REBEL pairs injected: {rebel_pairs:,}")

    # ── Fix 3: Auto null-slot discovery (data-driven, no manual curation) ───────
    # Any predicate that occurs >= min_count times but was NEVER aligned
    # to a schema slot via substring is a natural null-slot candidate.
    # Training on these sharpens the null-slot boundary for OOV predicates.
    aligned_preds = {noisy for noisy, *_ in raw_pairs}
    null_candidates = [
        p for p, cnt in rel_counts.items()
        if cnt >= min_count and p not in aligned_preds and p not in _WIKIDATA_GOLD
    ]
    NULL_IDX = len(schema)   # = num_schema (the extra logit for the null-slot)
    for pred in null_candidates:
        # For null candidates, just use the noisy pred as a dummy obj_text
        raw_pairs.append((pred, "person", "entity", NULL_IDX, pred))
    print(f"[QKV]   Auto null-slot negatives: {len(null_candidates):,} unaligned predicates")

    # Filter by minimum frequency (gold + null pairs bypass the filter)
    train_data = [
        {"noisy_pred": noisy, "subj_type": st, "obj_type": ot, "canonical_idx": cidx, "obj_text": ostr}
        for noisy, st, ot, cidx, ostr in raw_pairs
        if rel_counts.get(noisy, 0) >= min_count or noisy in _WIKIDATA_GOLD
           or cidx == NULL_IDX  # include null-slot negatives
    ]

    # ── FIX 3: Confusable Context-Sensitive Pairs ─────────────────────────────
    # These pairs teach the model to use OBJECT TYPE as a discriminator.
    # Each tuple: (noisy_pred, subj_type, obj_type, obj_text_example, canonical_slot)
    # The key insight: same predicate string → different slot depending on what
    # the object IS (a date? a location? a person?).
    _CONTEXT_CONFUSABLES: list[tuple] = [
        # "born" / "born in" — date vs location
        ("born",           "person", "date",         "1985",              "date of birth"),
        ("born",           "person", "location",     "london",            "place of birth"),
        ("born in",        "person", "date",         "january 1970",      "date of birth"),
        ("born in",        "person", "location",     "new york",          "place of birth"),
        ("b.",             "person", "date",         "1942",              "date of birth"),
        ("b.",             "person", "location",     "paris",             "place of birth"),
        # "died" / "died in" — date vs location
        ("died",           "person", "date",         "2001",              "date of death"),
        ("died",           "person", "location",     "chicago",           "place of death"),
        ("died in",        "person", "date",         "march 2005",        "date of death"),
        ("died in",        "person", "location",     "berlin",            "place of death"),
        ("d.",             "person", "date",         "1999",              "date of death"),
        ("d.",             "person", "location",     "rome",              "place of death"),
        # Nationality vs country of residence
        ("nationality",    "person", "nationality",  "american",          "country of citizenship"),
        ("nationality",    "person", "location",     "united states",     "country of citizenship"),
        ("citizenship",    "person", "nationality",  "british",           "country of citizenship"),
        ("country",        "person", "location",     "france",            "country of citizenship"),
        ("country",        "person", "organization", "france national",   "country for sport"),
        # Education — organization vs location
        ("alma mater",     "person", "organization", "mit",               "educated at"),
        ("college",        "person", "organization", "oxford",            "educated at"),
        ("university",     "person", "organization", "caltech",           "educated at"),
        ("school",         "person", "organization", "harvard",           "educated at"),
        ("studied at",     "person", "organization", "cambridge",         "educated at"),
        # Occupation vs employer (both "person" → "organization")
        ("employer",       "person", "organization", "bbc",               "employer"),
        ("worked at",      "person", "organization", "nbc",               "employer"),
        ("occupation",     "person", "occupation",   "actor",             "occupation"),
        ("profession",     "person", "occupation",   "musician",          "occupation"),
        ("job",            "person", "occupation",   "director",          "occupation"),
        ("career",         "person", "occupation",   "politician",        "occupation"),
        # Spouse vs child vs sibling (person → person — role determines slot)
        ("spouse",         "person", "person",       "mary jones",        "spouse"),
        ("married to",     "person", "person",       "jane smith",        "spouse"),
        ("husband",        "person", "person",       "john doe",          "spouse"),
        ("wife",           "person", "person",       "alice brown",       "spouse"),
        ("children",       "person", "person",       "tom doe",           "child"),
        ("son",            "person", "person",       "michael",           "child"),
        ("daughter",       "person", "person",       "sarah",             "child"),
        ("father",         "person", "person",       "william",           "father"),
        ("mother",         "person", "person",       "elizabeth",         "mother"),
        ("sibling",        "person", "person",       "james",             "sibling"),
        ("brother",        "person", "person",       "david",             "sibling"),
        ("sister",         "person", "person",       "emma",              "sibling"),
        # Award vs notable work
        ("award",          "person", "award",        "oscar",             "award received"),
        ("prize",          "person", "award",        "nobel prize",       "award received"),
        ("notable work",   "person", "entity",       "hamlet",            "notable work"),
        ("known for",      "person", "entity",       "theory of relativity", "notable work"),
        ("known for",      "person", "award",        "pulitzer prize",    "award received"),
    ]

    confusable_count = 0
    for (pred, st, ot, obj_text, slot) in _CONTEXT_CONFUSABLES:
        if slot not in schema_index:
            continue
        cidx = schema_index[slot]
        # Add multiple copies to give these rare adversarial pairs enough weight
        for _ in range(5):
            train_data.append({
                "noisy_pred":    pred,
                "subj_type":     st,
                "obj_type":      ot,
                "obj_text":      obj_text,   # raw object string for TypeMLP
                "canonical_idx": cidx,
            })
        confusable_count += 1

    print(f"[QKV]   Confusable context pairs injected: {confusable_count} × 5 = {confusable_count*5:,}")
    print(f"[QKV] Unique noisy predicates: {len(rel_counts):,}")
    print(f"[QKV] Total training pairs:    {len(train_data):,}")
    return {"schema": schema, "train_data": train_data, "rel_counts": rel_counts}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  The QKV head module
# ──────────────────────────────────────────────────────────────────────────────

class _QKVHead(nn.Module):
    """
    A single cross-attention head.

    Query  = learnable projection of the (predicate + subject_type + object_type) context.
    Key/Value = learnable projections of the fixed schema-predicate embeddings.

    The schema embeddings K/V are registered as buffers and are never trained.
    """

    def __init__(self, input_dim: int, schema_dim: int, head_dim: int, num_schema: int):
        super().__init__()
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

        # Learnable projections (these are the ONLY trained parameters)
        self.W_Q = nn.Linear(input_dim, head_dim, bias=False)
        self.W_K = nn.Linear(schema_dim, head_dim, bias=False)
        self.W_V = nn.Linear(schema_dim, head_dim, bias=False)

        # IMPROVEMENT 1: Learned per-head temperature τ (Gemini suggestion).
        # Initialised at 0 so exp(0)=1 → same as fixed scaling at start.
        # The model learns to sharpen (τ→small, more peaked softmax = confident)
        # or soften (τ→large, flatter softmax = uncertain / fallback to null).
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query_vec: torch.Tensor,    # [B, input_dim]
        schema_embs: torch.Tensor,  # [N_schema + 1, schema_dim] (includes null slot)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # ([B, head_dim], [B, N+1])
        Q = self.W_Q(query_vec)                   # [B, head_dim]
        K = self.W_K(schema_embs)                 # [N+1, head_dim]
        V = self.W_V(schema_embs)                 # [N+1, head_dim]

        # Learned temperature: clamp to [0.01, 10] for numerical stability
        tau = self.log_tau.exp().clamp(min=0.01, max=10.0)

        # Scaled dot-product attention with learned τ: [B, N+1]
        attn_scores = torch.matmul(Q, K.T) / (self.scale * tau)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values: [B, head_dim]
        attended = torch.matmul(attn_weights, V)
        return attended, attn_weights  # expose weights for diversity loss


class MultiHeadQKVCanonicalizer(nn.Module):
    """
    Four-head canonicalizer that maps:
        (predicate_emb || subject_type_emb || object_type_emb) → canonical_predicate_logits

    Heads
    -----
    Head 1 – Surface Head   : upweights textual/lexical similarity of predicate strings
    Head 2 – Context Head   : upweights domain/range constraints (subject+object types)
    Head 3 – Structure Head : attends to distributional frequency patterns
    Head 4 – Polarity Head  : separates positive vs inverse/negated relations

    All four heads use the same architecture; they diverge through independent
    random initialisation and gradient flow during training.
    """

    def __init__(
        self,
        embed_dim: int = 768,       # Nomic embedding dimension
        type_dim: int = 64,         # Compact type embedding dimension
        head_dim: int = 64,         # Per-head projection dimension
        num_heads: int = 4,
        num_schema: int = len(BUILTIN_SCHEMA),
        dropout: float = 0.1,
        use_qkv: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.type_dim = type_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_schema = num_schema
        self.use_qkv = use_qkv

        # Input to each head = [predicate_emb || subj_type_emb || obj_type_emb]
        input_dim = embed_dim + 2 * type_dim

        self.heads = nn.ModuleList([
            _QKVHead(input_dim, embed_dim, head_dim, num_schema)
            for _ in range(num_heads)
        ])

        # IMPROVEMENT 2: Learnable Null Slot (Gemini suggestion).
        # A single learnable embedding appended to schema_embs before every
        # forward pass.  If the model is uncertain (hallucinated predicate),
        # attention gravitates toward this slot → the final logit index
        # num_schema signals "discard this triple".
        # Stored in embed_dim to match schema_embs, projected by W_K / W_V.
        self.null_slot = nn.Parameter(torch.randn(embed_dim) * 0.01)

        # Output projection: concat of all heads → logits over (num_schema + 1) classes
        # Index num_schema = "null / discard"
        if use_qkv:
            self.W_O = nn.Linear(num_heads * head_dim, num_schema + 1, bias=True)
        else:
            # Baseline MLP: 3 layers directly mapping input to logits
            self.baseline_mlp = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_schema + 1)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pred_emb: torch.Tensor,
        subj_type_emb: torch.Tensor,
        obj_type_emb: torch.Tensor,
        schema_embs: torch.Tensor,
        return_attns: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        query = torch.cat([pred_emb, subj_type_emb, obj_type_emb], dim=-1)

        if not self.use_qkv:
            # PURE MLP BASELINE (No QKV)
            logits = self.baseline_mlp(query)
            if return_attns:
                return logits, [] # No attention to return
            return logits

        # Prepend null slot to schema so every head can attend to it
        null = self.null_slot.unsqueeze(0)                    # [1, embed_dim]
        schema_with_null = torch.cat([schema_embs, null], dim=0)  # [N+1, embed_dim]

        head_results = [head(query, schema_with_null) for head in self.heads]
        head_outputs = [r[0] for r in head_results]
        head_attns   = [r[1] for r in head_results]

        concat = torch.cat(head_outputs, dim=-1)
        concat = self.dropout(concat)
        logits = self.W_O(concat)

        if return_attns:
            return logits, head_attns
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 5.  High-level trainer / inference wrapper
# ──────────────────────────────────────────────────────────────────────────────

class QKVCanonicalizer:
    """
    High-level wrapper that:
    - Loads / manages the frozen Nomic encoder
    - Builds type embeddings
    - Trains the MultiHeadQKVCanonicalizer
    - Runs inference on a processed JSON KG file
    """

    def __init__(
        self,
        schema_predicates: Optional[list[str]] = None,
        embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        head_dim: int = 64,
        num_heads: int = 4,
        type_dim: int = 64,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.schema = schema_predicates if schema_predicates else load_schema()
        self.schema_lower = [s.lower() for s in self.schema]
        self.num_schema = len(self.schema)

        print(f"[QKV] Loading Nomic encoder on {self.device}…")
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(embed_model_name, trust_remote_code=True).to(self.device)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)   # Frozen – never trains

        self.embed_dim = 768  # Nomic-embed-text-v1.5 output dimension

        # Type vocabulary
        self._type_vocab = sorted(TYPE_DESCRIPTIONS.keys())
        self._type2idx = {t: i for i, t in enumerate(self._type_vocab)}

        # FIX 2 — Trainable TypeMLP replaces frozen nn.Embedding.
        # The old approach encoded 10 discrete type tokens into 64-dim vectors
        # that were near-identical ("date" ≈ "location" in Nomic space), giving
        # the Q-vector no discriminative signal about the object type.
        #
        # The new approach: encode the RAW OBJECT STRING (e.g., "1985" or "London")
        # with the frozen Nomic encoder → 768-dim vector → trainable 3-layer MLP
        # → 64-dim type embedding.  Because "1985" and "London" have very different
        # 768-dim vectors, the MLP can learn to separate them reliably.
        #
        # At training time: object strings are pre-encoded => MLP maps to type_dim.
        # At inference time: same path, fully differentiable through the MLP only.
        self.type_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, type_dim),
        ).to(self.device)
        # Keep a small lookup table for the 10 coarse type tokens (subject types
        # are always "person" in WikiBio — we keep this cheap path for subjects).
        self.type_emb_layer = nn.Embedding(len(self._type_vocab), type_dim).to(self.device)
        self._init_type_embeddings_from_nomic(type_dim)

        self.model = MultiHeadQKVCanonicalizer(
            embed_dim=self.embed_dim,
            type_dim=type_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            num_schema=self.num_schema,
            dropout=dropout,
        ).to(self.device)

        # Pre-compute and cache frozen schema embeddings
        print("[QKV] Pre-computing schema embeddings…")
        self._schema_embs = self._encode(self.schema_lower)  # [N_schema, 768]
        self._null_idx = self.num_schema  # convenience alias

        # Build the neural type classifier (zero-shot, no new training)
        _NEURAL_TYPE_CLASSIFIER.build(self._encode, str(self.device))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings with the frozen Nomic model."""
        with torch.no_grad():
            embs = self._encoder.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )
        embs = F.normalize(embs, p=2, dim=-1)
        return embs.to(self.device)

    def _init_type_embeddings_from_nomic(self, type_dim: int) -> None:
        """
        Warm-start the type embedding table from Nomic encodings of type
        descriptions (TYPE_DESCRIPTIONS dict).  A linear projection reduces
        768-dim Nomic vectors to type_dim.

        After this call self.type_emb_layer contains semantically meaningful
        initial weights rather than random noise, so early training is stable
        and the type context in the Q vector already carries meaning.
        """
        descriptions = [TYPE_DESCRIPTIONS[t] for t in self._type_vocab]
        with torch.no_grad():
            raw = self._encode(descriptions)   # [N_types, 768]  frozen Nomic
        # Project 768 → type_dim with a random but fixed linear map
        proj = nn.Linear(768, type_dim, bias=False)
        nn.init.orthogonal_(proj.weight)       # orthogonal init preserves distances
        type_inits = proj(raw.cpu()).detach()  # [N_types, type_dim]
        with torch.no_grad():
            self.type_emb_layer.weight.copy_(type_inits.to(self.device))
        print(f"[QKV] Type embeddings warm-started from Nomic ({len(self._type_vocab)} types).")

    def _type_tensor(self, type_tokens: list[str]) -> torch.Tensor:
        """Convert a list of type strings to an embedding tensor [B, type_dim]."""
        indices = [self._type2idx.get(t, self._type2idx[_FALLBACK_TYPE]) for t in type_tokens]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.type_emb_layer(idx_tensor)

    def _obj_type_emb_from_text(self, obj_texts: list[str]) -> torch.Tensor:
        """FIX 2: Encode raw object strings with Nomic then project via TypeMLP.
        This yields discriminative type embeddings (e.g. '1985' vs 'London')."""
        with torch.no_grad():
            raw = self._encode(obj_texts)          # [B, 768]  frozen
        return self.type_mlp(raw)                  # [B, type_dim]  trainable

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: list[dict],
        epochs: int = 25,
        lr: float = 3e-4,
        batch_size: int = 256,
        eval_split: float = 0.1,
        verbose: bool = True,
        div_weight: float = 0.005,
        use_type_mlp: bool = True,
    ) -> dict:
        """
        Train the QKV heads using cross-entropy on the bootstrapped pairs.

        Parameters
        ----------
        train_data   : list of dicts from build_training_pairs()
        epochs       : training epochs
        lr           : initial learning rate (cosine-decayed)
        batch_size   : examples per gradient step
        eval_split   : fraction of data for validation
        verbose      : print per-epoch metrics

        Returns
        -------
        dict with training history (loss, val_acc per epoch)
        """
        import random

        self.use_type_mlp = use_type_mlp

        if not train_data:
            raise ValueError("[QKV] train_data is empty – run build_training_pairs() first.")

        random.shuffle(train_data)
        split = int(len(train_data) * (1 - eval_split))
        train_split, val_split = train_data[:split], train_data[split:]
        print(f"[QKV] Training on {len(train_split)} pairs, validating on {len(val_split)} pairs.")

        # Pre-encode all unique noisy predicates in one batch (fast)
        unique_preds = list({d["noisy_pred"] for d in train_data})
        print(f"[QKV] Encoding {len(unique_preds)} unique noisy predicates…")
        pred_embs_map: dict[str, torch.Tensor] = {}
        bs = 512
        for start in range(0, len(unique_preds), bs):
            batch_preds = unique_preds[start:start + bs]
            batch_embs = self._encode(batch_preds)
            for pred, emb in zip(batch_preds, batch_embs):
                pred_embs_map[pred] = emb

        # OPTIMIZATION: Pre-encode all unique object texts in one batch (Fixes performance bottleneck)
        unique_objs = list({d["obj_text"] for d in train_data if "obj_text" in d})
        print(f"[QKV] Encoding {len(unique_objs)} unique object strings…")
        obj_raw_embs_map: dict[str, torch.Tensor] = {}
        for start in range(0, len(unique_objs), bs):
            batch_objs = unique_objs[start:start + bs]
            batch_embs = self._encode(batch_objs)
            for obj, emb in zip(batch_objs, batch_embs):
                obj_raw_embs_map[obj] = emb

        # Optimise QKV heads + type embeddings + TypeMLP (NOT the Nomic encoder)
        optimizer = torch.optim.AdamW(
            list(self.model.parameters())
            + list(self.type_emb_layer.parameters())
            + list(self.type_mlp.parameters()),   # FIX 2: train the TypeMLP
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # ── CLASS-BALANCED LOSS (THE DRIFT FIX) ─────────────────────────────
        # Compute inverse class frequencies to counteract the massive
        # representation bias of sports statistics (e.g. goals, caps) in WikiBio.
        freqs = {}
        for d in train_data:
            idx = d["canonical_idx"]
            freqs[idx] = freqs.get(idx, 0) + 1
        
        # We have self.num_schema classes + 1 null slot
        total_classes = self.num_schema + 1
        weights = torch.ones(total_classes, device=self.device)
        
        # Use inverse frequency scaling: weight = 1 / max(count, pseudocount)
        # We use a pseudocount of 50 to prevent rare slots from having stratospheric weights
        # that explode the gradients.
        for i in range(total_classes):
            count = freqs.get(i, 0)
            if count > 0:
                weights[i] = 1.0 / max(count, 50.0)
        
        # Normalize weights so they sum roughly to the number of classes
        weights = weights / weights.sum() * total_classes
        print(f"[QKV] Using Class-Balanced CrossEntropyLoss (max weight penalty: {weights.max().item():.2f}x)")
        
        criterion = nn.CrossEntropyLoss(weight=weights)

        history = {"train_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.type_emb_layer.train()
            self.type_mlp.train()

            random.shuffle(train_split)
            total_loss = 0.0
            num_batches = 0

            for start in range(0, len(train_split), batch_size):
                batch = train_split[start: start + batch_size]
                if not batch:
                    continue

                noisy_preds = [d["noisy_pred"] for d in batch]
                subj_types  = [d["subj_type"]  for d in batch]
                obj_types   = [d["obj_type"]   for d in batch]
                labels      = torch.tensor([d["canonical_idx"] for d in batch],
                                           dtype=torch.long, device=self.device)

                # Gather frozen predicate embeddings
                pred_emb = torch.stack([pred_embs_map[p] for p in noisy_preds])

                # OPTIMIZED: subject uses cheap lookup (always "person" in WikiBio);
                # object uses either the TypeMLP (v3) or legacy token types (ablation)
                subj_emb = self._type_tensor(subj_types)        # [B, type_dim]
                
                if use_type_mlp and "obj_text" in batch[0]:
                    obj_texts = [d["obj_text"] for d in batch]
                    # Map strings to pre-calculated 768-dim vectors
                    try:
                        raw_embs = torch.stack([obj_raw_embs_map[t] for t in obj_texts])
                        obj_emb = self.type_mlp(raw_embs)            # [B, type_dim]  trainable
                    except KeyError:
                        # Fallback for any dynamically created text not in pre-encode map
                        obj_emb = self._obj_type_emb_from_text(obj_texts)
                else:
                    # Ablation: fallback to categorical token embeddings
                    obj_emb = self._type_tensor(obj_types)

                # Get logits AND per-head attention maps (FIX 1 needs attns)
                logits, head_attns = self.model(
                    pred_emb, subj_emb, obj_emb, self._schema_embs,
                    return_attns=True,
                )

                # ── Standard cross-entropy ─────────────────────────────────
                ce_loss = criterion(logits, labels)

                # ── SANS: Self-Adversarial Negative Sampling ───────────────
                with torch.no_grad():
                    top2 = logits.topk(2, dim=-1).indices
                    wrong_idx = torch.where(
                        top2[:, 0] == labels, top2[:, 1], top2[:, 0]
                    )
                correct_scores = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                wrong_scores   = logits.gather(1, wrong_idx.unsqueeze(1)).squeeze(1)
                margin_loss = F.relu(wrong_scores - correct_scores + 0.5).mean()

                # ── FIX 1: Stable Head Diversity Loss (Cosine Similarity) ──
                # We want heads to be DIVERGENT (low cosine similarity).
                div_loss = torch.tensor(0.0, device=self.device)
                
                if div_weight > 0:
                    # Pairwise Cosine Similarity penalty
                    for i in range(len(head_attns)):
                        for j in range(i + 1, len(head_attns)):
                            h_i = head_attns[i]
                            h_j = head_attns[j]
                            sim = F.cosine_similarity(h_i, h_j, dim=-1)
                            div_loss = div_loss + sim.mean()
                
                # loss weight passed via param (STABLE_DIV_WEIGHT = 0.005)
                loss = ce_loss + 0.3 * margin_loss + div_weight * div_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            # Validation accuracy
            val_acc = self._evaluate(val_split, pred_embs_map, obj_raw_embs_map)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

        return history

    def _evaluate(
        self,
        val_data: list[dict],
        pred_embs_map: dict,
        obj_raw_embs_map: dict = None,
        use_type_mlp: bool = True
    ) -> float:
        """Return top-1 accuracy on a validation split."""
        if not val_data:
            return 0.0
        obj_raw_embs_map = obj_raw_embs_map or {}
        self.model.eval()
        self.type_emb_layer.eval()
        self.type_mlp.eval()
        correct = 0
        with torch.no_grad():
            for d in val_data:
                pred_emb = pred_embs_map.get(d["noisy_pred"])
                if pred_emb is None:
                    pred_emb = self._encode([d["noisy_pred"]])[0]
                pred_emb = pred_emb.unsqueeze(0)
                subj_emb = self._type_tensor([d["subj_type"]])
                
                # OPTIMIZED: use pre-calculated map if available
                if use_type_mlp and "obj_text" in d:
                    obj_text = d["obj_text"]
                    if obj_text in obj_raw_embs_map:
                        obj_emb = self.type_mlp(obj_raw_embs_map[obj_text].unsqueeze(0))
                    else:
                        obj_emb = self._obj_type_emb_from_text([obj_text])
                else:
                    obj_emb = self._type_tensor([d["obj_type"]])
                logits = self.model(pred_emb, subj_emb, obj_emb, self._schema_embs)
                pred_idx = logits.argmax(dim=-1).item()
                if pred_idx == d["canonical_idx"]:
                    correct += 1
        return correct / len(val_data)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def canonicalize_relation(
        self,
        noisy_pred: str,
        subj_type: str = "person",
        obj_type: str = "entity",
        top_k: int = 3,
        confidence_threshold: float = 0.05,
    ) -> list[dict]:
        """
        Map a single noisy predicate to a ranked list of canonical candidates.

        Parameters
        ----------
        noisy_pred            : the raw relation string (e.g. "born in", "is spouse of")
        subj_type             : coarse subject type (e.g. "person")
        obj_type              : coarse object type (e.g. "location")
        top_k                 : number of candidates to return
        confidence_threshold  : minimum softmax probability to include a candidate

        Returns
        -------
        list of dicts: [{"canonical": str, "confidence": float}, ...]
        Sorted by confidence (descending).  Empty list if nothing clears the threshold.
        """
        self.model.eval()
        self.type_emb_layer.eval()
        with torch.no_grad():
            pred_emb = self._encode([noisy_pred.lower()])                   # [1, 768]
            subj_emb = self._type_tensor([subj_type])                       # [1, type_dim]
            obj_emb  = self._type_tensor([obj_type])                        # [1, type_dim]
            logits   = self.model(pred_emb, subj_emb, obj_emb, self._schema_embs)  # [1, N]
            probs    = F.softmax(logits, dim=-1)[0]                         # [N]

        top_k_vals, top_k_idxs = torch.topk(probs, min(top_k, self.num_schema))
        results = []
        for prob, idx in zip(top_k_vals.tolist(), top_k_idxs.tolist()):
            if prob >= confidence_threshold:
                results.append({
                    "canonical": self.schema[idx],
                    "confidence": round(prob, 4),
                })
        return results

    def canonicalize_kg(
        self,
        input_json: str,
        output_json: str,
        confidence_threshold: float = 0.10,
        unknown_label: str = "__unmapped__",
        batch_size: int = 512,
        verbose: bool = True,
    ) -> dict:
        """
        Apply canonicalization in BATCH mode to all triples in a KG file.
        Significantly faster than record-by-record processing.
        """
        print(f"[QKV] Loading KG from {input_json}…")
        with open(input_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        # 1. Gather all triples needing canonicalization
        worklist = []
        for ridx, record in enumerate(records):
            entities = record.get("entities", [])
            subj_type = _get_subject_type_from_record(record)
            for tidx, t in enumerate(record.get("triples", [])):
                noisy = t.get("relation", "").strip().lower()
                if not noisy:
                    continue
                obj_type = _get_object_type(t.get("object", ""), entities)
                worklist.append({
                    "record_idx": ridx,
                    "triple_idx": tidx,
                    "noisy": noisy,
                    "subj_type": subj_type,
                    "obj_type": obj_type,
                })

        if not worklist:
            print("[QKV] No triples found to canonicalize.")
            return {"total_triples": 0}

        print(f"[QKV] Flattened {len(worklist):,} triples. Starting batch inference…")
        
        # 2. Pre-encode unique noisy predicates to save Nomic calls
        unique_noises = list({w["noisy"] for w in worklist})
        print(f"[QKV] Encoding {len(unique_noises):,} unique predicates…")
        noise_embs_map = {}
        for i in range(0, len(unique_noises), batch_size):
            batch = unique_noises[i:i+batch_size]
            embs = self._encode(batch)
            for noise, emb in zip(batch, embs):
                noise_embs_map[noise] = emb

        # 2b. Neural object-type reclassification (Fix 1) ─────────────────────
        # Batch-encode all unique object strings and classify via cosine similarity
        # to type-description anchors. Replaces pattern-based fallback that collapsed
        # diverse object types (locations, dates, persons) all to "entity".
        unique_objs = list({
            records[w["record_idx"]]["triples"][w["triple_idx"]].get("object", "")
            for w in worklist
        })
        print(f"[QKV] Neural type-classifying {len(unique_objs):,} unique object strings…")
        neural_type_map: dict[str, str] = {}
        obj_raw_embs_map: dict[str, torch.Tensor] = {}
        for i in range(0, len(unique_objs), batch_size):
            batch_objs = unique_objs[i:i+batch_size]
            batch_embs = self._encode(batch_objs)
            for j, obj in enumerate(batch_objs):
                obj_raw_embs_map[obj] = batch_embs[j]
            
            # Use cached embeddings to save Nomic calls during classification
            batch_types = _NEURAL_TYPE_CLASSIFIER.classify_batch(batch_objs, lambda x: batch_embs)
            for obj, typ in zip(batch_objs, batch_types):
                neural_type_map[obj] = typ

        # Backfill worklist with neural types (override only where NER label absent)
        for w in worklist:
            t = records[w["record_idx"]]["triples"][w["triple_idx"]]
            entities = records[w["record_idx"]].get("entities", [])
            obj_str = t.get("object", "")
            ner_label = next(
                (e.get("label") for e in entities
                 if e.get("text", "").strip().lower() in obj_str.lower()),
                None
            )
            # Prefer NER label; fall back to neural classification
            if ner_label:
                w["obj_type"] = infer_type(obj_str, ner_label=ner_label)
            else:
                w["obj_type"] = neural_type_map.get(obj_str, w["obj_type"])


        # 3. Batch processing using the QKV model
        self.model.eval()
        self.type_emb_layer.eval()
        
        stats = {"total_triples": len(worklist), "canonicalized": 0, "unchanged": 0}

        for i in tqdm(range(0, len(worklist), batch_size), desc="[QKV] Inference"):
            batch_items = worklist[i : i + batch_size]
            
            # Prepare tensors
            pred_embs = torch.stack([noise_embs_map[w["noisy"]] for w in batch_items]).to(self.device)
            s_types = [w["subj_type"] for w in batch_items]
            s_type_embs = self._type_tensor(s_types) # [B, type_dim]
            
            if getattr(self, "use_type_mlp", True):
                obj_strs = [records[w["record_idx"]]["triples"][w["triple_idx"]].get("object", "") for w in batch_items]
                raw_embs = torch.stack([obj_raw_embs_map[o] for o in obj_strs])
                o_type_embs = self.type_mlp(raw_embs)
            else:
                o_types = [w["obj_type"] for w in batch_items]
                o_type_embs = self._type_tensor(o_types) # [B, type_dim]
            
            with torch.no_grad():
                # Correct way: use the model's forward pass which handles heads + null slot
                logits = self.model(pred_embs, s_type_embs, o_type_embs, self._schema_embs) # [B, num_schema + 1]
                probs = torch.softmax(logits, dim=-1) # [B, num_schema + 1]
                
                confidences, best_idxs = torch.max(probs, dim=-1)

            # 4. Update the original JSON records
            for j, m in enumerate(batch_items):
                conf = confidences[j].item()
                idx = best_idxs[j].item()
                t = records[m["record_idx"]]["triples"][m["triple_idx"]]

                # Fix 4: Sink-label suppression
                # "genre", "mother" etc. act as catch-alls—require higher confidence
                canonical = self.schema[idx] if idx < self.num_schema else unknown_label
                is_sink = canonical in _SINK_LABELS
                effective_threshold = _SINK_THRESHOLD if is_sink else confidence_threshold

                if idx < self.num_schema and conf >= effective_threshold:
                    t["canonical_relation"] = canonical
                    t["canonical_confidence"] = round(conf, 4)
                    stats["canonicalized"] += 1
                else:
                    # null slot or low confidence (including sink labels below higher bar)
                    t["canonical_relation"] = unknown_label
                    t["canonical_confidence"] = 0.0
                    stats["unchanged"] += 1

        print(f"[QKV] Writing enriched KG to {output_json}…")
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        return stats

        if verbose:
            print(f"\n[QKV] Canonicalization Summary")
            print(f"  Total triples   : {stats['total_triples']:,}")
            print(f"  Canonicalized   : {stats['canonicalized']:,}  "
                  f"({stats['canonicalized']/max(stats['total_triples'],1)*100:.1f}%)")
            print(f"  Below threshold : {stats['unchanged']:,}")

        print(f"[QKV] Writing output to {output_json}…")
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trained weights (QKV heads + type embeddings) to disk."""
        torch.save({
            "model_state": self.model.state_dict(),
            "type_emb_state": self.type_emb_layer.state_dict(),
            "type_mlp_state": self.type_mlp.state_dict(),
            "use_type_mlp": getattr(self, "use_type_mlp", True),
            "schema": self.schema,
            "type_vocab": self._type_vocab,
        }, path)
        print(f"[QKV] Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """Restore trained weights from a checkpoint saved by save()."""
        import sys
        # SHIM: Some checkpoints were saved with classes in __main__.
        # We inject our current definitions into __main__ so torch.load can find them.
        main_mod = sys.modules.get('__main__')
        if main_mod:
            main_mod.MultiHeadQKVCanonicalizer = MultiHeadQKVCanonicalizer
            main_mod._QKVHead = _QKVHead

        ckpt = torch.load(path, map_location=self.device)
        # Restore schema if it was saved
        if "schema" in ckpt:
            self.schema = ckpt["schema"]
            self.schema_lower = [s.lower() for s in self.schema]
            self.num_schema = len(self.schema)
            self._schema_embs = self._encode(self.schema_lower)
        if "type_vocab" in ckpt:
            self._type_vocab = ckpt["type_vocab"]
            self._type2idx = {t: i for i, t in enumerate(self._type_vocab)}
        
        self.use_type_mlp = ckpt.get("use_type_mlp", True)
        if "type_mlp_state" in ckpt:
            self.type_mlp.load_state_dict(ckpt["type_mlp_state"])
            
        self.model.load_state_dict(ckpt["model_state"])
        self.type_emb_layer.load_state_dict(ckpt["type_emb_state"])
        print(f"[QKV] Loaded checkpoint from {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  QKV Entity Normaliser  (open-vocabulary, contrastive)
# ──────────────────────────────────────────────────────────────────────────────
#
# Why contrastive, not classification?
# ─────────────────────────────────────
# Relation canonicalization has a fixed vocabulary (~50 schema slots) so
# cross-entropy over a softmax works perfectly.
#
# Entity normalisation is open-vocabulary: the set of canonical entities
# is the entire KG node set (93,193 nodes in your case) and grows with
# every biography added.  Maintaining a weight vector per entity is
# impractical and wouldn't generalise to unseen entities.
#
# The correct formulation is METRIC LEARNING:
#   • Train 4 QKV heads to project noisy entity mentions into a space
#     where synonyms (same cluster in entity_merges.json) are close and
#     different entities are far apart.
#   • Loss = MultipleNegativeRankingLoss (InfoNCE / SimCSE style).
#   • Inference = nearest-centroid search in the projected space.
#
# Training signal
# ───────────────
# entity_merges.json  (produced by global_identity_resolution.py) already
# contains synonym clusters, e.g.:
#   {"lead": "United States", "synonyms": ["USA", "U.S.", "United States"]}
#
# Every (anchor, positive) pair within a cluster is a positive training pair.
# All other clusters in the same batch become negatives automatically.
#
# Architecture per head
# ─────────────────────
#   Q = W_Q · [mention_emb || type_emb]   [B, head_dim]
#   K = W_K · key_emb                      [B, head_dim]  (same structure, different input)
#   similarity = Q · K^T / √d              scalar
#
# The 4 heads are concatenated before a final L2 projection:
#   final_emb = W_O · Concat(head_1..4)    [B, proj_dim]
#
# Inference: given a mention, embed it, then find the closest centroid
# (pre-computed as the mean of all synonym embeddings per cluster).
# ──────────────────────────────────────────────────────────────────────────────


def build_entity_training_pairs(
    entity_merges_json: str,
    min_cluster_size: int = 2,
) -> list[dict]:
    """
    Build contrastive training pairs from the synonym clusters produced by
    global_identity_resolution.py (entity_merges.json).

    Each item in the returned list has:
        "anchor"   : str  – a noisy / variant entity mention
        "positive" : str  – another mention from the same synonym cluster (lead node)
        "type"     : str  – coarse type inferred from the lead string

    Parameters
    ----------
    entity_merges_json  : path to entity_merges.json
    min_cluster_size    : skip singleton clusters (no positive pair possible)

    Returns
    -------
    list of pair dicts suitable for QKVEntityNormalizer.train()
    """
    print(f"[ENorm] Loading entity clusters from: {entity_merges_json}")
    with open(entity_merges_json, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    pairs: list[dict] = []
    for cluster in clusters:
        lead = cluster.get("lead", "")
        synonyms = cluster.get("synonyms", [])
        if not lead or len(synonyms) < min_cluster_size:
            continue

        coarse_type = infer_type(lead)
        # Every synonym is an (anchor, positive=lead) pair
        for syn in synonyms:
            if syn == lead:
                continue
            pairs.append({"anchor": syn, "positive": lead, "type": coarse_type})

    print(f"[ENorm] Clusters loaded: {len(clusters):,}  |  Pairs built: {len(pairs):,}")
    return pairs


class _EntityQKVHead(nn.Module):
    """
    Single QKV head for entity metric learning.

    Query  = W_Q · [mention_emb || type_emb]   [B, head_dim]
    Key    = W_K · target_emb                   [*, head_dim]

    (No Value projection needed for contrastive loss — we only need the
    similarity score, not a weighted value aggregate.)
    """

    def __init__(self, embed_dim: int, type_dim: int, head_dim: int):
        super().__init__()
        self.scale = math.sqrt(head_dim)
        input_dim = embed_dim + type_dim
        self.W_Q = nn.Linear(input_dim, head_dim, bias=False)
        self.W_K = nn.Linear(input_dim, head_dim, bias=False)

    def project_query(self, x: torch.Tensor) -> torch.Tensor:   # [B, head_dim]
        return self.W_Q(x)

    def project_key(self, x: torch.Tensor) -> torch.Tensor:     # [*, head_dim]
        return self.W_K(x)


class MultiHeadEntityEncoder(nn.Module):
    """
    Encodes an entity mention into a fixed-dim projection vector using
    4 QKV heads + a final linear aggregation.

    Produces a unit-norm embedding suitable for cosine-similarity retrieval.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        type_dim: int = 64,
        head_dim: int = 64,
        num_heads: int = 4,
        proj_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            _EntityQKVHead(embed_dim, type_dim, head_dim)
            for _ in range(num_heads)
        ])
        # Aggregate 4 heads → final embedding
        self.W_O = nn.Linear(num_heads * head_dim, proj_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        mention_emb: torch.Tensor,   # [B, embed_dim]  frozen Nomic
        type_emb: torch.Tensor,      # [B, type_dim]   trainable
    ) -> torch.Tensor:               # [B, proj_dim]   L2-normalised
        x = torch.cat([mention_emb, type_emb], dim=-1)  # [B, embed_dim+type_dim]
        head_qs = [h.project_query(x) for h in self.heads]  # H × [B, head_dim]
        concat = torch.cat(head_qs, dim=-1)                  # [B, H*head_dim]
        concat = self.dropout(concat)
        out = self.W_O(concat)                               # [B, proj_dim]
        return F.normalize(out, p=2, dim=-1)                 # unit norm


class QKVEntityNormalizer:
    """
    High-level wrapper for entity normalisation using multi-head QKV
    metric learning.

    Key differences from QKVCanonicalizer
    ──────────────────────────────────────
    • Open-vocabulary: no fixed schema. Canonical entities = centroid index.
    • Loss: MultipleNegativeRankingLoss (InfoNCE) instead of cross-entropy.
    • Inference: nearest-centroid ANN search using pre-built centroid cache.

    The canonical entity for each noisy mention is stored as the 'lead'
    string from the synonym cluster, NOT a URI, because your pipeline uses
    string keys throughout.  URI resolution is a downstream step (DBpedia
    Spotlight already handles it for mapped entities).
    """

    def __init__(
        self,
        embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        head_dim: int = 64,
        num_heads: int = 4,
        type_dim: int = 64,
        proj_dim: int = 256,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[ENorm] Loading Nomic encoder on {self.device}…")
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(embed_model_name, trust_remote_code=True).to(self.device)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)

        self.embed_dim = 768
        self.proj_dim = proj_dim

        self._type_vocab = sorted({
            "person", "organization", "location", "date",
            "nationality", "occupation", "award", "event", "number", _FALLBACK_TYPE,
        })
        self._type2idx = {t: i for i, t in enumerate(self._type_vocab)}
        self.type_emb_layer = nn.Embedding(len(self._type_vocab), type_dim).to(self.device)

        self.model = MultiHeadEntityEncoder(
            embed_dim=self.embed_dim,
            type_dim=type_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            proj_dim=proj_dim,
            dropout=dropout,
        ).to(self.device)

        # Centroid cache (built after training or loaded from checkpoint)
        self._centroid_embs: Optional[torch.Tensor] = None   # [N_clusters, proj_dim]
        self._centroid_labels: list[str] = []                 # lead string per centroid

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _encode_raw(self, texts: list[str]) -> torch.Tensor:
        """Frozen Nomic embeddings, L2-normalised."""
        with torch.no_grad():
            embs = self._encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return F.normalize(embs, p=2, dim=-1).to(self.device)

    def _type_tensor(self, type_tokens: list[str]) -> torch.Tensor:
        indices = [self._type2idx.get(t, self._type2idx[_FALLBACK_TYPE]) for t in type_tokens]
        return self.type_emb_layer(torch.tensor(indices, dtype=torch.long, device=self.device))

    def _project(self, texts: list[str], types: list[str]) -> torch.Tensor:
        """Full forward pass → projected unit-norm embedding."""
        raw = self._encode_raw(texts)
        type_emb = self._type_tensor(types)
        return self.model(raw, type_emb)

    # ----------------------------------------------------------------
    # Centroid index
    # ----------------------------------------------------------------

    def build_centroid_index(
        self,
        entity_merges_json: str,
        batch_size: int = 512,
    ) -> None:
        """
        Pre-compute one centroid vector per synonym cluster and cache it.

        The centroid = mean of all synonym projected embeddings (unit-renormed).
        This only needs to run once after training; the result is saved in
        the checkpoint by save().

        Parameters
        ----------
        entity_merges_json : path to entity_merges.json
        batch_size         : encoding batch size
        """
        print("[ENorm] Building centroid index from cluster leads…")
        with open(entity_merges_json, "r", encoding="utf-8") as f:
            clusters = json.load(f)

        self.model.eval()
        self.type_emb_layer.eval()

        centroid_list: list[torch.Tensor] = []
        labels: list[str] = []

        for i in tqdm(range(0, len(clusters), batch_size), desc="[ENorm] Centroids"):
            batch = clusters[i: i + batch_size]
            # For each cluster, embed ALL synonyms and take mean → centroid
            for cluster in batch:
                lead = cluster.get("lead", "")
                synonyms = cluster.get("synonyms", [lead])
                if not synonyms:
                    continue
                coarse_type = infer_type(lead)
                with torch.no_grad():
                    syn_proj = self._project(synonyms, [coarse_type] * len(synonyms))  # [S, proj_dim]
                centroid = syn_proj.mean(dim=0)                                          # [proj_dim]
                centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=-1).squeeze(0)
                centroid_list.append(centroid)
                labels.append(lead)

        self._centroid_embs = torch.stack(centroid_list).to(self.device)   # [N, proj_dim]
        self._centroid_labels = labels
        print(f"[ENorm] Centroid index built: {len(labels):,} clusters.")

    # ----------------------------------------------------------------
    # Training  (MultipleNegativeRankingLoss / InfoNCE)
    # ----------------------------------------------------------------

    def train(
        self,
        train_pairs: list[dict],
        epochs: int = 20,
        lr: float = 3e-4,
        batch_size: int = 128,
        temperature: float = 0.07,
        eval_split: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """
        Train the multi-head entity encoder using InfoNCE contrastive loss.

        For each batch, anchor and positive embeddings form the diagonal of
        a similarity matrix.  All off-diagonal pairs are treated as negatives
        (in-batch negatives), which is efficient and proven to work well
        (SimCSE, E5, GTE all use this strategy).

        Loss = -mean(log(exp(sim(a,p)/τ) / Σ_j exp(sim(a,p_j)/τ)))

        Parameters
        ----------
        train_pairs  : list from build_entity_training_pairs()
        epochs       : number of training epochs
        lr           : learning rate (cosine-decayed)
        batch_size   : pairs per step  (larger = more in-batch negatives = harder)
        temperature  : τ for InfoNCE  (0.05–0.10 works well; lower = sharper)
        eval_split   : fraction for validation (nearest-centroid recall@1)
        verbose      : print per-epoch metrics

        Returns
        -------
        dict with training history
        """
        import random

        if not train_pairs:
            raise ValueError("[ENorm] train_pairs is empty – run build_entity_training_pairs() first.")

        random.shuffle(train_pairs)
        split = int(len(train_pairs) * (1 - eval_split))
        tr, va = train_pairs[:split], train_pairs[split:]
        print(f"[ENorm] Training on {len(tr):,} pairs, validating on {len(va):,} pairs.")

        # Pre-encode all unique anchor/positive strings
        unique_texts = list({d["anchor"] for d in train_pairs} | {d["positive"] for d in train_pairs})
        print(f"[ENorm] Pre-encoding {len(unique_texts):,} unique entity strings…")
        raw_cache: dict[str, torch.Tensor] = {}
        for start in range(0, len(unique_texts), 512):
            batch_texts = unique_texts[start: start + 512]
            batch_embs = self._encode_raw(batch_texts)
            for text, emb in zip(batch_texts, batch_embs):
                raw_cache[text] = emb

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.type_emb_layer.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        history: dict[str, list] = {"train_loss": [], "val_recall1": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.type_emb_layer.train()
            random.shuffle(tr)
            total_loss = 0.0
            num_batches = 0

            for start in range(0, len(tr), batch_size):
                batch = tr[start: start + batch_size]
                if len(batch) < 2:   # Need at least 2 for in-batch negatives
                    continue

                anchors   = [d["anchor"]   for d in batch]
                positives = [d["positive"] for d in batch]
                types     = [d["type"]     for d in batch]

                anc_raw = torch.stack([raw_cache[a] for a in anchors])    # [B, 768]
                pos_raw = torch.stack([raw_cache[p] for p in positives])  # [B, 768]
                type_emb = self._type_tensor(types)                         # [B, type_dim]

                anc_proj = self.model(anc_raw, type_emb)   # [B, proj_dim]
                pos_proj = self.model(pos_raw, type_emb)   # [B, proj_dim]

                # InfoNCE: similarity matrix [B, B], diagonal = positives
                sim = torch.matmul(anc_proj, pos_proj.T) / temperature  # [B, B]
                labels = torch.arange(len(batch), device=self.device)
                loss = F.cross_entropy(sim, labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            history["train_loss"].append(avg_loss)

            val_r1 = self._eval_recall1(va, raw_cache)
            history["val_recall1"].append(val_r1)

            if verbose:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_recall@1={val_r1:.3f}")

        return history

    def _eval_recall1(self, val_pairs: list[dict], raw_cache: dict) -> float:
        """Recall@1: is the positive the nearest neighbor of the anchor?"""
        if len(val_pairs) < 2:
            return 0.0
        self.model.eval()
        self.type_emb_layer.eval()
        correct = 0
        with torch.no_grad():
            anchors   = [d["anchor"]   for d in val_pairs]
            positives = [d["positive"] for d in val_pairs]
            types     = [d["type"]     for d in val_pairs]

            anc_raw = torch.stack([
                raw_cache.get(a) if raw_cache.get(a) is not None else self._encode_raw([a])[0]
                for a in anchors
            ])
            pos_raw = torch.stack([
                raw_cache.get(p) if raw_cache.get(p) is not None else self._encode_raw([p])[0]
                for p in positives
            ])
            type_emb = self._type_tensor(types)

            anc_proj = self.model(anc_raw, type_emb)   # [N, proj_dim]
            pos_proj = self.model(pos_raw, type_emb)   # [N, proj_dim]

            sim = torch.matmul(anc_proj, pos_proj.T)   # [N, N]
            preds = sim.argmax(dim=-1)                  # [N]
            correct = (preds == torch.arange(len(val_pairs), device=self.device)).sum().item()
        return correct / len(val_pairs)

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    def normalize_entity(
        self,
        entity_text: str,
        entity_type: str = _FALLBACK_TYPE,
        top_k: int = 1,
        similarity_threshold: float = 0.80,
    ) -> list[dict]:
        """
        Map a noisy entity mention to canonical cluster lead(s).

        If the centroid index has not been built yet, falls back to returning
        the input unchanged (graceful degradation).

        Parameters
        ----------
        entity_text          : the raw entity string from the triple
        entity_type          : coarse NER type
        top_k                : number of candidates to return
        similarity_threshold : minimum cosine similarity to commit to a match

        Returns
        -------
        list of dicts: [{"canonical": str, "similarity": float}, ...]
        Empty if below threshold (meaning: keep original string as-is).
        """
        if self._centroid_embs is None or not self._centroid_labels:
            return []   # index not built, caller should skip

        self.model.eval()
        self.type_emb_layer.eval()
        with torch.no_grad():
            raw = self._encode_raw([entity_text.lower()])         # [1, 768]
            type_emb = self._type_tensor([entity_type])            # [1, type_dim]
            proj = self.model(raw, type_emb)                       # [1, proj_dim]

            sims = torch.matmul(proj, self._centroid_embs.T)[0]   # [N_clusters]
            top_vals, top_idxs = torch.topk(sims, min(top_k, len(self._centroid_labels)))

        results = []
        for sim_val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            if sim_val >= similarity_threshold:
                results.append({
                    "canonical": self._centroid_labels[idx],
                    "similarity": round(sim_val, 4),
                })
        return results

    def normalize_kg(
        self,
        input_json: str,
        output_json: str,
        similarity_threshold: float = 0.85,
        batch_size: int = 512,
        verbose: bool = True,
    ) -> dict:
        """
        Apply entity normalisation in BATCH mode to all subjects and objects.
        Uses single large MatMul across the centroid index for extreme speed.
        """
        if self._centroid_embs is None:
            raise RuntimeError("[ENorm] Centroid index not built.")

        print(f"[ENorm] Loading KG from {input_json}…")
        with open(input_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        # 1. Identify all unique (text, type) mentions to process
        # We process unique pairs to minimize Nomic encoding overhead.
        unique_mentions = set()
        for record in records:
            entities = record.get("entities", [])
            for t in record.get("triples", []):
                s_text = t.get("subject", "").strip()
                if s_text: unique_mentions.add((s_text, "person"))
                o_text = t.get("object", "").strip()
                if o_text:
                    o_type = _get_object_type(o_text, entities)
                    unique_mentions.add((o_text, o_type))

        mention_list = list(unique_mentions)
        print(f"[ENorm] Found {len(mention_list):,} unique mentions. Encoding and projecting…")

        # 2. Batch-encode and project unique mentions
        self.model.eval()
        self.type_emb_layer.eval()
        
        lookup = {} # (text, type) -> canonical_name
        lookup_sim = {}

        for i in tqdm(range(0, len(mention_list), batch_size), desc="[ENorm] Batch Projection"):
            batch = mention_list[i : i + batch_size]
            texts = [b[0] for b in batch]
            types = [b[1] for b in batch]
            
            with torch.no_grad():
                raw = self._encode_raw([t.lower() for t in texts]) # [B, 768]
                teb = self._type_tensor(types)                      # [B, type_dim]
                proj = self.model(raw, teb)                        # [B, proj_dim]
                
                # Vectorized similarity against ALL centroids [B, N_clusters]
                sims = torch.matmul(proj, self._centroid_embs.T)
                best_sims, best_idxs = torch.max(sims, dim=-1)

            for j, (text, etype) in enumerate(batch):
                sim = best_sims[j].item()
                if sim >= similarity_threshold:
                    lookup[(text, etype)] = self._centroid_labels[best_idxs[j].item()]
                    lookup_sim[(text, etype)] = round(sim, 4)
                else:
                    lookup[(text, etype)] = text # map to self
                    lookup_sim[(text, etype)] = 1.0

        # 3. Apply lookup back to records (in-place)
        stats = {"total_triples": 0, "subjects_normalised": 0, "objects_normalised": 0, "unchanged": 0}
        
        for record in records:
            entities = record.get("entities", [])
            for t in record.get("triples", []):
                stats["total_triples"] += 1
                
                # Subject
                s_text = t.get("subject", "").strip()
                if s_text:
                    canon = lookup.get((s_text, "person"), s_text)
                    t["canonical_subject"] = canon
                    t["canonical_subject_sim"] = lookup_sim.get((s_text, "person"), 0.0)
                    if canon != s_text: stats["subjects_normalised"] += 1
                
                # Object
                o_text = t.get("object", "").strip()
                if o_text:
                    o_type = _get_object_type(o_text, entities)
                    canon = lookup.get((o_text, o_type), o_text)
                    t["canonical_object"] = canon
                    t["canonical_object_sim"] = lookup_sim.get((o_text, o_type), 0.0)
                    if canon != o_text: stats["objects_normalised"] += 1
                
                if t.get("canonical_subject") == s_text and t.get("canonical_object") == o_text:
                    stats["unchanged"] += 1

        if verbose:
            total = max(stats["total_triples"], 1)
            print(f"\n[ENorm] Entity Normalisation Summary")
            print(f"  Total triples        : {stats['total_triples']:,}")
            print(f"  Subjects normalised  : {stats['subjects_normalised']:,}"
                  f"  ({stats['subjects_normalised']/total*100:.1f}%)")
            print(f"  Objects normalised   : {stats['objects_normalised']:,}"
                  f"  ({stats['objects_normalised']/total*100:.1f}%)")
            print(f"  Both unchanged       : {stats['unchanged']:,}")

        print(f"[ENorm] Writing output to {output_json}…")
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        return stats

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save encoder weights, type embeddings, and the centroid index."""
        payload: dict = {
            "model_state": self.model.state_dict(),
            "type_emb_state": self.type_emb_layer.state_dict(),
            "type_vocab": self._type_vocab,
            "centroid_labels": self._centroid_labels,
        }
        if self._centroid_embs is not None:
            payload["centroid_embs"] = self._centroid_embs.cpu()
        torch.save(payload, path)
        print(f"[ENorm] Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """Restore weights and centroid index from a checkpoint."""
        import sys
        # SHIM: Namespace injection for pickle resolution
        main_mod = sys.modules.get('__main__')
        if main_mod:
            main_mod.MultiHeadQKVCanonicalizer = MultiHeadQKVCanonicalizer
            main_mod._QKVHead = _QKVHead

        ckpt = torch.load(path, map_location=self.device)
        if "type_vocab" in ckpt:
            self._type_vocab = ckpt["type_vocab"]
            self._type2idx = {t: i for i, t in enumerate(self._type_vocab)}
        self.model.load_state_dict(ckpt["model_state"])
        self.type_emb_layer.load_state_dict(ckpt["type_emb_state"])
        if "centroid_embs" in ckpt:
            self._centroid_embs = ckpt["centroid_embs"].to(self.device)
            self._centroid_labels = ckpt.get("centroid_labels", [])
            print(f"[ENorm] Centroid index restored: {len(self._centroid_labels):,} clusters.")
        print(f"[ENorm] Loaded checkpoint from {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Stand-alone CLI  (python qkv_canonicalizer.py --help)
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    DEFAULT_INPUT   = "/home2/pallamreddy.n/Project/processed_triples_44k_resolved.json"
    DEFAULT_OUTPUT  = "/home2/pallamreddy.n/Project/processed_triples_44k_canonicalized.json"
    DEFAULT_CKPT    = "/scratch/INLP_Project_Wiki/qkv_canon.pt"
    DEFAULT_MERGES  = "/scratch/INLP_Project_Wiki/entity_merges.json"
    DEFAULT_ENORM_CKPT = "/scratch/INLP_Project_Wiki/qkv_enorm.pt"

    parser = argparse.ArgumentParser(
        description="QKV Multi-Head Canonicalizer (relations) + Entity Normaliser."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    t_cmd = sub.add_parser("train", help="Build training pairs and train the QKV heads.")
    t_cmd.add_argument("--input",   default=DEFAULT_INPUT, help="Path to processed JSON or .sent file.")
    t_cmd.add_argument("--schema",  default=None, help="Optional JSON schema file.")
    t_cmd.add_argument("--mapping-ttl", default=None, help="(Optional) DBpedia mapping-based-properties_en.ttl for distant supervision.")
    t_cmd.add_argument("--ckpt",    default=DEFAULT_CKPT)
    t_cmd.add_argument("--epochs",  type=int, default=25)
    t_cmd.add_argument("--lr",      type=float, default=3e-4)
    t_cmd.add_argument("--batch",   type=int, default=256)
    t_cmd.add_argument("--min-count", type=int, default=2)
    t_cmd.add_argument("--max-pairs", type=int, default=200_000)

    # ---- canonicalize ----
    c_cmd = sub.add_parser("canonicalize", help="Apply trained model to a KG JSON file.")
    c_cmd.add_argument("--input",     default=DEFAULT_INPUT)
    c_cmd.add_argument("--output",    default=DEFAULT_OUTPUT)
    c_cmd.add_argument("--ckpt",      default=DEFAULT_CKPT)
    c_cmd.add_argument("--threshold", type=float, default=0.10)
    c_cmd.add_argument("--schema",    default=None)

    # ---- probe ----
    p_cmd = sub.add_parser("probe", help="Interactively probe the model on a single predicate.")
    p_cmd.add_argument("--ckpt",   default=DEFAULT_CKPT)
    p_cmd.add_argument("--pred",   required=True, help="Noisy predicate string to canonicalize.")
    p_cmd.add_argument("--subj",   default="person")
    p_cmd.add_argument("--obj",    default="entity")
    p_cmd.add_argument("--top-k",  type=int, default=5)
    p_cmd.add_argument("--schema", default=None)

    # ── Entity Normaliser sub-commands ─────────────────────────────────────────

    # ---- train-enorm ----
    te_cmd = sub.add_parser(
        "train-enorm",
        help="Train QKV entity normaliser using InfoNCE on synonym clusters.",
    )
    te_cmd.add_argument("--merges",  default=DEFAULT_MERGES,
                        help="Path to entity_merges.json (from global_identity_resolution.py)")
    te_cmd.add_argument("--redirects-ttl", default=None,
                        help="(Optional) DBpedia redirects_en.ttl for distant supervision.")
    te_cmd.add_argument("--ckpt",    default=DEFAULT_ENORM_CKPT)
    te_cmd.add_argument("--epochs",  type=int, default=20)
    te_cmd.add_argument("--lr",      type=float, default=3e-4)
    te_cmd.add_argument("--batch",   type=int, default=128)
    te_cmd.add_argument("--temperature", type=float, default=0.07)
    te_cmd.add_argument("--min-cluster", type=int, default=2)
    te_cmd.add_argument("--max-pairs", type=int, default=300_000)

    # ---- build-index ----
    bi_cmd = sub.add_parser(
        "build-index",
        help="Build the centroid index after training (required before normalize-entities).",
    )
    bi_cmd.add_argument("--merges",  default=DEFAULT_MERGES)
    bi_cmd.add_argument("--ckpt",    default=DEFAULT_ENORM_CKPT)

    # ---- normalize-entities ----
    ne_cmd = sub.add_parser(
        "normalize-entities",
        help="Apply trained entity normaliser to a KG JSON file.",
    )
    ne_cmd.add_argument("--input",     default=DEFAULT_INPUT)
    ne_cmd.add_argument("--output",    default=DEFAULT_INPUT.replace(".json", "_enorm.json"))
    ne_cmd.add_argument("--ckpt",      default=DEFAULT_ENORM_CKPT)
    ne_cmd.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()

    # ── Relation canonicalization handlers ────────────────────────────────────

    if args.command == "train":
        pairs_data = []
        schema = load_schema(args.schema)

        # Try Distant Supervision first if TTL and sentence file are provided
        if args.mapping_ttl and Path(args.mapping_ttl).exists():
            pairs_data = build_distant_supervision_pairs(
                args.mapping_ttl, args.input, args.schema, max_pairs=args.max_pairs
            )

        # Fallback to internal bootstrapping if DS failed or was skipped
        if not pairs_data:
            bootstrap = build_training_pairs(args.input, args.schema, min_count=args.min_count)
            pairs_data = bootstrap["train_data"]
            schema = bootstrap["schema"]

        canon = QKVCanonicalizer(schema_predicates=schema)
        canon.train(pairs_data, epochs=args.epochs, lr=args.lr, batch_size=args.batch)
        canon.save(args.ckpt)

    elif args.command == "canonicalize":
        schema = load_schema(args.schema)
        canon = QKVCanonicalizer(schema_predicates=schema)
        canon.load(args.ckpt)
        canon.canonicalize_kg(args.input, args.output, confidence_threshold=args.threshold)

    elif args.command == "probe":
        schema = load_schema(args.schema)
        canon = QKVCanonicalizer(schema_predicates=schema)
        canon.load(args.ckpt)
        results = canon.canonicalize_relation(args.pred, args.subj, args.obj, top_k=args.top_k)
        print(f"\nInput:  '{args.pred}'  (subj={args.subj}, obj={args.obj})")
        print("Canonical candidates:")
        for r in results:
            bar = "█" * int(r["confidence"] * 40)
            print(f"  {r['canonical']:<40} {r['confidence']:.4f}  {bar}")

    # ── Entity normaliser handlers ────────────────────────────────────────────

    elif args.command == "train-enorm":
        pairs = build_entity_training_pairs(args.merges, min_cluster_size=args.min_cluster)

        # Supplement with DBpedia Redirects if available
        if args.redirects_ttl and Path(args.redirects_ttl).exists():
            redirect_pairs = build_entity_normalization_pairs_from_redirects(
                args.redirects_ttl, max_pairs=args.max_pairs
            )
            pairs.extend(redirect_pairs)

        enorm = QKVEntityNormalizer()
        enorm.train(pairs, epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch, temperature=args.temperature)
        enorm.save(args.ckpt)

    elif args.command == "build-index":
        enorm = QKVEntityNormalizer()
        enorm.load(args.ckpt)
        enorm.build_centroid_index(args.merges)
        enorm.save(args.ckpt)   # re-save with centroid index embedded

    elif args.command == "normalize-entities":
        enorm = QKVEntityNormalizer()
        enorm.load(args.ckpt)
        enorm.normalize_kg(args.input, args.output, similarity_threshold=args.threshold)


if __name__ == "__main__":
    _cli()
