import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL = "Babelscape/rebel-large"
DEFAULT_SPACY_MODEL = "en_core_web_trf"


# ---------------------------------------------------------------------------
# Phase 1 – Preprocessing (WikiBio token cleanup)
# ---------------------------------------------------------------------------

WIKIBIO_REPLACEMENTS = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lsb-": "[",
    "-rsb-": "]",
    "-lcb-": "{",
    "-rcb-": "}",
    "``": '"',
    "''": '"',
}


def clean_sentence(text: str) -> str:
    """Clean a raw WikiBio sentence: normalise unicode, replace tokens, strip citations."""
    text = unicodedata.normalize("NFKC", text)
    for old, new in WIKIBIO_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\[[^\]]*\]", "", text)          # strip citation brackets
    text = re.sub(r"\s+--\s+", " \u2013 ", text)      # em-dash
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)      # fix punctuation spacing
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_title(title: str) -> str:
    """Extract person name from WikiBio title, removing disambiguation brackets."""
    title = re.sub(r"\s*-lrb-.*?-rrb-\s*", "", title)
    title = re.sub(r"\s*\(.*?\)\s*", "", title)
    return title.strip()


def is_skip_sentence(text: str) -> bool:
    """Return True for sentences too short or non-informative for RE."""
    tokens = text.split()
    if len(tokens) < 4:
        return True
    if text.startswith("http") or text.startswith("[") or text.startswith("--"):
        return True
    return False


# ---------------------------------------------------------------------------
# Phase 2 – Lightweight coreference (title-based pronoun replacement)
# ---------------------------------------------------------------------------

_SUBJ_PRONOUNS = {"he", "she", "they"}
_POSS_PRONOUNS = {"his", "her", "their"}


def resolve_pronouns(sentence: str, title: str) -> str:
    """Replace sentence-initial pronoun with the record's person name."""
    if not title:
        return sentence
    first_space = sentence.find(" ")
    if first_space == -1:
        return sentence
    first_word = sentence[:first_space].lower()
    rest = sentence[first_space:]
    if first_word in _SUBJ_PRONOUNS:
        return title + rest
    if first_word in _POSS_PRONOUNS:
        return title + "'s" + rest
    return sentence


# ---------------------------------------------------------------------------
# Phase 3 – Post-processing: split concatenated REBEL relations
# ---------------------------------------------------------------------------

SEED_RELATIONS: set[str] = {
    "date of birth", "place of birth", "country of citizenship",
    "date of death", "place of death", "cause of death",
    "occupation", "position held", "employer", "work location",
    "member of sports team", "sport", "position played on team",
    "educated at", "alma mater", "academic degree",
    "spouse", "child", "father", "mother", "sibling", "relative",
    "award received", "nominated for", "member of",
    "genre", "instrument", "record label", "performer",
    "military rank", "conflict", "member of political party",
    "country", "nationality", "place of burial", "residence",
    "notable work", "author", "publisher", "language of work or name",
    "participant of", "coach of sports team", "religion",
    "country for sport", "participant in",
    "languages spoken written or signed",
    "field of work", "instance of", "sex or gender",
    "different from", "parent organization", "subsidiary", "inception",
    "followed by", "follows", "part of", "has part", "subclass of",
    "sports season of league or competition", "point in time",
    "located in the administrative territorial entity",
    "contains administrative territorial entity",
    "applies to jurisdiction", "legislated by",
}


def discover_relations(
    all_raw_triples: list[list[dict[str, str]]],
    max_tokens: int = 5,
) -> set[str]:
    """Collect all atomic relation strings (≤ *max_tokens* words) observed in raw output."""
    discovered: set[str] = set()
    for triples in all_raw_triples:
        for t in triples:
            rel = t["relation"].strip().lower()
            if len(rel.split()) <= max_tokens and rel:
                discovered.add(rel)
    return discovered


def _split_object(obj_str: str, num_parts: int, sentence: str) -> list[str]:
    """Split a concatenated object string into *num_parts* guided by the source sentence.

    Tries every way to partition the object words into *num_parts* non-empty
    segments and picks the partition where the most segments actually appear
    (as substrings) in *sentence*.
    """
    words = obj_str.split()
    n = len(words)
    if n < num_parts or num_parts <= 1 or n > 15:
        return [obj_str] * max(num_parts, 1)

    sent_lower = sentence.lower()
    best_parts: list[str] | None = None
    best_score = -1

    def _search(start: int, remaining: int, acc: list[str]) -> None:
        nonlocal best_parts, best_score
        if remaining == 1:
            part = " ".join(words[start:])
            candidate = acc + [part]
            score = sum(1 for p in candidate if p.lower() in sent_lower)
            if score > best_score:
                best_score = score
                best_parts = candidate
            return
        for i in range(start + 1, n - remaining + 2):
            part = " ".join(words[start:i])
            _search(i, remaining - 1, acc + [part])

    _search(0, num_parts, [])
    if best_parts and best_score > 0:
        return best_parts
    return [obj_str] * num_parts


def split_concatenated_triples(
    triples: list[dict[str, str]],
    sentence: str = "",
    known_relations: set[str] | None = None,
) -> list[dict[str, str]]:
    """Split triples whose relation is a concatenation of known relations.

    When *sentence* is provided, the object string is also partitioned so
    that each split triple gets the correct sub-object.

    *known_relations* defaults to SEED_RELATIONS when not supplied.
    """
    rels = known_relations if known_relations is not None else SEED_RELATIONS
    sorted_rels = sorted(rels, key=len, reverse=True)

    result: list[dict[str, str]] = []
    for triple in triples:
        rel = triple["relation"].strip().lower()
        if rel in rels:
            result.append(triple)
            continue
        # Greedy left-to-right matching of known relations.
        found: list[str] = []
        remaining = rel
        while remaining:
            matched = False
            for known in sorted_rels:
                if remaining.startswith(known):
                    found.append(known)
                    remaining = remaining[len(known):].strip()
                    matched = True
                    break
            if not matched:
                break
        if len(found) >= 2 and not remaining:
            objects = _split_object(triple["object"], len(found), sentence)
            for r, o in zip(found, objects):
                result.append(
                    {"subject": triple["subject"], "relation": r, "object": o}
                )
        else:
            result.append(triple)
    return result


# ---------------------------------------------------------------------------
# Record metadata helpers (.nb / .title companion files)
# ---------------------------------------------------------------------------


def load_record_metadata(sent_path: Path) -> list[tuple[str, int]] | None:
    """Auto-detect and load .nb + .title files alongside a .sent file.

    Returns a list of (cleaned_title, sentence_count) tuples, or None if
    the companion files are not found.
    """
    stem = sent_path.stem
    parent = sent_path.parent
    nb_path = parent / f"{stem}.nb"
    title_path = parent / f"{stem}.title"
    if not nb_path.exists() or not title_path.exists():
        return None
    with nb_path.open("r", encoding="utf-8") as f:
        counts = [int(line.strip()) for line in f if line.strip()]
    with title_path.open("r", encoding="utf-8") as f:
        titles = [clean_title(line.strip()) for line in f if line.strip()]
    if len(counts) != len(titles):
        print(
            f"Warning: .nb has {len(counts)} entries but .title has "
            f"{len(titles)} – metadata ignored"
        )
        return None
    return list(zip(titles, counts))


def normalize_text(text: str) -> str:
    """Normalize text for lightweight matching between triples and named entities."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r\"'`.,;:!?()[]{}")


def iter_sentences(
    path: Path,
    text_column: str,
    metadata: list[tuple[str, int]] | None = None,
) -> Iterable[tuple[int, str, int, str]]:
    """Yield (sentence_id, sentence, record_id, title) from an input file.

    When *metadata* (from load_record_metadata) is provided and the input is
    a .sent file, each sentence is annotated with its record index and the
    person's name.
    """
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or text_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{text_column}' not found in CSV: {path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            for sentence_id, row in enumerate(reader, start=1):
                raw_text = row.get(text_column, "")
                sentence = raw_text.strip() if isinstance(raw_text, str) else ""
                if sentence:
                    yield sentence_id, sentence, 0, ""
        return

    # Plain .sent / .txt file – one sentence per line.
    record_id = 0
    sent_in_record = 0
    title = ""
    count = 0
    if metadata:
        title, count = metadata[0]

    with path.open("r", encoding="utf-8") as handle:
        for sentence_id, line in enumerate(handle, start=1):
            sentence = line.rstrip("\n").strip()
            if not sentence:
                continue
            if metadata:
                while sent_in_record >= count and record_id < len(metadata) - 1:
                    record_id += 1
                    title, count = metadata[record_id]
                    sent_in_record = 0
            yield sentence_id, sentence, record_id, title
            sent_in_record += 1


def chunked(records: Iterable[tuple[int, str, int, str]], batch_size: int) -> Iterable[list[tuple[int, str, int, str]]]:
    batch: list[tuple[int, str, int, str]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_rebel_output(decoded_text: str) -> list[dict[str, str]]:
    """
    Parse REBEL output format with special tokens:
    <triplet> subject <subj> object <obj> relation
    """
    decoded_text = (
        decoded_text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("\n", " ")
        .strip()
    )

    triplets: list[dict[str, str]] = []
    subject = ""
    obj = ""
    relation = ""
    state = None

    for token in decoded_text.split():
        if token == "<triplet>":
            if subject and relation and obj:
                triplets.append(
                    {
                        "subject": subject.strip(),
                        "relation": relation.strip(),
                        "object": obj.strip(),
                    }
                )
            subject = ""
            obj = ""
            relation = ""
            state = "subject"
        elif token == "<subj>":
            state = "object"
        elif token == "<obj>":
            state = "relation"
        else:
            if state == "subject":
                subject += f" {token}"
            elif state == "object":
                obj += f" {token}"
            elif state == "relation":
                relation += f" {token}"

    if subject and relation and obj:
        triplets.append(
            {
                "subject": subject.strip(),
                "relation": relation.strip(),
                "object": obj.strip(),
            }
        )

    # Drop duplicate triples while preserving order.
    deduped: list[dict[str, str]] = []
    seen = set()
    for triple in triplets:
        key = (triple["subject"], triple["relation"], triple["object"])
        if key not in seen:
            deduped.append(triple)
            seen.add(key)
    return deduped


def extract_entities(nlp, sentences: list[str]) -> list[list[dict[str, object]]]:
    all_entities: list[list[dict[str, object]]] = []
    for doc in nlp.pipe(sentences, batch_size=64):
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
            for ent in doc.ents
        ]
        all_entities.append(entities)
    return all_entities


def extract_rebel_triples(
    tokenizer,
    model,
    sentences: list[str],
    device: torch.device,
    max_input_length: int,
    max_output_length: int,
    num_beams: int,
    num_return_sequences: int,
) -> list[list[dict[str, str]]]:
    encoded = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_length=max_output_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)

    # For each input sentence, merge triplets from all returned generations.
    all_results: list[list[dict[str, str]]] = []
    for i in range(len(sentences)):
        merged: list[dict[str, str]] = []
        seen = set()
        start = i * num_return_sequences
        end = start + num_return_sequences
        for text in decoded[start:end]:
            for triple in parse_rebel_output(text):
                key = (triple["subject"], triple["relation"], triple["object"])
                if key not in seen:
                    merged.append(triple)
                    seen.add(key)
        all_results.append(merged)
    return all_results


def is_entity_aligned(text: str, entity_texts: set[str]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False

    if normalized in entity_texts:
        return True

    # Allow small textual variation by containment checks.
    for entity in entity_texts:
        if normalized in entity or entity in normalized:
            return True
    return False


def filter_triples(
    triples: list[dict[str, str]], entities: list[dict[str, object]], filter_mode: str
) -> list[dict[str, str]]:
    if filter_mode == "none":
        return triples

    entity_texts: set[str] = set()
    for entity in entities:
        value = entity.get("text")
        if isinstance(value, str) and value.strip():
            entity_texts.add(normalize_text(value))

    filtered: list[dict[str, str]] = []
    for triple in triples:
        subj_ok = is_entity_aligned(triple["subject"], entity_texts)
        obj_ok = is_entity_aligned(triple["object"], entity_texts)
        if filter_mode == "both" and subj_ok and obj_ok:
            filtered.append(triple)
        elif filter_mode == "either" and (subj_ok or obj_ok):
            filtered.append(triple)

    return filtered


def quality_filter_triples(
    triples: list[dict[str, str]],
    entities: list[dict[str, object]],
    quality_mode: str,
) -> list[dict[str, str]]:
    if quality_mode == "off":
        return triples

    entity_texts: set[str] = set()
    for entity in entities:
        value = entity.get("text")
        if isinstance(value, str) and value.strip():
            entity_texts.add(normalize_text(value))

    enforce_entity_alignment = quality_mode == "strict" and bool(entity_texts)

    filtered: list[dict[str, str]] = []
    for triple in triples:
        subject = normalize_text(triple.get("subject", ""))
        relation = normalize_text(triple.get("relation", ""))
        obj = normalize_text(triple.get("object", ""))

        if not subject or not relation or not obj:
            continue

        # Structural quality checks (not phrase blacklist based).
        relation_tokens = relation.split()
        if not relation_tokens:
            continue
        if len(relation_tokens) > 6:
            # REBEL sometimes emits malformed relation strings like chained concepts.
            continue
        if any(char.isdigit() for char in relation):
            continue

        # Tautology: subject and object are identical.
        if subject == obj:
            continue

        # Reject relations that repeat subject/object terms (often low-value artifacts).
        if relation in subject or relation in obj or subject in relation or obj in relation:
            continue

        # Require at least one meaningful token (len >= 3) in relation phrase.
        if not any(len(token) >= 3 and token.isalpha() for token in relation_tokens):
            continue

        if enforce_entity_alignment:
            subj_ok = is_entity_aligned(subject, entity_texts)
            obj_ok = is_entity_aligned(obj, entity_texts)
            if not (subj_ok and obj_ok):
                continue

            # In strict mode, also require relation phrase to be concise and natural.
            if len(relation_tokens) > 2:
                continue

        filtered.append(triple)

    return filtered


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is not available.")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "wikipedia-biography-dataset" / "test" / "test.sent"
    default_output = base_dir / "outputs" / "triples.jsonl"

    parser = argparse.ArgumentParser(
        description="Generate entity-filtered relation triplets from WikiBio sentences."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to input file (.csv with text column, or .sent/.txt line file)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="sent",
        help="Column name containing sentence text when --input is a CSV",
    )
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["ner_rebel", "rebel_only"],
        default="ner_rebel",
        help="Run full NER+REBEL pipeline or direct REBEL-only extraction",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output JSONL path")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="REBEL model name")
    parser.add_argument(
        "--spacy-model",
        type=str,
        default=DEFAULT_SPACY_MODEL,
        help="spaCy model used for NER",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for REBEL inference")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Process only first N non-empty lines (0 = full file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Max token length for REBEL input",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=512,
        help="Max token length for REBEL generation output",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam width for REBEL generation (higher can improve recall but is slower)",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=3,
        help="Number of generated candidates per sentence to merge for recall",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["both", "either", "none"],
        default="either",
        help="NER filter strictness for keeping triples",
    )
    parser.add_argument(
        "--quality-mode",
        type=str,
        choices=["off", "balanced", "strict"],
        default="balanced",
        help="Post-filter quality mode for relation cleanup",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        default=False,
        help="Skip WikiBio token cleaning and sentence filtering",
    )
    parser.add_argument(
        "--no-resolve-pronouns",
        action="store_true",
        default=False,
        help="Skip title-based pronoun replacement",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_beams <= 0:
        raise ValueError("--num-beams must be > 0")
    if args.num_return_sequences <= 0:
        raise ValueError("--num-return-sequences must be > 0")
    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences cannot exceed --num-beams")

    device = resolve_device(args.device)

    # Auto-detect record metadata (.nb / .title companion files)
    metadata = load_record_metadata(args.input)
    if metadata:
        print(f"Loaded record metadata: {len(metadata)} records from .nb/.title files")

    nlp = None
    if args.pipeline_mode == "ner_rebel":
        print(f"Loading spaCy model: {args.spacy_model}")
        nlp = spacy.load(args.spacy_model)
    else:
        print("Pipeline mode: rebel_only (spaCy NER skipped)")

    print(f"Loading REBEL model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    raw_stream = iter_sentences(
        args.input, text_column=args.text_column, metadata=metadata
    )

    # Phase 1 + 2: preprocess sentences and resolve pronouns.
    def preprocess(stream: Iterable[tuple[int, str, int, str]]) -> Iterable[tuple[int, str, int, str]]:
        for sentence_id, raw_sentence, record_id, title in stream:
            if not args.no_preprocess:
                sentence = clean_sentence(raw_sentence)
            else:
                sentence = raw_sentence
            if is_skip_sentence(sentence):
                continue
            if not args.no_resolve_pronouns and title:
                sentence = resolve_pronouns(sentence, title)
            yield sentence_id, sentence, record_id, title

    sentence_stream: Iterable[tuple[int, str, int, str]] = preprocess(raw_stream)

    if args.sample > 0:
        sentence_stream = (
            rec for idx, rec in enumerate(sentence_stream, start=1) if idx <= args.sample
        )

    total_processed = 0
    total_raw_triples = 0
    total_filtered_triples = 0
    total_dropped_triples = 0

    # ----------------------------------------------------------------
    # Phase A: Run REBEL extraction, buffer results, discover relations
    # ----------------------------------------------------------------
    buffered: list[dict] = []  # stores per-sentence data for Phase B

    for batch in tqdm(chunked(sentence_stream, args.batch_size), desc="Extracting triples"):
        sentence_ids = [sid for sid, _, _, _ in batch]
        sentences = [s for _, s, _, _ in batch]
        record_ids = [rid for _, _, rid, _ in batch]
        titles = [t for _, _, _, t in batch]

        if nlp is not None:
            entities_per_sentence = extract_entities(nlp, sentences)
        else:
            entities_per_sentence = [[] for _ in sentences]
        raw_triples_per_sentence = extract_rebel_triples(
            tokenizer=tokenizer,
            model=model,
            sentences=sentences,
            device=device,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
        )

        for sid, sent, rid, title, ents, raws in zip(
            sentence_ids, sentences, record_ids, titles,
            entities_per_sentence, raw_triples_per_sentence,
        ):
            buffered.append({
                "sentence_id": sid,
                "sentence": sent,
                "record_id": rid,
                "title": title,
                "entities": ents,
                "raw_triples": raws,
            })

    # Build relation vocabulary from raw outputs.
    all_raw = [item["raw_triples"] for item in buffered]
    discovered = discover_relations(all_raw)
    known_relations = SEED_RELATIONS | discovered
    print(
        f"Relation vocabulary: {len(SEED_RELATIONS)} seed + "
        f"{len(discovered - SEED_RELATIONS)} discovered = {len(known_relations)} total"
    )

    # ----------------------------------------------------------------
    # Phase B: Post-process (split, filter, ground) and write output
    # ----------------------------------------------------------------
    with args.output.open("w", encoding="utf-8") as writer:
        for item in tqdm(buffered, desc="Post-processing"):
            sentence_id = item["sentence_id"]
            sentence = item["sentence"]
            record_id = item["record_id"]
            title = item["title"]
            entities = item["entities"]
            raw_triples = item["raw_triples"]

            # Split concatenated relation strings using discovered vocab.
            split_triples = split_concatenated_triples(
                raw_triples, sentence, known_relations=known_relations,
            )

            # Deduplicate after splitting.
            _seen_keys: set[tuple[str, str, str]] = set()
            _deduped: list[dict[str, str]] = []
            for t in split_triples:
                _key = (t["subject"], t["relation"], t["object"])
                if _key not in _seen_keys:
                    _deduped.append(t)
                    _seen_keys.add(_key)
            split_triples = _deduped

            if args.pipeline_mode == "ner_rebel":
                ner_filtered = filter_triples(
                    split_triples, entities, filter_mode=args.filter_mode
                )
            else:
                ner_filtered = split_triples

            quality_filtered = quality_filter_triples(
                ner_filtered, entities, quality_mode=args.quality_mode,
            )

            # Grounding check: drop triples where neither subject
            # nor object text appears in the input sentence.
            sent_lower = sentence.lower()
            final_triples = [
                t for t in quality_filtered
                if normalize_text(t["subject"]) in sent_lower
                or normalize_text(t["object"]) in sent_lower
            ]

            dropped_triples = [
                t
                for t in raw_triples
                if (t["subject"], t["relation"], t["object"])
                not in {
                    (f["subject"], f["relation"], f["object"])
                    for f in final_triples
                }
            ]
            record = {
                "record_id": record_id,
                "title": title,
                "sentence_id": sentence_id,
                "sentence": sentence,
                "pipeline_mode": args.pipeline_mode,
                "entities": entities,
                "raw_triples": raw_triples,
                "filtered_triples": final_triples,
                "dropped_triples": dropped_triples,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_processed += 1
            total_raw_triples += len(raw_triples)
            total_filtered_triples += len(final_triples)
            total_dropped_triples += len(dropped_triples)

    print(f"Saved JSONL output to: {args.output}")
    print(
        f"Processed sentences: {total_processed} | "
        f"Raw triples: {total_raw_triples} | "
        f"Filtered triples: {total_filtered_triples} | "
        f"Dropped triples: {total_dropped_triples}"
    )


if __name__ == "__main__":
    main()
