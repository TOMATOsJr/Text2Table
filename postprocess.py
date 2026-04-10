"""
postprocess.py
==============
Utility for relation mapping and normalization in the Text2Table pipeline.
Provides the RELATION_TO_ATTRIBUTE dictionary and normalization functions.
"""

from typing import Optional

# --- Relation to Attribute Mapping -------------------------------------------
# Maps noisy REBEL predicates to canonical WikiBio attribute keys.
RELATION_TO_ATTRIBUTE = {
    "date of birth": "birth_date",
    "place of birth": "birth_place",
    "country of citizenship": "nationality",
    "nationality": "nationality",
    "date of death": "death_date", 
    "place of death": "death_place",
    "cause of death": "death_cause",
    "occupation": "occupation",
    "position held": "position",
    "employer": "team",
    "work location": "location",
    "member of sports team": "clubs",
    "sport": "sport",
    "position played on team": "position",
    "educated at": "education",
    "alma mater": "alma_mater",
    "academic degree": "education",
    "spouse": "spouse",
    "child": "children",
    "father": "parents",
    "mother": "parents",
    "sibling": "relations",
    "relative": "relations",
    "award received": "awards",
    "nominated for": "awards",
    "member of": "member_of",
    "genre": "genre",
    "instrument": "instrument",
    "record label": "label",
    "performer": "associated_acts",
    "military rank": "position",
    "conflict": "known_for",
    "member of political party": "party",
    "country": "country",
    "place of burial": "location",
    "residence": "residence",
    "notable work": "notable_work",
    "author": "notable_work",
    "publisher": "label",
    "language of work or name": "language",
    "participant of": "known_for",
    "participant in": "known_for",
    "coach of sports team": "occupation",
    "religion": "religion",
    "sex or gender": "gender",
    "instance of": "instance_of",
    "subclass of": "subclass_of",
    "inception": "birth_date",
}

def normalise_relation(relation: str) -> str:
    """Map a raw predicate string to a canonical attribute key."""
    rel = relation.lower().strip()
    
    # 1. Direct mapping
    if rel in RELATION_TO_ATTRIBUTE:
        return RELATION_TO_ATTRIBUTE[rel]
    
    # 2. Heuristic: snake_case standardisation
    clean = rel.replace(" ", "_").replace("-", "_")
    
    # 3. Handle 'born', 'died' variants
    if "born" in rel:
        if "place" in rel or "in" in rel: return "birth_place"
        return "birth_date"
    if "died" in rel or "death" in rel:
        if "place" in rel or "in" in rel: return "death_place"
        return "death_date"
        
    return clean

def _auto_normalise(rel: str) -> str:
    """Alias for normalise_relation used in older pipeline scripts."""
    return normalise_relation(rel)
