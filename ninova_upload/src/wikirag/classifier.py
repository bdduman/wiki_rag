from __future__ import annotations

import re

from .entities import ENTITY_TYPES, PEOPLE, PLACES


PERSON_HINTS = {
    "person",
    "people",
    "who",
    "born",
    "discover",
    "discovered",
    "invent",
    "famous for",
    "known for",
    "associated with",
    "footballer",
    "artist",
    "scientist",
    "writer",
}

PLACE_HINTS = {
    "place",
    "places",
    "where",
    "located",
    "country",
    "city",
    "built",
    "used for",
    "landmark",
    "monument",
    "mountain",
    "museum",
}

COMPARISON_HINTS = {"compare", "versus", " vs ", "difference", "similarities", "both"}


def classify_query(query: str) -> str:
    """Return person, place, both, or unknown for a user query."""
    lowered = f" {query.lower()} "
    matched_types = set()

    for title, entity_type in ENTITY_TYPES.items():
        if title in lowered:
            matched_types.add(entity_type)

    person_hits = sum(1 for hint in PERSON_HINTS if hint in lowered)
    place_hits = sum(1 for hint in PLACE_HINTS if hint in lowered)
    has_comparison = any(hint in lowered for hint in COMPARISON_HINTS)

    capitalized_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", query)
    if len(capitalized_terms) >= 2 and has_comparison:
        matched_types.update(
            ENTITY_TYPES.get(term.lower(), "unknown") for term in capitalized_terms
        )
        matched_types.discard("unknown")

    if len(matched_types) > 1:
        return "both"
    if matched_types:
        detected = next(iter(matched_types))
        if has_comparison and (person_hits and place_hits):
            return "both"
        return detected
    if has_comparison:
        return "both"
    if person_hits and place_hits:
        return "both"
    if person_hits:
        return "person"
    if place_hits:
        return "place"
    return "unknown"


def all_entity_titles() -> list[str]:
    return PEOPLE + PLACES


def mentioned_entities(query: str) -> list[tuple[str, str]]:
    lowered = f" {query.lower()} "
    matches = []
    for title in PEOPLE + PLACES:
        if title.lower() in lowered:
            matches.append((title, ENTITY_TYPES[title.lower()]))
    return matches
