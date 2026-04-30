from wikirag.classifier import classify_query


def test_classifies_known_people():
    assert classify_query("What did Marie Curie discover?") == "person"


def test_classifies_known_places():
    assert classify_query("Where is the Eiffel Tower located?") == "place"


def test_classifies_mixed_comparison():
    assert classify_query("Compare Albert Einstein and Nikola Tesla") == "person"
    assert classify_query("Compare the Eiffel Tower and the Statue of Liberty") == "place"
    assert classify_query("Compare Nikola Tesla and the Eiffel Tower") == "both"


def test_failure_query_uses_person_hint_before_retrieval_fallback():
    assert classify_query("Who is the president of Mars?") == "person"
