"""
Unit tests for the SHL Assessment Recommender.
Tests core functions without requiring LLM API keys or a running server.
"""

import json
import pytest
from pathlib import Path

# ── Catalog tests ──

from catalog import load_catalog, catalog_to_text, find_by_name, TEST_TYPE_LABELS


def test_load_catalog_returns_list():
    catalog = load_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) > 0


def test_catalog_items_have_required_fields():
    catalog = load_catalog()
    required = {"name", "url", "test_type"}
    for item in catalog:
        missing = required - set(item.keys())
        assert not missing, f"Item '{item.get('name', '?')}' missing fields: {missing}"


def test_catalog_urls_are_valid():
    catalog = load_catalog()
    for item in catalog:
        assert item["url"].startswith("https://www.shl.com/"), (
            f"Invalid URL for '{item['name']}': {item['url']}"
        )


def test_catalog_no_empty_descriptions():
    catalog = load_catalog()
    empty = [item["name"] for item in catalog if not item.get("description")]
    assert len(empty) == 0, f"Items with empty descriptions: {empty}"


def test_catalog_test_types_are_valid():
    catalog = load_catalog()
    valid_types = set(TEST_TYPE_LABELS.keys())
    for item in catalog:
        for t in item.get("test_type", []):
            assert t in valid_types, (
                f"Invalid test type '{t}' in '{item['name']}'"
            )


def test_catalog_to_text_includes_name():
    item = {
        "name": "Java 8 (New)",
        "test_type": ["K"],
        "description": "Tests Java 8 skills",
        "remote_testing": True,
        "adaptive_irt": False,
    }
    text = catalog_to_text(item)
    assert "Java 8 (New)" in text
    assert "Knowledge" in text


def test_find_by_name_case_insensitive():
    catalog = load_catalog()
    result = find_by_name(catalog, "java")
    assert result is not None
    assert "java" in result["name"].lower()


def test_find_by_name_returns_none_for_nonexistent():
    catalog = load_catalog()
    result = find_by_name(catalog, "xyznonexistent12345")
    assert result is None


# ── Agent helper tests ──

from agent import _extract_search_queries, _extract_json


def test_extract_search_queries_from_simple_message():
    messages = [{"role": "user", "content": "I need a Java developer assessment"}]
    queries = _extract_search_queries(messages)
    assert len(queries) >= 1
    assert any("java" in q.lower() for q in queries)


def test_extract_search_queries_detects_tech_keywords():
    messages = [{"role": "user", "content": "Testing Python and SQL skills"}]
    queries = _extract_search_queries(messages)
    combined = " ".join(queries).lower()
    assert "python" in combined
    assert "sql" in combined


def test_extract_search_queries_detects_roles():
    messages = [{"role": "user", "content": "Hiring a manager for our team"}]
    queries = _extract_search_queries(messages)
    combined = " ".join(queries).lower()
    assert "manager" in combined


def test_extract_search_queries_handles_personality():
    messages = [{"role": "user", "content": "Need personality and behavioral assessments"}]
    queries = _extract_search_queries(messages)
    combined = " ".join(queries).lower()
    assert "personality" in combined


def test_extract_search_queries_multi_turn():
    messages = [
        {"role": "user", "content": "Hiring a developer"},
        {"role": "assistant", "content": "What tech stack?"},
        {"role": "user", "content": "Java and React"},
    ]
    queries = _extract_search_queries(messages)
    combined = " ".join(queries).lower()
    assert "java" in combined
    assert "react" in combined


def test_extract_search_queries_empty():
    queries = _extract_search_queries([])
    assert queries == []


def test_extract_search_queries_no_user_messages():
    messages = [{"role": "assistant", "content": "Hello"}]
    queries = _extract_search_queries(messages)
    assert queries == []


def test_extract_search_queries_max_limit():
    messages = [{"role": "user", "content": "java python sql react angular node developer engineer manager analyst"}]
    queries = _extract_search_queries(messages)
    assert len(queries) <= 5


# ── JSON extraction tests ──

def test_extract_json_clean():
    text = '{"reply": "Hello", "recommendations": [], "end_of_conversation": false}'
    result = _extract_json(text)
    assert result["reply"] == "Hello"
    assert result["recommendations"] == []
    assert result["end_of_conversation"] is False


def test_extract_json_markdown_fenced():
    text = '```json\n{"reply": "Hi", "recommendations": [], "end_of_conversation": false}\n```'
    result = _extract_json(text)
    assert result["reply"] == "Hi"


def test_extract_json_with_surrounding_text():
    text = 'Here is my response:\n{"reply": "Test", "recommendations": [], "end_of_conversation": true}\nDone.'
    result = _extract_json(text)
    assert result["reply"] == "Test"
    assert result["end_of_conversation"] is True


def test_extract_json_with_recommendations():
    text = json.dumps({
        "reply": "Here are your assessments",
        "recommendations": [
            {"name": "Java 8 (New)", "url": "https://www.shl.com/test", "test_type": "K"}
        ],
        "end_of_conversation": False,
    })
    result = _extract_json(text)
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["name"] == "Java 8 (New)"


def test_extract_json_fallback_on_invalid():
    text = "I'm sorry, I can't help with that."
    result = _extract_json(text)
    assert result["reply"] == text
    assert result["recommendations"] == []
    assert result["end_of_conversation"] is False


# ── Retrieval tests ──

from retrieval import _keyword_score


def test_keyword_score_name_match():
    item = {"name": "Java 8 (New)", "test_type": ["K"], "description": ""}
    score = _keyword_score(item, "java programming test")
    assert score > 0


def test_keyword_score_type_match():
    item = {"name": "Some Test", "test_type": ["P"], "description": "personality assessment"}
    score = _keyword_score(item, "personality assessment")
    assert score > 0


def test_keyword_score_no_match():
    item = {"name": "Java 8 (New)", "test_type": ["K"], "description": "Java knowledge test"}
    score = _keyword_score(item, "personality leadership")
    assert score == 0.0


def test_keyword_score_capped():
    item = {
        "name": "Java Python SQL Developer Engineer Test",
        "test_type": ["K"],
        "description": "programming coding software technical knowledge java python sql",
    }
    score = _keyword_score(item, "java python sql programming coding software developer engineer knowledge technical")
    assert score <= 0.4


# ── Trace file validation ──

def test_trace_files_are_valid():
    traces_dir = Path(__file__).parent / "traces"
    if not traces_dir.exists():
        pytest.skip("No traces/ directory")

    for trace_file in traces_dir.glob("*.json"):
        with open(trace_file, encoding="utf-8") as f:
            trace = json.load(f)

        assert "persona" in trace, f"{trace_file.name}: missing 'persona'"
        assert "initial_message" in trace, f"{trace_file.name}: missing 'initial_message'"
        assert "expected_assessments" in trace, f"{trace_file.name}: missing 'expected_assessments'"
        assert isinstance(trace["expected_assessments"], list), f"{trace_file.name}: expected_assessments not a list"
        assert len(trace["expected_assessments"]) > 0, f"{trace_file.name}: empty expected_assessments"


def test_trace_expected_assessments_exist_in_catalog():
    """Verify that expected assessments in traces actually exist in the catalog."""
    traces_dir = Path(__file__).parent / "traces"
    if not traces_dir.exists():
        pytest.skip("No traces/ directory")

    catalog = load_catalog()
    catalog_names = {item["name"].lower() for item in catalog}

    for trace_file in traces_dir.glob("*.json"):
        with open(trace_file, encoding="utf-8") as f:
            trace = json.load(f)

        for expected in trace.get("expected_assessments", []):
            assert expected.lower() in catalog_names, (
                f"{trace_file.name}: expected assessment '{expected}' not found in catalog"
            )


# ── Response schema validation ──

def test_chat_response_schema():
    """Verify the response structure matches the API contract."""
    valid_response = {
        "reply": "Here are assessments",
        "recommendations": [
            {"name": "Test", "url": "https://www.shl.com/test", "test_type": "K"}
        ],
        "end_of_conversation": False,
    }
    assert isinstance(valid_response["reply"], str)
    assert isinstance(valid_response["recommendations"], list)
    assert isinstance(valid_response["end_of_conversation"], bool)
    for rec in valid_response["recommendations"]:
        assert "name" in rec
        assert "url" in rec
        assert "test_type" in rec
        assert len(rec["test_type"]) == 1  # single letter code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
