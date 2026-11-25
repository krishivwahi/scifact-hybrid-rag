from src.agent import Verdict


def test_numeric_doc_ids_parse_as_strings():
    # Gemini emits doc ids as JSON numbers; rejecting them once silently turned verdicts into NEI
    v = Verdict.model_validate_json('{"label": "SUPPORT", "doc_ids": [13734012], "rationale": "r"}')
    assert v.doc_ids == ["13734012"]
