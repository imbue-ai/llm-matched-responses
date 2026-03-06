"""Unit tests for the llm-matched-responses plugin."""

import json
from pathlib import Path

import llm
import pytest
from llm_matched_responses import MatchedResponsesModel
from llm_matched_responses import _try_parse_tool_calls
from llm_matched_responses import resolve_response


def test_default_echo() -> None:
    assert resolve_response("Hello world") == "Echo: Hello world"


def test_empty_message() -> None:
    assert resolve_response("") == "Echo: (empty message)"


def test_static_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MATCHED_RESPONSE", "Static reply")
    assert resolve_response("anything") == "Static reply"


def test_static_env_override_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MATCHED_RESPONSE", "Always this")
    assert resolve_response("") == "Always this"


def test_static_env_takes_precedence_over_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.json"
    responses_file.write_text(json.dumps({"hello": "from file"}))
    monkeypatch.setenv("LLM_MATCHED_RESPONSE", "from env")
    monkeypatch.setenv("LLM_MATCHED_RESPONSES_FILE", str(responses_file))
    assert resolve_response("hello") == "from env"


def test_responses_file_substring_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.json"
    responses_file.write_text(json.dumps({"hello": "Hi!", "help": "I can help."}))
    monkeypatch.setenv("LLM_MATCHED_RESPONSES_FILE", str(responses_file))
    assert resolve_response("hello world") == "Hi!"


def test_responses_file_no_match_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses_file = tmp_path / "responses.json"
    responses_file.write_text(json.dumps({"hello": "Hi!"}))
    monkeypatch.setenv("LLM_MATCHED_RESPONSES_FILE", str(responses_file))
    assert resolve_response("goodbye") == "Echo: goodbye"


def test_responses_file_missing_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MATCHED_RESPONSES_FILE", "/nonexistent/path.json")
    assert resolve_response("hello") == "Echo: hello"


def test_responses_file_invalid_json_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json {{{")
    monkeypatch.setenv("LLM_MATCHED_RESPONSES_FILE", str(bad_file))
    with pytest.raises(json.JSONDecodeError):
        resolve_response("hello")


def test_supports_tools_attribute() -> None:
    model = MatchedResponsesModel()
    assert model.supports_tools is True


def test_try_parse_tool_calls_valid() -> None:
    reply = json.dumps({
        "tool_calls": [{"name": "my_tool", "arguments": {"x": 1}}],
        "text": "hello",
    })
    result = _try_parse_tool_calls(reply)
    assert result is not None
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "my_tool"
    assert result["text"] == "hello"


def test_try_parse_tool_calls_plain_text() -> None:
    assert _try_parse_tool_calls("Echo: hello") is None


def test_try_parse_tool_calls_json_without_tool_calls() -> None:
    assert _try_parse_tool_calls(json.dumps({"foo": "bar"})) is None


def test_try_parse_tool_calls_not_a_dict() -> None:
    assert _try_parse_tool_calls(json.dumps([1, 2, 3])) is None


def test_execute_with_tools_returns_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """When tools are provided and response contains tool_calls JSON, the model emits tool calls."""
    tool_response = json.dumps({
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "SF"}}],
        "text": "Calling tool",
    })
    monkeypatch.setenv("LLM_MATCHED_RESPONSE", tool_response)

    model = MatchedResponsesModel()
    response = model.prompt("what's the weather?", tools=[_dummy_tool()])
    text = response.text()
    assert text == "Calling tool"
    assert len(response.tool_calls()) == 1
    assert response.tool_calls()[0].name == "get_weather"
    assert response.tool_calls()[0].arguments == {"city": "SF"}


def test_execute_with_tools_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """When tools are provided but response is plain text, no tool calls are emitted."""
    monkeypatch.setenv("LLM_MATCHED_RESPONSE", "just text")

    model = MatchedResponsesModel()
    response = model.prompt("hello", tools=[_dummy_tool()])
    assert response.text() == "just text"
    assert response.tool_calls() == []


def test_execute_without_tools_ignores_tool_calls_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no tools are provided, tool_calls JSON is returned as plain text."""
    tool_response = json.dumps({
        "tool_calls": [{"name": "get_weather", "arguments": {}}],
    })
    monkeypatch.setenv("LLM_MATCHED_RESPONSE", tool_response)

    model = MatchedResponsesModel()
    response = model.prompt("hello")
    assert response.text() == tool_response


def _dummy_tool() -> llm.Tool:
    """Create a minimal Tool for testing."""
    return llm.Tool(
        name="get_weather",
        description="Get weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
        implementation=lambda city: f"Weather in {city}: sunny",
    )
