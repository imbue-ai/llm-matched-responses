# llm-matched-responses

An [llm](https://llm.datasette.io/) plugin that provides a `matched-responses` model for testing. Returns configurable responses based on input patterns, so you can test LLM-based systems without real API keys.

## Installation

```bash
llm install llm-matched-responses
```

## Usage

```bash
# Default: echoes back the input
llm -m matched-responses "Hello world"
# Output: Echo: Hello world

# Static response for all inputs (via env var)
LLM_MATCHED_RESPONSE="Fixed reply" llm -m matched-responses "anything"
# Output: Fixed reply

# Substring-matched responses from a JSON file
echo '{"hello": "Hi there!", "help": "I can help."}' > responses.json
LLM_MATCHED_RESPONSES_FILE=responses.json llm -m matched-responses "hello world"
# Output: Hi there!
```

## Response resolution order

1. **`LLM_MATCHED_RESPONSE`** env var: if set, always return this exact string
2. **`LLM_MATCHED_RESPONSES_FILE`** env var: path to a JSON file mapping input substrings to responses. The first matching key wins.
3. **Default**: returns `"Echo: <input>"` (or `"Echo: (empty message)"` for empty input)

## Responses file format

```json
{
    "hello": "Hello! I am the test model.",
    "help": "I am a test model that returns matched responses.",
    "code": "Here is some code:\n```python\nprint('hello')\n```"
}
```

Keys are matched as substrings of the user's message. The first match wins.
