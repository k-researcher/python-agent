from agent.llm import LLMClient


def test_parse_chat_completion() -> None:
    response = LLMClient._parse_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["id"] == "call-1"
    assert response.prompt_tokens == 10

