"""Official Agnes API access; credentials stay outside logs and command arguments."""
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]


def read_config():
    config = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip("\"'")
    if not config.get("AGNES_API_KEY"):
        raise ValueError("AGNES_API_KEY is empty")
    if config.get("AGNES_BASE_URL") not in ["https://apihub.agnes-ai.com/v1", "https://api.agnes-ai.cn/v1"]:
        raise ValueError("Expected the configured official Agnes endpoint")
    return config


def chat(messages, tools=None, max_tokens=1024, tool_choice=None):
    config = read_config()
    payload = {"model": config["AGNES_MODEL"], "messages": messages,
        "temperature": 0, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    start = time.perf_counter()
    try:
        response = requests.post(config["AGNES_BASE_URL"] + "/chat/completions",
            headers={"Authorization": "Bearer " + config["AGNES_API_KEY"], "Content-Type": "application/json"},
            json=payload, timeout=(15, 60))
    except requests.RequestException as error:
        raise RuntimeError(f"Agnes connection failure: {type(error).__name__}") from None
    if response.status_code != 200:
        raise RuntimeError(f"Agnes HTTP status {response.status_code}; provider body omitted")
    data = response.json()
    if not data.get("choices"):
        raise RuntimeError("Agnes response has no choices")
    return data, {"seconds": time.perf_counter() - start, "model": data.get("model"),
        "usage": data.get("usage", {}), "response_id": data.get("id")}


def main():
    data, plain = chat([{"role": "user", "content": "Reply with only OK."}], max_tokens=32)
    tool = {"type": "function", "function": {"name": "confirm_ready",
        "description": "Confirm that the connection can execute a typed tool call.",
        "parameters": {"type": "object", "properties": {"ready": {"type": "boolean"}},
            "required": ["ready"], "additionalProperties": False}}}
    result, tool_meta = chat([{"role": "user", "content": "Call confirm_ready with ready=true. Do not answer in plain text."}],
        tools=[tool], tool_choice={"type": "function", "function": {"name": "confirm_ready"}}, max_tokens=256)
    calls = result["choices"][0]["message"].get("tool_calls", [])
    tool_ok = bool(calls and calls[0]["function"]["name"] == "confirm_ready" and
        json.loads(calls[0]["function"]["arguments"]) == {"ready": True})
    check = {"text_connection": bool(data["choices"][0]["message"].get("content")),
        "typed_tool_call": tool_ok, "plain_request": plain, "tool_request": tool_meta,
        "credentials_logged": False, "official_endpoint": read_config()["AGNES_BASE_URL"]}
    (ROOT / "experiments/agnes_connection_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))
    if not tool_ok:
        raise RuntimeError("Typed tool call did not validate")


if __name__ == "__main__":
    main()
