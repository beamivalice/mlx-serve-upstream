#!/usr/bin/env python3
"""Exercise the original Xing BF16 checkpoint through a real native server.

python3 tests/test_xing4.py /path/to/Xing4.0-29B-A4B
Build the binary with zig build -Doptimize=ReleaseFast first.
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/mlx-serve"))
    parser.add_argument("--port", type=int, default=18991)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    assert json.loads((model / "config.json").read_text())["model_type"] == "xing4_0"
    binary = args.binary.resolve(strict=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    if args.output:
        root = args.output.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        Path("zig-out").mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="xing4-", dir="zig-out")).resolve()
    home = root / "home"
    empty = root / "empty-models"
    home.mkdir()
    empty.mkdir()
    base = f"http://127.0.0.1:{args.port}"
    log_path = root / "server.log"
    command = [
        str(binary), "--serve", "--host", "127.0.0.1", "--port", str(args.port),
        "--model-dir", str(empty), "--ctx-size", "4096", "--prefill-chunk", "512",
        "--max-concurrent", "2", "--no-pld", "--no-mtp", "--no-drafter",
        "--no-decode-attn-quant", "--prefix-cache-entries", "0",
        "--log-level", "debug", "--log-file", str(log_path),
    ]
    (root / "command.json").write_text(json.dumps(command, indent=2))
    env = {**os.environ, "HOME": str(home)}
    env.pop("MLX_SERVE_CONFIG_OVERRIDES", None)
    results = {}

    def request(path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise AssertionError(f"{path}: HTTP {error.code}: {error.read().decode()}") from error

    def record(name, value):
        results[name] = value
        (root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"PASS {name}", flush=True)

    def clean(text):
        assert isinstance(text, str) and text.strip(), repr(text)
        for marker in ("<_bot>", "<_end>", "<think>", "</think>", "<tool_call>", "<param_key>"):
            assert marker not in text, (marker, text)
        return text

    def chat(prompt, **extra):
        return {
            "model": model.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
            "enable_thinking": False,
            **extra,
        }

    def streaming(body):
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({**body, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        content, reasoning = [], []
        done = False
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                assert "error" not in event, event
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
        assert done, "stream ended without [DONE]"
        return {"content": "".join(content), "reasoning_content": "".join(reasoning)}

    with (root / "console.log").open("w") as console:
        server = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 30
            while True:
                assert server.poll() is None, "server exited before readiness"
                try:
                    if request("/health")["status"] == "ok":
                        break
                except (OSError, AssertionError):
                    assert time.monotonic() < deadline, "server readiness timeout"
                    time.sleep(0.2)
            record("load-original-bf16", request("/v1/load-model", {"model": str(model)}))
            rows = request("/v1/models")["data"]
            row = next(row for row in rows if row["id"] == model.name)
            assert row["context_length"] == 4096, row
            record("advertised-context", row)

            body = chat("What is 2 + 2? Answer with only the number.")
            plain = request("/v1/chat/completions", body)
            message = plain["choices"][0]["message"]
            assert "4" in clean(message.get("content")), plain
            assert plain["choices"][0]["finish_reason"] == "stop", plain
            record("plain-arithmetic", plain)
            stream = streaming(body)
            assert stream["content"] == message["content"], (stream, message)
            assert stream["reasoning_content"] == (message.get("reasoning_content") or ""), (stream, message)
            record("stream-nonstream-parity", stream)

            chinese = request("/v1/chat/completions", chat("请用中文简短回答：中国的首都是哪里？"))
            assert "北京" in clean(chinese["choices"][0]["message"].get("content")), chinese
            record("chinese", chinese)
            thought_body = chat("What is 7 + 5?", enable_thinking=True, reasoning_budget_tokens=64)
            thought = request("/v1/chat/completions", thought_body)
            clean(thought["choices"][0]["message"].get("content"))
            record("bounded-thinking", thought)
            thought_stream = streaming(thought_body)
            clean(thought_stream["content"])
            assert thought_stream["content"] == thought["choices"][0]["message"]["content"], (thought_stream, thought)
            # Non-stream reasoning removes the think block's framing whitespace.
            assert thought_stream["reasoning_content"].strip("\n ") == (thought["choices"][0]["message"].get("reasoning_content") or "").strip("\n "), (thought_stream, thought)
            record("thinking-stream-parity", thought_stream)

            tool_body = chat(
                "Call echo with text exactly XING_TOOL_42. Do not answer directly.",
                tools=[{"type": "function", "function": {
                    "name": "echo", "description": "Echo the text.",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                }}],
            )
            tool = request("/v1/chat/completions", tool_body)
            assistant = tool["choices"][0]["message"]
            calls = assistant.get("tool_calls", [])
            assert len(calls) == 1, tool
            assert calls[0]["function"]["name"] == "echo", tool
            assert json.loads(calls[0]["function"]["arguments"]) == {"text": "XING_TOOL_42"}, tool
            record("tool-call", tool)
            follow = {**tool_body, "messages": tool_body["messages"] + [
                assistant,
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "XING_TOOL_42"},
                {"role": "user", "content": "The tool succeeded. Say done without calling it again."},
            ]}
            follow_reply = request("/v1/chat/completions", follow)
            clean(follow_reply["choices"][0]["message"].get("content"))
            record("tool-history", follow_reply)

            anthropic = request("/v1/messages", {
                "model": model.name, "max_tokens": 128, "temperature": 0,
                "enable_thinking": False,
                "messages": [{"role": "user", "content": "Say hello briefly."}],
            })
            clean("".join(block.get("text", "") for block in anthropic["content"] if block["type"] == "text"))
            record("anthropic", anthropic)
            responses = request("/v1/responses", {
                "model": model.name, "input": "Say hello briefly.",
                "temperature": 0, "max_output_tokens": 128, "enable_thinking": False,
            })
            clean("".join(
                part.get("text", "") for item in responses["output"]
                if item["type"] == "message" for part in item["content"]
                if part["type"] == "output_text"
            ))
            record("responses", responses)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                replies = list(pool.map(
                    lambda prompt: request("/v1/chat/completions", chat(prompt)),
                    ["What is 3 + 4? Answer briefly.", "What is 8 + 1? Answer briefly."],
                ))
            for reply in replies:
                clean(reply["choices"][0]["message"].get("content"))
            record("concurrent-requests", replies)
            record("unload", request("/v1/unload-model", {"model": model.name}))
            record("health-after-unload", request("/health"))
            log = log_path.read_text()
            assert "[dtype-trace] residual widened" not in log, "residual widened; inspect server.log"
            assert "jinja error:" not in log, "template fallback; inspect server.log"
            print(f"All Xing BF16 API checks passed. Artifacts: {root}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            print(f"Test artifacts: {root}", flush=True)


if __name__ == "__main__":
    main()
