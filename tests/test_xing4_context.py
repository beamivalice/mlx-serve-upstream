#!/usr/bin/env python3
"""Check repeated long prompts and latent prefix reuse on a running Xing server."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--max-active-gib", type=float)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def request(path, body=None):
        req = urllib.request.Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)

    def prompt(nonce, count):
        records = "\n".join(
            f"Record {i}: The amber notebook contains sketches of birch trees beside a quiet river."
            for i in range(count)
        )
        return f"{nonce}\n{records}\nReply with exactly LATENT_OK."

    lo, hi = 1, args.tokens
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        tokens = request("/tokenize", {"content": prompt("alpha", mid)})["tokens"]
        if len(tokens) <= args.tokens:
            lo = mid
        else:
            hi = mid
    results = []
    for nonce, label in (("alpha", "cold"), ("alpha", "warm"), ("beta", "divergent"), ("beta", "warm-divergent")):
        result = request("/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt(nonce, lo)}],
            "temperature": 0, "enable_thinking": False, "max_tokens": 32,
        })
        usage = result["usage"]
        assert args.tokens * 0.95 <= usage["prompt_tokens"] <= args.tokens * 1.05, usage
        assert "LATENT_OK" in result["choices"][0]["message"].get("content", ""), result
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        if label.startswith("warm"):
            assert cached > args.tokens // 2, (label, usage)
        props = request("/props")
        if args.max_active_gib:
            assert props["memory"]["active_bytes"] < args.max_active_gib * 1024**3, props["memory"]
        results.append({
            "case": label, "usage": usage, "memory": props["memory"],
            "timings": result.get("timings"), "content": result["choices"][0]["message"]["content"],
        })
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=2))
        print(f"PASS {label}: prompt={usage['prompt_tokens']} cached={cached} active={props['memory']['active_bytes'] / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
