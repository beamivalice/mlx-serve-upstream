#!/usr/bin/env python3
"""Build and run a tiny Xing4.0 reference checkpoint.

The reference files are kept outside this repository.  The generated directory
contains the original HF tensor names and BF16 checkpoint layout, so the Zig
test exercises `model.loadModelWeights`'s raw-layout preparation instead of a
converted mirror.

Examples:

  python3 tests/dump_xing4_fixtures.py --reference-dir /path/to/Xing4.0-29B-A4B \
      build --out /absolute/path/to/xing4-mini
  python3 tests/dump_xing4_fixtures.py --reference-dir /path/to/Xing4.0-29B-A4B \
      dump --model /absolute/path/to/xing4-mini \
      --out /absolute/path/to/xing4-mini/fixtures.json
  XING4_TEST_MODEL=/absolute/path/to/xing4-mini \
      zig build test -Doptimize=ReleaseFast -Dtest-filter="xing4 fixture"

The default dump includes one 4096-token prefix and a position-4096 decode
row.  Use `--no-long` when iterating on the short fixture only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


DEFAULT_SEED = 0x4A1F04

# Keep the real release's YaRN fields.  The native parser consumes this legacy
# `rope_scaling` spelling, while Transformers derives `rope_parameters` from it.
ROPE_SCALING = {
    "type": "yarn",
    "factor": 64.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
}

# The dimensions retain all of the architecture's distinct paths while keeping
# CPU reference and GPU parity runs small: MLA, one dense layer, then two MoE
# layers, plus the four-stream mHC around both sublayers.
TINY = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 3,
    "num_nextn_predict_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "routed_scaling_factor": 2.0,
    "kv_lora_rank": 8,
    "q_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 8,
    "qk_nope_head_dim": 8,
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "hidden_act": "silu",
    "max_position_embeddings": 262144,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "use_cache": True,
    "pad_token_id": None,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "rope_scaling": ROPE_SCALING,
    "rope_interleave": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "mhc_h_res_clamp_min": -30.0,
    "mhc_h_res_clamp_max": 30.0,
}

PREFILL_IDS = [1, 7, 3, 14, 8, 22, 11, 4, 19, 2, 5, 29]
DECODE_IDS = [31, 6, 17, 9]


def import_reference(reference_dir: Path):
    """Load the two trusted reference files without copying them into the repo."""

    config_path = reference_dir / "configuration_xing4_0.py"
    modeling_path = reference_dir / "modeling_xing4_0.py"
    if not config_path.is_file() or not modeling_path.is_file():
        raise FileNotFoundError(
            f"Xing reference files are missing under {reference_dir}"
        )

    package_name = "_mlx_serve_xing4_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(reference_dir)]
    sys.modules[package_name] = package
    loaded = {}
    for name, path in (
        ("configuration_xing4_0", config_path),
        ("modeling_xing4_0", modeling_path),
    ):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        # Do not create __pycache__ in the read-only reference checkpoint.
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        loaded[name] = module
    return loaded["configuration_xing4_0"], loaded["modeling_xing4_0"]


def patch_reference_init(modeling):
    """Work around the source file's stale `module.fn` init hook.

    This changes construction only.  The forward implementation remains the
    read-only reference under test.
    """

    original = modeling.Xing4_0PreTrainedModel._init_weights

    @torch.no_grad()
    def fixture_init(self, module):
        if isinstance(module, modeling.Xing4_0TopkRouter):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            torch.nn.init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, modeling.Xing4_0HyperConnection):
            torch.nn.init.normal_(
                module.hc_fn, mean=0.0, std=self.config.initializer_range
            )
            torch.nn.init.zeros_(module.hc_base)
            torch.nn.init.ones_(module.hc_scale)
        else:
            original(self, module)

    modeling.Xing4_0PreTrainedModel._init_weights = fixture_init


def make_config(configuration):
    config = configuration.Xing4_0Config(**TINY)
    config._attn_implementation = "eager"
    return config


@torch.no_grad()
def randomize_model(model, modeling):
    """Use non-degenerate deterministic values for every exercised component."""

    for name, parameter in model.named_parameters():
        if name.endswith("hc_fn"):
            torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
        elif name.endswith("hc_base"):
            torch.nn.init.normal_(parameter, mean=0.0, std=0.12)
        elif name.endswith("hc_scale"):
            parameter.copy_(
                torch.tensor(
                    [0.75, 1.25, -0.6015625],
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )
        elif ".mlp.gate.weight" in name:
            torch.nn.init.normal_(parameter, mean=0.0, std=0.25)
        elif parameter.ndim >= 2:
            std = 0.08 if "embed_tokens" in name else 0.05
            torch.nn.init.normal_(parameter, mean=0.0, std=std)
        elif "norm" in name or name.endswith(".weight"):
            parameter.copy_(1.0 + torch.randn_like(parameter) * 0.05)
        else:
            torch.nn.init.normal_(parameter, mean=0.0, std=0.05)

    # The correction is selection-only.  Values are BF16-rounded before being
    # stored in the F32 checkpoint buffer, so both reference precision arms use
    # exactly the same selection keys.
    correction = torch.linspace(
        -0.35,
        0.30,
        TINY["n_routed_experts"],
        dtype=torch.float32,
    ).to(torch.bfloat16).float()
    for layer in model.model.layers:
        if isinstance(layer.mlp, modeling.Xing4_0MoE):
            layer.mlp.gate.e_score_correction_bias.copy_(correction)


def checkpoint_state(model):
    """Match the real checkpoint's BF16/F32 storage split."""

    out = {}
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        if name.endswith(".hc_scale") or name.endswith(
            ".e_score_correction_bias"
        ):
            out[name] = value.float()
        elif value.is_floating_point():
            out[name] = value.to(torch.bfloat16)
        else:
            out[name] = value
    return out


def config_json(config):
    data = config.to_dict()
    # Keep the original legacy object for mlx-serve's Xing-specific parser.
    data.pop("rope_parameters", None)
    data["model_type"] = "xing4_0"
    data["architectures"] = ["Xing4_0ForCausalLM"]
    data["rope_scaling"] = dict(ROPE_SCALING)
    data["rope_interleave"] = True
    data["torch_dtype"] = "bfloat16"
    return data


def build(out: Path, reference_dir: Path):
    configuration, modeling = import_reference(reference_dir)
    patch_reference_init(modeling)
    torch.manual_seed(DEFAULT_SEED)
    config = make_config(configuration)
    model = modeling.Xing4_0ForCausalLM(config).eval()
    randomize_model(model, modeling)
    state = checkpoint_state(model)
    assert "model.layers.0.self_attn.q_a_proj.weight" in state
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in state
    assert "model.layers.0.attention.q_a_proj.weight" not in state

    out.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(out / "model.safetensors"),
        metadata={
            "format": "pt",
            "xing4_fixture": "original-layout-bf16",
            "weights_dtype": "bfloat16",
        },
    )
    (out / "config.json").write_text(
        json.dumps(config_json(config), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[xing4] wrote {len(state)} original-layout tensors to {out} "
        f"(BF16 projections, F32 hc_scale/router correction)"
    )


def load_reference_model(model_dir: Path, reference_dir: Path, dtype):
    configuration, modeling = import_reference(reference_dir)
    patch_reference_init(modeling)
    config = make_config(configuration)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    model = modeling.Xing4_0ForCausalLM(config).eval()
    model.load_state_dict(state, strict=True)
    model = model.to(dtype).eval()
    if dtype == torch.bfloat16:
        # The native loader keeps these two checkpoint F32 control tensors F32.
        for name, parameter in model.named_parameters():
            if name.endswith(".hc_scale"):
                parameter.data = state[name].float().clone()
        for name, buffer in model.named_buffers():
            if name.endswith(".e_score_correction_bias"):
                buffer.data = state[name].float().clone()
    return model


def run_sequence(model, input_ids, decode_ids):
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        first = model(ids, use_cache=True, return_dict=True)
        prefill = first.logits.detach().cpu().float()
        cache = first.past_key_values
        decoded = []
        for token in decode_ids:
            step = model(
                torch.tensor([[token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = step.past_key_values
            decoded.append(step.logits[:, -1].detach().cpu().float())
    return prefill, torch.cat(decoded, dim=0)


def flattened(values):
    values = values.detach().cpu().float().contiguous()
    if not torch.isfinite(values).all():
        raise RuntimeError("reference produced a non-finite fixture value")
    return values.reshape(-1).tolist()


def dump(model_dir: Path, out: Path, reference_dir: Path, include_long: bool):
    torch.set_num_threads(1)
    # The f32 arm is CPU truth over BF16-rounded checkpoint weights.  The
    # checkpoint arm keeps BF16 activations and the checkpoint's F32 controls.
    ref_f32 = load_reference_model(model_dir, reference_dir, torch.float32)
    ref_bf16 = load_reference_model(model_dir, reference_dir, torch.bfloat16)
    pre_f32, dec_f32 = run_sequence(ref_f32, PREFILL_IDS, DECODE_IDS)
    pre_bf16, dec_bf16 = run_sequence(ref_bf16, PREFILL_IDS, DECODE_IDS)

    vocab = TINY["vocab_size"]
    expected_prefill = [1, len(PREFILL_IDS), vocab]
    expected_decode = [len(DECODE_IDS), vocab]
    if list(pre_f32.shape) != expected_prefill:
        raise RuntimeError(f"unexpected prefill shape: {tuple(pre_f32.shape)}")
    if list(dec_f32.shape) != expected_decode:
        raise RuntimeError(f"unexpected decode shape: {tuple(dec_f32.shape)}")

    fixture = {
        "fixture_version": 1,
        "model_type": "xing4_0",
        "vocab_size": vocab,
        "input_ids": PREFILL_IDS,
        "decode_ids": DECODE_IDS,
        "prefill_shape": expected_prefill,
        "decode_shape": expected_decode,
        "prefill_logits_f32": flattened(pre_f32),
        "decode_logits_f32": flattened(dec_f32),
        "prefill_logits_bf16": flattened(pre_bf16),
        "decode_logits_bf16": flattened(dec_bf16),
    }

    if include_long:
        # Position 4096 is beyond the declared original YaRN window.  Only the
        # final decode row is retained; the input ids are the reproducible key.
        long_ids = [((i * 37 + 11) % vocab) for i in range(4096)]
        long_ids[0] = 1
        long_ids[-1] = 2
        _, long_bf16 = run_sequence(ref_bf16, long_ids, [17])
        _, long_f32 = run_sequence(ref_f32, long_ids, [17])
        fixture.update(
            {
                "long_prefix_ids": long_ids,
                "long_decode_id": 17,
                "long_logits_f32": flattened(long_f32[-1]),
                "long_logits_bf16": flattened(long_bf16[-1]),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"[xing4] wrote fixture {out} "
        f"(short prefill={len(PREFILL_IDS)}, decode={len(DECODE_IDS)}, "
        f"long={'yes' if include_long else 'no'})"
    )
    print(
        "[xing4] checkpoint/reference max short abs delta: "
        f"prefill={(pre_bf16 - pre_f32).abs().max().item():.6g}, "
        f"decode={(dec_bf16 - dec_f32).abs().max().item():.6g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="read-only Xing reference directory",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--out", type=Path, required=True)

    p_dump = sub.add_parser("dump")
    p_dump.add_argument("--model", type=Path, required=True)
    p_dump.add_argument("--out", type=Path, required=True)
    p_dump.add_argument("--no-long", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        build(args.out, args.reference_dir)
    else:
        dump(args.model, args.out, args.reference_dir, not args.no_long)


if __name__ == "__main__":
    main()
