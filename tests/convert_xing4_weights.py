#!/usr/bin/env python3
"""Convert the read-only Xing4.0-29B-A4B checkpoint to a native MLX pack.

The output is a normal MLX affine checkpoint.  It has no Python dependency at
serve time:

* routed trunk experts (layers 2..39) are stacked as affine 4-bit, group 64;
* every other 2-D trunk linear is affine 8-bit, group 64;
* norms, router correction vectors, mHC bases/scales, and all other 1-D or
  scalar tensors stay byte-for-byte unchanged;
* the separate layer-40 MTP subtree is omitted.

The converter reads safetensors by header and seeks one tensor at a time.  A
whole source shard is never loaded into RAM.  MLX is used only on its CPU
stream for the affine pack/depack operation.

Examples:

    python3 tests/convert_xing4_weights.py --self-test
    python3 tests/convert_xing4_weights.py \
        --src /Users/beam/llm/models/Xing4.0-29B-A4B \
        --out zig-out/models/Xing4.0-29B-A4B-Affine-MoE4-Dense8-G64 \
        --verify
    python3 tests/convert_xing4_weights.py \
        --verify-existing zig-out/models/Xing4.0-29B-A4B-Affine-MoE4-Dense8-G64 \
        --src /Users/beam/llm/models/Xing4.0-29B-A4B
    python3 tests/convert_xing4_weights.py \
        --verify-existing zig-out/models/old-pack --write-checksums
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePath
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np


CONVERTER_VERSION = "xing4-mixed-affine-v1"
WEIGHT_MAP_NAME = "model.safetensors.index.json"
AFFINE_GROUP = 64
DENSE_BITS = 8
EXPERT_BITS = 4
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U8": 1,
    "I8": 1,
    "U32": 4,
    "I32": 4,
    "I64": 8,
}
FLOAT_SOURCE_DTYPES = {"BF16", "F16", "F32"}
EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")

_mx = None


class InvalidShardPathError(ValueError):
    """An index shard name is not a local path below its checkpoint."""


def mlx():
    """Import MLX lazily, and force every conversion operation onto the CPU."""

    global _mx
    if _mx is None:
        import mlx.core as mx

        _mx = mx
        _mx.set_default_device(_mx.cpu)
    return _mx


def f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    """Round f32 to the BF16 bit pattern used by MLX's quantizer."""

    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_u16_to_f32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint16)
    return (values.astype(np.uint32) << 16).view(np.float32)


@dataclass(frozen=True)
class RawTensor:
    """A safetensors tensor without a second decoded copy."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @property
    def nbytes(self) -> int:
        count = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        return count * DTYPE_BYTES[self.dtype]

    def validate(self, name: str) -> None:
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"{name}: unsupported dtype {self.dtype}")
        if len(self.data) != self.nbytes:
            raise ValueError(
                f"{name}: {self.dtype}{self.shape} has {len(self.data)} bytes, "
                f"expected {self.nbytes}"
            )

    def numpy(self) -> np.ndarray:
        self.validate("tensor")
        if self.dtype == "BF16":
            dt = np.dtype("<u2")
        elif self.dtype == "F16":
            dt = np.dtype("<f2")
        elif self.dtype == "F32":
            dt = np.dtype("<f4")
        elif self.dtype == "U8":
            dt = np.dtype("u1")
        elif self.dtype == "I8":
            dt = np.dtype("i1")
        elif self.dtype == "U32":
            dt = np.dtype("<u4")
        elif self.dtype == "I32":
            dt = np.dtype("<i4")
        elif self.dtype == "I64":
            dt = np.dtype("<i8")
        else:
            raise ValueError(f"cannot decode {self.dtype}")
        return np.frombuffer(self.data, dtype=dt).reshape(self.shape)

    def f32(self) -> np.ndarray:
        if self.dtype == "BF16":
            return bf16_u16_to_f32(self.numpy())
        if self.dtype == "F16":
            return self.numpy().astype(np.float32)
        if self.dtype == "F32":
            return self.numpy().astype(np.float32, copy=False)
        raise ValueError(f"cannot interpret {self.dtype} as floating point")

    def f32_rows(self, start: int, stop: int) -> np.ndarray:
        """Decode only a row range; full-matrix f32 copies break the RAM bound."""

        if len(self.shape) == 0:
            return self.f32()
        rows = self.numpy()[start:stop]
        if self.dtype == "BF16":
            return bf16_u16_to_f32(rows)
        if self.dtype in {"F16", "F32"}:
            return rows.astype(np.float32)
        raise ValueError(f"cannot interpret {self.dtype} as floating point")

    def as_bf16_mlx(self):
        """Create one CPU MLX BF16 array from this tensor."""

        mx = mlx()
        if self.dtype == "BF16":
            # MLX copies the NumPy view into its own CPU buffer.  Keeping this
            # view zero-copy avoids a second full BF16 staging buffer for the
            # 64-expert bank and the 131k-row embedding.
            bits = np.array(self.numpy(), dtype=np.uint16, copy=False)
        elif self.dtype in {"F16", "F32"}:
            bits = f32_to_bf16_u16(self.f32())
        else:
            raise ValueError(f"quantization needs a floating tensor, got {self.dtype}")
        return mx.array(bits).reshape(self.shape).view(mx.bfloat16)


class ShardReader:
    """Header-validated, random-access reader for one safetensors shard."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{path}: truncated safetensors length")
            header_len = struct.unpack("<Q", prefix)[0]
            self.header_bytes = stream.read(header_len)
            if len(self.header_bytes) != header_len:
                raise ValueError(f"{path}: truncated safetensors header")
        try:
            self.header = json.loads(self.header_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid safetensors header") from exc
        if not isinstance(self.header, dict):
            raise ValueError(f"{path}: safetensors header is not an object")
        self.data_offset = 8 + len(self.header_bytes)
        self.file_size = path.stat().st_size
        self.validate_ranges()

    def names(self) -> list[str]:
        return [name for name in self.header if name != "__metadata__"]

    def metadata(self, name: str) -> tuple[str, tuple[int, ...], int, int]:
        """Return one tensor's typed geometry and payload-relative range."""

        meta = self.header.get(name)
        if not isinstance(meta, dict):
            raise KeyError(f"{self.path}: missing tensor {name}")
        dtype = meta.get("dtype")
        raw_shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(raw_shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"{self.path}: malformed metadata for {name}")
        try:
            shape = tuple(int(x) for x in raw_shape)
            begin, end = (int(x) for x in offsets)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.path}: malformed metadata for {name}") from exc
        if any(x < 0 for x in shape) or begin < 0 or end < begin:
            raise ValueError(f"{self.path}: invalid metadata for {name}")
        return dtype, shape, begin, end

    def validate_ranges(self) -> None:
        """Validate every declared payload range against the actual file."""

        payload_bytes = self.file_size - self.data_offset
        if payload_bytes < 0:
            raise ValueError(f"{self.path}: file ends inside safetensors header")
        ranges = []
        for name in self.names():
            dtype, shape, begin, end = self.metadata(name)
            expected = int(np.prod(shape, dtype=np.int64)) * DTYPE_BYTES[dtype]
            if end - begin != expected or end > payload_bytes:
                raise ValueError(f"{self.path}:{name}: invalid data range")
            ranges.append((begin, end, name))
        ranges.sort()
        previous_end = 0
        for begin, end, name in ranges:
            if begin != previous_end:
                relation = "overlapping" if begin < previous_end else "gapped"
                raise ValueError(f"{self.path}:{name}: {relation} data range")
            previous_end = end
        if previous_end != payload_bytes:
            raise ValueError(
                f"{self.path}: file length does not end at the last tensor "
                f"(payload={payload_bytes}, last={previous_end})"
            )

    def validate_readback(self, sample_bytes: int = 64) -> None:
        """Read both ends of every declared payload range from disk."""

        with self.path.open("rb") as stream:
            for name in self.names():
                _dtype, _shape, begin, end = self.metadata(name)
                length = end - begin
                if length == 0:
                    continue
                for offset, count in (
                    (begin, min(length, sample_bytes)),
                    (max(begin, end - sample_bytes), min(length, sample_bytes)),
                ):
                    stream.seek(self.data_offset + offset)
                    data = stream.read(count)
                    if len(data) != count:
                        raise ValueError(f"{self.path}:{name}: payload readback truncated")

    def read(self, name: str) -> RawTensor:
        dtype, shape, begin, end = self.metadata(name)
        with self.path.open("rb") as stream:
            stream.seek(self.data_offset + begin)
            data = stream.read(end - begin)
        tensor = RawTensor(dtype, shape, data)
        tensor.validate(f"{self.path}:{name}")
        return tensor


def write_safetensors_raw(
    path: Path, tensors: dict[str, RawTensor], metadata: dict[str, str] | None = None
) -> None:
    """Write a deterministic safetensors file without materializing tensors."""

    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in sorted(metadata.items())}
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        tensor.validate(name)
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(tensor.data)],
        }
        offset += len(tensor.data)
    encoded = json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name in sorted(tensors):
            stream.write(tensors[name].data)
    os.replace(temporary, path)


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int
    role: str


def layer_number(name: str) -> int | None:
    match = LAYER_RE.match(name)
    return int(match.group("layer")) if match else None


def expert_parts(name: str) -> tuple[int, int, str] | None:
    match = EXPERT_RE.match(name)
    if not match:
        return None
    return (
        int(match.group("layer")),
        int(match.group("expert")),
        match.group("projection"),
    )


def role_for_linear(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    if name.endswith(".mlp.gate.weight"):
        return "router"
    if ".mlp.shared_experts." in name:
        return "shared_expert"
    if name.endswith(".attn_hc.hc_fn") or name.endswith(".ffn_hc.hc_fn"):
        return "mhc_projection"
    return "dense_linear"


def allocation_for(name: str, shape: Iterable[int], num_hidden_layers: int = 40) -> QuantSpec | None:
    """Return the one quantization rule allowed for a source tensor."""

    shape = tuple(int(x) for x in shape)
    li = layer_number(name)
    if len(shape) != 2 or (li is not None and li >= num_hidden_layers):
        return None
    if expert_parts(name) is not None:
        return QuantSpec(EXPERT_BITS, AFFINE_GROUP, "routed_expert")
    return QuantSpec(DENSE_BITS, AFFINE_GROUP, role_for_linear(name))


def canonical_name(name: str) -> str:
    """Map source HF names onto transformer.zig's native Xing names."""

    match = re.match(r"^(model\.layers\.\d+)\.self_attn\.(.+)$", name)
    if match:
        tail = match.group(2)
        if tail.startswith("o_proj."):
            tail = "dense." + tail[len("o_proj.") :]
        return f"{match.group(1)}.attention.{tail}"
    if name.endswith(".mlp.gate.e_score_correction_bias"):
        return name[: -len("e_score_correction_bias")] + "expert_bias"
    return name


def quantized_shapes(shape: Iterable[int], bits: int, group_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return MLX affine packed and per-group shapes for one 2-D matrix."""

    shape = tuple(int(x) for x in shape)
    if len(shape) < 2:
        raise ValueError(f"matrix geometry {shape} has no input axis")
    leading, cols = shape[:-1], shape[-1]
    if any(x <= 0 for x in leading) or cols <= 0 or cols % group_size != 0:
        raise ValueError(f"matrix geometry {tuple(shape)} is not divisible by group size {group_size}")
    packed_bits = cols * bits
    if packed_bits % 32:
        raise ValueError(f"matrix geometry {tuple(shape)} cannot pack {bits}-bit values into u32")
    return leading + (packed_bits // 32,), leading + (cols // group_size,)


def affine_quantize(source: RawTensor, spec: QuantSpec) -> tuple[RawTensor, RawTensor, RawTensor]:
    """Quantize one source matrix with MLX's native affine packing."""

    if len(source.shape) < 2:
        raise ValueError(f"{spec.role}: only matrices can be quantized, got {source.shape}")
    q_shape, side_shape = quantized_shapes(source.shape, spec.bits, spec.group_size)
    mx = mlx()
    values = source.as_bf16_mlx()
    packed, scales, biases = mx.quantize(
        values, group_size=spec.group_size, bits=spec.bits
    )
    mx.eval(packed, scales, biases)
    q = RawTensor("U32", tuple(int(x) for x in packed.shape), np.array(packed).tobytes())
    sc = RawTensor(
        "BF16",
        tuple(int(x) for x in scales.shape),
        np.array(scales.view(mx.uint16)).tobytes(),
    )
    bi = RawTensor(
        "BF16",
        tuple(int(x) for x in biases.shape),
        np.array(biases.view(mx.uint16)).tobytes(),
    )
    if q.shape != q_shape or sc.shape != side_shape or bi.shape != side_shape:
        raise ValueError(
            f"{spec.role}: MLX returned {q.shape}/{sc.shape}/{bi.shape}, "
            f"expected {q_shape}/{side_shape}/{side_shape}"
        )
    return q, sc, bi


def _sample_row_ranges(rows: int) -> list[tuple[int, int]]:
    if rows <= 16:
        return [(0, rows)]
    starts = sorted({0, rows // 3, (2 * rows) // 3, rows - 8})
    return [(start, min(rows, start + 8)) for start in starts if start < rows]


def verify_dequantized(
    source: RawTensor,
    quantized: tuple[RawTensor, RawTensor, RawTensor],
    spec: QuantSpec,
    name: str,
) -> None:
    """Check packed geometry and deterministic row samples without a full copy."""

    q, sc, bi = quantized
    expected_q, expected_side = quantized_shapes(
        source.shape, spec.bits, spec.group_size
    )
    if q.shape != expected_q or sc.shape != expected_side or bi.shape != expected_side:
        raise AssertionError(
            f"{name}: bad affine geometry {q.shape}/{sc.shape}/{bi.shape}, "
            f"expected {expected_q}/{expected_side}/{expected_side}"
        )
    mx = mlx()
    q_np = np.frombuffer(q.data, dtype="<u4").reshape(q.shape)
    sc_np = np.frombuffer(sc.data, dtype="<u2").reshape(sc.shape)
    bi_np = np.frombuffer(bi.data, dtype="<u2").reshape(bi.shape)
    for start, stop in _sample_row_ranges(source.shape[0]):
        q_rows = mx.array(np.array(q_np[start:stop], copy=True))
        sc_rows = mx.array(np.array(sc_np[start:stop], copy=True)).view(mx.bfloat16)
        bi_rows = mx.array(np.array(bi_np[start:stop], copy=True)).view(mx.bfloat16)
        back = mx.dequantize(
            q_rows,
            sc_rows,
            bi_rows,
            group_size=spec.group_size,
            bits=spec.bits,
        )
        mx.eval(back)
        got = np.array(back.astype(mx.float32))
        expected = source.f32_rows(start, stop)
        if not np.isfinite(got).all():
            raise AssertionError(f"{name}: dequantized sample contains non-finite values")
        denominator = float(np.linalg.norm(expected) * np.linalg.norm(got))
        if denominator == 0.0:
            error = float(np.max(np.abs(expected - got)))
            if error > (0.02 if spec.bits == 8 else 0.35):
                raise AssertionError(f"{name}: zero-norm dequant error {error}")
        else:
            cosine = float(np.sum(expected * got) / denominator)
            threshold = 0.995 if spec.bits == 8 else 0.90
            if cosine < threshold:
                raise AssertionError(
                    f"{name}: {spec.bits}-bit dequant cosine {cosine:.6f} < {threshold}"
                )
    del q_np, sc_np, bi_np


def _header_for(path: Path) -> dict:
    return ShardReader(path).header


def _local_shard_path(root: Path, shard: str, label: str) -> Path:
    """Resolve an index filename without permitting an external dependency."""

    if not isinstance(shard, str):
        raise InvalidShardPathError(f"{label} shard path {shard!r} is not local")
    relative = PurePath(shard)
    if relative.is_absolute() or ".." in relative.parts:
        raise InvalidShardPathError(f"{label} shard path {shard!r} is not local")
    root = root.resolve()
    path = (root / Path(shard)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise InvalidShardPathError(
            f"{label} shard path {shard!r} is not local to {root}"
        ) from exc
    return path


def _indexed_readers(
    root: Path, weight_map: dict[str, str], label: str
) -> dict[str, ShardReader]:
    """Open indexed shards and share header/range validation for both sides."""

    readers: dict[str, ShardReader] = {}
    reverse: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError(f"{label}: weight_map names and shards must be strings")
        reverse.setdefault(shard, []).append(name)
    for shard in sorted(reverse):
        path = _local_shard_path(root, shard, label)
        if not path.is_file():
            raise FileNotFoundError(f"{root}: index names missing shard {shard}")
        reader = ShardReader(path)
        header_names = set(reader.names())
        indexed_names = set(reverse[shard])
        if header_names != indexed_names:
            missing = sorted(indexed_names - header_names)
            extra = sorted(header_names - indexed_names)
            raise ValueError(
                f"{shard}: index/header mismatch "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        readers[shard] = reader
    return readers


def _config_int(config: dict, key: str, src: Path, *, positive: bool = True) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{src}: config missing integer geometry scalar {key}")
    if positive and value <= 0:
        raise ValueError(f"{src}: config geometry scalar {key} must be positive")
    return value


def _expected_source_tensors(
    config: dict, src: Path
) -> dict[str, tuple[tuple[int, ...], frozenset[str]]]:
    """Derive the base Xing trunk contract from config geometry."""

    hidden = _config_int(config, "hidden_size", src)
    vocab = _config_int(config, "vocab_size", src)
    layers = _config_int(config, "num_hidden_layers", src)
    heads = _config_int(config, "num_attention_heads", src)
    q_lora = _config_int(config, "q_lora_rank", src)
    kv_lora = _config_int(config, "kv_lora_rank", src)
    qk_nope = _config_int(config, "qk_nope_head_dim", src)
    qk_rope = _config_int(config, "qk_rope_head_dim", src)
    v_head = _config_int(config, "v_head_dim", src)
    intermediate = _config_int(config, "intermediate_size", src)
    moe_intermediate = _config_int(config, "moe_intermediate_size", src)
    experts = _config_int(config, "n_routed_experts", src)
    first_dense = int(config.get("first_k_dense_replace", 0))
    shared_experts = int(config.get("n_shared_experts", 1))
    hc_mult = _config_int(config, "hc_mult", src)
    if first_dense < 0 or first_dense > layers:
        raise ValueError(f"{src}: first_k_dense_replace is outside the layer range")
    if shared_experts <= 0:
        raise ValueError(f"{src}: n_shared_experts must be positive")
    if "qk_head_dim" in config and config["qk_head_dim"] != qk_nope + qk_rope:
        raise ValueError(f"{src}: qk_head_dim disagrees with MLA component dimensions")

    bf16 = frozenset({"BF16"})
    f32 = frozenset({"F32"})
    expected: dict[str, tuple[tuple[int, ...], frozenset[str]]] = {
        "model.embed_tokens.weight": ((vocab, hidden), bf16),
        "lm_head.weight": ((vocab, hidden), bf16),
        "model.norm.weight": ((hidden,), bf16),
    }
    hc_mix = (2 + hc_mult) * hc_mult
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.self_attn.q_a_proj.weight": ((q_lora, hidden), bf16),
                f"{prefix}.self_attn.q_a_layernorm.weight": ((q_lora,), bf16),
                f"{prefix}.self_attn.q_b_proj.weight": (
                    (heads * (qk_nope + qk_rope), q_lora),
                    bf16,
                ),
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": (
                    (kv_lora + qk_rope, hidden),
                    bf16,
                ),
                f"{prefix}.self_attn.kv_a_layernorm.weight": ((kv_lora,), bf16),
                f"{prefix}.self_attn.kv_b_proj.weight": (
                    (heads * (qk_nope + v_head), kv_lora),
                    bf16,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    (hidden, heads * v_head),
                    bf16,
                ),
                f"{prefix}.input_layernorm.weight": ((hidden,), bf16),
                f"{prefix}.post_attention_layernorm.weight": ((hidden,), bf16),
                f"{prefix}.attn_hc.hc_fn": ((hc_mix, hc_mult * hidden), bf16),
                f"{prefix}.attn_hc.hc_base": ((hc_mix,), bf16),
                f"{prefix}.attn_hc.hc_scale": ((3,), f32),
                f"{prefix}.ffn_hc.hc_fn": ((hc_mix, hc_mult * hidden), bf16),
                f"{prefix}.ffn_hc.hc_base": ((hc_mix,), bf16),
                f"{prefix}.ffn_hc.hc_scale": ((3,), f32),
            }
        )
        if layer < first_dense:
            expected.update(
                {
                    f"{prefix}.mlp.gate_proj.weight": ((intermediate, hidden), bf16),
                    f"{prefix}.mlp.up_proj.weight": ((intermediate, hidden), bf16),
                    f"{prefix}.mlp.down_proj.weight": ((hidden, intermediate), bf16),
                }
            )
        else:
            shared_width = shared_experts * moe_intermediate
            expected.update(
                {
                    f"{prefix}.mlp.gate.weight": ((experts, hidden), bf16),
                    f"{prefix}.mlp.gate.e_score_correction_bias": ((experts,), f32),
                    f"{prefix}.mlp.shared_experts.gate_proj.weight": (
                        (shared_width, hidden),
                        bf16,
                    ),
                    f"{prefix}.mlp.shared_experts.up_proj.weight": (
                        (shared_width, hidden),
                        bf16,
                    ),
                    f"{prefix}.mlp.shared_experts.down_proj.weight": (
                        (hidden, shared_width),
                        bf16,
                    ),
                }
            )
            for expert in range(experts):
                for projection, shape in (
                    ("gate_proj", (moe_intermediate, hidden)),
                    ("up_proj", (moe_intermediate, hidden)),
                    ("down_proj", (hidden, moe_intermediate)),
                ):
                    expected[
                        f"{prefix}.mlp.experts.{expert}.{projection}.weight"
                    ] = (shape, bf16)
    return expected


def _validate_expected_source_tensors(
    src: Path,
    config: dict,
    weight_map: dict[str, str],
    readers: dict[str, ShardReader],
) -> None:
    expected = _expected_source_tensors(config, src)
    for name, (expected_shape, expected_dtypes) in expected.items():
        shard = weight_map.get(name)
        if shard is None:
            raise ValueError(f"{src}: required tensor {name} is missing")
        dtype, shape, _begin, _end = readers[shard].metadata(name)
        if shape != expected_shape:
            raise ValueError(
                f"{src}:{name}: shape {shape} does not match config-derived "
                f"shape {expected_shape}"
            )
        if dtype not in expected_dtypes:
            allowed = ",".join(sorted(expected_dtypes))
            raise ValueError(f"{src}:{name}: dtype {dtype} is not one of {allowed}")


def validate_source(src: Path, strict: bool = True) -> tuple[dict, dict[str, str], dict[str, ShardReader]]:
    """Validate config, index, every shard header, and every byte range."""

    src = src.resolve()
    config_path = src / "config.json"
    index_path = src / WEIGHT_MAP_NAME
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"{src}: config.json and {WEIGHT_MAP_NAME} are required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "xing4_0":
        raise ValueError(f"{src}: expected model_type=xing4_0")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{src}: malformed safetensors index")
    weight_map = index["weight_map"]
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
        raise ValueError(f"{src}: weight_map names and shards must be strings")
    readers = _indexed_readers(src, weight_map, "source")
    if strict:
        expected = {
            "num_hidden_layers": 40,
            "n_routed_experts": 64,
            "first_k_dense_replace": 2,
            "n_shared_experts": 1,
            "num_nextn_predict_layers": 1,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{src}: expected {key}={value}, got {config.get(key)!r}")
    _validate_expected_source_tensors(src, config, weight_map, readers)
    return config, weight_map, readers


def source_state(src: Path) -> dict[str, tuple[int, int, int]]:
    """Capture metadata for a no-source-mutation guard without reading payloads."""

    result = {}
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        stat = path.stat()
        result[str(path.relative_to(src))] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return result


def header_digest(readers: dict[str, ShardReader]) -> str:
    digest = hashlib.sha256()
    for shard in sorted(readers):
        digest.update(shard.encode())
        digest.update(readers[shard].header_bytes)
    return digest.hexdigest()


def read_source_tensor(
    name: str, weight_map: dict[str, str], readers: dict[str, ShardReader]
) -> RawTensor:
    return readers[weight_map[name]].read(name)


def add_tensor(targets: dict[str, RawTensor], name: str, tensor: RawTensor) -> None:
    if name in targets:
        raise ValueError(f"duplicate output tensor {name}")
    targets[name] = tensor


def quant_metadata_entry(spec: QuantSpec) -> dict[str, object]:
    return {"bits": spec.bits, "group_size": spec.group_size, "mode": "affine", "role": spec.role}


def copy_sidecars(src: Path, out: Path) -> list[str]:
    copied = []
    names = (
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    )
    for name in names:
        source = src / name
        if source.is_file():
            shutil.copyfile(source, out / name)
            copied.append(name)
    return copied


def output_header_digest(out: Path, shards: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for shard in sorted(shards):
        path = _local_shard_path(out, shard, "output")
        reader = ShardReader(path)
        digest.update(shard.encode())
        digest.update(reader.header_bytes)
    return digest.hexdigest()


def shard_checksums(reader: ShardReader) -> dict[str, str]:
    """Hash one emitted shard without materializing its payload."""

    full = hashlib.sha256()
    payload = hashlib.sha256()
    offset = 0
    with reader.path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            full.update(chunk)
            chunk_end = offset + len(chunk)
            begin = max(offset, reader.data_offset)
            end = min(chunk_end, reader.file_size)
            if begin < end:
                payload.update(chunk[begin - offset : end - offset])
            offset = chunk_end
    if offset != reader.file_size:
        raise ValueError(f"{reader.path}: file readback ended at {offset}, expected {reader.file_size}")
    return {"sha256": full.hexdigest(), "payload_sha256": payload.hexdigest()}


def _expected_output_contract(
    config: dict,
    weight_map: dict[str, str],
    readers: dict[str, ShardReader],
) -> tuple[set[str], dict[str, tuple[tuple[int, ...], QuantSpec]]]:
    """Derive converted names and affine geometry from the validated source."""

    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["n_routed_experts"])
    first_dense = int(config.get("first_k_dense_replace", 0))
    names: set[str] = set()
    quantized: dict[str, tuple[tuple[int, ...], QuantSpec]] = {}
    for source_name, shard in weight_map.items():
        layer = layer_number(source_name)
        if layer is not None and layer >= num_layers:
            continue
        parts = expert_parts(source_name)
        if parts is not None:
            continue
        dtype, shape, _begin, _end = readers[shard].metadata(source_name)
        del dtype
        target = canonical_name(source_name)
        if len(shape) == 2:
            spec = allocation_for(source_name, shape, num_layers)
            if spec is None:
                raise ValueError(f"{source_name}: no allocation rule")
            base = target[: -len(".weight")] if target.endswith(".weight") else target
            for suffix in (".weight", ".scales", ".biases"):
                names.add(base + suffix)
            quantized[base] = (shape, spec)
        else:
            names.add(target)

    for layer in range(first_dense, num_layers):
        prefix = f"model.layers.{layer}."
        for projection in ("gate_proj", "up_proj", "down_proj"):
            expert_name = f"{prefix}mlp.experts.0.{projection}.weight"
            if expert_name not in weight_map:
                raise ValueError(f"missing routed expert tensor {expert_name}")
            _dtype, shape, _begin, _end = readers[weight_map[expert_name]].metadata(expert_name)
            stacked_shape = (num_experts,) + shape
            base = f"{prefix}mlp.switch_mlp.{projection}"
            for suffix in (".weight", ".scales", ".biases"):
                names.add(base + suffix)
            quantized[base] = (
                stacked_shape,
                QuantSpec(EXPERT_BITS, AFFINE_GROUP, "routed_expert"),
            )
    return names, quantized


def _verify_affine_output(
    weight_map: dict[str, str],
    readers: dict[str, ShardReader],
    quant_metadata: dict[str, dict[str, object]],
) -> None:
    """Check every on-disk U32 affine triple, not only manifest-listed entries."""

    bases: set[str] = set()
    for name, shard in weight_map.items():
        dtype, _shape, _begin, _end = readers[shard].metadata(name)
        if name.endswith(".weight") and dtype == "U32":
            bases.add(name[: -len(".weight")])
    for base in quant_metadata:
        if base not in bases:
            raise AssertionError(f"{base}: manifest affine tensor is absent on disk")
    for base in sorted(bases):
        shard = weight_map.get(base + ".weight")
        if shard is None:
            raise AssertionError(f"{base}: affine weight is not indexed")
        reader = readers[shard]
        try:
            q_dtype, q_shape, _q_begin, _q_end = reader.metadata(base + ".weight")
            sc_dtype, sc_shape, _sc_begin, _sc_end = reader.metadata(base + ".scales")
            bi_dtype, bi_shape, _bi_begin, _bi_end = reader.metadata(base + ".biases")
        except KeyError as exc:
            raise AssertionError(f"{base}: missing affine side tensor") from exc
        if q_dtype != "U32" or sc_dtype != "BF16" or bi_dtype != "BF16":
            raise AssertionError(f"{base}: affine triple dtype mismatch")
        if not q_shape or sc_shape != bi_shape or q_shape[:-1] != sc_shape[:-1]:
            raise AssertionError(f"{base}: affine triple shape mismatch")
        if not sc_shape or sc_shape[-1] <= 0:
            raise AssertionError(f"{base}: affine side shape is empty")
        packed_bits = q_shape[-1] * 32
        grouped_width = sc_shape[-1] * AFFINE_GROUP
        if packed_bits % grouped_width:
            raise AssertionError(f"{base}: affine packed geometry is not integral")
        bits = packed_bits // grouped_width
        if bits not in (EXPERT_BITS, DENSE_BITS):
            raise AssertionError(f"{base}: unsupported emitted affine bit width {bits}")
        if base in quant_metadata:
            spec = quant_metadata[base]
            if int(spec["bits"]) != bits or int(spec["group_size"]) != AFFINE_GROUP:
                raise AssertionError(f"{base}: manifest affine geometry disagrees with disk")
            if bits == EXPERT_BITS and "routed_expert" not in str(spec["role"]):
                raise AssertionError(f"{base}: non-expert accidentally allocated at 4-bit")


def _manifest_shards(manifest: dict | None) -> dict[str, dict]:
    if not isinstance(manifest, dict):
        return {}
    entries = manifest.get("output_shards")
    if not isinstance(entries, list):
        return {}
    result = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            result[entry["name"]] = entry
    return result


def verify_output(
    out: Path,
    index: dict[str, object],
    quant_metadata: dict[str, dict[str, object]] | None = None,
    *,
    manifest: dict | None = None,
    source: Path | None = None,
) -> dict[str, object]:
    """Verify emitted ranges, readback samples, affine triples, and checksums."""

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AssertionError("output index has no weight_map")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
        raise AssertionError("output index weight_map names and shards must be strings")
    out = out.resolve()
    if manifest is None:
        manifest_path = out / "conversion_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readers = _indexed_readers(out, weight_map, "output")
    for reader in readers.values():
        reader.validate_readback()
    quant_metadata = quant_metadata or {}
    _verify_affine_output(weight_map, readers, quant_metadata)
    expected_files = set(weight_map.values())
    actual_files = {path.name for path in out.glob("*.safetensors")}
    if actual_files != expected_files:
        raise AssertionError(f"output shard set mismatch: {sorted(actual_files ^ expected_files)}")
    manifest_shards = _manifest_shards(manifest)
    actual_checksums: dict[str, dict[str, str]] = {}
    missing_checksums = []
    for shard, reader in sorted(readers.items()):
        actual = shard_checksums(reader)
        actual_checksums[shard] = actual
        info = manifest_shards.get(shard, {})
        if info.get("size") is not None and int(info["size"]) != reader.file_size:
            raise AssertionError(f"{shard}: manifest size disagrees with disk")
        expected_file = info.get("sha256")
        expected_payload = info.get("payload_sha256")
        if expected_file is None or expected_payload is None:
            missing_checksums.append(shard)
            continue
        if expected_file != actual["sha256"]:
            raise AssertionError(f"{shard}: file checksum mismatch")
        if expected_payload != actual["payload_sha256"]:
            raise AssertionError(f"{shard}: payload checksum mismatch")

    if source is not None:
        source = source.resolve()
        source_config, source_map, source_readers = validate_source(source, strict=False)
        expected_names, expected_quantized = _expected_output_contract(
            source_config, source_map, source_readers
        )
        actual_names = set(weight_map)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise AssertionError(
                f"source/output coverage mismatch missing={missing[:3]} extra={extra[:3]}"
            )
        for base, (shape, spec) in sorted(expected_quantized.items()):
            shard = weight_map[base + ".weight"]
            reader = readers[shard]
            expected_q, expected_side = quantized_shapes(
                shape, spec.bits, spec.group_size
            )
            _dtype, q_shape, _begin, _end = reader.metadata(base + ".weight")
            _dtype, sc_shape, _begin, _end = reader.metadata(base + ".scales")
            if q_shape != expected_q or sc_shape != expected_side:
                raise AssertionError(
                    f"{base}: output geometry {q_shape}/{sc_shape} does not match "
                    f"source geometry {expected_q}/{expected_side}"
                )
        if manifest is not None:
            source_digest = manifest.get("source_header_sha256")
            if source_digest is not None and source_digest != header_digest(source_readers):
                raise AssertionError("source header checksum does not match output manifest")
    return {
        "checksum_status": "missing" if missing_checksums else "verified",
        "checksums": actual_checksums,
        "missing_checksum_shards": missing_checksums,
    }


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_output(out: Path) -> tuple[dict[str, object], dict, dict | None]:
    out = out.resolve()
    if not out.is_dir():
        raise FileNotFoundError(f"output pack does not exist: {out}")
    config_path = out / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{out}: config.json is required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "xing4_0":
        raise ValueError(f"{out}: expected output model_type=xing4_0")
    index_path = out / WEIGHT_MAP_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"{out}: {WEIGHT_MAP_NAME} is required")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    manifest_path = out / "conversion_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    quant_metadata = (
        manifest.get("quantized_tensors", {})
        if isinstance(manifest, dict)
        else {}
    )
    if not isinstance(quant_metadata, dict):
        quant_metadata = {}
    return index, quant_metadata, manifest


def verify_existing(out: Path, source: Path | None = None) -> dict[str, object]:
    """Audit an existing output pack without changing any file."""

    out = out.resolve()
    index, quant_metadata, manifest = _load_output(out)
    result = verify_output(
        out,
        index,
        quant_metadata,
        manifest=manifest,
        source=source,
    )
    result["output"] = str(out)
    result["manifest_checksums"] = result["checksum_status"] == "verified"
    return result


def write_checksums(out: Path, source: Path | None = None) -> dict[str, object]:
    """Explicitly adopt SHA256 metadata without rewriting weight shards."""

    out = out.resolve()
    manifest_path = out / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{out}: conversion_manifest.json is required for checksum adoption"
        )
    index, quant_metadata, manifest = _load_output(out)
    assert manifest is not None
    result = verify_output(
        out,
        index,
        quant_metadata,
        manifest=manifest,
        source=source,
    )
    entries = manifest.get("output_shards")
    if not isinstance(entries, list):
        entries = []
    by_name = {
        entry["name"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    for shard, checksums in sorted(result["checksums"].items()):
        info = by_name.setdefault(shard, {"name": shard})
        info["size"] = int(
            ShardReader(_local_shard_path(out, shard, "output")).file_size
        )
        info.update(checksums)
    manifest["output_shards"] = [by_name[name] for name in sorted(by_name)]
    manifest["integrity"] = {
        "algorithm": "sha256",
        "scopes": ["file", "payload"],
    }
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    _json_write(temporary, manifest)
    os.replace(temporary, manifest_path)
    result["checksum_status"] = "verified"
    result["missing_checksum_shards"] = []
    result["checksums_written"] = True
    result["output"] = str(out)
    result["manifest_checksums"] = True
    return result


def convert(
    src: Path,
    out: Path,
    *,
    verify: bool = False,
    strict_source: bool = True,
) -> dict[str, object]:
    """Convert one source directory without ever writing beneath it."""

    src = src.resolve()
    out = out.resolve()
    if out == src or src in out.parents:
        raise ValueError("output must not be inside the read-only source directory")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output {out}")
    config, weight_map, readers = validate_source(src, strict=strict_source)
    before = source_state(src)
    out.mkdir(parents=True)
    source_headers = header_digest(readers)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["n_routed_experts"])
    first_dense = int(config.get("first_k_dense_replace", 0))
    copied_sidecars = copy_sidecars(src, out)
    output_index: dict[str, str] = {}
    quant_metadata: dict[str, dict[str, object]] = {}
    shard_info: list[dict[str, object]] = []
    processed: set[str] = set()
    excluded: set[str] = set()
    quantized_counts = {"4": 0, "8": 0}
    output_bytes = 0

    def process_nonexpert(
        source_name: str, targets: dict[str, RawTensor], layer: int | None
    ) -> None:
        raw = read_source_tensor(source_name, weight_map, readers)
        raw.validate(source_name)
        target = canonical_name(source_name)
        processed.add(source_name)
        if len(raw.shape) == 2 and (layer is None or layer < num_layers):
            spec = allocation_for(source_name, raw.shape, num_layers)
            if spec is None:
                raise ValueError(f"{source_name}: no allocation rule")
            triple = affine_quantize(raw, spec)
            if verify:
                verify_dequantized(raw, triple, spec, source_name)
            base = target[: -len(".weight")] if target.endswith(".weight") else target
            for suffix, value in zip((".weight", ".scales", ".biases"), triple):
                add_tensor(targets, base + suffix, value)
            quant_metadata[base] = quant_metadata_entry(spec)
            quantized_counts[str(spec.bits)] += 1
        else:
            add_tensor(targets, target, raw)

    def write_group(filename: str, targets: dict[str, RawTensor]) -> None:
        nonlocal output_bytes
        if not targets:
            raise ValueError(f"{filename}: empty output shard")
        path = out / filename
        write_safetensors_raw(
            path,
            targets,
            {
                "format": "mlx",
                "converter": CONVERTER_VERSION,
                "quantization": "affine",
            },
        )
        reader = ShardReader(path)
        checksums = shard_checksums(reader)
        shard_size = reader.file_size
        output_bytes += shard_size
        for name in targets:
            output_index[name] = filename
        shard_info.append(
            {
                "name": filename,
                "size": shard_size,
                "tensors": [
                    {
                        "name": name,
                        "dtype": targets[name].dtype,
                        "shape": list(targets[name].shape),
                    }
                    for name in sorted(targets)
                ],
                **checksums,
            }
        )

    root_targets: dict[str, RawTensor] = {}
    root_names = sorted(
        name for name in weight_map if layer_number(name) is None
    )
    for name in root_names:
        process_nonexpert(name, root_targets, None)
    write_group("model-root.safetensors", root_targets)

    for layer in range(num_layers):
        targets: dict[str, RawTensor] = {}
        prefix = f"model.layers.{layer}."
        names = sorted(name for name in weight_map if name.startswith(prefix))
        for name in names:
            if expert_parts(name) is not None:
                continue
            process_nonexpert(name, targets, layer)
        if layer < first_dense:
            write_group(f"model-layer-{layer:03d}.safetensors", targets)
            continue
        for projection in ("gate_proj", "up_proj", "down_proj"):
            source_data = bytearray()
            source_dtype: str | None = None
            source_shape: tuple[int, ...] | None = None
            for expert in range(num_experts):
                source_name = (
                    f"{prefix}mlp.experts.{expert}.{projection}.weight"
                )
                if source_name not in weight_map:
                    raise ValueError(f"missing routed expert tensor {source_name}")
                raw = read_source_tensor(source_name, weight_map, readers)
                if len(raw.shape) != 2 or raw.dtype not in FLOAT_SOURCE_DTYPES:
                    raise ValueError(f"{source_name}: expected floating 2-D source tensor")
                if source_dtype is None:
                    source_dtype = raw.dtype
                    source_shape = raw.shape
                elif raw.dtype != source_dtype or raw.shape != source_shape:
                    raise ValueError(f"{source_name}: routed expert geometry differs from expert 0")
                source_data.extend(raw.data)
                del raw
                processed.add(source_name)
            assert source_dtype is not None and source_shape is not None
            stacked_source = RawTensor(
                source_dtype,
                (num_experts,) + source_shape,
                source_data,
            )
            spec = QuantSpec(EXPERT_BITS, AFFINE_GROUP, "routed_expert")
            triple = affine_quantize(stacked_source, spec)
            if verify:
                verify_dequantized(stacked_source, triple, spec, f"{prefix}mlp.experts.*.{projection}.weight")
            del stacked_source, source_data
            q, sc, bi = triple
            target_base = (
                f"{prefix}mlp.switch_mlp.{projection}"
            )
            add_tensor(targets, target_base + ".weight", q)
            add_tensor(targets, target_base + ".scales", sc)
            add_tensor(targets, target_base + ".biases", bi)
            quant_metadata[target_base] = quant_metadata_entry(
                QuantSpec(EXPERT_BITS, AFFINE_GROUP, "routed_expert")
            )
            quantized_counts["4"] += 1
        write_group(f"model-layer-{layer:03d}.safetensors", targets)

    for name in weight_map:
        li = layer_number(name)
        if li is not None and li >= num_layers:
            excluded.add(name)
    expected_names = set(weight_map)
    if processed | excluded != expected_names:
        missing = sorted(expected_names - (processed | excluded))
        extra = sorted((processed | excluded) - expected_names)
        raise AssertionError(f"conversion coverage mismatch missing={missing[:5]} extra={extra[:5]}")

    converted_config = copy.deepcopy(config)
    original_mtp_layers = converted_config.get("num_nextn_predict_layers", 0)
    converted_config["num_nextn_predict_layers"] = 0
    converted_config["quantization"] = {
        "bits": DENSE_BITS,
        "group_size": AFFINE_GROUP,
        "mode": "affine",
        "scheme": CONVERTER_VERSION,
        "overrides": {
            "routed_experts": {
                "bits": EXPERT_BITS,
                "group_size": AFFINE_GROUP,
                "mode": "affine",
            },
            "all_other_2d_linears": {
                "bits": DENSE_BITS,
                "group_size": AFFINE_GROUP,
                "mode": "affine",
            },
        },
    }
    _json_write(out / "config.json", converted_config)

    index = {
        "metadata": {
            "format": "mlx",
            "total_size": output_bytes,
            "converter": CONVERTER_VERSION,
        },
        "weight_map": {name: output_index[name] for name in sorted(output_index)},
    }
    _json_write(out / WEIGHT_MAP_NAME, index)
    manifest = {
        "format": "mlx-serve-xing4",
        "converter": CONVERTER_VERSION,
        "source_model": src.name,
        "source_header_sha256": source_headers,
        "architecture": {
            "model_type": config.get("model_type"),
            "num_hidden_layers": num_layers,
            "n_routed_experts": num_experts,
            "first_k_dense_replace": config.get("first_k_dense_replace"),
        },
        "allocation": {
            "routed_experts": {
                "bits": EXPERT_BITS,
                "group_size": AFFINE_GROUP,
                "count": quantized_counts["4"],
            },
            "all_other_2d_linears": {
                "bits": DENSE_BITS,
                "group_size": AFFINE_GROUP,
                "count": quantized_counts["8"],
            },
            "norms_and_scalars": "verbatim",
        },
        "quantized_tensors": {name: quant_metadata[name] for name in sorted(quant_metadata)},
        "excluded": {
            "layer": num_layers,
            "pattern": f"model.layers.{num_layers}.*",
            "reason": "MTP subtree excluded from base-only serving pack",
            "source_num_nextn_predict_layers": original_mtp_layers,
            "source_tensor_count": len(excluded),
        },
        "copied_sidecars": copied_sidecars,
        "output_shards": shard_info,
        "output_bytes": output_bytes,
        "integrity": {
            "algorithm": "sha256",
            "scopes": ["file", "payload"],
        },
    }
    _json_write(out / "conversion_manifest.json", manifest)
    if verify:
        verify_output(out, index, quant_metadata, manifest=manifest)
    after = source_state(src)
    if before != after:
        raise AssertionError("source checkpoint metadata changed during conversion")
    print(
        f"[xing4] wrote {output_bytes} bytes: "
        f"{quantized_counts['4']} routed 4-bit banks, "
        f"{quantized_counts['8']} dense 8-bit matrices, "
        f"{len(excluded)} MTP tensors omitted"
    )
    return manifest


def _tiny_source(path: Path) -> None:
    """Create a complete tiny original-layout checkpoint for hermetic tests."""

    hidden = 64
    intermediate = 64
    hc = 4
    hc_width = hc * hidden
    q_lora = 64
    kv_lora = 64
    qk_nope = 32
    qk_rope = 64
    v_head = 16
    heads = 4
    hc_mix = (2 + hc) * hc
    experts = 2
    layers = 3
    tensors: dict[str, RawTensor] = {}
    rng = np.random.default_rng(19)

    def bf16(shape: tuple[int, ...], scale: float = 0.05) -> RawTensor:
        values = (rng.standard_normal(shape).astype(np.float32) * scale)
        return RawTensor("BF16", shape, f32_to_bf16_u16(values).tobytes())

    def f32(shape: tuple[int, ...]) -> RawTensor:
        return RawTensor("F32", shape, rng.standard_normal(shape).astype(np.float32).tobytes())

    tensors["model.embed_tokens.weight"] = bf16((32, hidden))
    tensors["lm_head.weight"] = bf16((32, hidden))
    tensors["model.norm.weight"] = bf16((hidden,))
    for li in range(layers):
        prefix = f"model.layers.{li}"
        for name, shape in (
            ("self_attn.q_a_proj.weight", (q_lora, hidden)),
            ("self_attn.q_a_layernorm.weight", (q_lora,)),
            ("self_attn.q_b_proj.weight", (heads * (qk_nope + qk_rope), q_lora)),
            ("self_attn.kv_a_proj_with_mqa.weight", (kv_lora + qk_rope, hidden)),
            ("self_attn.kv_a_layernorm.weight", (kv_lora,)),
            ("self_attn.kv_b_proj.weight", (heads * (qk_nope + v_head), kv_lora)),
            ("self_attn.o_proj.weight", (hidden, heads * v_head)),
            ("input_layernorm.weight", (hidden,)),
            ("post_attention_layernorm.weight", (hidden,)),
            ("attn_hc.hc_fn", (hc_mix, hc_width)),
            ("attn_hc.hc_base", (hc_mix,)),
            ("attn_hc.hc_scale", (3,)),
            ("ffn_hc.hc_fn", (hc_mix, hc_width)),
            ("ffn_hc.hc_base", (hc_mix,)),
            ("ffn_hc.hc_scale", (3,)),
        ):
            tensors[f"{prefix}.{name}"] = f32(shape) if name.endswith("hc_scale") else bf16(shape)
        if li < 1:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = (intermediate, hidden) if projection != "down_proj" else (hidden, intermediate)
                tensors[f"{prefix}.mlp.{projection}.weight"] = bf16(shape)
        else:
            tensors[f"{prefix}.mlp.gate.weight"] = bf16((experts, hidden))
            tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = f32((experts,))
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = (intermediate, hidden) if projection != "down_proj" else (hidden, intermediate)
                for expert in range(experts):
                    tensors[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = bf16(shape)
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = (intermediate, hidden) if projection != "down_proj" else (hidden, intermediate)
                tensors[f"{prefix}.mlp.shared_experts.{projection}.weight"] = bf16(shape)
    write_safetensors_raw(path / "model.safetensors", tensors, {"format": "pt"})
    index = {"metadata": {"total_size": 0}, "weight_map": {
        name: "model.safetensors" for name in sorted(tensors)
    }}
    _json_write(path / WEIGHT_MAP_NAME, index)
    _json_write(
        path / "config.json",
        {
            "model_type": "xing4_0",
            "vocab_size": 32,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "moe_intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "qk_head_dim": qk_nope + qk_rope,
            "v_head_dim": v_head,
            "n_routed_experts": experts,
            "first_k_dense_replace": 1,
            "n_shared_experts": 1,
            "hc_mult": hc,
            "num_nextn_predict_layers": 1,
        },
    )
    (path / "tokenizer.model").write_bytes(b"tiny tokenizer")
    (path / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")


def _same_tree_bytes(first: Path, second: Path) -> bool:
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    if first_files != second_files:
        return False
    return all(
        (first / relative).read_bytes() == (second / relative).read_bytes()
        for relative in first_files
    )


def _rewrite_source(
    source: Path,
    *,
    remove: Iterable[str] = (),
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
    shard_name: str | None = None,
) -> None:
    """Rewrite one tiny source shard for a negative converter test."""

    source_shard = source / "model.safetensors"
    reader = ShardReader(source_shard)
    removed = set(remove)
    overrides = shape_overrides or {}
    tensors = {}
    for name in reader.names():
        if name in removed:
            continue
        tensor = reader.read(name)
        shape = overrides.get(name, tensor.shape)
        tensors[name] = RawTensor(tensor.dtype, shape, tensor.data)
    write_safetensors_raw(source_shard, tensors, {"format": "pt"})
    index = json.loads((source / WEIGHT_MAP_NAME).read_text(encoding="utf-8"))
    index["weight_map"] = {
        name: (shard_name or "model.safetensors") for name in sorted(tensors)
    }
    _json_write(source / WEIGHT_MAP_NAME, index)


def _expect_failure(operation, needle: str, label: str, check) -> None:
    try:
        operation()
    except Exception as exc:
        check(needle in str(exc), f"{label}: names the failure")
    else:
        check(False, f"{label}: rejects invalid input")


def self_test() -> None:
    """Hermetic allocation, geometry, binding, conversion, and immutability tests."""

    checks = 0

    def check(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1
        print(f"PASS {label}")

    check(
        allocation_for("model.layers.2.mlp.experts.0.gate_proj.weight", (1024, 3584)).bits
        == 4,
        "routed experts allocate 4-bit",
    )
    for name in (
        "model.layers.2.mlp.gate.weight",
        "model.layers.2.mlp.shared_experts.gate_proj.weight",
        "model.layers.2.attn_hc.hc_fn",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ):
        check(allocation_for(name, (64, 64)).bits == 8, f"{name} allocates 8-bit")
    check(
        allocation_for("model.layers.40.mlp.experts.0.gate_proj.weight", (1024, 3584))
        is None,
        "MTP layer 40 is excluded",
    )
    check(
        canonical_name("model.layers.2.self_attn.o_proj.weight")
        == "model.layers.2.attention.dense.weight",
        "attention output uses native dense name",
    )
    check(
        canonical_name("model.layers.2.mlp.gate.e_score_correction_bias")
        == "model.layers.2.mlp.gate.expert_bias",
        "router correction uses native expert_bias name",
    )
    check(
        quantized_shapes((1024, 3584), 4, 64) == ((1024, 448), (1024, 56)),
        "4-bit expert packed geometry",
    )
    check(
        quantized_shapes((131072, 3584), 8, 64) == ((131072, 896), (131072, 56)),
        "8-bit embedding packed geometry",
    )
    check(
        quantized_shapes((24, 14336), 8, 64) == ((24, 3584), (24, 224)),
        "8-bit mHC packed geometry",
    )

    with tempfile.TemporaryDirectory(prefix="xing4-convert-test-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        _tiny_source(source)
        original = source_state(source)
        original_bytes = {
            path.relative_to(source): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        }
        first = root / "first"
        second = root / "second"
        manifest = convert(source, first, verify=True, strict_source=False)
        convert(source, second, verify=True, strict_source=False)
        check(_same_tree_bytes(first, second), "conversion is deterministic")
        check(source_state(source) == original, "source metadata is unchanged")
        check(
            all(
                (source / relative).read_bytes() == data
                for relative, data in original_bytes.items()
            ),
            "source bytes are unchanged",
        )
        output_config = json.loads((first / "config.json").read_text())
        check(output_config["quantization"]["bits"] == 8, "config base quantization is 8-bit")
        check(output_config["quantization"]["overrides"]["routed_experts"]["bits"] == 4,
              "config routed override is 4-bit")
        check(output_config["num_nextn_predict_layers"] == 0, "MTP is disabled in output config")
        output_index = json.loads((first / WEIGHT_MAP_NAME).read_text())["weight_map"]
        check("model.layers.2.mlp.switch_mlp.gate_proj.weight" in output_index,
              "stacked routed bank is indexed")
        check("model.layers.2.mlp.experts.0.gate_proj.weight" not in output_index,
              "raw routed expert names are absent")
        check(not any("model.layers.3." in name for name in output_index),
              "layer 40/MTP subtree is absent")
        expert_header = _header_for(first / output_index["model.layers.2.mlp.switch_mlp.gate_proj.weight"])
        expert = expert_header["model.layers.2.mlp.switch_mlp.gate_proj.weight"]
        expert_scales = expert_header["model.layers.2.mlp.switch_mlp.gate_proj.scales"]
        expert_biases = expert_header["model.layers.2.mlp.switch_mlp.gate_proj.biases"]
        check(expert["dtype"] == "U32" and expert["shape"] == [2, 64, 8],
              "stacked expert weight has 4-bit shape")
        check(expert_scales["dtype"] == "BF16" and expert_biases["dtype"] == "BF16"
              and expert_scales["shape"] == [2, 64, 1]
              and expert_scales["shape"] == expert_biases["shape"],
              "expert scales and biases bind to the same stacked rows")
        hc_name = "model.layers.2.attn_hc.hc_fn.weight"
        hc_header = _header_for(first / output_index[hc_name])
        check(hc_header[hc_name]["dtype"] == "U32", "mHC projection is quantized")
        norm_name = "model.layers.2.input_layernorm.weight"
        norm_header = _header_for(first / output_index[norm_name])
        check(norm_header[norm_name]["dtype"] == "BF16"
              and norm_name + ".scales" not in norm_header,
              "norm remains unquantized")
        allocation = manifest["allocation"]
        check(allocation["routed_experts"]["count"] == 6
              and allocation["all_other_2d_linears"]["count"] > 0,
              "manifest records 4-bit banks and 8-bit matrices")
        check(
            all(
                "sha256" in shard_info and "payload_sha256" in shard_info
                for shard_info in manifest["output_shards"]
            ),
            "manifest records full-file and payload checksums",
        )
        check(manifest["excluded"]["source_tensor_count"] == 0,
              "tiny source has no MTP subtree to exclude")

        missing = root / "missing-required"
        shutil.copytree(source, missing)
        _rewrite_source(
            missing,
            remove={"model.layers.0.self_attn.q_a_proj.weight"},
        )
        missing_out = root / "missing-required-out"
        _expect_failure(
            lambda: convert(missing, missing_out, verify=True, strict_source=False),
            "required tensor",
            "missing required q_a_proj is rejected",
            check,
        )
        check(not missing_out.exists(), "source validation runs before destination mkdir")

        wrong_shape = root / "wrong-shape"
        shutil.copytree(source, wrong_shape)
        _rewrite_source(
            wrong_shape,
            shape_overrides={"model.layers.0.self_attn.q_a_proj.weight": (32, 128)},
        )
        _expect_failure(
            lambda: validate_source(wrong_shape, strict=False),
            "shape",
            "config-derived source shape is enforced",
            check,
        )

        outside = root / "outside.safetensors"
        shutil.copyfile(source / "model.safetensors", outside)
        escaping = root / "escaping-shard"
        shutil.copytree(source, escaping)
        _rewrite_source(escaping, shard_name="../outside.safetensors")
        _expect_failure(
            lambda: validate_source(escaping, strict=False),
            "not local",
            "parent traversal shard is rejected",
            check,
        )
        absolute = root / "absolute-shard"
        shutil.copytree(source, absolute)
        _rewrite_source(absolute, shard_name=str(outside))
        _expect_failure(
            lambda: validate_source(absolute, strict=False),
            "not local",
            "absolute shard is rejected",
            check,
        )
        nested = root / "nested-shard"
        shutil.copytree(source, nested)
        (nested / "shards").mkdir()
        shutil.move(nested / "model.safetensors", nested / "shards" / "model.safetensors")
        nested_index = json.loads((nested / WEIGHT_MAP_NAME).read_text(encoding="utf-8"))
        nested_index["weight_map"] = {
            name: "shards/model.safetensors"
            for name in nested_index["weight_map"]
        }
        _json_write(nested / WEIGHT_MAP_NAME, nested_index)
        validate_source(nested, strict=False)
        check(True, "ordinary nested local shard is accepted")

        check(
            verify_existing(first, source)["checksum_status"] == "verified",
            "existing output verifies recorded checksums",
        )
        truncated = root / "truncated-output"
        shutil.copytree(first, truncated)
        truncated_shard = truncated / "model-root.safetensors"
        truncated_reader = ShardReader(truncated_shard)
        truncated_shard.write_bytes(truncated_shard.read_bytes()[: truncated_reader.data_offset])
        _expect_failure(
            lambda: verify_existing(truncated),
            "range",
            "truncated output is rejected",
            check,
        )
        bitflipped = root / "bitflipped-output"
        shutil.copytree(first, bitflipped)
        bitflip_shard = bitflipped / "model-root.safetensors"
        bitflip_reader = ShardReader(bitflip_shard)
        bitflip_data = bytearray(bitflip_shard.read_bytes())
        bitflip_data[bitflip_reader.data_offset] ^= 0x01
        bitflip_shard.write_bytes(bitflip_data)
        _expect_failure(
            lambda: verify_existing(bitflipped),
            "checksum",
            "payload bit flip is rejected by disk checksum",
            check,
        )

        legacy = root / "legacy-output"
        shutil.copytree(first, legacy)
        legacy_manifest_path = legacy / "conversion_manifest.json"
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        for shard_info in legacy_manifest["output_shards"]:
            shard_info.pop("sha256", None)
            shard_info.pop("payload_sha256", None)
        _json_write(legacy_manifest_path, legacy_manifest)
        check(
            verify_existing(legacy)["checksum_status"] == "missing",
            "pre-checksum manifest is distinguished from verified output",
        )
        weight_bytes_before = {
            path.relative_to(legacy): path.read_bytes()
            for path in legacy.glob("*.safetensors")
        }
        write_checksums(legacy)
        check(
            verify_existing(legacy)["checksum_status"] == "verified",
            "explicit checksum adoption verifies the legacy output",
        )
        check(
            all(
                path.read_bytes() == weight_bytes_before[path.relative_to(legacy)]
                for path in legacy.glob("*.safetensors")
            ),
            "checksum adoption does not rewrite weight shards",
        )
    print(f"ALL SELF-TESTS PASS ({checks} checks)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-existing", type=Path, metavar="OUT")
    parser.add_argument(
        "--write-checksums",
        nargs="?",
        const=True,
        default=False,
        metavar="OUT",
        help="explicitly add SHA256 metadata (optionally name the output here)",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    verify_target = args.verify_existing
    write_requested = bool(args.write_checksums)
    if isinstance(args.write_checksums, str):
        if verify_target is not None:
            parser.error("--write-checksums OUT cannot be combined with --verify-existing")
        verify_target = Path(args.write_checksums)
    if verify_target is not None:
        if args.out is not None or args.verify or args.dry_run:
            parser.error("--verify-existing cannot be combined with --out/--verify/--dry-run")
        result = (
            write_checksums(verify_target, args.src)
            if write_requested
            else verify_existing(verify_target, args.src)
        )
        summary = {
            key: value
            for key, value in result.items()
            if key != "checksums"
        }
        summary["shard_count"] = len(result["checksums"])
        print(json.dumps(summary, sort_keys=True))
        return 0
    if write_requested:
        parser.error("--write-checksums requires an output pack")
    if args.src is None or args.out is None:
        parser.error("--src and --out are required")
    if args.dry_run:
        config, weight_map, readers = validate_source(args.src.resolve())
        counts = {"4": 0, "8": 0, "excluded": 0}
        for name, shard in weight_map.items():
            li = layer_number(name)
            if li is not None and li >= int(config["num_hidden_layers"]):
                counts["excluded"] += 1
            else:
                _dtype, shape, _begin, _end = readers[shard].metadata(name)
                spec = allocation_for(name, shape, int(config["num_hidden_layers"]))
                if spec is not None:
                    counts[str(spec.bits)] += 1
        print(json.dumps({"source_tensors": len(weight_map), "plan": counts}, sort_keys=True))
        return 0
    convert(args.src, args.out, verify=args.verify)
    return 0


if __name__ == "__main__":
    main()
