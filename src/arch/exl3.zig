//! Draft config adapter using the pinned sushi format definitions.
//! Only tests call this adapter; it does not enable EXL3 model serving.
//! JSON-to-rate conversion mirrors sushi's parser until that parser is shared.

const std = @import("std");
const engine = @import("sushi_exl3");

/// The engine's own refusal names, so one pack is reported in one vocabulary
/// whichever side refuses it.
pub const Refusal = error{
    /// No `expert_quant` block, or one this is not how we read.
    ExpertLayoutUnsupported,
    /// A codeword window the engine does not decode. Never rounded: the same
    /// bitstream decodes to different weights at each width.
    Exl3WindowUnsupported,
};

/// What the engine needs to know to read a pack's routed experts.
pub const Spec = struct {
    rate: engine.format.Rate,
    codebook: engine.format.Codebook,
    window: engine.format.Window,
};

/// Null means a valid config without `expert_quant`; malformed configs are
/// refused, while allocation failures propagate to the caller.
pub fn specFromConfig(allocator: std.mem.Allocator, raw: []const u8) (Refusal || std.mem.Allocator.Error)!?Spec {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ExpertLayoutUnsupported,
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.ExpertLayoutUnsupported;
    const root = parsed.value.object;
    const block = root.get("expert_quant") orelse return null;
    if (block != .object) return error.ExpertLayoutUnsupported;

    const format = block.object.get("format") orelse return error.ExpertLayoutUnsupported;
    if (format != .string or !std.mem.eql(u8, format.string, "exl3")) return error.ExpertLayoutUnsupported;

    // k is a rate, integer or fractional: admitted only when 16k is an even
    // whole number of halfwords per tile, which is what the reader indexes by.
    const k_v = block.object.get("k") orelse return error.ExpertLayoutUnsupported;
    const rate = rateFromConfigK(k_v) orelse return error.ExpertLayoutUnsupported;

    const cb_v = block.object.get("codebook") orelse return error.ExpertLayoutUnsupported;
    if (cb_v != .string) return error.ExpertLayoutUnsupported;
    const codebook = engine.format.Codebook.fromName(cb_v.string) orelse return error.ExpertLayoutUnsupported;

    // Absent means 16, the whole sliding window.
    const window = windowFromConfig(block.object.get("window")) orelse return error.Exl3WindowUnsupported;

    return Spec{ .rate = rate, .codebook = codebook, .window = window };
}

fn rateFromConfigK(k_v: std.json.Value) ?engine.format.Rate {
    const scaled: f64 = switch (k_v) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |f| f * 16.0,
        else => return null,
    };
    if (!std.math.isFinite(scaled)) return null;
    const rounded = @round(scaled);
    if (@abs(scaled - rounded) > 1e-6) return null;
    if (rounded < 0 or rounded > 1024) return null;
    return engine.format.kFromPackedDim(@intFromFloat(rounded));
}

fn windowFromConfig(v: ?std.json.Value) ?engine.format.Window {
    const raw = v orelse return engine.format.Window.w16;
    if (raw != .integer) return null;
    return engine.format.Window.fromBits(raw.integer);
}

const t = std.testing;

test "exl3: a pack's expert_quant block becomes a spec the engine accepts" {
    const spec = (try specFromConfig(t.allocator,
        \\{"model_type":"qwen4_exp","expert_quant":{"format":"exl3","k":3,"codebook":"mul1"}}
    )).?;
    try t.expectEqual(engine.format.Rate.fromK(3), spec.rate);
    try t.expectEqual(engine.format.Codebook.mul1, spec.codebook);
    try t.expectEqual(engine.format.Window.w16, spec.window);
}

test "exl3: a fractional rate and a narrower codeword window are the pack's, not a guess" {
    const narrow = (try specFromConfig(t.allocator,
        \\{"expert_quant":{"format":"exl3","k":2.5,"codebook":"mcg","window":12}}
    )).?;
    try t.expectEqual(engine.format.kFromPackedDim(40).?, narrow.rate);
    try t.expectEqual(engine.format.Codebook.mcg, narrow.codebook);
    try t.expectEqual(engine.format.Window.w12, narrow.window);
}

test "exl3: a pack with no expert_quant block is not an EXL3 pack" {
    try t.expectEqual(@as(?Spec, null), try specFromConfig(t.allocator,
        \\{"model_type":"qwen4_exp","quantization":{"mode":"affine","bits":8}}
    ));
}

test "exl3: malformed configs are refused and allocation failures propagate" {
    for ([_][]const u8{
        "not json",
        "[]",
        "{\"expert_quant\":{},\"expert_quant\":{}}",
        "{\"expert_quant\":{\"format\":\"exl3\",\"k\":1e309,\"codebook\":\"mcg\"}}",
    }) |raw| {
        try t.expectError(error.ExpertLayoutUnsupported, specFromConfig(t.allocator, raw));
    }
    var memory: [0]u8 = .{};
    var allocator = std.heap.FixedBufferAllocator.init(&memory);
    try t.expectError(error.OutOfMemory, specFromConfig(allocator.allocator(), "{\"expert_quant\":{}}"));
}

test "exl3: an unreadable stamp is refused by name, never defaulted" {
    const cases = [_]struct { config: []const u8, err: anyerror }{
        .{ .config =
        \\{"expert_quant":{"format":"mxfp4","k":3,"codebook":"mcg"}}
        , .err = error.ExpertLayoutUnsupported },
        .{ .config =
        \\{"expert_quant":{"format":"exl3","codebook":"mcg"}}
        , .err = error.ExpertLayoutUnsupported },
        .{ .config =
        \\{"expert_quant":{"format":"exl3","k":0.1,"codebook":"mcg"}}
        , .err = error.ExpertLayoutUnsupported },
        .{ .config =
        \\{"expert_quant":{"format":"exl3","k":3,"codebook":"mul2"}}
        , .err = error.ExpertLayoutUnsupported },
        .{ .config =
        \\{"expert_quant":{"format":"exl3","k":3,"codebook":"mcg","window":17}}
        , .err = error.Exl3WindowUnsupported },
        .{ .config =
        \\{"expert_quant":{"format":"exl3","k":3,"codebook":"mcg","window":"12"}}
        , .err = error.Exl3WindowUnsupported },
    };
    for (cases) |c| try t.expectError(c.err, specFromConfig(t.allocator, c.config));
}
