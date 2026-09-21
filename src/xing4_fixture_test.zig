// Reference fixtures: tests/dump_xing4_fixtures.py build/dump.
// Set XING4_TEST_MODEL to the checkpoint directory and optionally XING4_FIXTURE
// to a fixture file; absent env skips this GPU test.
const std = @import("std");
const model = @import("model.zig");
const transformer = @import("transformer.zig");
const mlx = @import("mlx.zig");

const testing = std.testing;

fn readAll(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var reader = file.reader(io, &rb);
    return reader.interface.allocRemaining(allocator, .limited(1 << 28));
}

fn jsonNumber(value: std.json.Value) !f64 {
    return switch (value) {
        .float => |v| v,
        .integer => |v| @floatFromInt(v),
        else => error.BadXingFixture,
    };
}

fn fixtureArray(
    allocator: std.mem.Allocator,
    root: std.json.ObjectMap,
    key: []const u8,
) ![]f32 {
    const value = root.get(key) orelse return error.BadXingFixture;
    if (value != .array) return error.BadXingFixture;
    const out = try allocator.alloc(f32, value.array.items.len);
    errdefer allocator.free(out);
    for (out, value.array.items) |*dst, src| dst.* = @floatCast(try jsonNumber(src));
    return out;
}

fn fixtureIds(
    allocator: std.mem.Allocator,
    root: std.json.ObjectMap,
    key: []const u8,
) ![]i32 {
    const value = root.get(key) orelse return error.BadXingFixture;
    if (value != .array) return error.BadXingFixture;
    const out = try allocator.alloc(i32, value.array.items.len);
    errdefer allocator.free(out);
    for (out, value.array.items) |*dst, src| {
        if (src != .integer) return error.BadXingFixture;
        if (src.integer < std.math.minInt(i32) or src.integer > std.math.maxInt(i32)) {
            return error.BadXingFixture;
        }
        dst.* = @intCast(src.integer);
    }
    return out;
}

fn isRowMajorContiguous(arr: mlx.mlx_array) bool {
    const ndim = mlx.mlx_array_ndim(arr);
    if (ndim == 0) return true;
    const shape = mlx.mlx_array_shape(arr);
    const strides = mlx.mlx_array_strides(arr);
    var expected: usize = 1;
    var i = ndim;
    while (i > 0) {
        i -= 1;
        const dim: usize = @intCast(shape[i]);
        if (dim != 1 and strides[i] != expected) return false;
        expected *= dim;
    }
    return true;
}

fn toHostF32(
    allocator: std.mem.Allocator,
    arr: mlx.mlx_array,
    count: usize,
    stream: mlx.mlx_stream,
) ![]f32 {
    var f32_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_arr);
    try mlx.check(mlx.mlx_astype(&f32_arr, arr, .float32, stream));
    try mlx.check(mlx.mlx_array_eval(f32_arr));
    if (!isRowMajorContiguous(f32_arr)) {
        const flat_shape = [_]c_int{@intCast(mlx.mlx_array_size(f32_arr))};
        var materialized = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(materialized);
        try mlx.check(mlx.mlx_reshape(&materialized, f32_arr, &flat_shape, 1, stream));
        try mlx.check(mlx.mlx_array_set(&f32_arr, materialized));
        try mlx.check(mlx.mlx_array_eval(f32_arr));
    }
    if (!isRowMajorContiguous(f32_arr)) return error.NonContiguousFixtureOutput;
    const ptr = mlx.mlx_array_data_float32(f32_arr) orelse return error.NoData;
    const out = try allocator.alloc(f32, count);
    @memcpy(out, ptr[0..count]);
    return out;
}

fn makeTokenArray(ids: []const i32) mlx.mlx_array {
    const shape = [_]c_int{ 1, @intCast(ids.len) };
    return mlx.mlx_array_new_data(ids.ptr, &shape, 2, .int32);
}

fn expectShape(arr: mlx.mlx_array, expected: []const c_int) !void {
    try testing.expectEqualSlices(c_int, expected, mlx.getShape(arr));
}

const CompareStats = struct {
    max_abs: f64,
    rms_err: f64,
    rms_ref: f64,
};

fn compare(
    label: []const u8,
    got: []const f32,
    want: []const f32,
    max_abs_limit: f64,
    rms_relative_limit: f64,
) !CompareStats {
    try testing.expectEqual(want.len, got.len);
    var max_abs: f64 = 0;
    var worst_index: usize = 0;
    var err_sq: f64 = 0;
    var ref_sq: f64 = 0;
    for (got, want, 0..) |g, w, i| {
        try testing.expect(std.math.isFinite(g));
        try testing.expect(std.math.isFinite(w));
        const diff = @abs(@as(f64, g) - @as(f64, w));
        if (diff > max_abs) {
            max_abs = diff;
            worst_index = i;
        }
        err_sq += diff * diff;
        ref_sq += @as(f64, w) * @as(f64, w);
    }
    const n: f64 = @floatFromInt(got.len);
    const rms_err = @sqrt(err_sq / n);
    const rms_ref = @sqrt(ref_sq / n);
    const rms_limit = @max(1e-6, rms_ref * rms_relative_limit);
    std.debug.print(
        "[xing4-fixture] {s}: max_abs={d:.6} at={d} rms_err={d:.6} rms_ref={d:.6} limit={d:.6}\n",
        .{ label, max_abs, worst_index, rms_err, rms_ref, rms_limit },
    );
    try testing.expect(max_abs <= max_abs_limit);
    try testing.expect(rms_err <= rms_limit);
    return .{ .max_abs = max_abs, .rms_err = rms_err, .rms_ref = rms_ref };
}

fn forward(
    xfm: *transformer.Transformer,
    allocator: std.mem.Allocator,
    ctx: *transformer.ForwardCtx,
    ids: []const i32,
    vocab: usize,
) ![]f32 {
    const token_ids = makeTokenArray(ids);
    defer _ = mlx.mlx_array_free(token_ids);
    const logits = try xfm.forwardWith(ctx, token_ids);
    defer _ = mlx.mlx_array_free(logits);
    try expectShape(logits, &.{ 1, @intCast(ids.len), @intCast(vocab) });
    return toHostF32(allocator, logits, ids.len * vocab, mlx.gpuStream());
}

fn forwardDecode(
    xfm: *transformer.Transformer,
    allocator: std.mem.Allocator,
    ctx: *transformer.ForwardCtx,
    ids: []const i32,
    vocab: usize,
) ![]f32 {
    const out = try allocator.alloc(f32, ids.len * vocab);
    errdefer allocator.free(out);
    for (ids, 0..) |token, i| {
        const one = [_]i32{token};
        const token_ids = makeTokenArray(&one);
        defer _ = mlx.mlx_array_free(token_ids);
        const logits = try xfm.forwardWith(ctx, token_ids);
        defer _ = mlx.mlx_array_free(logits);
        try expectShape(logits, &.{ 1, 1, @intCast(vocab) });
        const row = try toHostF32(allocator, logits, vocab, mlx.gpuStream());
        defer allocator.free(row);
        @memcpy(out[i * vocab ..][0..vocab], row);
    }
    return out;
}

fn runShort(
    xfm: *transformer.Transformer,
    allocator: std.mem.Allocator,
    input_ids: []const i32,
    decode_ids: []const i32,
    vocab: usize,
) !struct { prefill: []f32, decode: []f32 } {
    var ctx = xfm.defaultCtx();
    const prefill = try forward(xfm, allocator, &ctx, input_ids, vocab);
    errdefer allocator.free(prefill);
    const decode = try forwardDecode(xfm, allocator, &ctx, decode_ids, vocab);
    return .{ .prefill = prefill, .decode = decode };
}

test "xing4 fixture: original BF16 checkpoint matches CPU reference" {
    const model_dir_z = std.c.getenv("XING4_TEST_MODEL") orelse return;
    const model_dir = std.mem.span(model_dir_z);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    const cfg_json = try readAll(io, allocator, cfg_path);
    defer allocator.free(cfg_json);
    var config = try model.parseConfigFromJson(allocator, cfg_json);
    defer config.deinit(allocator);
    try testing.expect(config.isXing4());
    try testing.expectEqual(@as(u32, 4), config.xing_hc_mult);
    try testing.expect(config.isMla());
    try testing.expectEqual(@as(u32, 1), config.first_k_dense_replace);

    const fixture_path = if (std.c.getenv("XING4_FIXTURE")) |p|
        try allocator.dupe(u8, std.mem.span(p))
    else
        try std.fmt.allocPrint(allocator, "{s}/fixtures.json", .{model_dir});
    defer allocator.free(fixture_path);
    const fixture_json = try readAll(io, allocator, fixture_path);
    defer allocator.free(fixture_json);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, fixture_json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const vocab_value = root.get("vocab_size") orelse return error.BadXingFixture;
    if (vocab_value != .integer) return error.BadXingFixture;
    const vocab: usize = @intCast(vocab_value.integer);
    const input_ids = try fixtureIds(allocator, root, "input_ids");
    defer allocator.free(input_ids);
    const decode_ids = try fixtureIds(allocator, root, "decode_ids");
    defer allocator.free(decode_ids);
    const want_prefill_bf16 = try fixtureArray(allocator, root, "prefill_logits_bf16");
    defer allocator.free(want_prefill_bf16);
    const want_decode_bf16 = try fixtureArray(allocator, root, "decode_logits_bf16");
    defer allocator.free(want_decode_bf16);
    const want_prefill_f32 = try fixtureArray(allocator, root, "prefill_logits_f32");
    defer allocator.free(want_prefill_f32);
    const want_decode_f32 = try fixtureArray(allocator, root, "decode_logits_f32");
    defer allocator.free(want_decode_f32);

    var weights = try model.loadModelWeights(io, allocator, model_dir, &config, false);
    defer weights.deinit();
    var xfm = try transformer.Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    const short = try runShort(&xfm, allocator, input_ids, decode_ids, vocab);
    defer allocator.free(short.prefill);
    defer allocator.free(short.decode);
    // The cache stores exactly the configured representation in both A/B arms.
    for (xfm.cache.entries) |entry| {
        const heads: c_int = @intCast(config.kvCacheHeads());
        const key_width: c_int = @intCast(if (config.mla_latent_kv) config.mla_qk_rope_head_dim else config.mlaQkHeadDim());
        const value_width: c_int = @intCast(if (config.mla_latent_kv) config.mla_kv_lora_rank else config.mla_v_head_dim);
        try expectShape(entry.key_view, &.{ 1, heads, @intCast(input_ids.len + decode_ids.len), key_width });
        try expectShape(entry.value_view, &.{ 1, heads, @intCast(input_ids.len + decode_ids.len), value_width });
    }
    _ = try compare("prefill vs BF16 reference", short.prefill, want_prefill_bf16, 0.02, 0.01);
    _ = try compare("decode vs BF16 reference", short.decode, want_decode_bf16, 0.02, 0.01);
    _ = try compare("prefill vs F32 truth", short.prefill, want_prefill_f32, 0.02, 0.01);
    _ = try compare("decode vs F32 truth", short.decode, want_decode_f32, 0.02, 0.01);

    // The long fixture is optional for quick local iteration, but the default
    // dump includes it so the position-4096 YaRN path remains executable.
    if (root.get("long_prefix_ids") != null) {
        const long_ids = try fixtureIds(allocator, root, "long_prefix_ids");
        defer allocator.free(long_ids);
        const long_decode_value = root.get("long_decode_id") orelse return error.BadXingFixture;
        if (long_decode_value != .integer) return error.BadXingFixture;
        const long_decode_id: i32 = @intCast(long_decode_value.integer);
        const long_dec = [_]i32{long_decode_id};
        const want_long_bf16 = try fixtureArray(allocator, root, "long_logits_bf16");
        defer allocator.free(want_long_bf16);
        const want_long_f32 = try fixtureArray(allocator, root, "long_logits_f32");
        defer allocator.free(want_long_f32);

        try xfm.resetCache();
        var long_ctx = xfm.defaultCtx();
        const long_prefill = try forward(&xfm, allocator, &long_ctx, long_ids, vocab);
        defer allocator.free(long_prefill);
        const long_logits = try forwardDecode(&xfm, allocator, &long_ctx, &long_dec, vocab);
        defer allocator.free(long_logits);
        _ = try compare("position-4096 decode vs BF16 reference", long_logits, want_long_bf16, 0.02, 0.01);
        _ = try compare("position-4096 decode vs F32 truth", long_logits, want_long_f32, 0.02, 0.01);
    }
}
