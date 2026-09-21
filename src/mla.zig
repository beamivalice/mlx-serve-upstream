const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const EXPAND_PREFILL_MIN_ROWS = 256;
var prefill_expand_enabled: ?bool = null;
var prefill_expand_logged = false;

/// Wide prefills amortize one bank projection and then score the smaller
/// per-head K/V. Decode must never re-expand the whole cached prefix.
pub fn expandedPrefillEligible(rows: c_int, nope: c_int, value_dim: c_int, rope: c_int) bool {
    if (rows < EXPAND_PREFILL_MIN_ROWS) return false;
    if (nope <= 0 or value_dim <= 0 or rope <= 0) return false;
    // Three BF16 score sheets plus expanded KV must fit the billed three
    // F32 sheets; the extra bank buffers cost 4*(NoPE+V)+2*RoPE per head/key.
    const bank_bytes = 4 * (@as(u64, @intCast(nope)) + @as(u64, @intCast(value_dim))) + 2 * @as(u64, @intCast(rope));
    if (6 * @as(u64, @intCast(rows)) < bank_bytes) return false;
    if (prefill_expand_enabled == null) {
        const raw = std.c.getenv("MLX_SERVE_MLA_PREFILL_EXPAND");
        prefill_expand_enabled = raw == null or raw.?[0] != '0';
    }
    if (!prefill_expand_enabled.?) return false;
    if (!prefill_expand_logged) {
        prefill_expand_logged = true;
        log.info("[mla] expanded prefill engaged: q={d}, cache remains latent; MLX_SERVE_MLA_PREFILL_EXPAND=0 restores absorbed prefill\n", .{rows});
    }
    return true;
}

test "MLA prefill expansion declines decode and unbilled wide heads" {
    const previous = prefill_expand_enabled;
    prefill_expand_enabled = true;
    defer prefill_expand_enabled = previous;
    try std.testing.expect(!expandedPrefillEligible(1, 128, 128, 64));
    try std.testing.expect(!expandedPrefillEligible(255, 128, 128, 64));
    try std.testing.expect(expandedPrefillEligible(256, 128, 128, 64));
    try std.testing.expect(!expandedPrefillEligible(256, 512, 512, 64));
    try std.testing.expect(expandedPrefillEligible(1024, 512, 512, 64));
}

/// Absorb the key expansion into Q and the value expansion into the attended
/// latent. Keep o_proj separate: folding it would enlarge every decode read.
pub const Projections = struct {
    key: mlx.mlx_array, // [heads, nope, rank], f32
    value: mlx.mlx_array, // [heads, rank, value_dim], f32

    pub fn init(raw: mlx.mlx_array, heads: c_int, nope: c_int, value_dim: c_int, rank: c_int, s: mlx.mlx_stream) !Projections {
        const shape = mlx.getShape(raw);
        if (shape.len != 2 or heads <= 0 or nope <= 0 or value_dim <= 0 or rank <= 0 or
            shape[0] != heads * (nope + value_dim) or shape[1] != rank)
            return error.InvalidMlaProjectionShape;
        const wide = try cast(raw, .float32, s);
        defer _ = mlx.mlx_array_free(wide);
        var bank = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bank);
        try mlx.check(mlx.mlx_reshape(&bank, wide, &[_]c_int{ heads, nope + value_dim, rank }, 3, s));
        var key_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key_view);
        try mlx.check(mlx.mlx_slice(&key_view, bank, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ heads, nope, rank }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var value_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(value_view);
        try mlx.check(mlx.mlx_slice(&value_view, bank, &[_]c_int{ 0, nope, 0 }, 3, &[_]c_int{ heads, nope + value_dim, rank }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var value_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(value_t);
        try mlx.check(mlx.mlx_transpose_axes(&value_t, value_view, &[_]c_int{ 0, 2, 1 }, 3, s));
        var key = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_contiguous(&key, key_view, false, s));
        var value = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(value);
        try mlx.check(mlx.mlx_contiguous(&value, value_t, false, s));
        const arrays = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ key, value }, 2);
        defer _ = mlx.mlx_vector_array_free(arrays);
        try mlx.check(mlx.mlx_eval(arrays));
        return .{ .key = key, .value = value };
    }

    pub fn deinit(self: Projections) void {
        _ = mlx.mlx_array_free(self.key);
        _ = mlx.mlx_array_free(self.value);
    }
};

fn cast(x: mlx.mlx_array, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, x, dtype, s));
    return out;
}

fn transposeBank(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_transpose_axes(&out, x, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
    return out;
}

/// Fold heads into GEMM rows so a shared bank is tiled once, rather than
/// issuing a separate matrix-vector read of the same bank for every head.
fn sharedBankProduct(queries: mlx.mlx_array, bank: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const q = mlx.getShape(queries);
    const b = mlx.getShape(bank);
    if (q.len != 4 or b.len != 4 or b[1] != 1 or q[0] != b[0] or q[3] != b[2])
        return error.InvalidMlaAttentionShape;
    var q_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rows);
    try mlx.check(mlx.mlx_reshape(&q_rows, queries, &[_]c_int{ q[0], q[1] * q[2], q[3] }, 3, s));
    var shared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared);
    try mlx.check(mlx.mlx_reshape(&shared, bank, &[_]c_int{ b[0], b[2], b[3] }, 3, s));
    var product = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(product);
    try mlx.check(mlx.mlx_matmul(&product, q_rows, shared, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, product, &[_]c_int{ q[0], q[1], q[2], b[3] }, 4, s));
    return out;
}

/// q_nope/q_rope are [B,H,Q,D]; cached banks are [B,1,T,D]. Rotary keys
/// have already been rotated at their absolute positions. No bank expansion
/// or concatenation is needed, and both score terms share one softmax.
pub fn attention(
    q_nope: mlx.mlx_array,
    q_rope: mlx.mlx_array,
    rope_bank: mlx.mlx_array,
    latent_bank: mlx.mlx_array,
    projections: Projections,
    scale: f32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const q_shape = mlx.getShape(q_nope);
    const k_shape = mlx.getShape(rope_bank);
    if (q_shape.len != 4 or k_shape.len != 4 or k_shape[1] != 1 or q_shape[2] > k_shape[2])
        return error.InvalidMlaAttentionShape;
    const q32 = try cast(q_nope, .float32, s);
    defer _ = mlx.mlx_array_free(q32);
    const qr32 = try cast(q_rope, .float32, s);
    defer _ = mlx.mlx_array_free(qr32);
    const kr32 = try cast(rope_bank, .float32, s);
    defer _ = mlx.mlx_array_free(kr32);
    const latent32 = try cast(latent_bank, .float32, s);
    defer _ = mlx.mlx_array_free(latent32);
    const latent_t = try transposeBank(latent32, s);
    defer _ = mlx.mlx_array_free(latent_t);
    const rope_t = try transposeBank(kr32, s);
    defer _ = mlx.mlx_array_free(rope_t);
    var q_latent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_latent);
    try mlx.check(mlx.mlx_matmul(&q_latent, q32, projections.key, s));
    const latent_scores = try sharedBankProduct(q_latent, latent_t, s);
    defer _ = mlx.mlx_array_free(latent_scores);
    const rope_scores = try sharedBankProduct(qr32, rope_t, s);
    defer _ = mlx.mlx_array_free(rope_scores);
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_add(&scores, latent_scores, rope_scores, s));
    const scale_arr = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scale_arr);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, scores, scale_arr, s));
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (q_shape[2] > 1) {
        var q_positions = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_positions);
        try mlx.check(mlx.mlx_arange(&q_positions, @floatFromInt(k_shape[2] - q_shape[2]), @floatFromInt(k_shape[2]), 1, .int32, s));
        var k_positions = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_positions);
        try mlx.check(mlx.mlx_arange(&k_positions, 0, @floatFromInt(k_shape[2]), 1, .int32, s));
        var q_column = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_column);
        try mlx.check(mlx.mlx_expand_dims(&q_column, q_positions, 1, s));
        var allowed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(allowed);
        try mlx.check(mlx.mlx_less_equal(&allowed, k_positions, q_column, s));
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        try mlx.check(mlx.mlx_where(&masked, allowed, scaled, neg_inf, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&masked, scaled));
    }
    var probabilities = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probabilities);
    try mlx.check(mlx.mlx_softmax_axis(&probabilities, masked, -1, true, s));
    const attended_latent = try sharedBankProduct(probabilities, latent32, s);
    defer _ = mlx.mlx_array_free(attended_latent);
    var projected = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(projected);
    try mlx.check(mlx.mlx_matmul(&projected, attended_latent, projections.value, s));
    return cast(projected, mlx.mlx_array_dtype(q_nope), s);
}

test "MLA latent attention equals expanded f64 math with offset causal masking" {
    // Isolate the algebra from Metal's TF32 GEMM rounding; the Xing fixture
    // compares the complete BF16 GPU forward against the reference.
    const H = 2;
    const N = 3;
    const R = 5;
    const V = 4;
    const P = 2;
    const Q = 3;
    const T = 7;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var random = std.Random.DefaultPrng.init(0x4d4c41);
    const rng = random.random();
    var weights: [H * (N + V) * R]f32 = undefined;
    var latent: [T * R]f32 = undefined;
    var rope: [T * P]f32 = undefined;
    var qn: [H * Q * N]f32 = undefined;
    var qr: [H * Q * P]f32 = undefined;
    for (&weights) |*v| v.* = rng.float(f32) - 0.5;
    for (&latent) |*v| v.* = rng.float(f32) - 0.5;
    for (&rope) |*v| v.* = rng.float(f32) - 0.5;
    for (&qn) |*v| v.* = rng.float(f32) - 0.5;
    for (&qr) |*v| v.* = rng.float(f32) - 0.5;
    const w = mlx.mlx_array_new_data(&weights, &[_]c_int{ H * (N + V), R }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);
    const l = mlx.mlx_array_new_data(&latent, &[_]c_int{ 1, 1, T, R }, 4, .float32);
    defer _ = mlx.mlx_array_free(l);
    const k = mlx.mlx_array_new_data(&rope, &[_]c_int{ 1, 1, T, P }, 4, .float32);
    defer _ = mlx.mlx_array_free(k);
    const q = mlx.mlx_array_new_data(&qn, &[_]c_int{ 1, H, Q, N }, 4, .float32);
    defer _ = mlx.mlx_array_free(q);
    const r = mlx.mlx_array_new_data(&qr, &[_]c_int{ 1, H, Q, P }, 4, .float32);
    defer _ = mlx.mlx_array_free(r);
    const projections = try Projections.init(w, H, N, V, R, s);
    defer projections.deinit();
    const scale: f32 = 0.75;
    const out = try attention(q, r, k, l, projections, scale, s);
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expectEqualSlices(c_int, &.{ 1, H, Q, V }, mlx.getShape(out));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, out, &[_]c_int{H * Q * V}, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    try std.testing.expectEqual(@as(usize, 1), mlx.mlx_array_strides(flat)[0]);
    const actual = mlx.mlx_array_data_float32(flat).?;
    for (0..H) |h| {
        for (0..Q) |row| {
            var scores: [T]f64 = @splat(-std.math.inf(f64));
            for (0..T - Q + row + 1) |t| {
                var score: f64 = 0;
                for (0..N) |n| {
                    var key: f64 = 0;
                    for (0..R) |c| key += @as(f64, latent[t * R + c]) * weights[(h * (N + V) + n) * R + c];
                    score += @as(f64, qn[(h * Q + row) * N + n]) * key;
                }
                for (0..P) |p| score += @as(f64, qr[(h * Q + row) * P + p]) * rope[t * P + p];
                scores[t] = score * scale;
            }
            var total: f64 = 0;
            for (&scores) |*score| {
                score.* = @exp(score.*);
                total += score.*;
            }
            for (0..V) |v| {
                var expected: f64 = 0;
                for (0..T) |t| {
                    var value: f64 = 0;
                    for (0..R) |c| value += @as(f64, latent[t * R + c]) * weights[(h * (N + V) + N + v) * R + c];
                    expected += scores[t] / total * value;
                }
                const got = actual[(h * Q + row) * V + v];
                try std.testing.expect(std.math.isFinite(got));
                try std.testing.expectApproxEqAbs(expected, @as(f64, got), 1e-5);
            }
        }
    }
}
