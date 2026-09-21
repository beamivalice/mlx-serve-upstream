//! Optional Xing4.0 mHC Sinkhorn normalization candidate.
//!
//! The caller supplies the already-exponentiated f32 comb matrices. Keeping
//! exp out of this kernel avoids the JIT-versus-metallib transcendental
//! discrepancy; this module only replaces the repeated row/column reductions.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const HC: c_int = 4;
const MATRIX: c_int = HC * HC;

// One SIMD group owns a matrix; register shuffles avoid shared-memory races.
// The ascending sums match mlx_sum_axis. Rows/iterations are runtime inputs.
const KERNEL_SOURCE =
    \\const int idx = thread_position_in_grid.x;
    \\if (idx >= rows * 16) return;
    \\const int elem = idx % 16;
    \\const int matrix = idx / 16;
    \\float value = comb[idx];
    \\for (int iter = 0; iter < iters; ++iter) {
    \\  const int r = elem / 4;
    \\  float sum = 0.0f;
    \\  for (int c = 0; c < 4; ++c) sum += simd_shuffle(value, r * 4 + c);
    \\  value = value / (sum + eps);
    \\  const int c = elem % 4;
    \\  sum = 0.0f;
    \\  for (int r2 = 0; r2 < 4; ++r2) sum += simd_shuffle(value, r2 * 4 + c);
    \\  value = value / (sum + eps);
    \\}
    \\out[matrix * 16 + elem] = value;
;

const ShapeKey = struct {
    b: c_int,
    s: c_int,
    h: c_int,
    w: c_int,
};

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
var config_cache: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
var config_key: ?ShapeKey = null;
var enabled_env: ?bool = null;
var engagement_logged = false;

fn enabled() bool {
    if (enabled_env) |value| return value;
    const raw = std.c.getenv("MLX_SERVE_XING_SINKHORN");
    const value = raw == null or raw.?[0] != '0';
    enabled_env = value;
    return value;
}

fn keyFor(comb_exp: mlx.mlx_array) ?ShapeKey {
    if (comb_exp.ctx == null or mlx.mlx_array_dtype(comb_exp) != .float32) return null;
    const shape = mlx.getShape(comb_exp);
    if (shape.len != 4 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != HC or shape[3] != HC) return null;
    const b: usize = @intCast(shape[0]);
    const seq: usize = @intCast(shape[1]);
    const rows = std.math.mul(usize, b, seq) catch return null;
    const elements = std.math.mul(usize, rows, @as(usize, @intCast(MATRIX))) catch return null;
    if (elements > std.math.maxInt(c_int) or elements != mlx.mlx_array_size(comb_exp)) return null;
    return .{ .b = shape[0], .s = shape[1], .h = shape[2], .w = shape[3] };
}

fn rowsFor(key: ShapeKey) c_int {
    const rows: usize = @as(usize, @intCast(key.b)) * @as(usize, @intCast(key.s));
    return @intCast(rows);
}

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_cache) |kernel| return kernel;
    const input_names = [_][*:0]const u8{ "comb", "rows", "iters", "eps" };
    const output_names = [_][*:0]const u8{"out"};
    const inputs = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(inputs);
    const outputs = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(outputs);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_xing_sinkhorn",
        inputs,
        outputs,
        KERNEL_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    kernel_cache = kernel;
    return kernel;
}

fn configFor(key: ShapeKey) !mlx.mlx_fast_metal_kernel_config {
    if (config_key) |previous| if (std.meta.eql(previous, key)) return config_cache;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const shape = [_]c_int{ key.b, key.s, key.h, key.w };
    const rows = rowsFor(key);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &shape, shape.len, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, rows * MATRIX, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, MATRIX, 1, 1));
    if (config_cache.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(config_cache);
    config_cache = config;
    config_key = key;
    return config;
}

/// Normalize `[B,S,4,4]` post-exp comb matrices with Xing's denominator-eps
/// row/column rounds. Returns null when the GPU-only hc4/f32 arm declines.
pub fn normalize(
    comb_exp: mlx.mlx_array,
    iters: u32,
    eps: f32,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamIsGpu(s) or iters == 0 or iters > std.math.maxInt(c_int) or !std.math.isFinite(eps) or eps < 0) return null;
    const key = keyFor(comb_exp) orelse return null;
    const input_rows = mlx.mlx_array_new_int(rowsFor(key));
    defer _ = mlx.mlx_array_free(input_rows);
    const input_iters = mlx.mlx_array_new_int(@intCast(iters));
    defer _ = mlx.mlx_array_free(input_iters);
    const input_eps = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(input_eps);

    const inputs_arr = [_]mlx.mlx_array{ comb_exp, input_rows, input_iters, input_eps };
    const inputs = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, try getKernel(), inputs, try configFor(key), s));
    if (mlx.mlx_vector_array_size(outputs) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    if (!engagement_logged) {
        engagement_logged = true;
        log.info("[xing-hc] fused Sinkhorn engaged: shape=[{d},{d},4,4] iters={d}\n", .{ key.b, key.s, iters });
    }
    return out;
}

const BenchClock = struct {
    io: std.Io,
    mark: std.Io.Timestamp,

    fn init() BenchClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .mark = std.Io.Timestamp.now(io, .boot) };
    }

    fn lap(self: *BenchClock) u64 {
        const ns: i64 = @intCast(self.mark.untilNow(self.io, .boot).nanoseconds);
        self.mark = std.Io.Timestamp.now(self.io, .boot);
        return @intCast(@max(ns, 0));
    }
};

fn composed(
    comb_exp: mlx.mlx_array,
    iters: u32,
    eps: f32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    try mlx.check(mlx.mlx_array_set(&current, comb_exp));
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    var iter: u32 = 0;
    while (iter < iters) : (iter += 1) {
        for ([_]c_int{ -1, -2 }) |axis| {
            var sum = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_sum_axis(&sum, current, axis, true, s));
            var denominator = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(denominator);
            try mlx.check(mlx.mlx_add(&denominator, sum, eps_arr, s));
            var next = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(next);
            try mlx.check(mlx.mlx_divide(&next, current, denominator, s));
            _ = mlx.mlx_array_free(current);
            current = next;
            next = .{ .ctx = null };
        }
    }
    return current;
}

fn hostF32(allocator: std.mem.Allocator, arr: mlx.mlx_array) ![]f32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const ptr = mlx.mlx_array_data_float32(arr) orelse return error.NoData;
    const out = try allocator.alloc(f32, mlx.mlx_array_size(arr));
    @memcpy(out, ptr[0..out.len]);
    return out;
}

fn hostBf16(allocator: std.mem.Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]u16 {
    var cast = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, arr, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(cast));
    const ptr = mlx.mlx_array_data_bfloat16(cast) orelse return error.NoData;
    const out = try allocator.alloc(u16, mlx.mlx_array_size(cast));
    @memcpy(out, ptr[0..out.len]);
    return out;
}

fn combInput(allocator: std.mem.Allocator, b: c_int, seq: c_int) !struct { data: []f32, array: mlx.mlx_array } {
    const rows: usize = @as(usize, @intCast(b)) * @as(usize, @intCast(seq));
    const data = try allocator.alloc(f32, rows * @as(usize, @intCast(MATRIX)));
    errdefer allocator.free(data);
    const patterns = [_][4]f64{
        .{ 0.0, -0.0000001, -0.00001, -60.0 },
        .{ 0.0, -0.0001, -20.0, -60.0 },
        .{ 0.0, -0.5, -1.0, -2.0 },
        .{ 0.0, -20.0, -40.0, -60.0 },
        .{ 0.0, -0.000001, -0.000001, -0.000001 },
        .{ 0.0, -0.0000002, -0.0000002, -60.0 },
        .{ 0.0, -10.0, -10.0, -10.0001 },
        .{ 0.0, -60.0, -60.0, -60.0 },
        .{ 0.0, -0.0123, -0.0124, -3.7 },
    };
    for (0..rows) |row| {
        const logits = patterns[row % patterns.len];
        var maximum = logits[0];
        for (logits[1..]) |value| maximum = @max(maximum, value);
        for (0..@as(usize, @intCast(MATRIX))) |element| {
            const value = std.math.exp(logits[element % 4] - maximum);
            data[row * @as(usize, @intCast(MATRIX)) + element] = @floatCast(value);
        }
    }
    const shape = [_]c_int{ b, seq, HC, HC };
    const array = mlx.mlx_array_new_data(data.ptr, &shape, shape.len, .float32);
    return .{ .data = data, .array = array };
}

fn f64Reference(allocator: std.mem.Allocator, data: []const f32, iters: u32, eps: f32) ![]f64 {
    const out = try allocator.alloc(f64, data.len);
    for (data, 0..) |value, i| out[i] = value;
    const rows = data.len / @as(usize, @intCast(MATRIX));
    for (0..rows) |row| {
        const base = row * @as(usize, @intCast(MATRIX));
        var iter: u32 = 0;
        while (iter < iters) : (iter += 1) {
            for (0..@as(usize, @intCast(HC))) |r| {
                var sum: f64 = 0;
                for (0..@as(usize, @intCast(HC))) |c| sum += out[base + r * @as(usize, @intCast(HC)) + c];
                for (0..@as(usize, @intCast(HC))) |c| {
                    const index = base + r * @as(usize, @intCast(HC)) + c;
                    out[index] /= sum + @as(f64, eps);
                }
            }
            for (0..@as(usize, @intCast(HC))) |c| {
                var sum: f64 = 0;
                for (0..@as(usize, @intCast(HC))) |r| sum += out[base + r * @as(usize, @intCast(HC)) + c];
                for (0..@as(usize, @intCast(HC))) |r| {
                    const index = base + r * @as(usize, @intCast(HC)) + c;
                    out[index] /= sum + @as(f64, eps);
                }
            }
        }
    }
    return out;
}

const ParityStats = struct {
    f32_mismatches: usize = 0,
    bf16_mismatches: usize = 0,
    max_abs: f64 = 0,
    max_f64_error: f64 = 0,
};

fn parityStats(got: []const f32, want: []const f32, reference: []const f64, got_bf16: []const u16, want_bf16: []const u16) ParityStats {
    var stats = ParityStats{};
    for (got, want, reference) |g, w, ref| {
        const gbits: u32 = @bitCast(g);
        const wbits: u32 = @bitCast(w);
        if (gbits != wbits) stats.f32_mismatches += 1;
        stats.max_abs = @max(stats.max_abs, @abs(@as(f64, g) - @as(f64, w)));
        stats.max_f64_error = @max(stats.max_f64_error, @abs(@as(f64, g) - ref));
    }
    for (got_bf16, want_bf16) |g, w| {
        if (g != w) stats.bf16_mismatches += 1;
    }
    return stats;
}

test "xing hc fused Sinkhorn matches the composed f32 chain across shapes and values" {
    if (mlx.noGpuBackend() or !enabled()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const allocator = std.testing.allocator;
    const cases = [_]struct { b: c_int, seq: c_int, iters: u32 }{
        .{ .b = 1, .seq = 1, .iters = 1 },
        .{ .b = 1, .seq = 1, .iters = 3 },
        .{ .b = 1, .seq = 1, .iters = 20 },
        .{ .b = 1, .seq = 2, .iters = 20 },
        .{ .b = 1, .seq = 16, .iters = 20 },
        .{ .b = 1, .seq = 512, .iters = 20 },
        .{ .b = 2, .seq = 1, .iters = 20 },
        .{ .b = 2, .seq = 8, .iters = 20 },
    };
    for (cases) |shape_case| {
        const input_parts = try combInput(allocator, shape_case.b, shape_case.seq);
        defer allocator.free(input_parts.data);
        defer _ = mlx.mlx_array_free(input_parts.array);
        const chain = try composed(input_parts.array, shape_case.iters, 1e-6, s);
        defer _ = mlx.mlx_array_free(chain);
        const fused = (try normalize(input_parts.array, shape_case.iters, 1e-6, s)) orelse return error.FusedSinkhornDeclined;
        defer _ = mlx.mlx_array_free(fused);
        const got = try hostF32(allocator, fused);
        defer allocator.free(got);
        const want = try hostF32(allocator, chain);
        defer allocator.free(want);
        const reference = try f64Reference(allocator, input_parts.data, shape_case.iters, 1e-6);
        defer allocator.free(reference);
        const got_bf16 = try hostBf16(allocator, fused, s);
        defer allocator.free(got_bf16);
        const want_bf16 = try hostBf16(allocator, chain, s);
        defer allocator.free(want_bf16);
        const stats = parityStats(got, want, reference, got_bf16, want_bf16);
        std.debug.print(
            "[xing-hc-parity] B={d} S={d} iters={d}: f32_mismatch={d}/{d} bf16_mismatch={d}/{d} max_abs={d:.9} max_f64={d:.9}\n",
            .{ shape_case.b, shape_case.seq, shape_case.iters, stats.f32_mismatches, got.len, stats.bf16_mismatches, got_bf16.len, stats.max_abs, stats.max_f64_error },
        );
        try std.testing.expectEqual(@as(usize, 0), stats.f32_mismatches);
        try std.testing.expectEqual(@as(usize, 0), stats.bf16_mismatches);
        // The f64 loop is a numerical bound, not the parity oracle. The
        // clamp-range inputs stay comfortably below one f32 ulp-scale bound.
        try std.testing.expect(stats.max_f64_error <= 0.000001);
    }
}

test "xing hc fused Sinkhorn declines CPU streams" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const values = [_]f32{
        1.0,  0.9,  0.8,  0.7,
        0.6,  0.5,  0.4,  0.3,
        0.2,  0.1,  0.05, 0.025,
        0.75, 0.65, 0.55, 0.45,
    };
    const shape = [_]c_int{ 1, 1, HC, HC };
    const input = mlx.mlx_array_new_data(&values, &shape, shape.len, .float32);
    defer _ = mlx.mlx_array_free(input);
    const got = try normalize(input, 20, 1e-6, s);
    try std.testing.expect(got == null);
}

test "xing hc fused Sinkhorn declines non-hc4 and non-f32 inputs" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);

    const f32_values = [_]f32{
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    };
    const bad_shape = [_]c_int{ 1, 1, 4, 3 };
    const wrong_shape_input = mlx.mlx_array_new_data(&f32_values, &bad_shape, bad_shape.len, .float32);
    defer _ = mlx.mlx_array_free(wrong_shape_input);
    try std.testing.expect((try normalize(wrong_shape_input, 20, 1e-6, s)) == null);

    const bf16_values = [_]u16{
        0x3f80, 0x3f80, 0x3f80, 0x3f80,
        0x3f80, 0x3f80, 0x3f80, 0x3f80,
        0x3f80, 0x3f80, 0x3f80, 0x3f80,
        0x3f80, 0x3f80, 0x3f80, 0x3f80,
    };
    const good_shape = [_]c_int{ 1, 1, HC, HC };
    const wrong_dtype_input = mlx.mlx_array_new_data(&bf16_values, &good_shape, good_shape.len, .bfloat16);
    defer _ = mlx.mlx_array_free(wrong_dtype_input);
    try std.testing.expect((try normalize(wrong_dtype_input, 20, 1e-6, s)) == null);
}

fn benchmarkEnabled() bool {
    const raw = std.c.getenv("XING_HC_BENCH") orelse return false;
    return raw[0] != '0';
}

fn timedComposed(input: mlx.mlx_array, s: mlx.mlx_stream) !u64 {
    var clock = BenchClock.init();
    const output = try composed(input, 20, 1e-6, s);
    defer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_array_eval(output));
    return clock.lap();
}

fn timedFused(input: mlx.mlx_array, s: mlx.mlx_stream) !u64 {
    var clock = BenchClock.init();
    const output = (try normalize(input, 20, 1e-6, s)) orelse return error.FusedSinkhornDeclined;
    defer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_array_eval(output));
    return clock.lap();
}

test "xing hc fused Sinkhorn benchmark (XING_HC_BENCH=1)" {
    if (!benchmarkEnabled() or mlx.noGpuBackend() or !enabled()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const allocator = std.testing.allocator;
    const input_parts = try combInput(allocator, 1, 512);
    defer allocator.free(input_parts.data);
    defer _ = mlx.mlx_array_free(input_parts.array);

    // Compile the kernel and materialize both graph arms before timing. Each
    // timed arm performs exactly one eval of its complete output graph.
    for (0..4) |_| {
        _ = try timedComposed(input_parts.array, s);
        _ = try timedFused(input_parts.array, s);
    }
    const reps = 12;
    var composed_ns: [reps]u64 = undefined;
    var fused_ns: [reps]u64 = undefined;
    for (0..reps) |i| {
        if ((i & 1) == 0) {
            composed_ns[i] = try timedComposed(input_parts.array, s);
            fused_ns[i] = try timedFused(input_parts.array, s);
        } else {
            fused_ns[i] = try timedFused(input_parts.array, s);
            composed_ns[i] = try timedComposed(input_parts.array, s);
        }
    }
    std.sort.pdq(u64, &composed_ns, {}, std.sort.asc(u64));
    std.sort.pdq(u64, &fused_ns, {}, std.sort.asc(u64));
    const chain_median = composed_ns[reps / 2];
    const fused_median = fused_ns[reps / 2];
    const ratio = @as(f64, @floatFromInt(chain_median)) / @as(f64, @floatFromInt(fused_median));
    std.debug.print(
        "[xing-hc-bench] rows=512 reps={d} chain_median={d}ns fused_median={d}ns speedup={d:.3}x\n",
        .{ reps, chain_median, fused_median, ratio },
    );
}
