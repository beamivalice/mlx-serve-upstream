const std = @import("std");

/// The on-disk SentencePiece `ModelProto.SentencePiece.Type` values.
pub const PieceType = enum {
    normal,
    unknown,
    control,
    user_defined,
    byte,
};

pub const Piece = struct {
    text: []const u8,
    score: f32,
    kind: PieceType,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    /// All piece strings borrow from this one owned protobuf copy.
    storage: []u8,
    pieces: []Piece,
    add_dummy_prefix: bool,
    byte_fallback: bool,
    unk_id: u32,
    bos_id: ?u32,
    eos_id: ?u32,
    pad_id: ?u32,

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.pieces);
        self.allocator.free(self.storage);
    }
};

/// Parse the protobuf wire format used by SentencePiece `tokenizer.model`.
///
/// This intentionally implements only the ModelProto fields needed by a BPE
/// tokenizer. Unknown protobuf fields are skipped, but an unknown model type or
/// normalizer is rejected rather than guessed.
pub fn parse(allocator: std.mem.Allocator, input: []const u8) !Model {
    var model = Model{
        .allocator = allocator,
        .storage = try allocator.dupe(u8, input),
        .pieces = &.{},
        .add_dummy_prefix = true,
        .byte_fallback = false,
        .unk_id = 0,
        .bos_id = 1,
        .eos_id = 2,
        .pad_id = null,
    };
    errdefer model.deinit();

    var pieces: std.ArrayList(Piece) = .empty;
    defer pieces.deinit(allocator);

    var trainer_seen = false;
    var trainer_model_type: ?u64 = null;
    var normalizer_seen = false;
    var normalizer_name: ?[]const u8 = null;
    var normalizer_has_charsmap = false;
    var normalizer_rule_tsv: ?[]const u8 = null;
    var normalizer_remove_extra = true;
    var normalizer_escape = true;

    var reader = Reader{ .data = model.storage };
    while (!reader.done()) {
        const key = try reader.readVarint();
        const field_no = key >> 3;
        const wire = key & 7;
        if (field_no == 0) return error.InvalidSentencePieceModel;

        switch (field_no) {
            1 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                const payload = try reader.readBytes();
                try pieces.append(allocator, try parsePiece(payload));
            },
            2 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                trainer_seen = true;
                try parseTrainer(
                    try reader.readBytes(),
                    &trainer_model_type,
                    &model.byte_fallback,
                    &model.unk_id,
                    &model.bos_id,
                    &model.eos_id,
                    &model.pad_id,
                );
            },
            3 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                normalizer_seen = true;
                try parseNormalizer(
                    try reader.readBytes(),
                    &normalizer_name,
                    &normalizer_has_charsmap,
                    &normalizer_rule_tsv,
                    &model.add_dummy_prefix,
                    &normalizer_remove_extra,
                    &normalizer_escape,
                );
            },
            else => try reader.skip(wire),
        }
    }

    if (!trainer_seen or trainer_model_type == null or trainer_model_type.? != 2) {
        return error.UnsupportedSentencePieceModel;
    }
    if (!normalizer_seen or normalizer_name == null or
        !std.mem.eql(u8, normalizer_name.?, "identity") or
        normalizer_has_charsmap or
        (normalizer_rule_tsv != null and normalizer_rule_tsv.?.len != 0) or
        normalizer_remove_extra or
        !normalizer_escape)
    {
        return error.UnsupportedSentencePieceNormalizer;
    }
    if (pieces.items.len == 0) return error.InvalidSentencePieceModel;

    model.pieces = try pieces.toOwnedSlice(allocator);
    pieces = .empty;
    return model;
}

const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    fn done(self: *const Reader) bool {
        return self.pos == self.data.len;
    }

    fn readVarint(self: *Reader) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        for (0..10) |i| {
            if (self.pos >= self.data.len) return error.InvalidSentencePieceModel;
            const byte = self.data[self.pos];
            self.pos += 1;
            if (i == 9 and byte > 1) return error.InvalidSentencePieceModel;
            result |= @as(u64, byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) return result;
            shift += 7;
        }
        return error.InvalidSentencePieceModel;
    }

    fn readBytes(self: *Reader) ![]const u8 {
        const raw_len = try self.readVarint();
        const len: usize = std.math.cast(usize, raw_len) orelse return error.InvalidSentencePieceModel;
        if (len > self.data.len - self.pos) return error.InvalidSentencePieceModel;
        const out = self.data[self.pos .. self.pos + len];
        self.pos += len;
        return out;
    }

    fn readFixed32(self: *Reader) !u32 {
        if (self.data.len - self.pos < 4) return error.InvalidSentencePieceModel;
        const out = @as(u32, self.data[self.pos]) |
            (@as(u32, self.data[self.pos + 1]) << 8) |
            (@as(u32, self.data[self.pos + 2]) << 16) |
            (@as(u32, self.data[self.pos + 3]) << 24);
        self.pos += 4;
        return out;
    }

    fn skip(self: *Reader, wire: u64) !void {
        switch (wire) {
            0 => _ = try self.readVarint(),
            1 => {
                if (self.data.len - self.pos < 8) return error.InvalidSentencePieceModel;
                self.pos += 8;
            },
            2 => _ = try self.readBytes(),
            5 => {
                if (self.data.len - self.pos < 4) return error.InvalidSentencePieceModel;
                self.pos += 4;
            },
            else => return error.InvalidSentencePieceModel,
        }
    }
};

fn parsePiece(input: []const u8) !Piece {
    var piece = Piece{ .text = "", .score = 0, .kind = .normal };
    var have_text = false;
    var reader = Reader{ .data = input };
    while (!reader.done()) {
        const key = try reader.readVarint();
        const field_no = key >> 3;
        const wire = key & 7;
        switch (field_no) {
            1 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                piece.text = try reader.readBytes();
                have_text = true;
            },
            2 => {
                if (wire != 5) return error.InvalidSentencePieceModel;
                piece.score = @bitCast(try reader.readFixed32());
                if (!std.math.isFinite(piece.score)) return error.InvalidSentencePieceModel;
            },
            3 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                piece.kind = switch (try reader.readVarint()) {
                    1 => .normal,
                    2 => .unknown,
                    3 => .control,
                    4 => .user_defined,
                    6 => .byte,
                    else => return error.UnsupportedSentencePieceModel,
                };
            },
            else => try reader.skip(wire),
        }
    }
    if (!have_text or piece.text.len == 0) return error.InvalidSentencePieceModel;
    return piece;
}

fn parseTrainer(
    input: []const u8,
    model_type: *?u64,
    byte_fallback: *bool,
    unk_id: *u32,
    bos_id: *?u32,
    eos_id: *?u32,
    pad_id: *?u32,
) !void {
    var reader = Reader{ .data = input };
    while (!reader.done()) {
        const key = try reader.readVarint();
        const field_no = key >> 3;
        const wire = key & 7;
        switch (field_no) {
            3 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                model_type.* = try reader.readVarint();
            },
            35 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                byte_fallback.* = (try reader.readVarint()) != 0;
            },
            40 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                unk_id.* = try parseId(try reader.readVarint()) orelse return error.InvalidSentencePieceModel;
            },
            41 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                bos_id.* = try parseOptionalId(try reader.readVarint());
            },
            42 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                eos_id.* = try parseOptionalId(try reader.readVarint());
            },
            43 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                pad_id.* = try parseOptionalId(try reader.readVarint());
            },
            else => try reader.skip(wire),
        }
    }
}

fn parseId(raw: u64) !?u32 {
    const signed: i32 = @bitCast(@as(u32, @truncate(raw)));
    if (signed < 0) return null;
    return @intCast(signed);
}

fn parseOptionalId(raw: u64) !?u32 {
    return parseId(raw);
}

fn parseNormalizer(
    input: []const u8,
    name: *?[]const u8,
    has_charsmap: *bool,
    rule_tsv: *?[]const u8,
    add_dummy_prefix: *bool,
    remove_extra: *bool,
    escape_whitespaces: *bool,
) !void {
    var reader = Reader{ .data = input };
    while (!reader.done()) {
        const key = try reader.readVarint();
        const field_no = key >> 3;
        const wire = key & 7;
        switch (field_no) {
            1 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                name.* = try reader.readBytes();
            },
            2 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                has_charsmap.* = (try reader.readBytes()).len != 0;
            },
            3 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                add_dummy_prefix.* = (try reader.readVarint()) != 0;
            },
            4 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                remove_extra.* = (try reader.readVarint()) != 0;
            },
            5 => {
                if (wire != 0) return error.InvalidSentencePieceModel;
                escape_whitespaces.* = (try reader.readVarint()) != 0;
            },
            6 => {
                if (wire != 2) return error.InvalidSentencePieceModel;
                rule_tsv.* = try reader.readBytes();
            },
            else => try reader.skip(wire),
        }
    }
}

test "SentencePiece parser rejects non-BPE model types" {
    // ModelProto.trainer_spec.model_type = UNIGRAM.
    const bytes = [_]u8{ 0x12, 0x02, 0x18, 0x01 };
    try std.testing.expectError(error.UnsupportedSentencePieceModel, parse(std.testing.allocator, &bytes));
}

test "SentencePiece parser rejects non-identity normalizers" {
    // One normal piece, a BPE trainer spec, and NormalizerSpec.name=nmt_nfkc.
    const bytes = [_]u8{
        0x0a, 0x03, 0x0a, 0x01, 'x',
        0x12, 0x02, 0x18, 0x02, 0x1a,
        0x0a, 0x0a, 0x08, 'n',  'm',
        't',  '_',  'n',  'f',  'k',
        'c',
    };
    try std.testing.expectError(error.UnsupportedSentencePieceNormalizer, parse(std.testing.allocator, &bytes));
}
