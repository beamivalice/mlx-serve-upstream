#!/bin/bash
# A cached passing test must not hide a newly requested fixture failure.
set -euo pipefail
cd "$(dirname "$0")/.."
zig="${ZIG:-$PWD/.zig-toolchain/zig}"
unset XING4_TEST_MODEL XING4_FIXTURE
"$zig" build test -Doptimize=ReleaseFast -Dtest-filter="xing4 fixture"
missing="$PWD/zig-out/nonexistent-xing-fixture-$$"
if XING4_TEST_MODEL="$missing" "$zig" build test -Doptimize=ReleaseFast -Dtest-filter="xing4 fixture"; then
    echo "FAIL: cached success hid the missing fixture" >&2
    exit 1
fi
echo "PASS: environment-gated fixture was executed again"
