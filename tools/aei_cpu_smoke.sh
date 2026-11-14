#!/usr/bin/env bash
# Compile and run every AEI host-only regression without CUDA or an engine binary.
set -euo pipefail
cd "$(dirname "$0")/.."
compiler="${CXX:-g++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

common=(-std=c++20 -O2 -Wall -Wextra -Werror -pthread -Iinclude)
count=0
for test_src in tests/core/aei_*_test.cpp; do
  [[ -f "$test_src" ]] || continue
  name="$(basename "$test_src" .cpp)"
  extra=()
  case "$name" in
    aei_writer_test) extra=(src/platform/aei_trace_writer.cpp src/platform/file.cpp) ;;
    aei_policy_checkpoint_test|aei_profile_binding_test) extra=(src/platform/file.cpp) ;;
  esac
  echo "AEI CPU compile: $name"
  "$compiler" "${common[@]}" "$test_src" "${extra[@]}" -o "$out/$name"
  echo "AEI CPU run: $name"
  "$out/$name"
  count=$((count+1))
done
if (( count < 10 )); then
  echo "AEI CPU: only $count tests found (expected >= 10)" >&2
  exit 1
fi
echo "AEI CPU PASS: $count standalone tests"
