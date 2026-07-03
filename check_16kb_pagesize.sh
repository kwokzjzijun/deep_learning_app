#!/usr/bin/env bash
set -euo pipefail

APK_PATH="${1:-/Users/gzj/deep_learning_app/app/build/outputs/apk/debug/app-debug.apk}"
READELF="${READELF:-$HOME/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-readelf}"

if [[ ! -f "$APK_PATH" ]]; then
  echo "APK not found: $APK_PATH" >&2
  exit 2
fi
if [[ ! -x "$READELF" ]]; then
  echo "llvm-readelf not found: $READELF" >&2
  exit 2
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

unzip -q "$APK_PATH" 'lib/arm64-v8a/*.so' -d "$tmpdir"

bad=0
printf "%-34s %-8s\n" "Library" "MaxAlign"
printf "%-34s %-8s\n" "----------------------------------" "--------"

for so in $(find "$tmpdir/lib/arm64-v8a" -name '*.so' | sort); do
  aligns=$($READELF -l "$so" | awk '/LOAD/{print $NF}')
  max_align=$(python3 - "$aligns" <<'PY'
import sys
vals=sys.argv[1].split()
m=0
for v in vals:
    try:
        m=max(m,int(v,16))
    except Exception:
        pass
print(hex(m))
PY
)
  printf "%-34s %-8s\n" "$(basename "$so")" "$max_align"
  if [[ "$max_align" != "0x4000" && "$max_align" != "0x8000" && "$max_align" != "0x10000" ]]; then
    bad=1
  fi
done

if [[ "$bad" -eq 1 ]]; then
  echo "[WARN] Some native libs are not 16KB-page compatible yet (MaxAlign < 0x4000)."
  exit 1
fi

echo "[OK] All arm64 native libs look 16KB-page compatible."
