#!/usr/bin/env bash
set -euo pipefail

ADB_BIN="${ADB_BIN:-$HOME/Library/Android/sdk/platform-tools/adb}"
PKG="com.example.deep_learning_app"

if [[ ! -x "$ADB_BIN" ]]; then
  echo "adb not found: $ADB_BIN" >&2
  exit 1
fi

"$ADB_BIN" devices >/dev/null
"$ADB_BIN" logcat -c

echo "[INFO] collecting crash-related logs for $PKG ..."
"$ADB_BIN" logcat -v time \
  | grep -E "($PKG|AndroidRuntime|Fatal signal|YoloLocateProcessor|MainActivity|Locate|UricAcidProcessor)"
