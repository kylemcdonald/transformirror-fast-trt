#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run this as the desktop user, not with sudo." >&2
  exit 1
fi

known_units=(
  transformirror.service
  transformirror-local.service
  transformirror-fast-trt.service
)

append_detected_units() {
  local scope="$1"
  local line unit
  local command=(systemctl)

  if [[ -n "$scope" ]]; then
    command+=("$scope")
  fi

  while read -r line; do
    [[ -n "$line" ]] || continue
    unit="${line%% *}"
    [[ "$unit" == *transformirror*.service* ]] || continue
    known_units+=("$unit")
  done < <("${command[@]}" list-unit-files --type=service --all --no-pager 2>/dev/null || true)
}

unique_units() {
  local unit
  declare -A seen=()

  for unit in "$@"; do
    [[ -n "$unit" ]] || continue
    [[ -n "${seen[$unit]:-}" ]] && continue
    seen[$unit]=1
    printf '%s\n' "$unit"
  done
}

append_detected_units --user
append_detected_units ""

mapfile -t units < <(unique_units "${known_units[@]}")

user_unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

for unit in "${units[@]}"; do
  systemctl --user stop "$unit" >/dev/null 2>&1 || true
  systemctl --user disable "$unit" >/dev/null 2>&1 || true
  rm -f "$user_unit_dir/$unit"
  rm -f "$user_unit_dir/default.target.wants/$unit"
done

systemctl --user daemon-reload >/dev/null 2>&1 || true
for unit in "${units[@]}"; do
  systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
done

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  for unit in "${units[@]}"; do
    sudo systemctl stop "$unit" >/dev/null 2>&1 || true
    sudo systemctl disable "$unit" >/dev/null 2>&1 || true
    sudo rm -f "/etc/systemd/system/$unit"
    sudo rm -f "/etc/systemd/system/multi-user.target.wants/$unit"
  done
  sudo systemctl daemon-reload
  for unit in "${units[@]}"; do
    sudo systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  done
else
  echo "Skipping system-wide unit removal because passwordless sudo is unavailable." >&2
fi

echo "Removed Transformirror systemd units: ${units[*]}"
