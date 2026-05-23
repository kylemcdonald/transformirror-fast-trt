#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run this as the desktop user, not with sudo." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/.." && pwd)"
service_name="${TRANSFORMIRROR_SERVICE_NAME:-transformirror-fast-trt.service}"
service_user="$(id -un)"
service_uid="$(id -u)"
user_unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
service_path="$user_unit_dir/$service_name"

"$script_dir/uninstall_systemd_services.sh"

mkdir -p "$user_unit_dir"

cat > "$service_path" <<EOF
[Unit]
Description=Transformirror Fast TRT
Wants=network-online.target
After=network-online.target graphical-session.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$root_dir
Environment=DISPLAY=${TRANSFORMIRROR_DISPLAY:-:0}
Environment=XDG_RUNTIME_DIR=/run/user/$service_uid
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$service_uid/bus
Environment=PYTHONNOUSERSITE=1
Environment=HF_HUB_ENABLE_HF_TRANSFER=1
ExecStart=$root_dir/run-transformirror.sh
Restart=always
RestartSec=5
TimeoutStopSec=20
KillSignal=SIGINT

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$service_name"

if command -v loginctl >/dev/null 2>&1; then
  if loginctl enable-linger "$service_user" >/dev/null 2>&1; then
    :
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo loginctl enable-linger "$service_user" >/dev/null 2>&1 || true
  fi
fi

echo "Installed $service_name at $service_path"
echo "Status: systemctl --user status $service_name --no-pager"
