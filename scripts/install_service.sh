#!/usr/bin/env bash
# Pískoviště jako uživatelská služba systemd: spustí se po přihlášení do grafického
# sezení (projektor musí být připojený) a po pádu se samo restartuje.
#
#   ./scripts/install_service.sh            # nainstalovat a zapnout
#   ./scripts/install_service.sh --remove   # vypnout a odebrat
#   journalctl --user -u piskoviste -f      # živý log (podrobný log je i v logs/)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT="$HOME/.config/systemd/user/piskoviste.service"

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now piskoviste.service 2>/dev/null || true
  rm -f "$UNIT"
  systemctl --user daemon-reload
  echo "Služba odebrána."
  exit 0
fi

mkdir -p "$(dirname "$UNIT")"
cat >"$UNIT" <<UNIT
[Unit]
Description=Interaktivní AR pískoviště
After=graphical-session.target
PartOf=graphical-session.target
# po pádu Kinect potřebuje chvilku, než se znovu přihlásí na USB – restartovat donekonečna
StartLimitIntervalSec=0

[Service]
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/python -m sandbox
Restart=always
RestartSec=5

[Install]
WantedBy=graphical-session.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now piskoviste.service
echo "Služba piskoviste zapnuta (log: journalctl --user -u piskoviste -f, soubor: $ROOT/logs/)."
