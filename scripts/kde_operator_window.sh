#!/usr/bin/env bash
# KDE Plasma (Wayland): aplikace na Waylandu nesmí sama umístit okno na monitor,
# a KWin otevře okno obsluhy tam, kde je zrovna myš – často na projektoru.
# Tohle přidá pravidlo KWin: okno "Pískoviště – obsluha" vždy na monitor notebooku.
#
#   ./scripts/kde_operator_window.sh           # monitor 0 (obvykle vestavěný displej)
#   ./scripts/kde_operator_window.sh 1         # jiný monitor (pořadí viz: python -m sandbox monitors)
set -euo pipefail

SCREEN="${1:-0}"
GROUP="sandar-operator"
FILE="kwinrulesrc"

command -v kwriteconfig6 >/dev/null || { echo "Chybí kwriteconfig6 (není to KDE Plasma 6?)"; exit 1; }

kwriteconfig6 --file "$FILE" --group "$GROUP" --key Description "SandAR – okno obsluhy na notebook"
kwriteconfig6 --file "$FILE" --group "$GROUP" --key title "Pískoviště – obsluha"
kwriteconfig6 --file "$FILE" --group "$GROUP" --key titlematch 1        # přesná shoda
kwriteconfig6 --file "$FILE" --group "$GROUP" --key wmclassmatch 0      # třída okna libovolná
kwriteconfig6 --file "$FILE" --group "$GROUP" --key screen "$SCREEN"
kwriteconfig6 --file "$FILE" --group "$GROUP" --key screenrule 2        # vynutit
kwriteconfig6 --file "$FILE" --group "$GROUP" --key maximizehoriz true
kwriteconfig6 --file "$FILE" --group "$GROUP" --key maximizehorizrule 3 # použít na začátku
kwriteconfig6 --file "$FILE" --group "$GROUP" --key maximizevert true
kwriteconfig6 --file "$FILE" --group "$GROUP" --key maximizevertrule 3

# zařadit do seznamu pravidel (bez duplicit)
rules="$(kreadconfig6 --file "$FILE" --group General --key rules)"
if [[ ",$rules," != *",$GROUP,"* ]]; then
  rules="${rules:+$rules,}$GROUP"
  kwriteconfig6 --file "$FILE" --group General --key rules "$rules"
fi
count=$(tr ',' '\n' <<<"$rules" | grep -c .)
kwriteconfig6 --file "$FILE" --group General --key count "$count"

gdbus call --session --dest org.kde.KWin --object-path /KWin --method org.kde.KWin.reconfigure >/dev/null
echo "Pravidlo KWin nastaveno: okno obsluhy -> monitor $SCREEN (maximalizované)."
