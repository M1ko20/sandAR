# CLAUDE.md – Interaktivní AR pískoviště (Den otevřených dveří)

AR sandbox: Kinect v2 nad bednou s polystyrenovou drtí snímá hloubku, projektor promítá
obraz, který sedí na kopce a údolí. Návštěvníci modelují terén rukama, aplikace reaguje
v reálném čase. Cíl: stabilní, hezká instalace, která celý den běží bez zásahu.

Hlavní princip: návštěvník musí okamžitě pochopit, že rukama ovládá živý virtuální svět.

## Skutečná architektura (stav 11/2026)

```
Kinect v2 ─► build/kinect2_bridge (C++ proces, libfreenect2) ─► sandbox/sensor/ ─► terrain/ ─► HeightMap
                                                                 sim / replay        kalibrace, filtr rukou
HeightMap ─► games/<mód>.py ─► plátno (pohled shora) ─► projector/compositor (homografie + paralaxa) ─► projektor
                                └─► ui/operator.py (okno obsluhy na notebooku)
```

* `sandbox/games/` – **jeden soubor = jeden mód** (potomek `Game` z `games/base.py`, načte se sám).
  ClassVars: `id, title, description, order, keys, category, intro, attract`; metody
  `start/stop/update(hm, dt)/render/key/status/reset`. Jádro se kvůli módu nemění.
* `sandbox/sim/` – znovupoužitelné simulace: `water.py` (virtual pipes), `lava.py`, `ball.py`, `ecosystem.py`.
* `sandbox/render/` – `glsl.py` (sdílená GLSL knihovna: height_at, elevation_color, contours, hillshade),
  `host.py` (RenderHost pro módy), `painter.py` (2D kreslení, text), `window.py` (GLFW; na Linuxu
  nativní Wayland, OpenGL přes vlastní loader).
* `sandbox/terrain/` – kalibrace Kinectu (rohy podle stěn bedny), `processor.py` (výšková mapa,
  časový filtr, ruce → `hm.hands`, `hm.hand_points`).
* `sandbox/terrain/playarea.py` – hrací plocha: bedna bez úkosu stěn ∩ obraz projektoru; aplikace
  pracuje s oříznutou kalibrací, soubor drží celou bednu + okraje (`play_area`).
* `sandbox/autocal.py` – automatická kalibrace a kontrola přes barevnou kameru Kinectu
  (`build/kinect2_bridge/kinect2_snap`).
* `sandbox/attract.py` – úvodní slajd módu, attract mode, auto-reset.
* `sandbox/share.py` – LAN server pro stažení fotek (Digitální otisk) + QR (cv2.QRCodeEncoder).
* Konfigurace: dataclassy v `sandbox/config.py`, lokální přepisy v `config/settings.json`.

Rozhodnutí oproti původnímu plánu: místo DLT 3×4 matice máme homografii roviny písku +
model paralaxy (výška a průmět projektoru) – změřená přesnost 2–3 mm (`checkcal`).
Místo pylibfreenect2 C++ most přes rouru (stabilní, restartovatelný). Žádné pydantic/FastAPI –
zbytečné závislosti; config jsou dataclassy + JSON.

## Hotovo

- Kinect v2 přes most s auto-restartem, hlídač rodiče (žádné osiřelé procesy), sim + replay + record
- Rovina, filtrace, vyplnění děr, okraj u stěn (`terrain.edge_margin`), detekce rukou + středy rukou
- Automatická kalibrace (`python -m sandbox autocalibrate`) a kontrola (`checkcal`), ruční kalibrace (C)
- Korekce paralaxy v kompozitoru, sdílené shadery (výšková rampa, vrstevnice přes fwidth, stínování)
- Úvodní slajd, attract mode, Backspace = reset, pád módu nespadne celou aplikaci
- Log do `logs/`, systemd user služba (`scripts/install_service.sh`), KDE pravidlo okna obsluhy
- Módy: topo, mapa, výška, předloha, voda, kulička (minigolf), poklad, sopka, tsunami,
  ekosystém, záchrana města, digitální otisk (QR), nejvyšší hora / nejhlubší propast
- Hry pro dva hráče (`games/versus.py` = půlky, dělicí čára, text otočený k hráči u kratší strany):
  vodní válka, král kopce, potopa ostrovů, modeluj podle předlohy (P = 1 hráč s mírným hodnocením)

## Zbývá (podle priority)

- Ortofoto (data ČÚZK – nutné stáhnout předem, offline)
- Webový admin z mobilu (přepínání módů, rekalibrace) – zvážit stdlib server jako `share.py`
- Fyzické QR karty pro volbu módu (barevná kamera – ale tu za běhu drží most; most by musel posílat i barvu)
- Další módy z plánu: meteority, podmořský svět, dinosauři, světelné částice, dva týmy, závod vody
- Celodenní test na akci

## Pravidla

- Nový mód = nový soubor v `sandbox/games/`, fyzika do `sandbox/sim/`, testy do `tests/`.
- Texty pro návštěvníky česky (s diakritikou), čitelné ze 2 m (font ≥ výška plátna / 30), jasné barvy (tmavá místnost).
- Mód musí reagovat na skutečný terén, mít `intro`, `reset()` a běžet s rezervou na 60 FPS (numpy vektorizovaně).
- Klávesy, které bere jádro: Esc, Tab, 0–9 (i se Shiftem), C, H, Z, Backspace (pokud ho mód nepoužije), v simulaci F, N.
- Ověření: `pytest`, `python -m sandbox snapshot --game <id> --duration <s>` (headless PNG),
  na hardwaru `checkcal` a běh s `--duration N --snapshot-dir DIR`.
- Font nemá všechny symboly (např. ⚠) – kresli je painterem.

## Bezpečnost na akci

Polystyren: dozor u bedny (malé děti), hořlavý – projektor a zdroje mimo drť, antistatika,
zabezpečené kabely, co nejtmavší místnost. Share server běží na LAN IP počítače – na akci
pustit vlastní Wi-Fi/hotspot a otevřít port 8765 ve firewallu jen tam.
