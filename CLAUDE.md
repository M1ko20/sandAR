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
* `sandbox/sim/` – znovupoužitelné simulace: `water.py` (virtual pipes), `lava.py`, `ball.py`, `ecosystem.py`,
  `fauna.py` (12 druhů zvířat podle prostředí).
* `sandbox/render/` – `glsl.py` (sdílená GLSL knihovna: height_at, elevation_color, contours, hillshade),
  `host.py` (RenderHost pro módy), `painter.py` (2D kreslení, text; na plátně vynucuje písmo ≥ výška/30),
  `hud.py` (jednotný vzhled textů: label, legend, bar, fit), `fx.py` (déšť z ruky), `fauna.py` (kreslení
  zvířat), `window.py` (GLFW; na Linuxu nativní Wayland, OpenGL přes vlastní loader).
* `sandbox/terrain/` – kalibrace Kinectu (rohy podle stěn bedny), `processor.py` (výšková mapa,
  časový filtr; pixely ruky vyřadí), `hands.py` (detekce a sledování rukou z hloubky: pozadí po pixelech,
  hloubkové hrany, paže = oblast přes stěnu bedny, dlaň = geodetický konec paže, One Euro;
  → `hm.hand_list` [Hand: id, dlaň, špička, výška, rychlost], `hm.hand_points`, maska `hm.hands`).
* `sandbox/terrain/playarea.py` – hrací plocha: bedna bez úkosu stěn ∩ obraz projektoru; aplikace
  pracuje s oříznutou kalibrací, soubor drží celou bednu + okraje (`play_area`).
* `sandbox/autocal.py` – automatická kalibrace a kontrola přes barevnou kameru Kinectu
  (`build/kinect2_bridge/kinect2_snap`).
* `sandbox/attract.py` – úvodní slajd módu, attract mode, auto-reset.
* `sandbox/ortofoto.py` – stažení ortofota a DMR 5G z bezplatných služeb ČÚZK (`python -m sandbox ortofoto`)
  do `data/ortofoto/` pro mód Skutečná krajina (offline na akci).
* `sandbox/ui/help_overlay.py` – nápověda na písku (H), písmo se přizpůsobí, aby se vše vešlo.
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
- Detekce rukou z hloubky (`terrain/hands.py`), naladěná na skutečné nahrávce `recordings/ruce.npz`
  (více lidí z libovolné strany, ruka na písku se nepropíše do terénu)
- Módy (pořadí = klávesy): mapy (topo, mapa, výška) → příroda (skutečná krajina/ortofoto, voda,
  ekosystém, sopka s vesnicemi, tsunami) → hry (kulička, poklad, záchrana města, nejvyšší hora,
  postav podle předlohy – 16 předloh, hodnotí se jen hlavní část) → pro dva (níže)
- Hry pro dva hráče (`games/versus.py` = půlky, dělicí čára, text otočený k hráči u kratší strany):
  vodní válka, král kopce, potopa ostrovů, modeluj podle předlohy (P = 1 hráč s mírným hodnocením)

## Zbývá (podle priority)

- Webový admin z mobilu (přepínání módů, rekalibrace) – stdlib http.server
- Fyzické QR karty pro volbu módu (barevná kamera – ale tu za běhu drží most; most by musel posílat i barvu)
- Další módy z plánu: meteority, podmořský svět, dinosauři, světelné částice, dva týmy, závod vody
- Celodenní test na akci

## Pravidla

- Nový mód = nový soubor v `sandbox/games/`, fyzika do `sandbox/sim/`, testy do `tests/`.
- Texty pro návštěvníky česky (s diakritikou), čitelné ze 2 m (font ≥ výška plátna / 30), jasné barvy (tmavá místnost).
- Mód musí reagovat na skutečný terén, mít `intro`, `reset()` a běžet s rezervou na 60 FPS (numpy vektorizovaně).
- Klávesy, které bere jádro: Esc, Tab, 0–9 (i se Shiftem), C, H, Z, Backspace (pokud ho mód nepoužije), v simulaci F, N.
- Ruce v módech: `hm.hand_list` (dlaň `pos`, špička `tip`, `height` nad pískem, `id`), přítomnost `hm.has_hands`.
  Texty přes `render/hud.py` (label, legend), velikost `hud.size(canvas, k)`.
- Ověření: `pytest`, `python -m sandbox snapshot --game <id> --duration <s>` (headless PNG),
  na hardwaru `checkcal` a běh s `--duration N --snapshot-dir DIR`.
- Font nemá všechny symboly (např. ⚠) – kresli je painterem.

## Bezpečnost na akci

Polystyren: dozor u bedny (malé děti), hořlavý – projektor a zdroje mimo drť, antistatika,
zabezpečené kabely, co nejtmavší místnost. Data ortofota stáhnout předem (`python -m sandbox ortofoto`).
