# Interaktivní AR pískoviště

Kinect v2 snímá písek, projektor (BenQ) na něj promítá barevnou mapu, vrstevnice,
vodu a výukové úkoly. Běží na macOS i Linuxu, napsáno v Pythonu (+ malý C++ most pro Kinect).

## Instalace

```bash
# 1) Python prostředí
uv venv --python 3.13 .venv
uv pip install --python .venv -e '.[dev]'

# 2) Ovladač Kinectu v2 (libfreenect2) + most kinect2_bridge – vše lokálně do third_party/ a build/
./scripts/install_kinect2.sh

# test Kinectu: vypíše sériové číslo
./build/kinect2_bridge/kinect2_bridge --list
```

Kinect v2 musí být v **USB 3** portu a mít zapojený napájecí adaptér.
Na Linuxu skript nainstaluje udev pravidlo (Kinect pak funguje bez sudo).

## Spuštění

```bash
.venv/bin/python -m sandbox                 # Kinect, když je; jinak simulace
.venv/bin/python -m sandbox --sensor sim    # simulace bez hardwaru (písek se hrabe myší)
.venv/bin/python -m sandbox calibrate       # rovnou kalibrace
.venv/bin/python -m sandbox monitors        # seznam monitorů (výběr projektoru)
.venv/bin/python -m sandbox games           # seznam her
.venv/bin/python -m sandbox snapshot        # vykreslí všechny hry do snapshots/*.png bez okna
.venv/bin/python -m sandbox record --seconds 20   # nahrávka z Kinectu → recordings/
.venv/bin/python -m sandbox --replay recordings/nahravka.npz   # přehrání nahrávky
```

Projekce se automaticky otevře přes celou obrazovku na druhém monitoru (projektoru),
na notebooku se otevře **operátorské okno** (hloubka z Kinectu, náhled projekce, stav, nápověda).

## Kalibrace (jednou po postavení, klávesa C)

1. Uhlaď písek.
2. V obraze projektoru přetáhni myší barevné úchyty **1–4** do rohů bedny (na úrovni písku).
   Šipky posouvají vybraný úchyt o 1 px (Shift = 10 px).
3. Rohy bedny v hloubkovém obraze se najdou **automaticky** (klávesa A = znovu), nebo je klikni ručně.
   Musí to být **stejné fyzické rohy** jako úchyty 1–4 na projektoru – když nesedí pořadí,
   **O** pootočí pořadí rohů a **X** ho zrcadlí.
4. **Enter** – Kinect změří rovinu písku, **rozměry pískoviště** a výšku nad pískem; uloží se do
   `config/calibration_kinect2.json`.
5. Kontrola: projektor ukazuje mřížku po 10 cm a živé vrstevnice. Polož na písek hrnek –
   vrstevnice musí obkreslit jeho obrys. Když je obraz na vysokých místech posunutý,
   dolaď **PageUp/PageDown** (výška projektoru, korekce paralaxy) a znovu Enter.

## Hry a ovládání

| Klávesa | Všude |
|---|---|
| 1–9, Tab | výběr hry |
| C | kalibrace |
| H | nápověda promítnutá na písek |
| Z | zmrazit terén (výklad nad hotovým modelem) |
| Esc 2× | konec |
| F / N | (simulace) uhladit / nový terén, myš = hrabání |

1. **Topografická mapa** – barvy podle výšky, vrstevnice, stínování. K vrstevnice, S stín, L legenda, ↑↓ hladina moře, ←→ hustota vrstevnic.
2. **Turistická mapa** – hnědé vrstevnice na „papíře“, vodní plochy, kóty vrcholů v m n. m.
3. **Výšková mapa** – výška jako barva/šedá se stupnicí v cm (G, A).
4. **Postav podle předlohy** – červená = přidej písek, modrá = uber, čáry = vrstevnice předlohy.
   Průběžné skóre shody 0–100 %, při ≥ 80 % „Výborně!“. M přepne režim *Mapa* (jen vrstevnice,
   Enter = vyhodnotit), ←→ předloha, ↑↓ převýšení, **U uloží aktuální písek jako novou předlohu**
   (učitel postaví krajinu, uloží, uhladí a žáci ji staví podle mapy).
5. **Voda a povodně** – podrž ruku nad pískem a prší; voda stéká, plní údolí a jezera.
   R déšť všude, P povodeň, X vysušit, D odtok na okrajích.

Vlastní předlohy: PNG v šedi (bílá = nejvýš) do složky `targets/`.

**Skóre shody** = 1 − RMSE / RMSE(rovný písek): uhlazený písek má 0 %, přesná kopie 100 %.
Celkový posun výšky se neodečítá do chyby (množství písku žák neovlivní), okraj 5 % se nehodnotí.

## Nastavení

Vytvoř `config/settings.json` a uveď jen to, co chceš změnit (všechny volby a výchozí hodnoty
jsou v `sandbox/config.py`):

```json
{
  "display": {"monitor": "BenQ"},
  "map": {"contour_interval": 0.01, "vertical_scale": 5000, "base_altitude": 300},
  "terrain": {"max_height": 0.22, "stable_frames": 8},
  "challenge": {"success_score": 80, "target_relief": 0.10}
}
```

## Architektura

```
Kinect v2 ─► kinect2_bridge (C++ proces) ─► sensor/ ─► terrain/ ─► HeightMap ─► games/ ─► plátno ─► projector/ ─► BenQ
                                            simulace   kalibrace,    (metry)     hry       pohled     homografie +
                                            nahrávka   filtr rukou                         shora      paralaxa
```

* **`sandbox/sensor/`** – jen hloubkové snímky. Kinect běží v samostatném procesu
  (`native/kinect2_bridge`), takže pád ovladače neshodí aplikaci a most se sám restartuje.
* **`sandbox/terrain/`** – kalibrace Kinectu (rovina písku, rohy, rozměry), převod hloubky na
  výškovou mapu v metrech, časový filtr (projíždějící ruka se do terénu nepropíše), maska rukou.
* **`sandbox/projector/`** – kalibrace projektoru (4 rohy) a promítnutí s korekcí paralaxy.
* **`sandbox/games/`** – hry. Dostávají jen `HeightMap` a kreslí na plátno = pohled shora
  na bednu. O Kinectu ani projektoru nic nevědí.

### Nová hra

Ulož soubor do `sandbox/games/`, načte se sám:

```python
from .base import Game

class Lava(Game):
    id = "lava"
    title = "Sopka a láva"
    order = 60

    def start(self):
        self.prog = self.host.program("""
            void main() {
                float h = height_at(v_uv);
                vec3 c = elevation_color(h);
                if (h > 0.08) c = mix(c, vec3(1.0, 0.3, 0.0), 0.5 + 0.5 * sin(u_time * 4.0));
                vec2 cl = contours(h);
                f_color = vec4(mix(c, vec3(0.0), cl.x * 0.6 + cl.y), 1.0);
            }""")

    def render(self):
        self.host.draw_fullscreen(self.prog)
```

K dispozici v shaderech (`sandbox/render/glsl.py`): `height_at`, `height_grad`, `elevation_color`,
`land_color`, `contours`, `contour_line`, `hillshade`, `heat_color`, `u_time`, `u_box`.
V Pythonu: `self.host.painter` (text s diakritikou, čáry, kruhy), `self.host.heightmap`,
`hm.hands` (kde je ruka), `self.host.float_texture`, `self.host.notify`.
Kontrola bez projektoru: `python -m sandbox snapshot`.

## Testy

```bash
.venv/bin/python -m pytest
```

Testy ověřují kalibraci a rekonstrukci terénu proti simulaci se známou geometrií
(přesnost < 1 mm), filtr rukou, simulaci vody (zachování objemu), skóre předloh a paralaxu.
