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

**Fedora a další distribuce bez apt:** skript závislosti neinstaluje, doinstaluj je ručně:
`sudo dnf install libusb1-devel turbojpeg-devel glfw-devel`.

**KDE Plasma na Waylandu:** aplikace na Waylandu nesmí sama umístit okno, takže KWin otevře okno obsluhy
tam, kde je myš (často na projektoru). Jednorázově spusť `./scripts/kde_operator_window.sh`.
Přidá pravidlo KWin, které okno obsluhy vždy otevře maximalizované na monitoru 0 (notebook).
HiDPI (např. 2880×1800 při škálování 2×) funguje nativně, projektor se vykresluje 1:1.

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

## Automatická kalibrace (doporučeno)

```bash
.venv/bin/python -m sandbox autocalibrate   # ~2 min, pískoviště musí být vypnuté (Kinect volný)
.venv/bin/python -m sandbox checkcal        # kontrola: chyba v cm + fotky do snapshots/kontrola/
```

Rohy bedny se najdou podle vnitřních hran stěn v hloubce (písek nemusí být uhlazený).
Projektor pak promítne ~70 bílých čtverců, **barevná kamera Kinectu** je najde a z hloubky
v těch místech se spočítá model projektoru (rohy 1–4 + výška a poloha pro korekci paralaxy).
`checkcal` promítá body na náhodná místa (i na kopce) a měří, kam skutečně dopadly; navíc
vyfotí promítnutou 10cm mřížku a vrstevnice po 2 cm proti vrstevnicím spočítaným z hloubky
(`kontrola_vrstevnice.png`: zelená = promítnuto, červená = skutečný terén – mají se krýt).
Typický výsledek: medián chyby 2–3 mm, na kopcích bez korekce paralaxy by to byly 3–4 cm.

**Hrací plocha:** z bedny se automaticky vyřízne obdélník bez úkosu stěn (Kinect tam měří
falešně, projekce by padala na stěny) a jen tam, kam projektor dosvítí. Plátno = tato plocha,
takže mapy, texty i hry jsou vždy celé na písku. Okraje se měří při `autocalibrate`;
po přestavění bedny nebo posunu projektoru stačí `python -m sandbox fitarea` (pár sekund).

Nástroj `build/kinect2_bridge/kinect2_snap` (barva srovnaná s hloubkou) se sestaví spolu s mostem.

## Ruční kalibrace (klávesa C)

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
| 1–9, 0, Shift+1–9 | výběr módu (1–9, 10, 11–19); nebo klik na tlačítko v okně obsluhy |
| Tab | další mód (Shift+Tab předchozí) |
| C | kalibrace |
| H | nápověda promítnutá na písek |
| Z | zmrazit terén (výklad nad hotovým modelem) |
| Esc 2× | konec |
| F / N | (simulace) uhladit / nový terén, myš = hrabání |

**Mapy**

1. **Topografická mapa** – barvy podle výšky, vrstevnice, stínování. K vrstevnice, S stín, L legenda, ↑↓ hladina moře, ←→ hustota vrstevnic.
2. **Turistická mapa** – hnědé vrstevnice na „papíře“, vodní plochy, kóty vrcholů v m n. m.
3. **Výšková mapa** – výška jako barva/šedá se stupnicí v cm (G, A).

**Příroda**

4. **Skutečná krajina** – letecký snímek skutečného místa v ČR (Říp, Milešovka, Bezděz, Macocha,
   Lysá hora, Pražský hrad) promítnutý na písek a stínovaný podle tvaru písku. Návštěvníci podle
   skutečného výškového modelu stavějí kopce a údolí (červená = přidej, modrá = uber), při shodě ≥ 70 %
   nápověda zmizí a zůstane čistá fotka. ←→ místo, M nápověda, V vrstevnice skutečného terénu.
   Data (ČÚZK – Ortofoto ČR a DMR 5G, zdarma, CC BY 4.0) se stahují předem: `python -m sandbox ortofoto`
   (potřebuje internet; uloží se do `data/ortofoto/`, na akci pak mód běží offline).
5. **Voda a povodně** – z dlaně držené nad pískem prší (na ruku se promítne mráček, kolem dopadají kapky);
   ruka v písku neprší, ať jde kopat koryta. R déšť všude, P povodeň, X vysušit, D odtok na okrajích.
6. **Živý ekosystém** – biomy, řeky, jezera, rostoucí lesy, sníh a 12 druhů zvířat, každý ve svém
   prostředí: ryby a kachny ve vodě, krabi a želvy na pláži, srnky, zajíci, lišky a motýli na loukách,
   medvěd v lese, kamzíci na skalách, orel nad horami, ptáci nad krajinou. Počet roste s velikostí
   prostředí, plachá zvířata utíkají před rukou.
7. **Živá sopka** – láva vyteče z vrcholu, stéká po svahu a chladne v černou krustu. Pod sopkou stojí
   vesnice – láva je zapálí, hrázemi a koryty je lze zachránit (E erupce, A automatický cyklus, V vesnice).
8. **Mega tsunami** – z moře přijde vlna a zaplaví krajinu; hráze chrání města (T = vlna hned).

**Hry (jeden hráč nebo tým)**

9. **Kulička (minigolf)** – kulička se kutálí jen po skutečném svahu, postav dráhu do jamky.
   K zapne odpalování kuličky rukou (výchozí vypnuto).
10. **Hledání pokladu** – pod pískem jsou poklady, kopej; kruh kolem ruky napovídá „Zima / Teplo / Hoří!“.
11. **Záchrana města** – povodeň z hor, postav hráze a zachraň co nejvíc domů (Enter = start).
12. **Nejvyšší hora** – minutová soutěž, rekord dne a všech dob; V = nejhlubší propast.
13. **Postav podle předlohy** – 16 předloh (kopec, sopka, kráter, řeka, hrad s příkopem, srdce, terasy,
    kaňon…). Hodnotí se jen **hlavní část** v čárkovaném rámečku (tvar + 5 cm okolo), zbytek bedny je
    ztmavený a nezáleží na něm. Červená = přidej, modrá = uber, zelená = sedí, cíl 80 %.
    M režim *Mapa* (jen vrstevnice, Enter = vyhodnotit), ←→ předloha, ↑↓ převýšení,
    **U uloží aktuální písek jako novou předlohu**.

**Hry pro dva hráče** – bedna je čarou rozdělená přes střed delší strany, každý hráč stojí u své
kratší strany (červení vlevo, modří vpravo na plátně) a texty má otočené k sobě:

14. **Vodní válka** – uprostřed vyvěrá voda, každý má u své strany vesnici s 5 domky. Hráze k sobě,
    koryta k soupeři; prohrává ten, komu se zatopí všechny domky. Pramen postupně sílí, po 2 minutách
    vyhrává víc suchých domků. Enter = pustit vodu hned, ↑↓ síla pramene.
15. **Král kopce** – 60 s kolo (začne, když někdo sáhne do písku); body dostává ten, kdo má ve své
    půlce nejvyšší vrchol. Pás 3 cm u čáry je neutrální.
16. **Potopa ostrovů** – 20 s na stavbu ostrovů, pak stoupá moře (čím dál rychleji). Vyhrává poslední
    ostrov nad hladinou (ostrov = aspoň ~20 cm² souše v hráčově půlce). ↑↓ rychlost moře.
17. **Modeluj podle předlohy** – závod: oba hráči mají stejnou předlohu otočenou k sobě, barevnou
    nápovědu (zapnutá) a hodnotí se jen hlavní část. Vyhrává, kdo první udrží shodu 80 %.
    P přepne na jednoho hráče s mírným hodnocením, M nápověda, ←→ předloha.

Mód 10 = klávesa 0, módy 11–17 = Shift+1 až Shift+7, Tab = další; v okně obsluhy jdou všechny vybrat klikem.
Úvodní slajd každého módu vysvětlí návštěvníkům, co dělat. **H** promítne na písek nápovědu
(jak hrát, ovládání, seznam módů) – velikost písma se přizpůsobí, vše se vejde.

Vlastní předlohy: PNG v šedi (bílá = nejvýš) do složky `targets/`.

**Skóre shody** = 1 − RMSE / RMSE(rovný písek) v hodnocené oblasti: uhlazený písek má 0 %, přesná
kopie 100 %. Celkový posun výšky se do chyby nepočítá (množství písku žák neovlivní).

## Detekce rukou

Ruce se hledají v hloubce Kinectu (`sandbox/terrain/hands.py`), funguje i v úplné tmě a s obrazem
promítnutým na rukou. Princip: **ruka je vždy konec paže, která vede přes okraj bedny** – z jakékoli
strany, libovolný počet lidí. Hromádka písku okraje bedny nedosáhne, takže ji nikdy nezamění za ruku.

1. model pozadí po pixelech (vyhrabaný důlek se přijme hned, nová hromádka po ⅓ s, ruka nikdy),
2. popředí rozříznuté podle hloubkových hran (ruka nad hromádkou se s ní nespojí),
3. paže = oblast, která sahá za stěnu (nebo k tělu nakloněnému nad bednou); dlaň = konec paže
   podle geodetické vzdálenosti od místa vstupu, dvě ruce na jedné paži/těle se rozliší větvením,
4. sledování se stálým číslem ruky a vyhlazením One Euro.

Pixely ruky terén vůbec nepoužije – ruka položená na písku se do mapy nepropíše. V okně obsluhy je
ruka v hloubkovém obraze fialová a v promítání má kroužek s číslem a výškou nad pískem. Naladěno
na nahrávce ze skutečné bedny (`recordings/ruce.npz`, přehrání: `--sensor replay --replay recordings/ruce.npz`).
V simulaci prostřední tlačítko myši vytvoří virtuální ruku. Parametry: sekce `hands` v nastavení.

## Provoz na akci

* **Úvodní slajd**: po přepnutí módu se na písek na 5 s promítne název a věta „jak hrát“.
* **Attract mode**: když se 90 s nikdo nedotkne písku (žádné ruce, žádná změna terénu),
  módy se samy střídají po 45 s a každý začíná s čistým světem. Ruka nad pískem střídání zastaví.
  Nastavení v sekci `attract` (viz `sandbox/config.py`).
* **Backspace** = reset aktuálního módu (nový začátek pro dalšího návštěvníka).
* Chyba v jednom módu instalaci neshodí – zaloguje se a přepne se na topografii.
* Log: `logs/piskoviste.log` (rotuje se), v okně obsluhy je stav senzoru, FPS, ruce a odpočet attract mode.
* Služba s automatickým restartem: `./scripts/install_service.sh` (odebrání `--remove`).
* Okraj u stěn řeší hrací plocha (viz kalibrace); `terrain.edge_margin` je už jen malá rezerva.

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

**Chvějí se vrstevnice na klidném písku?** Proti šumu Kinectu jsou dvě volby v `terrain`:
`temporal_window` (kolik posledních snímků se průměruje, výchozí 12 ≈ 0,4 s; 1 = vypnuto) a
`hysteresis` (menší změny výšky se ignorují, výchozí 0.0015 = 1,5 mm). Když se to pořád
chvěje, zvyš `hysteresis` na 0.002–0.003, případně `temporal_window` na 20. Velké změny
(hrabání rukou) tím nezpomalíš, jen se pomaleji propíšou jemné úpravy.

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
