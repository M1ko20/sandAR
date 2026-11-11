"""Základní třída pro hry / výukové režimy.

Hra dostává jen dvě věci:

* ``self.host`` – služby pro kreslení (``RenderHost``): plátno, shadery, text,
  nastavení, poslední výškovou mapu,
* ``update(hm, dt)`` – novou výškovou mapu (metry nad základní rovinou).
  ``hm.hands`` je maska rukou nad pískem, ``hm.hand_points`` jejich středy v metrech.

Znovupoužitelné simulace (voda, …) patří do ``sandbox/sim/``, ne do konkrétní hry.
Text pro návštěvníky kresli přes ``self.host.painter.text(...)``, krátké hlášky
přes ``self.host.notify(...)``.

Kreslí se na *plátno pískoviště* = pohled shora na bednu, rohy 1..4 jsou rohy
plátna. O Kinectu ani o projektoru hra nic neví, takže stejná hra funguje se
simulací, nahrávkou i skutečným pískovištěm.

Minimální hra::

    from sandbox.games.base import Game

    class MojeHra(Game):
        id = "moje"
        title = "Moje hra"

        def start(self):
            self.prog = self.host.program('''
                void main() {
                    float h = height_at(v_uv);
                    f_color = vec4(elevation_color(h), 1.0);
                }''')

        def render(self):
            self.host.draw_fullscreen(self.prog)

Soubor stačí uložit do ``sandbox/games/`` – hra se načte automaticky.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ..render.host import RenderHost
    from ..terrain.heightmap import HeightMap


class Game:
    id: ClassVar[str] = ""
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""
    order: ClassVar[int] = 100  # pořadí v menu (klávesy 1..9, dál Tab)
    keys: ClassVar[list[tuple[str, str]]] = []  # nápověda: (klávesa, popis)
    category: ClassVar[str] = "vizualizace"  # vizualizace | simulace | soutěž | hra | zážitek
    intro: ClassVar[str] = ""  # jedna věta pro návštěvníky „jak hrát“ – promítne se po spuštění
    attract: ClassVar[bool] = True  # smí se ukazovat v attract mode (samo se střídá, když nikdo nehraje)

    def __init__(self, host: "RenderHost") -> None:
        self.host = host

    # --- životní cyklus ----------------------------------------------------------
    def start(self) -> None:
        """Vytvoření shaderů a textur. Volá se při výběru hry a po změně kalibrace."""

    def stop(self) -> None:
        """Uvolnění prostředků."""

    def update(self, hm: "HeightMap", dt: float) -> None:
        """Nová výšková mapa (volá se každý snímek, hm může být stejná jako minule)."""

    def render(self) -> None:
        """Kreslení na plátno (je už nastavené jako cíl)."""

    def reset(self) -> None:
        """Vrátit svět do výchozího stavu (attract mode po odchodu návštěvníků, klávesa Backspace)."""

    def key(self, name: str, shift: bool = False) -> bool:
        """Stisk klávesy. Vrať True, pokud ji hra zpracovala."""
        return False

    def status(self) -> list[str]:
        """Řádky pro operátorské okno."""
        return []
