"""Sdílení obrázků do mobilu přes místní síť (Wi-Fi) a QR kód.

Malý HTTP server (jen knihovna Pythonu) běží ve vlákně na pozadí a vydává
*pouze* soubory ze složky ``shares/`` – žádný výpis adresářů, žádné cesty ven.
Návštěvník naskenuje QR kód promítnutý na písek a otevře se mu stránka
s obrázkem a tlačítkem ke stažení. Telefon musí být ve stejné síti jako počítač.

Použití::

    from sandbox import share
    sid = share.save_share(rgb_obrazek, "Pohoří Zlatého Draka")
    url = share.url_for(sid)            # http://192.168.x.y:8765/<sid>
    qr = share.qr_image(url, 400)       # RGBA numpy pro texturu
"""

from __future__ import annotations

import html
import logging
import re
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from .config import ROOT

log = logging.getLogger(__name__)

SHARES_DIR = ROOT / "shares"
DEFAULT_PORT = 8765

# povolená jména souborů: krátké id + přípona (nic jiného se nevydá)
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.(png|jpg|html)$")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TYPES = {"png": "image/png", "jpg": "image/jpeg", "html": "text/html; charset=utf-8"}
_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # bez 0/o, 1/l/i – kdyby to někdo opisoval


def sanitize_name(name: str) -> str | None:
    """Jméno souboru ve shares/ nebo None, pokud není povolené (cesty, divné znaky…)."""
    name = unquote(name).strip("/")
    if "/" in name or "\\" in name or not _NAME_RE.match(name):
        return None
    return name


def new_id(length: int = 6) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def local_ip() -> str:
    """IP adresa počítače v místní síti (trik s UDP socketem – nic se neodesílá)."""
    for probe in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 1))
                ip = s.getsockname()[0]
            if ip and not ip.startswith("127.") and ip != "0.0.0.0":
                return ip
        except OSError:
            continue
    return "127.0.0.1"


# --- HTTP server ------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "SandboxShare/1.0"
    directory: Path = SHARES_DIR  # přepíše ShareServer

    def do_GET(self) -> None:  # noqa: N802 – jméno dané knihovnou
        self._serve(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(send_body=False)

    def _serve(self, send_body: bool) -> None:
        path = urlsplit(self.path).path
        if path in ("/", ""):
            body = _INDEX_HTML.encode("utf-8")
            return self._send(200, _TYPES["html"], body, send_body)
        name = sanitize_name(path)
        if name is None and _ID_RE.match(unquote(path).strip("/")):
            name = unquote(path).strip("/") + ".html"  # krátká adresa /<id> -> stránka
        if name is None:
            return self._send(404, "text/plain; charset=utf-8", b"Nenalezeno", send_body)
        base = self.directory.resolve()
        file = (base / name).resolve()
        if file.parent != base or not file.is_file():
            return self._send(404, "text/plain; charset=utf-8", "Nenalezeno".encode(), send_body)
        try:
            body = file.read_bytes()
        except OSError:
            return self._send(404, "text/plain; charset=utf-8", b"Nenalezeno", send_body)
        self._send(200, _TYPES[name.rsplit(".", 1)[1]], body, send_body,
                   download=name if self._wants_download() else None)

    def _wants_download(self) -> bool:
        return "download=1" in (urlsplit(self.path).query or "")

    def _send(self, code: int, ctype: str, body: bytes, send_body: bool, download: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        log.debug("share %s: " + fmt, self.client_address[0], *args)


_INDEX_HTML = """<!doctype html><html lang="cs"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pískoviště</title></head>
<body style="font-family:sans-serif;text-align:center;padding:2em">
<h1>AR pískoviště</h1><p>Naskenuj QR kód promítnutý na písku.</p></body></html>"""


class ShareServer:
    """HTTP server ve vlákně, který vydává jen soubory z jedné složky."""

    def __init__(self, directory: Path, port: int = DEFAULT_PORT, host: str = "0.0.0.0", tries: int = 20) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        handler = type("Handler", (_Handler,), {"directory": self.directory})
        last: OSError | None = None
        for p in range(port, port + tries):  # obsazený port -> zkusit další
            try:
                self.httpd = ThreadingHTTPServer((host, p), handler)
                break
            except OSError as exc:
                last = exc
        else:
            raise OSError(f"Žádný volný port od {port}: {last}")
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="share-http", daemon=True)
        self.thread.start()
        log.info("Sdílení běží na http://%s:%d/ (složka %s)", local_ip(), self.port, self.directory)

    def url_for(self, name: str) -> str:
        return f"http://{local_ip()}:{self.port}/{name}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


_server: ShareServer | None = None
_lock = threading.Lock()


def ensure_server() -> ShareServer | None:
    """Spustí server (jen jednou). Při chybě vrátí None – hra pak QR jen nezobrazí."""
    global _server
    with _lock:
        if _server is None:
            try:
                _server = ShareServer(SHARES_DIR)
            except OSError:
                log.exception("Server pro sdílení se nepodařilo spustit")
                return None
        return _server


def url_for(filename: str) -> str:
    """Adresa souboru (nebo id sdílení) pro telefon v téže síti."""
    srv = ensure_server()
    port = srv.port if srv else DEFAULT_PORT
    return f"http://{local_ip()}:{port}/{filename}"


# --- uložení sdílení ---------------------------------------------------------------
def save_share(rgb: np.ndarray, title: str, subtitle: str = "", directory: Path | None = None,
               share_id: str | None = None) -> str:
    """Uloží obrázek (RGB uint8) jako <id>.png + mobilní stránku <id>.html. Vrací id."""
    directory = Path(directory or SHARES_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    sid = share_id or new_id()
    if not _ID_RE.match(sid):
        raise ValueError(f"Neplatné id sdílení: {sid!r}")
    png = directory / f"{sid}.png"
    tmp = directory / f".{sid}.tmp.png"
    if not cv2.imwrite(str(tmp), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)):
        raise OSError(f"Nelze uložit {png}")
    tmp.replace(png)
    page = _PAGE_HTML.format(title=html.escape(title), subtitle=html.escape(subtitle), png=f"{sid}.png")
    tmp_html = directory / f".{sid}.tmp.html"
    tmp_html.write_text(page, encoding="utf-8")
    tmp_html.replace(directory / f"{sid}.html")
    prune(directory)
    return sid


def prune(directory: Path | None = None, keep: int = 500) -> None:
    """Smaže nejstarší sdílení, ať složka neroste donekonečna (attract mode fotí i bez lidí)."""
    directory = Path(directory or SHARES_DIR)
    pngs = sorted(directory.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in pngs[keep:]:
        for f in (old, old.with_suffix(".html")):
            try:
                f.unlink()
            except OSError:
                pass


_PAGE_HTML = """<!doctype html>
<html lang="cs"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body {{ margin:0; background:#16222e; color:#f4f1e6; font-family:system-ui,sans-serif; text-align:center; }}
main {{ max-width:900px; margin:auto; padding:16px; }}
h1 {{ font-size:1.5em; margin:.4em 0 .1em; }}
p {{ opacity:.85; margin:.2em 0 1em; }}
img {{ width:100%; height:auto; border-radius:8px; box-shadow:0 4px 18px #0008; }}
a.btn {{ display:inline-block; margin:18px 0; padding:14px 28px; background:#f2b632; color:#16222e;
        font-weight:bold; font-size:1.15em; border-radius:999px; text-decoration:none; }}
small {{ display:block; opacity:.6; margin-bottom:2em; }}
</style></head>
<body><main>
<h1>{title}</h1><p>{subtitle}</p>
<a href="{png}"><img src="{png}" alt="{title}"></a>
<a class="btn" href="{png}?download=1" download>Stáhnout obrázek</a>
<small>Na mobilu můžeš také podržet prst na obrázku a zvolit „Uložit“.</small>
</main></body></html>
"""


# --- QR kód --------------------------------------------------------------------------
def qr_modules(text: str) -> np.ndarray:
    """Matice QR kódu (uint8, 0 = tmavý modul, 255 = světlý) včetně tiché zóny."""
    params = cv2.QRCodeEncoder_Params()
    params.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_M  # snese trochu nerovný písek
    q = cv2.QRCodeEncoder.create(params).encode(text)
    if q is None or q.size == 0:
        raise ValueError("QR kód se nepodařilo vytvořit")
    q = np.where(q > 127, 255, 0).astype(np.uint8)
    # OpenCV přidává tichou zónu 4 moduly; pro jistotu ji zaručíme
    if q[0].min() == 0 or q[:, 0].min() == 0:
        q = np.pad(q, 4, constant_values=255)
    return q


def qr_image(text: str, size_px: int) -> np.ndarray:
    """QR kód jako RGBA (size_px x size_px): černé moduly na bílém pozadí s tichou zónou.

    Moduly jsou ostré (nejbližší soused) a celé – obrázek může být o pár pixelů
    menší než ``size_px``, okraj se doplní bílou.
    """
    q = qr_modules(text)
    n = q.shape[0]
    scale = max(1, size_px // n)
    big = cv2.resize(q, (n * scale, n * scale), interpolation=cv2.INTER_NEAREST)
    out = np.full((size_px, size_px), 255, np.uint8)
    off = (size_px - big.shape[0]) // 2
    if off >= 0:
        out[off:off + big.shape[0], off:off + big.shape[1]] = big
    else:  # požadovaná velikost menší než počet modulů – zmenšit (nečitelné, ale nespadne)
        out = cv2.resize(q, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    rgba = np.empty((size_px, size_px, 4), np.uint8)
    rgba[..., :3] = out[..., None]
    rgba[..., 3] = 255
    return rgba
