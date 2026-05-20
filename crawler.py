"""
Ultra-Fast Smart Crawler (Direct Tunnel + Burst Mode + WARP Auto-Rotation)
==========================================================================
يعمل الخادم الذكي في الخلفية ويتولى تبديل البروكسيات دون إزعاج الكراولر.
WARP يتم تدويره تلقائياً كل N طلب أو عند الحظر — بدون توقف مرئي.sou  
Install: pip install aiohttp aiofiles beautifulsoup4 lxml curl_cffi
"""
import asyncio, hashlib, json, logging, random, re, ssl, time, base64, threading
import subprocess, os, sys, platform
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urldefrag, parse_qs, urlencode

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup, Comment
import aiofiles

try:
    from curl_cffi.requests import AsyncSession as CffiSession
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

from urllib.parse import urlparse

FACULTY_MAPPING = {
    # Main university
    "www.univ-setif.dz": "Farhat_Abbas_University",

    # Faculties
    "ft.univ-setif.dz": "ftechnologie",
    "fsciences.univ-setif.dz": "fsciences",
    "fsnv.univ-setif.dz": "fsnv",
    "eco.univ-setif.dz": "feco",
    "fmedecine.univ-setif.dz": "fmed",

    # Institutes
    "iomp.univ-setif.dz": "iomp",
    "iast.univ-setif.dz": "iast",
    "istm.univ-setif.dz": "istm",
}


def get_output_folder(url: str) -> str:
    domain = urlparse(url).netloc.lower()

    # remove port if exists
    domain = domain.split(":")[0]

    # exact match
    if domain in FACULTY_MAPPING:
        return f"./university_farhat_abaas/{FACULTY_MAPPING[domain]}"

    # fallback
    safe_name = domain.replace(".", "_")
    return f"./university_farhat_abaas/{safe_name}"
# ═══════════════════════════════════════════════════════════
# 🔹 مدير WARP (تدوير IP عبر Cloudflare WARP)
# ═══════════════════════════════════════════════════════════
class WarpManager:
    """
    يتحكم في Cloudflare WARP CLI لتغيير IP تلقائياً.
    يدعم: Linux, macOS, Windows.
    يعمل في thread منفصل ولا يوقف الكراولر أثناء التدوير.
    """
    # كم طلب قبل التدوير الاستباقي (قبل الحظر)
    ROTATE_EVERY_REQUESTS = 80        # غيّر هذا حسب حدة الموقع المستهدف
    # كم ثانية بين التدوير الاستباقي كحد أقصى
    ROTATE_EVERY_SECONDS  = 120

    def __init__(self):
        self._lock          = threading.Lock()
        self._rotating      = False          # هل نحن في منتصف تدوير الآن؟
        self._last_rotate   = 0.0            # وقت آخر تدوير (monotonic)
        self._req_since_rot = 0              # عدد الطلبات منذ آخر تدوير
        self._rotate_count  = 0             # عداد كلي للتدويرات
        self._current_ip    = ""
        self._warp_cmd      = self._find_warp()
        self._available     = self._warp_cmd is not None

        if self._available:
            log.info(f"✅ WARP CLI: {self._warp_cmd} | تدوير كل {self.ROTATE_EVERY_REQUESTS} طلب أو {self.ROTATE_EVERY_SECONDS}ث")
            self._current_ip = self._get_public_ip()
            if self._current_ip:
                log.info(f"   IP الحالي: {self._current_ip}")
        else:
            log.warning("⚠ WARP CLI غير موجود — سيعمل الكراولر بدون تدوير IP")

    # ────────────────────────────────────────────────────
    # اكتشاف warp-cli تلقائياً حسب نظام التشغيل
    # ────────────────────────────────────────────────────
    @staticmethod
    def _find_warp() -> Optional[str]:
        candidates = ["warp-cli"]
        if platform.system() == "Windows":
            candidates = [
                r"C:\Program Files\Cloudflare\Cloudflare WARP\warp-cli.exe",
                "warp-cli.exe",
            ]
        elif platform.system() == "Darwin":
            candidates = [
                "/Applications/Cloudflare WARP.app/Contents/Resources/warp-cli",
                "/usr/local/bin/warp-cli",
                "warp-cli",
            ]
        for cmd in candidates:
            try:
                r = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
                continue
        return None

    # ────────────────────────────────────────────────────
    # جلب IP العام الحالي (بدون مكتبات خارجية)
    # ────────────────────────────────────────────────────
    @staticmethod
    def _get_public_ip() -> str:
        services = [
            "https://api.ipify.org",
            "https://icanhazip.com",
            "https://checkip.amazonaws.com",
        ]
        for svc in services:
            try:
                import urllib.request
                with urllib.request.urlopen(svc, timeout=6) as r:
                    ip = r.read().decode().strip()
                    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
                        return ip
            except Exception:
                continue
        return ""

    # ────────────────────────────────────────────────────
    # التدوير الفعلي (يُشغَّل في thread منفصل)
    # ────────────────────────────────────────────────────
    def _do_rotate(self, reason: str = ""):
        """ينفّذ disconnect → connect ويتحقق من تغيير IP."""
        with self._lock:
            if self._rotating:
                return   # تجنب التدوير المتوازي
            self._rotating = True

        try:
            old_ip = self._current_ip
            reason_tag = f"[{reason}] " if reason else ""
            log.info(f"🔄 WARP تدوير #{self._rotate_count + 1} {reason_tag}— قطع الاتصال...")

            # ① قطع الاتصال
            subprocess.run([self._warp_cmd, "disconnect"],
                           capture_output=True, timeout=10)
            time.sleep(2)

            # ② إعادة الاتصال
            subprocess.run([self._warp_cmd, "connect"],
                           capture_output=True, timeout=10)

            # ③ انتظار حتى تنتهي عملية الاتصال (حد أقصى 20 ثانية)
            connected = False
            for attempt in range(20):
                time.sleep(1)
                try:
                    r = subprocess.run([self._warp_cmd, "status"],
                                       capture_output=True, timeout=5)
                    out = r.stdout.decode(errors="replace").lower()
                    if "connected" in out and "disconnected" not in out:
                        connected = True
                        break
                except Exception:
                    pass

            if not connected:
                log.warning("⚠ WARP: لم يتصل في 20 ثانية، إعادة المحاولة...")
                subprocess.run([self._warp_cmd, "connect"],
                               capture_output=True, timeout=10)
                time.sleep(5)

            # ④ تحقق من تغيير IP
            new_ip = self._get_public_ip()
            self._current_ip = new_ip
            self._rotate_count += 1
            self._last_rotate = time.monotonic()
            self._req_since_rot = 0

            if new_ip and new_ip != old_ip:
                log.info(f"✅ WARP: IP تغيّر  {old_ip} → {new_ip}")
            elif new_ip:
                log.warning(f"⚠ WARP: IP لم يتغير ({new_ip}) — WARP قد لا يدعم تغيير IP على هذه الشبكة")
            else:
                log.warning("⚠ WARP: لم يمكن التحقق من IP الجديد")

        except Exception as exc:
            log.error(f"❌ WARP خطأ أثناء التدوير: {exc}")
        finally:
            with self._lock:
                self._rotating = False

    # ────────────────────────────────────────────────────
    # الواجهة العامة: يستدعيها الكراولر
    # ────────────────────────────────────────────────────
    def notify_request(self):
        """يُستدعى بعد كل طلب ناجح — يقرر هل آن أوان التدوير."""
        if not self._available:
            return
        with self._lock:
            self._req_since_rot += 1
            should = (
                self._req_since_rot >= self.ROTATE_EVERY_REQUESTS
                or (time.monotonic() - self._last_rotate) >= self.ROTATE_EVERY_SECONDS
            )
            rotating = self._rotating
        if should and not rotating:
            threading.Thread(
                target=self._do_rotate,
                args=("استباقي",),
                daemon=True,
            ).start()

    def rotate_now(self, reason: str = "حظر") -> bool:
        """يُستدعى فوراً عند اكتشاف حظر — ينتظر انتهاء التدوير."""
        if not self._available:
            return False
        with self._lock:
            already = self._rotating
        if already:
            # انتظر انتهاء التدوير الجاري
            for _ in range(30):
                time.sleep(1)
                with self._lock:
                    if not self._rotating:
                        break
            return True
        # شغّل التدوير وانتظر في نفس الـ thread
        t = threading.Thread(target=self._do_rotate, args=(reason,), daemon=True)
        t.start()
        t.join(timeout=40)
        return True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_rotating(self) -> bool:
        with self._lock:
            return self._rotating


# ═══════════════════════════════════════════════════════════
# 🔹 الخادم الذكي (يعمل في Thread منفصل)
# ═══════════════════════════════════════════════════════════
class SmartLocalProxy:
    def __init__(self, proxy_file="proxies.txt", port=8080):
        self.proxy_file = proxy_file
        self.port = port
        self.proxies = []
        self.banned = set()
        self.current_index = 0
        self._load_proxies()

    def _load_proxies(self):
        if not Path(self.proxy_file).exists(): return
        with open(self.proxy_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if not line.startswith(('http://', 'https://')):
                        parts = line.split(':')
                        if len(parts) == 4:
                            line = f'http://{parts[2]}:{parts[3]}@{parts[0]}:{parts[1]}'
                        else:
                            line = f'http://{line}'
                    self.proxies.append(line)
        random.shuffle(self.proxies)

    def get_next_proxy(self):
        if not self.proxies: return None
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            if proxy not in self.banned: return proxy
            attempts += 1
        self.banned.clear()
        return self.proxies[self.current_index]

    async def handle_client(self, client_reader, client_writer):
        try:
            request_line = await client_reader.readuntil(b"\r\n")
            if not request_line.startswith(b"CONNECT"): return
            target_host = request_line.decode().split()[1]
            while True:
                line = await client_reader.readuntil(b"\r\n")
                if line == b"\r\n": break
            if not self.proxies:
                client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                await client_writer.drain()
                return
            up_reader, up_writer = None, None
            for _ in range(len(self.proxies)):
                upstream_proxy = self.get_next_proxy()
                if not upstream_proxy: break
                p = urlparse(upstream_proxy)
                try:
                    up_reader, up_writer = await asyncio.wait_for(
                        asyncio.open_connection(p.hostname, p.port or 8080), timeout=5.0)
                    if p.username and p.password:
                        auth = base64.b64encode(f"{p.username}:{p.password}".encode()).decode()
                        req = f"CONNECT {target_host} HTTP/1.1\r\nProxy-Authorization: Basic {auth}\r\n\r\n"
                    else:
                        req = f"CONNECT {target_host} HTTP/1.1\r\n\r\n"
                    up_writer.write(req.encode()); await up_writer.drain()
                    resp = await asyncio.wait_for(up_reader.readuntil(b"\r\n\r\n"), timeout=5.0)
                    if b"200" in resp.split(b"\r\n")[0]: break
                    else:
                        self.banned.add(upstream_proxy)
                        try: up_writer.close()
                        except: pass
                        up_reader, up_writer = None, None
                except:
                    self.banned.add(upstream_proxy)
                    try:
                        if up_writer: up_writer.close()
                    except: pass
                    up_reader, up_writer = None, None
            if not up_reader or not up_writer:
                client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await client_writer.drain()
                return
            client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await client_writer.drain()
            async def pipe(reader, writer):
                try:
                    while True:
                        data = await reader.read(131072)
                        if not data: break
                        writer.write(data); await writer.drain()
                except: pass
                finally:
                    try: writer.close()
                    except: pass
            await asyncio.gather(pipe(client_reader, up_writer), pipe(up_reader, client_writer))
        except: pass
        finally:
            try:
                if not client_writer.is_closing(): client_writer.close()
            except: pass

    async def _start_async(self):
        server = await asyncio.start_server(self.handle_client, '127.0.0.1', self.port)
        async with server:
            await server.serve_forever()

    def start_in_background(self):
        if not self.proxies:
            print("❌ لم يتم العثور على بروكسيات في proxies.txt")
            return False
        print(f"🚀 Smart Local Proxy تم تشغيله في الخلفية على المنفذ {self.port} ({len(self.proxies)} بروكسي)")
        threading.Thread(target=asyncio.run, args=(self._start_async(),), daemon=True).start()
        time.sleep(1)
        return True


# ═══════════════════════════════════════════════════════════
# 🔹 باقي كود الكراولر
# ═══════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

TLS_IMP = ["chrome124","chrome123","chrome120","chrome119","chrome116","chrome110","edge101","safari17_0"]
_UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
_AL = ["fr-FR,fr;q=0.9,ar;q=0.8,en-US;q=0.7","en-US,en;q=0.9","ar-SA,ar;q=0.9,en-US;q=0.8"]
_SC = ['"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"','"Chromium";v="121", "Not(A:Brand";v="24", "Google Chrome";v="121"']

PROFILES = {
    "turbo":  {"workers": 20, "burst": 30, "burst_pause": 1.0, "max_retries": 1, "backoff": 2.0, "desc": "EXTREME: No delay, 30 burst, 1s pause"},
    "fast":   {"workers": 12, "burst": 15, "burst_pause": 2.0, "max_retries": 2, "backoff": 3.0, "desc": "Fast: 15 burst, 2s pause"},
    "normal": {"workers": 5,  "burst": 8,  "burst_pause": 4.0, "max_retries": 3, "backoff": 8.0, "desc": "Normal: 8 burst, 4s pause"},
    "strict": {"workers": 2,  "burst": 3,  "burst_pause": 8.0, "max_retries": 4, "backoff": 15.0,"desc": "Strict: 3 burst, 8s pause"},
}

_RETRY_CODES = {429, 502, 503, 504}
_SKIP_EXT = (".jpg",".jpeg",".png",".gif",".svg",".ico",".webp",".bmp",".css",".js",".woff",".woff2",".ttf",".eot",".map",".pdf",".doc",".docx",".xls",".xlsx",".ppt",".pptx",".odt",".ods",".odp",".rtf",".txt",".csv",".zip",".rar",".7z",".tar",".gz",".bz2",".mp3",".mp4",".avi",".mov",".wmv",".mkv",".wav")
_SKIP_URL_RE = re.compile(r"(\?|&)(print=1|tmpl=component|format=feed|format=pdf|task=rss|view=feed|type=rss|output=pdf|mobile=1|amp=1|iccaldate=)", re.I)
_DOC_EXT = {".pdf":"pdfs",".doc":"docs",".docx":"docs",".xls":"docs",".xlsx":"docs",".ppt":"docs",".pptx":"docs",".odt":"docs",".ods":"docs",".odp":"docs",".rtf":"docs",".csv":"docs",".zip":"docs",".rar":"docs",".7z":"docs"}
_IMAGE_EXT = {".jpg",".jpeg",".png",".gif",".webp",".svg",".bmp"}
_ERR_HTML = re.compile(r'<html[^>]*class=["\'][^"\']*error[-_]page', re.I)
_ERR_TITLE = re.compile(r'خطأ\s*[:\s]?\d*|error\s*[:\s]?\d*|page\s+not\s+found|access\s+denied|bad\s+request|server\s+error|not\s+found|forbidden', re.I)
_ERR_BODY = re.compile(r'class=["\']error[-_]code["\']|class=["\']error[-_]message["\']|<h1[^>]*class=["\']error', re.I)
_ONCLICK = re.compile(r"(?:window\.location(?:\.href)?|document\.location(?:\.href)?|location\.(?:href|assign|replace)|window\.open)\s*[=(]\s*['\"]([^'\"]+)['\"]", re.I)
_CHROME_IDS = {"sp-top-bar","sp-top1","sp-top2","sp-header","sp-logo","sp-menu","sp-footer","sp-footer1","sp-bottom","sp-bottom1","sp-bottom2","sp-bottom3","sp-page-title","sp-left","sp-right","sp-position2","masthead","colophon","site-header","site-footer","secondary","sidebar","widget-area","wpadminbar"}
_CHROME_CLS = re.compile(r"\b(navbar[-_]toggle|icon[-_]bar|sr[-_]only|scroll[-_]up|sticky[-_]header|camera_wrap|slideshow|mod-slideshowck|sliderck|mod-finder|cookie[-_]banner|gdpr[-_]|popup[-_]modal)\b", re.I)
_SKIP_LINE = re.compile(r"^(toggle|navigation|menu|skip|log\s*in|sign\s*in|sign\s*out|عرض\s*المزيد|more\s*\.\.\.|read\s*more|\.\.\.|cookie|accept)$", re.I)
_META_CLS = re.compile(r"\b(card[-_]info|card[-_]grade|card[-_]email|card[-_]depart|card[-_]bureau|card[-_]title|card[-_]phone|card[-_]office|profile[-_]info|author[-_]info|user[-_]details|person[-_]details|contact[-_]info)\b", re.I)
_HIDDEN_CLS = re.compile(r"\b(sr-only|visually-hidden|d-none|hidden|invisible)\b", re.I)
_STRIP_PARAMS = frozenset({"utm_source","utm_medium","utm_campaign","utm_term","utm_content","fbclid","gclid","yclid","msclkid","dclid","li_fat_id","ttclid","twclid","mc_cid","mc_eid","vero_id","wickedid","_hsenc","_hsmi","hsCtaTracking","PHPSESSID","jsessionid","sid","sessionid","url","redirect","redirect_url","redirect_uri","redirecturl","redirect_to","return","returnurl","returnto","return_path","returnTo","return_url","destination","next","back","goto","continue","forward","Itemid","layout","ref","source","_ga","amp"})
_STATIC_RE = re.compile(r'(/about|/contact|/faq|/help|/terms|/privacy|/legal|/qui-sommes|/a-propos|/presentation|/organisation|/structure|/historique|/histoire|/organigramme|/page_professionnelles|/equipe|/staff|/enseignant|/administration|/formation|/departement|/laboratoire|/calendrier|/polycope|/programme|/module)', re.I)
_INDEX_END_RE = re.compile(r'(/news|/actualit\w*|/articles?|/blog|/posts?|/evenement\w*|/event\w*|/annonces?|/publication\w*|/activit\w*|/accueil|/home|/page/\d+|/search|/result|/categorie\w*|/category|/rubrique\w*|/archives?)/?$', re.I)
_INDEX_EXCEPT_RE = re.compile(r'/annonces?/\d+|/articles?/\d+|/news/\d+|/blog/\d+|/posts?/\d+|/evenement\w*/\d+|/event\w*/\d+|/publication\w*/\d+|/categorie\w*/\d+|/category/\d+|/rubrique\w*/\d+', re.I)

def _content_hash(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()[:16]
def _classify(url: str) -> str:
    p = urlparse(url); path = p.path
    if _STATIC_RE.search(path): return "static"
    if _INDEX_EXCEPT_RE.search(path): return "content"
    if _INDEX_END_RE.search(path.rstrip("/") or "/"): return "index"
    if p.query:
        qs = parse_qs(p.query)
        if any(k in qs for k in ("page","start","limit","offset","p")): return "index"
    if path.rstrip("/") in ("/",""): return "index"
    if re.search(r'/index\.php', path, re.I): return "index"
    return "content"
def _ua(): return random.choice(_UA)
def _al(): return random.choice(_AL)
def _sc(): return random.choice(_SC)
def _norm(u, force_scheme: str = ""):
    if not u: return ""
    u, _ = urldefrag(u); p = urlparse(u)
    if force_scheme and p.scheme and p.scheme != force_scheme: p = p._replace(scheme=force_scheme)
    path = p.path.rstrip("/") or "/"; q = ""
    if p.query:
        qs = parse_qs(p.query, keep_blank_values=True)
        for k in list(qs.keys()):
            if k in _STRIP_PARAMS: qs.pop(k, None)
        if qs: q = urlencode(sorted(qs.items()), doseq=True)
    return p._replace(path=path, query=q, fragment="").geturl()
def _pgh(ref=""):
    ua=_ua(); h={"User-Agent":ua,"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8","Accept-Language":_al(),"Accept-Encoding":"gzip, deflate, br","Connection":"keep-alive","Upgrade-Insecure-Requests":"1","Sec-Fetch-Dest":"document","Sec-Fetch-Mode":"navigate","Sec-Fetch-Site":"none","Sec-Fetch-User":"?1"}
    if "Chrome" in ua: h["Sec-Ch-Ua"]=_sc(); h["Sec-Ch-Ua-Mobile"]="?0"; h["Sec-Ch-Ua-Platform"]='"Windows"'
    if ref: h["Referer"]=ref; h["Sec-Fetch-Site"]="same-origin"
    return h
def _fgh(ref=""): return {"User-Agent":_ua(),"Accept":"application/pdf,application/octet-stream,*/*","Accept-Language":_al(),**({"Referer":ref} if ref else {})}
def _is_err(h):
    if len(h)<400: return True
    if _ERR_HTML.search(h): return True
    m=re.search(r'<title[^>]*>(.*?)</title>',h,re.I|re.S)
    if m and _ERR_TITLE.search(m.group(1).strip()): return True
    if _ERR_BODY.search(h): return True
    m=re.search(r'<body[^>]*>(.*?)</body>',h,re.I|re.S|re.DOTALL)
    if m:
        txt=re.sub(r'<[^>]+>',' ',m.group(1)).strip(); txt=re.sub(r'\s+',' ',txt).strip()
        vals=re.findall(r'<(?:input|button|option|textarea)[^>]+value=["\']([^"\']{2,})["\']',m.group(1),re.I)
        alts=re.findall(r'<img[^>]+alt=["\']([^"\']{2,})["\']',m.group(1),re.I)
        if len(txt+' '.join(vals)+' '.join(alts).strip())<30: return True
    return False
def _strip_chrome(s):
    for t in ("nav","footer","header"):
        for el in s.find_all(t): el.decompose()
    for eid in _CHROME_IDS:
        el=s.find(id=eid)
        if el: el.decompose()
    for el in s.find_all(True,class_=_CHROME_CLS): el.decompose()
    for el in s.find_all(True,class_=re.compile(r"\boffcanvas\b",re.I)): el.decompose()
    for el in s.find_all(True,id=re.compile(r"^offcanvas",re.I)): el.decompose()
    for t in s(["script","style","noscript","svg","canvas"]): t.decompose()
    for c in s.find_all(string=lambda t: isinstance(t,Comment)): c.extract()
def _clean_content(s):
    for t in s(["script","style","noscript","svg","canvas"]): t.decompose()
    for c in s.find_all(string=lambda t: isinstance(t,Comment)): c.extract()
    for el in s.find_all(True,style=re.compile(r"display\s*:\s*none",re.I)): el.decompose()
    for el in s.find_all(attrs={"hidden":True}): el.decompose()
    for el in s.find_all(True,class_=_HIDDEN_CLS): el.decompose()
    for el in s.find_all(True,class_=re.compile(r"\b(offcanvas|sliderck|swiper|breadcrumb)\b",re.I)): el.decompose()
    for el in s.find_all(True,id=re.compile(r"^offcanvas",re.I)): el.decompose()
    for el in s.find_all(True,attrs={"aria-label":re.compile(r"breadcrumb",re.I)}): el.decompose()
def _content_area(s):
    for prop in ("articleBody","text"):
        ab=s.find(attrs={"itemprop":prop})
        if ab: return ab
    for cls in ["com-content-article","item-page","blog-item","article-content","article-details","article-body","post-content","entry-content","page-content","content-area","content-body","content-article","item-body","item-content","node-content","view-content","field-name-body","field-name-field-body","td-content","main-content-area","js-item-content","js-main-content","article-full","article-featured","interface2","hovercard","card-info","annonce"]:
        el=s.find(True,class_=re.compile(r"\b"+re.escape(cls)+r"\b",re.I))
        if el: return el
    for eid in ["sp-component","sp-main-body","main-content","content","primary","main","article","js-main","main-article","content-article","centercol","maincol","contentcol","yui-main","main-content-wrapper"]:
        el=s.find(id=eid)
        if el: return el
    for role in ("main","article"):
        el=s.find(attrs={"role":role})
        if el: return el
    for tag in ["main","article"]:
        el=s.find(tag)
        if el: return el
    best = None; best_len = 0
    for div in s.find_all(["div","section","article","main"]):
        txt=div.get_text(strip=True)
        if len(txt)<150: continue
        links=div.find_all("a")
        if links and sum(len(a.get_text(strip=True)) for a in links)>0.6*len(txt): continue
        if len(links)>10 and len(txt)<2000: continue
        if len(txt)>best_len: best = div; best_len = len(txt)
    return best if best else s.body or s
def _jsonld(s):
    parts=[]
    for sc in s.find_all("script",type="application/ld+json"):
        try: data=json.loads(sc.string or "")
        except: continue
        items=data.get("@graph",[data]) if isinstance(data,dict) else (data if isinstance(data,list) else [data])
        for item in items:
            if not isinstance(item,dict): continue
            for f in ("articleBody","description","text","headline"):
                v=item.get(f,"")
                if isinstance(v,str) and len(v)>20: parts.append(v.strip())
    return "\n".join(parts)
def _clean(raw):
    lines=[re.sub(r"[ \t]+"," ",l).strip() for l in raw.splitlines()]
    return re.sub(r"\n{3,}","\n\n","\n".join(l for l in lines if l and not _SKIP_LINE.match(l))).strip()
def _extract_meta(s,area):
    if not area: return ""
    parts=[]
    for el in s.find_all(True,class_=_META_CLS):
        p=el.parent; is_d=False
        while p:
            if p is area: is_d=True; break
            p=p.parent
        if not is_d:
            t=el.get_text(strip=True)
            if t and len(t)>1: parts.append(t)
    return "\n".join(parts)
def _extract_inputs(s):
    parts,bt=[],s.get_text(" ",strip=True)
    for inp in s.find_all(["input","button"]):
        if inp.get("type","text").lower() in ("button","submit","reset"):
            v=inp.get("value","").strip()
            if v and len(v)>2 and v not in bt: parts.append(v)
    return "\n".join(parts)
def extract_text(s):
    area=_content_area(s)
    sc=BeautifulSoup(str(area if area else s),"html.parser")
    (lambda: _clean_content(sc) if area else _strip_chrome(sc))()
    text=_clean(sc.get_text(separator="\n",strip=True))
    om=_extract_meta(s,area)
    if om: text=om+"\n\n"+text
    iv=_extract_inputs(s)
    if iv: text=(text+"\n\n"+iv).strip() if text else iv
    if len(text)<200:
        jld=_jsonld(s)
        if jld: text=_clean(jld+"\n"+text) if text else _clean(jld)
    if len(text)<150 and area:
        sc2=BeautifulSoup(str(s),"html.parser"); _strip_chrome(sc2)
        t2=_clean(sc2.get_text(separator="\n",strip=True))
        if len(t2)>len(text): text=t2
    return text,len(text)<150


class CrawlState:
    def __init__(self,d):
        self.f=d/"_crawl_state.json"; self.v=set(); self.q=[]; self.queued=set(); self.failed=[]
    def save(self):
        with open(self.f,"w",encoding="utf-8") as f: json.dump({"v":list(self.v),"q":self.q,"queued":list(self.queued),"failed":self.failed,"ts":datetime.now().isoformat()},f,ensure_ascii=False,indent=2)
    def load(self):
        if not self.f.exists(): return False
        try:
            with open(self.f,"r",encoding="utf-8") as f: d=json.load(f)
            self.v=set(d.get("v",[])); self.q=d.get("q",[]); self.queued=set(d.get("queued",[])); self.failed=d.get("failed",[])
            return bool(self.v or self.q)
        except: return False
    def clear(self):
        if self.f.exists(): self.f.unlink()

class Renderer:
    def __init__(self): self._pw=self._br=None; self._sem=asyncio.Semaphore(1)
    async def start(self):
        if HAS_PLAYWRIGHT:
            self._pw=await async_playwright().start()
            self._br=await self._pw.chromium.launch(headless=True,args=["--no-sandbox","--disable-dev-shm-usage"])
            log.info("Playwright ready")
    async def stop(self):
        if self._br: await self._br.close()
        if self._pw: await self._pw.stop()
    async def render(self,u):
        if not self._br: return None
        async with self._sem:
            ctx=await self._br.new_context(user_agent=_ua(),ignore_https_errors=True)
            pg=await ctx.new_page()
            try:
                await pg.goto(u,wait_until="domcontentloaded",timeout=30_000)
                await pg.wait_for_timeout(2_000); return await pg.content()
            except: return None
            finally: await ctx.close()


class Extractor:
    def __init__(self,url,out,workers=12,burst=15,burst_pause=2.0,max_pages=500,full=False,
                 use_pw=True,max_retries=5,backoff=0.5,resume=False,cycle_work=0,cycle_pause=0,
                 use_curl=True,proxy_url=None,warp: Optional[WarpManager]=None,
                 warp_rotate_every: int=80):
        self.url=_norm(url); self.out=Path(out); self.bd=urlparse(url).netloc
        self.workers=workers; self.burst=burst; self.burst_pause=burst_pause
        self.max_pages=max_pages; self.full=full; self.max_retries=max_retries; self.backoff=backoff
        self.cycle_work=cycle_work; self.cycle_pause=cycle_pause
        self.use_curl=use_curl and HAS_CURL_CFFI; self.px=proxy_url
        self.st=CrawlState(self.out)
        if resume and self.st.load(): log.info(f"Resuming: {len(self.st.v)} visited")
        else:
            if resume: self.st.clear()
            self.st.queued.add(self.url); self.st.q.append(self.url)
        self.dl_p=set(); self.dl_d=set(); self.dl_i=set()
        self.t0=datetime.now(); self._renderer=Renderer() if (use_pw and HAS_PLAYWRIGHT) else None
        self._lock=asyncio.Lock(); self._req_cnt=0
        self._blk=0; self._to=0; self._500=0; self._danger=False; self._danger_until=0.0
        self._cffi=None
        # ─── WARP ───────────────────────────────────────────
        self._warp: Optional[WarpManager] = warp
        self._warp_rotate_every = warp_rotate_every   # طلبات بين كل تدوير استباقي
        self._total_requests    = 0                   # عداد الطلبات الكلية
        # ────────────────────────────────────────────────────
        self.stats={"fetched":0,"err_page":0,"skipped_500":0,"enqueued":0,"dup":0,"ext":0,"failed":0,
                    "warp_rotations":0}
        self._dl_sem = asyncio.Semaphore(2)
        self._initial_update_meta = {}
        self.out.mkdir(parents=True,exist_ok=True)
        for s in ("pages","tables","pdfs","docs","images"): (self.out/s).mkdir(exist_ok=True)

    async def _gcffi(self):
        if not self._cffi: self._cffi=CffiSession(impersonate=random.choice(TLS_IMP))
        return self._cffi

    # ────────────────────────────────────────────────────────
    # 🔄 تدوير WARP (async wrapper — لا يوقف event loop)
    # ────────────────────────────────────────────────────────
    async def _warp_rotate_async(self, reason: str = "حظر"):
        """ينفذ تدوير WARP في executor لكي لا يجمّد الـ event loop."""
        if not self._warp or not self._warp.available:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._warp.rotate_now, reason)
        self.stats["warp_rotations"] += 1
        # أعد بناء الـ CFFI session بعد تغيير IP
        if self._cffi:
            try: await self._cffi.close()
            except: pass
            self._cffi = None

    async def _maybe_rotate_warp_proactive(self):
        """تدوير استباقي: يُستدعى بعد كل طلب ناجح."""
        if not self._warp or not self._warp.available:
            return
        self._total_requests += 1
        # نفّذ في background thread بدون انتظار (غير مؤثر على السرعة)
        if self._total_requests % self._warp_rotate_every == 0:
            log.info(f"⏱ تدوير WARP استباقي بعد {self._total_requests} طلب")
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, self._warp.notify_request)
            self.stats["warp_rotations"] += 1
            if self._cffi:
                try: await self._cffi.close()
                except: pass
                self._cffi = None

    async def _wait(self):
        async with self._lock:
            if self._danger:
                w=self._danger_until-time.monotonic()
                if w>0: await asyncio.sleep(w)
                self._danger=False; self._blk=0; self._to=0
            self._req_cnt += 1
            if self._req_cnt <= self.burst: await asyncio.sleep(0.03)
            else:
                self._req_cnt = 0
                wait = self.burst_pause
                if self._blk > 0: wait *= (1 + self._blk * 2)
                if self._to > 0: wait *= (1 + self._to * 1.5)
                await asyncio.sleep(wait)

    def _chk_danger(self):
        """يُفعَّل عند اكتشاف حظر — يحاول تغيير IP أولاً."""
        if self._blk >= 2 or self._to >= 3:
            if not self._danger:
                if self._warp and self._warp.available:
                    # لا نوقف الكراولر — ندير تدويراً طارئاً في thread منفصل
                    # ونمسح العدادات فوراً
                    log.warning(f"🚨 DANGER: blk={self._blk} to={self._to} — تدوير WARP طارئ...")
                    threading.Thread(
                        target=self._warp.rotate_now,
                        args=("طارئ",),
                        daemon=True
                    ).start()
                    self.stats["warp_rotations"] += 1
                    self._blk = 0
                    self._to  = 0
                    self._500 = 0
                else:
                    # لا WARP → وضع الانتظار التقليدي
                    self._danger = True
                    self._danger_until = time.monotonic() + 180
                    log.warning("⚠ DANGER: لا WARP متاح — انتظار 3 دقائق...")

    def _id(self,t): return hashlib.md5(t.encode()).hexdigest()[:8]
    def _safe(self,t,n=50):
        if not t: return "untitled"
        return re.sub(r'[<>:"/\\|?*]',"_",re.sub(r"\s+","_",t))[:n].strip("_")

    def _is_same_domain(self, url_netloc: str, base_netloc: str) -> bool:
        if not url_netloc: return True
        if not base_netloc: return False
        u = url_netloc.lower(); b = base_netloc.lower()
        for prefix in ("www.", "m.", "mobile.", "wap."):
            if u.startswith(prefix): u = u[len(prefix):]; break
        for prefix in ("www.", "m.", "mobile.", "wap."):
            if b.startswith(prefix): b = b[len(prefix):]; break
        if not u or not b: return False
        if u == b: return True
        if u.endswith("." + b) or b.endswith("." + u): return True
        return False

    def _crawlable(self, u):
        if not u: return False
        p = urlparse(u)
        if p.scheme not in ("http", "https"): return False
        if not self._is_same_domain(p.netloc, self.bd): return False
        if p.path.lower().endswith(_SKIP_EXT): return False
        return not _SKIP_URL_RE.search(u)

    def _enqueue(self,u):
        u=_norm(u)
        if not u or u in self.st.queued:
            if u in self.st.queued: self.stats["dup"]+=1
            return False
        if not self._crawlable(u): return False
        self.st.queued.add(u); self.st.q.append(u); self.stats["enqueued"]+=1
        return True

    def _title(self,s,u):
        for fn in [lambda s:s.find("meta",property="og:title"),lambda s:s.find("meta",attrs={"name":"twitter:title"})]:
            el=fn(s)
            if el and el.get("content","").strip(): return el["content"].strip()
        tt=s.find("title")
        if tt:
            t=re.sub(r"\s*[|\-–—]\s*[^|]+$","",tt.get_text().strip())
            if len(t)>3: return t
        h1=s.find("h1") or s.find(attrs={"itemprop":"headline"})
        if h1:
            t=h1.get_text().strip()
            if len(t)>3: return t
        path=urlparse(u).path
        if path and path!="/": return path.rstrip("/").split("/")[-1].replace("-"," ").replace("_"," ").title() or "Untitled"
        return "Untitled"

    async def _psleep(self, secs: float, prefix: str = ""):
        log.info(f"    {prefix}sleep {secs:.1f}s")
        await asyncio.sleep(secs)

    async def _fetch(self,sess,u):
        last=0
        for att in range(1,self.max_retries+1):
            await self._wait()
            h=_pgh(self.url)
            if random.random()<0.3:
                vl=list(self.st.v)[:10]
                if vl: h["Referer"]=random.choice(vl)
            try:
                if self.use_curl:
                    s=await self._gcffi()
                    kw={"headers":h,"allow_redirects":True,"timeout":15,"verify":False}
                    if self.px: kw["proxy"]=self.px
                    r=await s.get(u,**kw); last=r.status_code

                    if last==200 and r.text:
                        if _is_err(r.text): self.stats["err_page"]+=1; return None,None,0
                        self._blk=self._to=self._500=0
                        self.stats["fetched"]+=1
                        # ─── تدوير استباقي بعد كل طلب ناجح ───
                        await self._maybe_rotate_warp_proactive()
                        return r.text,_norm(u),200

                    if self.px and "127.0.0.1" in self.px:
                        if att < self.max_retries: await asyncio.sleep(0.2); continue
                    else:
                        if last in (429, 503):
                            self._blk+=1; self._chk_danger()
                            # إذا WARP متاح، انتظر قليلاً ثم أعد المحاولة (لا ساعات!)
                            if self._warp and self._warp.available:
                                wait_t = min(self.backoff * att + random.uniform(1, 3), 15)
                                await self._psleep(wait_t, f"HTTP {last}: ")
                            else:
                                await self._psleep(self.backoff*att+random.uniform(3,10),f"HTTP {last}: ")
                            continue
                        if last==500:
                            self._500+=1
                            if self._500>=3: self.stats["skipped_500"]+=1; return None,None,500
                            await asyncio.sleep(self.backoff*att); continue
                        if last in _RETRY_CODES and att<self.max_retries:
                            await self._psleep(self.backoff*att); continue
                    return None,None,last
            except Exception as e:
                if self.px and "127.0.0.1" in self.px:
                    if att < self.max_retries: await asyncio.sleep(0.2)
                else:
                    err_str = str(e).lower()
                    if "timeout" in err_str:
                        self._blk+=1; self._to+=1; self._chk_danger()
                        if att<self.max_retries:
                            wait_t = self.backoff*att if (self._warp and self._warp.available) else self.backoff*att
                            await self._psleep(wait_t,"TO: ")
                    else:
                        self._blk+=1; self._chk_danger()
                        if att<self.max_retries: await asyncio.sleep(self.backoff*att)

        self.stats["failed"]+=1
        self.st.failed.append({"url":u,"reason":"max_retries","status":last,"ts":datetime.now().isoformat()})
        return None,None,last

    # ═══════════════════════════════════════════════════════════
    # 🔹 FIXED: استخراج روابط iframe و embed و object أيضاً
    # ═══════════════════════════════════════════════════════════
    def _extract_links(self, s, pu):
        sc = BeautifulSoup(str(s), "html.parser")
        for el in sc.find_all(True, id=re.compile(r"^offcanvas", re.I)): el.decompose()
        for el in sc.find_all(True, class_=re.compile(r"\boffcanvas-menu\b", re.I)): el.decompose()
        for el in sc.find_all(True, id=re.compile(r"^navbar\d+$", re.I)):
            if "collapse" in " ".join(el.get("class", [])): el.decompose()
        crawl: set = set(); all_lk: list = []; seen: set = set(); bd = urlparse(pu).netloc
        def reg(href, label=""):
            href = href.strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "data:", "ftp:", "file:")): return
            if href.startswith("#"): return
            try: full = _norm(urljoin(pu, href).split("#")[0])
            except: return
            if not full or full in seen: return
            p = urlparse(full)
            if p.scheme and p.scheme not in ("http", "https"): return
            seen.add(full)
            if not label: label = full.split("/")[-1].split("?")[0] or "Link"
            label = re.sub(r"\s+", " ", label).strip()[:200]
            int_ = self._is_same_domain(p.netloc, bd)
            if int_ and self._crawlable(full): crawl.add(full)
            elif not int_: self.stats["ext"] += 1
            all_lk.append({"url": full, "text": label, "is_internal": int_})
        for a in sc.find_all("a", href=True):
            lbl = a.get_text(" ", strip=True)
            for c in a.find_all(["input", "button"]):
                v = c.get("value", "").strip()
                if v: lbl = v; break
            reg(a["href"], lbl)
        for form in sc.find_all("form", action=True):
            label = ""
            for c in form.find_all(["input", "button"]):
                if c.get("type", "submit").lower() in ("submit", "button"):
                    v = c.get("value", "").strip() or c.get_text(strip=True)
                    if v: label = v; break
            reg(form["action"], label or "Form")
        for area in sc.find_all("area", href=True): reg(area["href"], area.get("alt", "").strip() or "Area")
        # ─── NEW: iframe / embed / object ─────────────────────
        for iframe in sc.find_all("iframe", src=True):
            reg(iframe["src"], iframe.get("title", "").strip() or "IFrame")
        for embed in sc.find_all("embed", src=True):
            reg(embed["src"], embed.get("title", "").strip() or "Embed")
        for obj in sc.find_all("object", data=True):
            reg(obj["data"], obj.get("title", "").strip() or "Object")
        # ───────────────────────────────────────────────────────
        for el in sc.find_all(True):
            for at in ("data-href", "data-url", "data-link", "data-target-url", "data-redirect", "data-action", "data-route", "data-permalink"):
                v = el.get(at, "").strip()
                if v and v not in ("#", "javascript:void(0)"): reg(v, el.get_text(" ", strip=True)[:200])
        for el in sc.find_all(True, onclick=True):
            m = _ONCLICK.search(el.get("onclick", ""))
            if m: reg(m.group(1), (el.get_text(" ", strip=True) or el.get("value", "") or el.get("title", ""))[:200])
        head = sc.find("head")
        if head:
            for link in head.find_all("link", href=True):
                rel = " ".join(link.get("rel", [])).lower()
                if any(r in rel for r in ("alternate", "next", "prev", "canonical", "shortlink", "amphtml")): reg(link["href"], rel)
            for meta in head.find_all("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)}):
                content = meta.get("content", "")
                m = re.search(r"url\s*=\s*['\"]?([^'\"\s;]+)", content, re.I)
                if m: reg(m.group(1), "Meta Refresh")
        cont_lk: list = []; area = _content_area(sc); sc2: set = set()
        if area:
            for a in area.find_all("a", href=True):
                f = _norm(urljoin(pu, a["href"]).split("#")[0]) if a.get("href") else ""
                if f and f not in sc2:
                    sc2.add(f); p = urlparse(f)
                    cont_lk.append({"url": f, "text": a.get_text(" ", strip=True)[:200] or f.split("/")[-1], "is_internal": self._is_same_domain(p.netloc, bd)})
        sa: set = set(); dedup = []
        for l in all_lk:
            if l["url"] not in sa: sa.add(l["url"]); dedup.append(l)
        return crawl, dedup, cont_lk

    def _tables(self,s,pid,bu):
        tbls=[]
        for i,tbl in enumerate(s.find_all("table"),1):
            try:
                cap=tbl.find("caption"); title=cap.get_text().strip() if cap else f"Table_{i}"
                if title==f"Table_{i}":
                    pr=tbl.find_previous(["h1","h2","h3","h4","h5","h6"])
                    if pr: title=pr.get_text().strip()
                hdrs=[]; th=tbl.find("thead")
                if th and th.find("tr"): hdrs=[c.get_text().strip() for c in th.find("tr").find_all(["th","td"])]
                if not hdrs:
                    fr=tbl.find("tr")
                    if fr: hdrs=[c.get_text().strip() for c in fr.find_all("th")] or [c.get_text().strip() for c in fr.find_all("td")]
                rows=[]
                for tr in (tbl.find("tbody") or tbl).find_all("tr"):
                    if tr is (th.find("tr") if th else None): continue
                    cells=[c.get_text().strip() for c in tr.find_all(["td","th"])]
                    if any(cells): rows.append(cells)
                if rows:
                    fn=f"{pid}_table_{i}_{self._safe(title,30)}.json"
                    tbls.append({"title":title[:100],"headers":hdrs,"row_count":len(rows),"filename":fn,"data":{"source_url":bu,"title":title,"headers":hdrs,"rows":rows,"row_count":len(rows),"col_count":len(hdrs) if hdrs else (len(rows[0]) if rows else 0)}})
            except: pass
        return tbls

    # ═══════════════════════════════════════════════════════════
    # 🔹 FIXED: استخراج الملفات من iframe و embed و object
    # ═══════════════════════════════════════════════════════════
    def _doc_links(self,s,bu):
        seen=set(); docs=[]
        # ─── الروابط العادية ─────────────────────────────────
        for a in s.find_all("a",href=True):
            f=urljoin(bu,a["href"].strip()).split("#")[0].split("?")[0]
            for ext,sub in _DOC_EXT.items():
                if f.lower().endswith(ext) and f not in seen:
                    seen.add(f); t=a.get_text().strip() or Path(urlparse(f).path).stem.replace("_"," ").replace("-"," ")
                    docs.append({"title":t[:150],"url":f,"type":ext.lstrip("."),"sub":sub}); break
        # ─── NEW: iframe / embed / object ─────────────────────
        for tag_name, attr in [("iframe","src"), ("embed","src"), ("object","data")]:
            for tag in s.find_all(tag_name, **{attr: True}):
                f = urljoin(bu, tag[attr].strip()).split("#")[0].split("?")[0]
                for ext, sub in _DOC_EXT.items():
                    if f.lower().endswith(ext) and f not in seen:
                        seen.add(f)
                        t = Path(urlparse(f).path).stem.replace("_", " ").replace("-", " ")
                        docs.append({"title": t[:150], "url": f, "type": ext.lstrip("."), "sub": sub})
                        break
        # ───────────────────────────────────────────────────────
        return docs

    def _img_links(self,s,bu):
        seen=set(); imgs=[]
        for img in s.find_all("img",src=True):
            f=urljoin(bu,img["src"].strip()).split("#")[0]
            if f in seen: continue
            seen.add(f); p=urlparse(f).path; ext=Path(p).suffix.lower()
            if ext not in _IMAGE_EXT: ext=".jpg"
            alt=img.get("alt","").strip(); desc=img.get("title","").strip() or alt or Path(p).stem.replace("_"," ").replace("-"," ")
            imgs.append({"desc":desc[:150],"url":f,"alt":alt[:100],"ext":ext})
        # 🔹 إضافة دعم الصور المخفية الموجودة في كود التحديث
        for el in s.find_all(True):
            for at in ("data-src","data-lazy-src","data-original"):
                v=el.get(at,"").strip()
                if not v: continue
                f=urljoin(bu,v).split("#")[0]
                if f in seen: continue
                seen.add(f); p=urlparse(f).path; ext=Path(p).suffix.lower()
                if ext not in _IMAGE_EXT: ext=".jpg"
                alt=el.get("alt","").strip() or ""; desc=el.get("title","").strip() or alt or Path(p).stem.replace("_"," ").replace("-"," ")
                imgs.append({"desc":desc[:150],"url":f,"alt":alt[:100],"ext":ext})
        return imgs
    
    async def _dl(self,sess,u,dest,ref):
        if dest.exists(): return True
        for att in range(1,self.max_retries+1):
            try:
                async with sess.get(u,headers=_fgh(ref),ssl=False,timeout=ClientTimeout(total=120)) as r:
                    if r.status==200:
                        d=await r.read()
                        if len(d)<100: return False
                        async with aiofiles.open(dest,"wb") as f: await f.write(d)
                        return True
                    if r.status in _RETRY_CODES and att<self.max_retries: await asyncio.sleep(self.backoff*att); continue
                    return False
            except: await asyncio.sleep(2*att)
        return False

    async def _load_sitemap(self,sess):
        p=urlparse(self.url); su=f"{p.scheme}://{p.netloc}/sitemap.xml"; loaded=0
        async def _parse(xml,src):
            c=0
            try:
                bs=BeautifulSoup(xml,"lxml-xml")
                for loc in bs.find_all("loc"):
                    href=loc.get_text().strip()
                    if not href: continue
                    if href.endswith(".xml") and href!=src:
                        try:
                            async with sess.get(href,headers=_pgh(),ssl=False,timeout=ClientTimeout(total=15)) as r:
                                if r.status==200: c+=await _parse(await r.text(errors="replace"),href)
                        except: pass
                    elif self._enqueue(href): c+=1
            except: pass
            return c
        try:
            await self._wait()
            async with sess.get(su,headers=_pgh(),ssl=False,timeout=ClientTimeout(total=15)) as r:
                if r.status==200: loaded=await _parse(await r.text(errors="replace"),su)
        except: pass
        return loaded

    async def _process(self,sess,u,idx):
        nu=_norm(u)
        if nu in self.st.v: return set()
        if self.st.v and len(self.st.v)>5:
            pid_c=self._id(nu)
            if list((self.out/"pages").glob(f"{pid_c}_*.json")):
                self.st.v.add(nu); return set()
        try:
            log.info(f"[{idx:03d}] {u[:80]}")
            html,fu,st=await self._fetch(sess,u)
            self.st.v.add(nu)
            if fu and fu!=nu: self.st.v.add(fu)
            if not html: return set()
            soup=BeautifulSoup(html,"html.parser")
            crawl,all_lk,cont_lk=self._extract_links(soup,fu or u)
            if st!=200:
                if crawl: log.debug(f"    HTTP {st} — {len(crawl)} links")
                return crawl
            pid=self._id(fu); title=self._title(soup,fu); text,thin=extract_text(soup)
            if thin and self._renderer:
                log.info(f"    Thin ({len(text)}c) — Playwright")
                pw=await self._renderer.render(fu)
                if pw and not _is_err(pw):
                    try:
                        ps=BeautifulSoup(pw,"html.parser"); pt,_=extract_text(ps)
                        if len(pt)>len(text):
                            log.info(f"    Playwright: {len(text)}→{len(pt)}c")
                            soup=ps; text=pt; crawl.update(self._extract_links(ps,fu)[0])
                    except: pass
            self._initial_update_meta[fu] = {"id": pid,"hash": _content_hash(text),"type": _classify(fu),"first_seen": datetime.now().isoformat(),"last_check": datetime.now().isoformat(),"last_change": "","etag": "","last_modified": "","check_count": 0,"change_count": 0}
            tbls=self._tables(soup,pid,fu); docs=self._doc_links(soup,fu); imgs=self._img_links(soup,fu)
            pdfs_out,docs_out,imgs_out=[],[],[]
            if self.full:
                async def _dld(d):
                    async with self._dl_sem:
                        gs=self.dl_p if d["sub"]=="pdfs" else self.dl_d
                        fn=f"{pid}_{self._safe(d['title'],40)}.{d['type']}"; dest=self.out/d["sub"]/fn
                    if d["url"] not in gs:
                            if await self._dl(sess,d["url"],dest,fu):
                                gs.add(d["url"]); log.info(f"      ↓ {d['sub']}/{fn}")
                    return {"title":d["title"],"url":d["url"],"local_file":f"{d['sub']}/{fn}" if dest.exists() else None}
                async def _dli(img):
                    async with self._dl_sem:
                        fn=f"{pid}_{self._safe(img['desc'],40)}{img['ext']}"; dest=self.out/"images"/fn
                        if img["url"] not in self.dl_i:
                            if await self._dl(sess,img["url"],dest,fu):
                                self.dl_i.add(img["url"]); log.info(f"      ↓ images/{fn}")
                    return {"url":img["url"],"local_file":f"images/{fn}" if dest.exists() else None}
                res=await asyncio.gather(*[_dld(d) for d in docs],*[_dli(i) for i in imgs],return_exceptions=True)
                nd=len(docs)
                for i,d in enumerate(docs):
                    e=res[i]; (pdfs_out if d["sub"]=="pdfs" else docs_out).append(e if not isinstance(e,Exception) else {"title":d["title"],"url":d["url"],"local_file":None})
                for i,img in enumerate(imgs):
                    e=res[nd+i] if nd+i<len(res) else None
                    imgs_out.append(e if e and not isinstance(e,Exception) else {"url":img["url"],"local_file":None})
            else:
                for d in docs: (pdfs_out if d["sub"]=="pdfs" else docs_out).append({"title":d["title"],"url":d["url"],"local_file":None})
                imgs_out=[{"url":i["url"],"local_file":None} for i in imgs]
            log.info(f"    OK '{title[:40]}' | {len(text):,}c | {len(crawl)} new | {len(tbls)} tbl | {len(pdfs_out)+len(docs_out)} files")
            for tbl in tbls:
                async with aiofiles.open(self.out/"tables"/tbl["filename"],"w",encoding="utf-8") as f:
                    await f.write(json.dumps(tbl["data"],ensure_ascii=False,indent=2))
            pd={"metadata": {"page": {"url": fu, "title": title, "id": pid, "type": _classify(fu)}, "update": {"fetched_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), "hash": _content_hash(text), "update_num": 1}},"content": {"text": text},"resources": {"links": [{"text":l["text"],"url":l["url"],"type":"internal" if l["is_internal"] else "external"} for l in cont_lk],"images": [{"url":i["url"],"alt":i.get("alt",""),"desc":i.get("desc",""), **({"local_file":i["local_file"]} if i.get("local_file") else {})} for i in imgs_out],"documents": [{"title":d["title"],"url":d["url"],**({"local_file":d["local_file"]} if d.get("local_file") else {})} for d in pdfs_out+docs_out],"tables": [{"file":f"tables/{t['filename']}","rows":t["row_count"]} for t in tbls]}}
            async with aiofiles.open(self.out/"pages"/f"{pid}_{self._safe(title)}.json","w",encoding="utf-8") as f:
                await f.write(json.dumps(pd,ensure_ascii=False,indent=2))
            return crawl
        except Exception as exc:
            log.error(f"    ERR {str(exc)[:80]}"); self.st.v.add(nu)
            self.st.failed.append({"url":u,"reason":str(exc)[:100],"status":0,"ts":datetime.now().isoformat()})
            return set()

    async def crawl(self):
        backend="curl_cffi" if self.use_curl else "aiohttp"
        log.info("="*80)
        log.info(f"ULTRA-FAST EXTRACTOR | Burst: {self.burst} | Pause: {self.burst_pause}s | Workers: {self.workers}")
        log.info(f"URL: {self.url} | Backend: {backend} | Full: {self.full}")
        if self._warp and self._warp.available:
            log.info(f"WARP: ✅ تدوير كل {self._warp_rotate_every} طلب + طارئ عند الحظر")
        if self.px: log.info(f"Smart Proxy: ENABLED (localhost)")
        else: log.info("Proxies: NONE (direct)")
        log.info("="*80)
        if self._renderer: await self._renderer.start()
        ssl_c=ssl.create_default_context(); ssl_c.check_hostname=False; ssl_c.verify_mode=ssl.CERT_NONE
        conn=TCPConnector(ssl=ssl_c,limit=200,limit_per_host=30,keepalive_timeout=30)
        try:
            async with aiohttp.ClientSession(connector=conn,timeout=ClientTimeout(total=60,connect=15)) as sess:
                sm=await self._load_sitemap(sess)
                if sm: log.info(f"Sitemap: +{sm} URLs")
                random.shuffle(self.st.q)
                idx=len(self.st.v); ce=(self.cycle_work>0 and self.cycle_pause>0); cs=time.monotonic(); pc=0
                while self.st.q and len(self.st.v)<self.max_pages:
                    if ce and time.monotonic()-cs>=self.cycle_work*60:
                        pc+=1; ps=self.cycle_pause*60*random.uniform(0.8,1.2)
                        log.info(f"⏸ PAUSE #{pc} ({ps/60:.1f}min)"); self.st.save()
                        await self._psleep(ps,f"PAUSE #{pc}: "); cs=time.monotonic()
                        self._blk=self._to=self._500=0
                    batch=[]
                    while self.st.q and len(batch)<self.workers:
                        u=self.st.q.pop(0)
                        if u not in self.st.v: batch.append(u)
                    if not batch: continue
                    results=await asyncio.gather(*[self._process(sess,u,idx+i+1) for i,u in enumerate(batch)])
                    for nu in results:
                        if nu:
                            for u in nu: self._enqueue(u)
                    idx+=len(batch)
                    if len(self.st.v)%20==0:
                        self.st.save(); el=(datetime.now()-self.t0).total_seconds()
                        h="✓"
                        if self._blk>0 or self._to>0: h=f"⚠ blk={self._blk} to={self._to}"
                        w_info = f" | WARP×{self.stats['warp_rotations']}" if (self._warp and self._warp.available) else ""
                        log.info(f"-- {len(self.st.v)} done | Q:{len(self.st.q)} | Enq:{self.stats['enqueued']} | {el:.0f}s | {h}{w_info}")
        finally:
            if self._renderer: await self._renderer.stop()
            if self._cffi: await self._cffi.close()
            self.st.save()
            if self.st.failed:
                with open(self.out/"_failed_urls.json","w",encoding="utf-8") as f:
                    json.dump(self.st.failed,f,ensure_ascii=False,indent=2)
            if self._initial_update_meta:
                uf = self.out/"_update_state.json"
                existing_history = []
                if uf.exists():
                    try:
                        with open(uf, encoding="utf-8") as f: existing_history = json.load(f).get("update_history", [])
                    except: pass
                with open(uf, "w", encoding="utf-8") as f:
                    json.dump({"page_meta": self._initial_update_meta,"index_links": {},"update_history": existing_history,"saved_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        el=(datetime.now()-self.t0).total_seconds()
        log.info("\n"+"="*80)
        log.info(f"DONE | {len(self.st.v)} pages | {el:.0f}s ({el/60:.1f}min)")
        log.info(f"Fetched: {self.stats['fetched']} | ErrPages: {self.stats['err_page']} | Failed: {len(self.st.failed)}")
        if self._warp and self._warp.available:
            log.info(f"WARP تدويرات: {self.stats['warp_rotations']}")
        log.info("="*80)

def main():
    p=ArgumentParser(description="Ultra-Fast Smart Crawler (WARP Auto-Rotation Edition)")
    p.add_argument("--url",required=True)
    p.add_argument("--out", default=None)    
    p.add_argument("--profile",choices=["turbo","fast","normal","strict"],default="fast")
    p.add_argument("--full",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--max-pages",type=int,default=10000)
    p.add_argument("--no-playwright",action="store_true")
    p.add_argument("--proxies",type=str,default="proxies.txt")
    p.add_argument("--no-proxies",action="store_true")
    p.add_argument("--no-curl-cffi",action="store_true")
    p.add_argument("--no-warp",action="store_true",
                   help="تعطيل تدوير WARP حتى لو كان warp-cli موجوداً")
    p.add_argument("--warp-every",type=int,default=80,
                   help="عدد الطلبات الناجحة بين كل تدوير WARP استباقي (default: 80)")
    p.add_argument("--debug",action="store_true")
    a=p.parse_args()
    if not a.out:
        a.out = get_output_folder(a.url)
    if a.debug: logging.getLogger().setLevel(logging.DEBUG)
    cfg=dict(PROFILES[a.profile])

    # ─── تهيئة WARP ─────────────────────────────────────
    warp_mgr: Optional[WarpManager] = None
    if not a.no_warp:
        warp_mgr = WarpManager()
        warp_mgr.ROTATE_EVERY_REQUESTS = a.warp_every
        warp_mgr.ROTATE_EVERY_SECONDS  = a.warp_every * 2   # ثانية ≈ ضعف الطلبات
        if not warp_mgr.available:
            log.warning("WARP CLI غير موجود — يمكن تثبيته من: https://1.1.1.1/")
            warp_mgr = None

    # ─── تهيئة بروكسي محلي ──────────────────────────────
    proxy_url = None
    if not a.no_proxies and Path(a.proxies).exists():
        smart_proxy = SmartLocalProxy(a.proxies, port=8081)
        if smart_proxy.start_in_background():
            proxy_url = "http://127.0.0.1:8081"

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    ex=Extractor(
        a.url, out,
        workers=cfg["workers"], burst=cfg["burst"], burst_pause=cfg["burst_pause"],
        max_pages=a.max_pages or 10_000_000, full=a.full, use_pw=not a.no_playwright,
        max_retries=5, backoff=0.5, resume=a.resume, cycle_work=0, cycle_pause=0,
        use_curl=not a.no_curl_cffi, proxy_url=proxy_url,
        warp=warp_mgr, warp_rotate_every=a.warp_every,
    )
    try: asyncio.run(ex.crawl())
    except KeyboardInterrupt: log.info("\nStopped."); ex.st.save()

if __name__=="__main__": main()