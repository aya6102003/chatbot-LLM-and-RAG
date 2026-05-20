"""
Smart Updater v3 — Stable Hash + Anti-Block Edition
================================================================
الإصلاحات الرئيسية:
  1. هاش مستقر: يُزيل tokens/counters/timestamps قبل الحساب
  2. مقارنة مزدوجة: هاش نصي + هاش هيكلي للتأكيد
  3. تحمّل ضجيج: تغيير أقل من 5% لا يُعدّ تغييراً حقيقياً
  4. تحسينات مضادة للحظر: تأخير تكيفي، تدوير UA/TLS، إدارة 429
Install: pip install aiohttp aiofiles beautifulsoup4 lxml curl_cffi
"""

import asyncio, hashlib, json, logging, random, re, ssl, time
from argparse import ArgumentParser
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 🔹 إعدادات الأوضاع — بدون تغيير
# ═══════════════════════════════════════════════════════════
MODES = {
    "fast-changed": {"desc": "⚡ يومي سريع (بدون بروكسي - فحص شامل بدون تنزيل ملفات)", "proxy": False, "full": False, "sem": 30, "sleep": 0.05, "dynamic_only": False},
    "fast-full":     {"desc": "🔍 أسبوعي سريع (بدون بروكسي - فحص شامل + تنزيل كل الملفات)", "proxy": False, "full": True,  "sem": 30, "sleep": 0.05, "dynamic_only": False},
    "proxy-changed": {"desc": "🛡️ يومي بالبروكسي (فحص شامل بدون تنزيل ملفات)", "proxy": True, "full": False, "sem": 20, "sleep": 0.1, "dynamic_only": False},
    "proxy-full":    {"desc": "🛡️🔍 أسبوعي بالبروكسي (فحص شامل + تنزيل كل الملفات)", "proxy": True, "full": True,  "sem": 20, "sleep": 0.1, "dynamic_only": False},
    "fast-dynamic":  {"desc": "🚀 تيربو ديناميك (فقط الصفحات كثيرة التغير + الجديدة - بدون بروكسي - تنزيل ملفات)", "proxy": False, "full": True,  "sem": 30, "sleep": 0.05, "dynamic_only": True},
    "proxy-dynamic": {"desc": "🚀🛡️ تيربو ديناميك بالبروكسي (فقط الصفحات كثيرة التغير + الجديدة - تنزيل ملفات)", "proxy": True, "full": True,  "sem": 20, "sleep": 0.1, "dynamic_only": True},
}

TLS_IMP = ["chrome124","chrome123","chrome120","chrome119","chrome116","chrome110","edge101","safari17_0"]
_UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]
_AL = ["fr-FR,fr;q=0.9,ar;q=0.8,en-US;q=0.7","en-US,en;q=0.9","ar-SA,ar;q=0.9,en-US;q=0.8"]
_SC = ['"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"','"Chromium";v="121", "Not(A:Brand";v="24", "Google Chrome";v="121"']

_STRIP_PARAMS = frozenset({
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "fbclid","gclid","yclid","msclkid","dclid","li_fat_id","ttclid","twclid",
    "mc_cid","mc_eid","vero_id","wickedid","_hsenc","_hsmi","hsCtaTracking",
    "PHPSESSID","jsessionid","sid","sessionid",
    "url","redirect","redirect_url","redirect_uri","redirecturl","redirect_to",
    "return","returnurl","returnto","return_path","returnTo","return_url",
    "destination","next","back","goto","continue","forward",
    "Itemid","layout","ref","source","_ga","amp",
})

_SKIP_EXT = (".jpg",".jpeg",".png",".gif",".svg",".ico",".webp",".bmp",".css",".js",
    ".woff",".woff2",".ttf",".eot",".map",".pdf",".doc",".docx",".xls",".xlsx",
    ".ppt",".pptx",".odt",".ods",".odp",".rtf",".txt",".csv",".zip",".rar",".7z",
    ".tar",".gz",".bz2","mp3",".mp4",".avi",".mov",".wmv",".mkv",".wav")
_SKIP_URL_RE = re.compile(r"(\?|&)(print=1|tmpl=component|format=feed|format=pdf|task=rss|view=feed|type=rss|output=pdf|mobile=1|iccaldate=)", re.I)
_DOC_EXT = {".pdf":"pdfs",".doc":"docs",".docx":"docs",".xls":"docs",".xlsx":"docs",".ppt":"docs",".pptx":"docs",".odt":"docs",".ods":"docs",".odp":"docs",".rtf":"docs",".csv":"docs",".zip":"docs",".rar":"docs",".7z":"docs"}
_IMAGE_EXT = {".jpg",".jpeg",".png",".gif",".webp",".svg",".bmp"}
_ERR_HTML  = re.compile(r'<html[^>]*class=["\'][^"\']*error[-_]page', re.I)
_ERR_TITLE = re.compile(r'خطأ\s*[:\s]?\d*|error\s*[:\s]?\d*|page\s+not\s+found|access\s+denied|bad\s+request|server\s+error|not\s+found|forbidden', re.I)
_ERR_BODY  = re.compile(r'class=["\']error[-_]code["\']|class=["\']error[-_]message["\']|<h1[^>]*class=["\']error', re.I)
_ONCLICK   = re.compile(r"(?:window\.location(?:\.href)?|document\.location(?:\.href)?|location\.(?:href|assign|replace)|window\.open)\s*[=(]\s*['\"]([^'\"]+)['\"]", re.I)
_STATIC_RE = re.compile(r'(/about|/contact|/faq|/help|/terms|/privacy|/legal|/qui-sommes|/a-propos|/presentation|/organisation|/structure|/historique|/histoire|/organigramme|/page_professionnelles|/equipe|/staff|/enseignant|/administration|/formation|/departement|/laboratoire|/calendrier|/polycope|/programme|/module)', re.I)
_INDEX_END_RE = re.compile(r'(/news|/actualit\w*|/articles?|/blog|/posts?|/evenement\w*|/event\w*|/annonces?|/publication\w*|/activit\w*|/accueil|/home|/page/\d+|/search|/result|/categorie\w*|/category|/rubrique\w*|/archives?)/?$', re.I)
_INDEX_EXCEPT_RE = re.compile(r'/annonces?/\d+|/articles?/\d+|/news/\d+|/blog/\d+|/posts?/\d+|/evenement\w*/\d+|/event\w*/\d+|/publication\w*/\d+|/categorie\w*/\d+|/category/\d+|/rubrique\w*/\d+', re.I)
_CHROME_IDS = {"sp-top-bar","sp-top1","sp-top2","sp-header","sp-logo","sp-menu","sp-footer","sp-footer1","sp-bottom","sp-bottom1","sp-bottom2","sp-bottom3","sp-page-title","sp-left","sp-right","sp-position2","masthead","colophon","site-header","site-footer","secondary","sidebar","widget-area","wpadminbar"}
_CHROME_CLS = re.compile(r"\b(navbar[-_]toggle|icon[-_]bar|sr[-_]only|scroll[-_]up|sticky[-_]header|camera_wrap|slideshow|mod-slideshowck|sliderck|mod-finder|cookie[-_]banner|gdpr[-_]|popup[-_]modal)\b", re.I)
_SKIP_LINE  = re.compile(r"^(toggle|navigation|menu|skip|log\s*in|sign\s*in|sign\s*out|عرض\s*المزيد|more\s*\.\.\.|read\s*more|\.\.\.|cookie|accept)$", re.I)
_META_CLS   = re.compile(r"\b(card[-_]info|card[-_]grade|card[-_]email|card[-_]depart|card[-_]bureau|card[-_]title|card[-_]phone|card[-_]office|profile[-_]info|author[-_]info|user[-_]details|person[-_]details|contact[-_]info)\b", re.I)
_HIDDEN_CLS = re.compile(r"\b(sr-only|visually-hidden|d-none|hidden|invisible)\b", re.I)
_RETRY_CODES = {429, 502, 503, 504}

# ═══════════════════════════════════════════════════════════
# 🔹 الإصلاح الرئيسي: تنظيف HTML قبل الهاش
#    يُزيل كل العناصر الديناميكية التي تتغير بين الطلبات
#    دون أن تمثّل تغييراً حقيقياً في المحتوى
# ═══════════════════════════════════════════════════════════

# أنماط تتغير بين الطلبات (tokens, sessions, counters, ads...)
_NOISE_INLINE = re.compile(
    r"""
    # CSRF / session tokens في الـ JS
    (?:csrfToken|csrf_token|_token|nonce|authenticity_token
      |__RequestVerificationToken|X-CSRF-TOKEN)
    \s*[=:]\s*["'][^"']{8,}["']
    |
    # معرّفات جلسة / طلب
    (?:sessionId|requestId|correlationId|traceId|uid)\s*[=:]\s*["'][^"']+["']
    |
    # NEXT.js / Nuxt / SPA initial state (كاملة)
    window\.__(?:INITIAL_STATE|NEXT_DATA|NUXT|APP_STATE|REDUX_STATE)__\s*=\s*\{[^;]{0,2000};\s*
    |
    # timestamps مضمّنة في JS
    (?:Date\.now\(\)|new\s+Date\(\)|timestamp|generatedAt|renderedAt)\s*[=:]\s*\d+
    |
    # Google Analytics / GTM / Meta Pixel calls
    (?:gtag|ga|fbq|_gaq\.push|googletag\.cmd\.push)\s*\([^)]{0,300}\)
    |
    # random IDs مضمّنة
    Math\.random\(\)\s*\*\s*\d+
    """,
    re.VERBOSE | re.IGNORECASE
)

# أنماط في قيم attributes
_NOISE_ATTR = re.compile(
    r'\b(?:nonce|data-token|data-csrf|data-session|data-request-id)=["\'][^"\']*["\']',
    re.IGNORECASE
)

# عناصر HTML ديناميكية بالكامل (عدادات زوار، إعلانات...)
_NOISE_TAGS_CLS = re.compile(
    r"""\b(
        visit[-_]?count|view[-_]?count|page[-_]?view|hit[-_]?count|
        online[-_]?count|counter[-_]?block|
        ad[-_]?banner|ad[-_]?slot|advertisement|adsbygoogle|
        cookie[-_]?notice|gdpr[-_]|consent[-_]|
        share[-_]?count|like[-_]?count|comment[-_]?count|
        weather[-_]?widget|clock[-_]?widget|
        random[-_]?banner|rotating[-_]?banner
    )\b""",
    re.VERBOSE | re.IGNORECASE
)

# أرقام في نص قد تكون عدادات (مثل "1,234 زيارة" أو "views: 567")
_NOISE_COUNTER_TEXT = re.compile(
    r'(?:(?:زيارة|مشاهدة|زائر|زوار|visit|view|hit)s?\s*:?\s*[\d,،\.]+|[\d,،\.]+\s*(?:زيارة|مشاهدة|زائر|views?|hits?))',
    re.IGNORECASE
)

# أرقام عشوائية طويلة (tokens مخفية في النص)
_NOISE_LONG_NUM = re.compile(r'\b[a-f0-9]{32,}\b|\b\d{13,}\b')  # MD5+ هاشات أو Unix ms timestamps
# أضف هذه التعريفات
_NOISE_DATE = re.compile(
    r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
    re.IGNORECASE
)
_NOISE_SMALL_NUM = re.compile(r'\b\d{1,3}\b')   # أعداد صغيرة تظهر بمفردها
_NOISE_DYNAMIC_TOKENS = re.compile(r'data-(?:token|nonce|counter|timestamp)=\\"[^"]+\\"', re.IGNORECASE)

def _stable_html(html: str) -> str:
    """
    يُنتج نسخة مستقرة من HTML بحذف كل العناصر الديناميكية.
    المبدأ: نُطبّق على HTML الخام قبل أي parsing، لأن بعض الـ tokens
    موجودة داخل تعليقات JS وليس في DOM مباشرة.
    """
    # 1) احذف كل كتل <script> بالكامل (أسرع مصدر للضجيج)
    h = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # 2) احذف كتل <style> (لا تؤثر على المحتوى)
    h = re.sub(r'<style\b[^>]*>.*?</style>', '', h, flags=re.DOTALL | re.IGNORECASE)
    # 3) احذف تعليقات HTML
    h = re.sub(r'<!--.*?-->', '', h, flags=re.DOTALL)
    # 4) احذف قيم input hidden (tokens)
    h = re.sub(r'<input[^>]+type=["\']hidden["\'][^>]*/?>',  '', h, flags=re.IGNORECASE)
    h = re.sub(r'<input[^>]+name=["\'][^"\']*(?:token|csrf|nonce|session)[^"\']*["\'][^>]*/?>',
               '', h, flags=re.IGNORECASE)
    # 5) احذف meta refresh و canonical التي قد تتغير
    h = re.sub(r'<meta[^>]+(?:http-equiv=["\']refresh["\'])[^>]*/?>',  '', h, flags=re.IGNORECASE)
    # 6) احذف attributes الديناميكية من كل العناصر
    h = _NOISE_ATTR.sub('', h)
    # 7) احذف الضجيج المضمّن في النص
    h = _NOISE_INLINE.sub('__DYNAMIC__', h)
    # 8) نظّف whitespace الزائد (قد يختلف بين الطلبات)
    h = re.sub(r'\s+', ' ', h)
    h = _NOISE_DYNAMIC_TOKENS.sub(' __TOKEN__ ', h)
    return h


def _stable_text(text: str) -> str:
    """
    يُنتج نسخة مستقرة من النص المستخرج.
    يُزيل الأرقام التي قد تكون عدادات متغيرة، وكذلك التواريخ.
    """
    t = _NOISE_COUNTER_TEXT.sub('__COUNT__', text)
    t = _NOISE_LONG_NUM.sub('__ID__', t)
    t = _NOISE_DATE.sub('__DATE__', t)          # إزالة التواريخ
    t = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '__TIME__', t)  # إزالة الساعات
    # إزالة الأعداد الصغيرة المنفردة (تجنب إزالة أجزاء من الكلمات)
    t = re.sub(r'(?<![a-zA-Z0-9])\d{1,3}(?![a-zA-Z0-9])', '__NUM__', t)
    # تنظيف whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _content_hash(text: str) -> str:
    """هاش مستقر للنص — يُطبّق التنظيف أولاً."""
    return hashlib.sha256(_stable_text(text).encode()).hexdigest()[:16]


def _html_structure_hash(html: str) -> str:
    """
    هاش هيكلي للـ HTML: يعتمد على بنية الصفحة (عدد العناصر، الروابط، العناوين)
    وليس على المحتوى الحرفي. يُستخدم للتأكيد المزدوج.
    """
    stable = _stable_html(html)
    # استخرج فقط: العناوين h1-h6، الفقرات، الروابط الداخلية
    headings = re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', stable, re.DOTALL | re.IGNORECASE)
    headings_text = ' || '.join(re.sub(r'<[^>]+>', '', h).strip() for h in headings[:20])
    link_hrefs = re.findall(r'href=["\']([^"\'#?][^"\']*)["\']', stable, re.IGNORECASE)
    links_sig = '|'.join(sorted(set(link_hrefs))[:50])
    sig = f"H:{headings_text}|L:{links_sig}"
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


def _is_real_change(old_text: str, new_text: str, old_html_sig: str, new_html_sig: str, page_type: str = "content") -> bool:
    """
    يقرر إذا كان التغيير حقيقياً أم مجرد ضجيج ديناميكي.
    
    المعامل page_type: "index" للصفحات الرئيسية، "content" لغيرها.
    """
    # 1) الهيكل تغيّر → تغيير حقيقي
    if old_html_sig and new_html_sig and old_html_sig != new_html_sig:
        return True

    # 2) لا يوجد نص قديم → جديد بالتأكيد
    if not old_text:
        return True

    # 3) متطابق تماماً بعد التنظيف → لا تغيير
    old_clean = _stable_text(old_text)
    new_clean = _stable_text(new_text)
    if old_clean == new_clean:
        return False

    # 4) قيس حجم التغيير الفعلي
    max_len = max(len(old_clean), len(new_clean), 1)
    old_words = set(old_clean.split())
    new_words  = set(new_clean.split())
    added   = len(new_words  - old_words)
    removed = len(old_words - new_words)
    total_words = max(len(old_words), len(new_words), 1)
    change_ratio = (added + removed) / (total_words * 2)

    # عتبة مختلفة للصفحات الفهرسية (5%) وللمحتوى العادي (3%)
    threshold = 0.05 if page_type == "index" else 0.03
    
    if change_ratio < threshold:
        return False

    return True


# ═══════════════════════════════════════════════════════════
# 🔹 دوال مساعدة — بدون تغيير جوهري
# ═══════════════════════════════════════════════════════════
def _ua(): return random.choice(_UA)
def _al(): return random.choice(_AL)
def _sc(): return random.choice(_SC)

def _get_base_domain(netloc: str) -> str:
    if not netloc: return ""
    n = netloc.lower()
    for prefix in ("www.", "m.", "mobile.", "wap."):
        if n.startswith(prefix): n = n[len(prefix):]; break
    return n

def _is_same_domain(url_netloc: str, base_netloc: str) -> bool:
    if not url_netloc: return True
    if not base_netloc: return False
    u = _get_base_domain(url_netloc); b = _get_base_domain(base_netloc)
    if not u or not b: return False
    if u == b: return True
    if u.endswith("." + b) or b.endswith("." + u): return True
    return False

def _norm(u: str, force_scheme: str = "") -> str:
    if not u: return ""
    u, _ = urldefrag(u); p = urlparse(u)
    if force_scheme and p.scheme and p.scheme != force_scheme:
        p = p._replace(scheme=force_scheme)
    path = p.path.rstrip("/") or "/"; q = ""
    if p.query:
        qs = parse_qs(p.query, keep_blank_values=True)
        for k in list(qs.keys()):
            if k in _STRIP_PARAMS: qs.pop(k, None)
        if qs: q = urlencode(sorted(qs.items()), doseq=True)
    return p._replace(path=path, query=q, fragment="").geturl()

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

def _page_id(url: str) -> str: return hashlib.md5(url.encode()).hexdigest()[:8]
def _safe(t: str, n: int = 50) -> str:
    if not t: return "untitled"
    return re.sub(r'[<>:"/\\|?*]', "_", re.sub(r"\s+", "_", t))[:n].strip("_")

def _pgh(ref=""):
    ua=_ua(); h={"User-Agent":ua,"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8","Accept-Language":_al(),"Accept-Encoding":"gzip, deflate, br","Connection":"keep-alive","Upgrade-Insecure-Requests":"1","Sec-Fetch-Dest":"document","Sec-Fetch-Mode":"navigate","Sec-Fetch-Site":"none","Sec-Fetch-User":"?1"}
    if "Chrome" in ua: h["Sec-Ch-Ua"]=_sc(); h["Sec-Ch-Ua-Mobile"]="?0"; h["Sec-Ch-Ua-Platform"]='"Windows"'
    if ref: h["Referer"]=ref; h["Sec-Fetch-Site"]="same-origin"
    return h

def _is_err(h):
    if len(h)<400: return True
    if _ERR_HTML.search(h): return True
    m=re.search(r'<title[^>]*>(.*?)</title>',h,re.I|re.S)
    if m and _ERR_TITLE.search(m.group(1).strip()): return True
    if _ERR_BODY.search(h): return True
    return False

def _crawlable(u, bd):
    if not u: return False
    p=urlparse(u)
    if p.scheme not in ("http","https"): return False
    if not _is_same_domain(p.netloc, bd): return False
    if p.path.lower().endswith(_SKIP_EXT): return False
    return not _SKIP_URL_RE.search(u)

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
    # إضافة: احذف عناصر العدادات والإعلانات
    for el in s.find_all(True, class_=_NOISE_TAGS_CLS): el.decompose()

def _content_area(s):
    for prop in ("articleBody","text"):
        ab = s.find(attrs={"itemprop": prop})
        if ab: return ab
    for cls in ["com-content-article","item-page","blog-item","article-content","article-details","article-body","post-content","entry-content","page-content","content-area","content-body","content-article","item-body","item-content","node-content","view-content","field-name-body","field-name-field-body","td-content","main-content-area","js-item-content","js-main-content","article-full","article-featured","interface2","hovercard","card-info","annonce"]:
        el = s.find(True, class_=re.compile(r"\b"+re.escape(cls)+r"\b", re.I))
        if el: return el
    for eid in ["sp-component","sp-main-body","main-content","content","primary","main","article","js-main","main-article","content-article","centercol","maincol","contentcol","yui-main","main-content-wrapper"]:
        el = s.find(id=eid)
        if el: return el
    for role in ("main","article"):
        el = s.find(attrs={"role": role})
        if el: return el
    for tag in ["main","article"]:
        el = s.find(tag)
        if el: return el
    best = None; best_len = 0
    for div in s.find_all(["div","section","article","main"]):
        txt = div.get_text(strip=True)
        if len(txt) < 150: continue
        links = div.find_all("a")
        if links and sum(len(a.get_text(strip=True)) for a in links) > 0.6 * len(txt): continue
        if len(links) > 10 and len(txt) < 2000: continue
        if len(txt) > best_len: best = div; best_len = len(txt)
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

def _clean_text(raw):
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
    if area: _clean_content(sc)
    else: _strip_chrome(sc)
    text=_clean_text(sc.get_text(separator="\n",strip=True))
    om=_extract_meta(s,area)
    if om: text=om+"\n\n"+text
    iv=_extract_inputs(s)
    if iv: text=(text+"\n\n"+iv).strip() if text else iv
    if len(text)<200:
        jld=_jsonld(s)
        if jld: text=_clean_text(jld+"\n"+text) if text else _clean_text(jld)
    if len(text)<150 and area:
        sc2=BeautifulSoup(str(s),"html.parser"); _strip_chrome(sc2)
        t2=_clean_text(sc2.get_text(separator="\n",strip=True))
        if len(t2)>len(text): text=t2
    return text,len(text)<150

def _title_from_soup(s,u):
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

def _extract_links(soup, page_url, base_domain, scheme=""):
    sc=BeautifulSoup(str(soup),"html.parser")
    for el in sc.find_all(True,id=re.compile(r"^offcanvas",re.I)): el.decompose()
    for el in sc.find_all(True,class_=re.compile(r"\boffcanvas-menu\b",re.I)): el.decompose()
    for el in sc.find_all(True,id=re.compile(r"^navbar\d+$",re.I)):
        if "collapse" in " ".join(el.get("class",[])): el.decompose()
    crawl:set=set(); all_lk:list=[]; seen:set=set()
    bd=urlparse(page_url).netloc or base_domain

    def reg(href,label=""):
        href=href.strip()
        if not href or href.startswith(("javascript:","mailto:","tel:","data:","ftp:","file:")): return
        if href.startswith("#"): return
        try: full=_norm(urljoin(page_url,href).split("#")[0],force_scheme=scheme)
        except: return
        if not full or full in seen: return
        p=urlparse(full)
        if p.scheme and p.scheme not in ("http","https"): return
        seen.add(full)
        if not label: label=full.split("/")[-1].split("?")[0] or "Link"
        label=re.sub(r"\s+"," ",label).strip()[:200]
        int_=_is_same_domain(p.netloc, bd)
        if int_ and _crawlable(full, bd): crawl.add(full)
        all_lk.append({"url":full,"text":label,"is_internal":int_})

    for a in sc.find_all("a",href=True):
        lbl=a.get_text(" ",strip=True)
        for c in a.find_all(["input","button"]):
            v=c.get("value","").strip()
            if v: lbl=v; break
        reg(a["href"],lbl)
    for form in sc.find_all("form",action=True):
        label=""
        for c in form.find_all(["input","button"]):
            if c.get("type","submit").lower() in ("submit","button"):
                v=c.get("value","").strip() or c.get_text(strip=True)
                if v: label=v; break
        reg(form["action"],label or "Form")
    for area in sc.find_all("area",href=True): reg(area["href"],area.get("alt","").strip() or "Area")
    for el in sc.find_all(True):
        for at in ("data-href","data-url","data-link","data-target-url","data-redirect","data-action","data-route","data-permalink"):
            v=el.get(at,"").strip()
            if v and v not in ("#","javascript:void(0)"): reg(v,el.get_text(" ",strip=True)[:200])
    for el in sc.find_all(True,onclick=True):
        m=_ONCLICK.search(el.get("onclick",""))
        if m: reg(m.group(1),(el.get_text(" ",strip=True) or el.get("value","") or el.get("title",""))[:200])
    head=sc.find("head")
    if head:
        for link in head.find_all("link",href=True):
            rel=" ".join(link.get("rel",[])).lower()
            if any(r in rel for r in ("alternate","next","prev","canonical","shortlink","amphtml")): reg(link["href"],rel)
        for meta in head.find_all("meta",attrs={"http-equiv":re.compile(r"refresh",re.I)}):
            content = meta.get("content","")
            m = re.search(r"url\s*=\s*['\"]?([^'\"\s;]+)", content, re.I)
            if m: reg(m.group(1), "Meta Refresh")
    cont_lk:list=[]
    area=_content_area(sc); sc2:set=set()
    if area:
        for a in area.find_all("a",href=True):
            f=_norm(urljoin(page_url,a["href"]).split("#")[0],force_scheme=scheme) if a.get("href") else ""
            if f and f not in sc2:
                sc2.add(f); p=urlparse(f)
                cont_lk.append({"url":f,"text":a.get_text(" ",strip=True)[:200] or f.split("/")[-1],"is_internal":_is_same_domain(p.netloc, bd)})
    sa:set=set(); dedup=[]
    for l in all_lk:
        if l["url"] not in sa: sa.add(l["url"]); dedup.append(l)
    return crawl,dedup,cont_lk

def _tables_from_soup(soup,pid,base_url):
    tbls=[]
    for i,tbl in enumerate(soup.find_all("table"),1):
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
                fn=f"{pid}_table_{i}_{_safe(title,30)}.json"
                tbls.append({"title":title[:100],"headers":hdrs,"row_count":len(rows),"filename":fn,"data":{"source_url":base_url,"title":title,"headers":hdrs,"rows":rows,"row_count":len(rows),"col_count":len(hdrs) if hdrs else (len(rows[0]) if rows else 0)}})
        except: pass
    return tbls


# ═══════════════════════════════════════════════════════════
# 🔹 UpdateState — إضافة حقل html_sig للمقارنة الهيكلية
# ═══════════════════════════════════════════════════════════
class UpdateState:
    def __init__(self, out: Path, scheme: str):
        self.out = out; self.scheme = scheme; self.pages_dir = out / "pages"
        self.visited: Set[str] = set(); self.page_meta: Dict[str, Dict] = {}; self.url_to_file: Dict[str, str] = {}
        self.index_pages_list: List[str] = []; self.index_links: Dict[str, Set[str]] = {}; self.index_discovered: Set[str] = set()
        self.update_history: List[Dict] = []

    def n(self, u: str) -> str: return _norm(u, force_scheme=self.scheme)

    def load(self) -> bool:
        if not self.pages_dir.exists(): return False
        log.info("Loading state...")
        uf = self.out / "_update_state.json"
        if uf.exists():
            try:
                with open(uf, encoding="utf-8") as f: ud = json.load(f)
                self.update_history = ud.get("update_history", [])
                for k, v in ud.get("page_meta", {}).items():
                    nk = self.n(k)
                    if nk: self.page_meta[nk] = v
                for k, v in ud.get("index_links", {}).items():
                    nk = self.n(k)
                    if not nk: continue
                    normed = {self.n(x) for x in v if self.n(x)}
                    if nk in self.index_links: self.index_links[nk] |= normed
                    else: self.index_links[nk] = normed
                    self.index_discovered |= normed
            except Exception as e: log.warning(f"State read err: {e}")

        file_count = 0
        for jf in self.pages_dir.glob("*.json"):
            try:
                with open(jf, encoding="utf-8") as f: d = json.load(f)
                raw_url = d.get("metadata",{}).get("page",{}).get("url","")
                if not raw_url: continue
                url = self.n(raw_url)
                if not url: continue
                self.visited.add(url); self.url_to_file[url] = jf.name
                text = d.get("content",{}).get("text","")
                upd_meta = d.get("metadata", {}).get("update", {})
                # اقرأ الهاش المستقر أو احسبه من النص القديم
                saved_hash = upd_meta.get("stable_hash","") or upd_meta.get("hash","") or _content_hash(text)
                saved_sig  = upd_meta.get("html_sig","")
                fetched = upd_meta.get("fetched_at", "")
                file_id = d.get("metadata",{}).get("page",{}).get("id","")
                if url in self.page_meta:
                    self.page_meta[url]["hash"]     = saved_hash
                    self.page_meta[url]["html_sig"] = saved_sig
                    if "id" not in self.page_meta[url] and file_id: self.page_meta[url]["id"] = file_id
                else:
                    self.page_meta[url] = {
                        "id": file_id or _page_id(url),
                        "hash": saved_hash,
                        "html_sig": saved_sig,
                        "text": text,   # نحتفظ بالنص لمقارنة الضجيج لاحقاً
                        "type": _classify(url),
                        "first_seen": fetched or datetime.now().isoformat(),
                        "last_check": fetched or datetime.now().isoformat(),
                        "last_change": "",
                        "etag": "", "last_modified": "",
                        "check_count": 0, "change_count": 0,
                    }
                ptype = self.page_meta[url].get("type", _classify(url))
                if ptype == "index":
                    links = set()
                    for link in d.get("resources",{}).get("links",[]):
                        if link.get("is_internal"):
                            lu = self.n(link["url"])
                            if lu and lu != url: links.add(lu)
                    self.index_links[url] = links; self.index_discovered |= links
                file_count += 1
            except Exception as e: log.debug(f"Skip {jf.name}: {e}")

        cf = self.out / "_crawl_state.json"
        if cf.exists():
            try:
                with open(cf, encoding="utf-8") as f: cd = json.load(f)
                added = 0
                for u in cd.get("v", []):
                    nn = self.n(u)
                    if nn and nn not in self.visited: self.visited.add(nn); added += 1
                if added: log.info(f"  +{added} URLs from crawl state")
            except: pass

        start = self.n(str(self.out))
        for u in self.visited:
            p = urlparse(u)
            if p.path.rstrip("/") in ("/","") or p.path == "/index.php":
                start = u; break
        self.index_pages_list = sorted([u for u in self.visited if _classify(u) == "index"], key=lambda u: (0 if u == start else 1, u))
        if start and start not in self.index_pages_list: self.index_pages_list.insert(0, start)
        log.info(f"  {file_count} files → {len(self.visited)} URLs | {len(self.index_pages_list)} Index | {len(self.index_discovered)} discovered")
        return bool(self.visited)

    def mark_checked(self, url, new_hash="", html_sig="", changed=False, etag="", last_modified="", new_text=""):
        now = datetime.now().isoformat()
        if url not in self.page_meta:
            self.page_meta[url] = {
                "id": _page_id(url), "hash": new_hash, "html_sig": html_sig,
                "text": new_text,
                "type": _classify(url), "first_seen": now, "last_check": now,
                "last_change": now if changed else "", "etag": etag,
                "last_modified": last_modified, "check_count": 1,
                "change_count": 1 if changed else 0,
            }
        else:
            if "id" not in self.page_meta[url]: self.page_meta[url]["id"] = _page_id(url)
            m = self.page_meta[url]
            m["last_check"] = now
            m["check_count"] = m.get("check_count", 0) + 1
            m["etag"] = etag or m.get("etag","")
            m["last_modified"] = last_modified or m.get("last_modified","")
            if html_sig: m["html_sig"] = html_sig
            if new_text:  m["text"] = new_text
            if changed:
                m["last_change"] = now; m["hash"] = new_hash
                m["change_count"] = m.get("change_count", 0) + 1
                ck,cc = m["check_count"], m["change_count"]
                if ck >= 3 and cc >= ck-1 and m["type"] not in ("dynamic","index","static"):
                    m["type"] = "dynamic"

    def mark_deleted(self, url):
        fname = self.url_to_file.get(url)
        if fname:
            fp = self.pages_dir / fname
            if fp.exists(): fp.unlink(); log.info(f"  🗑️ Deleted: {fname}")
            self.url_to_file.pop(url, None)
        else:
            pid = _page_id(url)
            for jf in self.pages_dir.glob(f"{pid}_*.json"):
                try:
                    with open(jf,encoding="utf-8") as f: d=json.load(f)
                    if self.n(d.get("metadata",{}).get("page",{}).get("url",""))==url:
                        jf.unlink(); log.info(f"  🗑️ Deleted: {jf.name}"); break
                except: pass
        self.visited.discard(url); self.page_meta.pop(url,None); self.index_discovered.discard(url)
        for k in self.index_links: self.index_links[k].discard(url)

    def register_file(self, url, filename): self.url_to_file[url] = filename

    def save(self):
        # لا نحفظ حقل "text" في الملف (كبير الحجم) — فقط في الذاكرة
        meta_to_save = {}
        for url, m in self.page_meta.items():
            meta_to_save[url] = {k: v for k, v in m.items() if k != "text"}
        with open(self.out/"_update_state.json","w",encoding="utf-8") as f:
            json.dump({
                "page_meta": meta_to_save,
                "index_links": {k:list(v) for k,v in self.index_links.items()},
                "index_discovered": list(self.index_discovered),
                "update_history": self.update_history[-100:],
                "saved_at": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 🔹 مدير التكيّف مع الحظر (Anti-Block Adapter)
# ═══════════════════════════════════════════════════════════
class BlockAdapter:
    """
    يتتبع أنماط الحظر ويُكيّف التأخير تلقائياً.
    - عند الحظر: يرفع التأخير تدريجياً
    - عند الاستقرار: يخفض التأخير تدريجياً
    - يتذكر آخر TLS impersonation ناجحة لكل domain
    """
    def __init__(self, base_sleep: float):
        self._base = base_sleep
        self._current = base_sleep
        self._blocks = 0          # عداد الحظر المتتالي
        self._successes = 0       # عداد النجاحات المتتالية
        self._last_tls = {}       # domain → last successful TLS profile
        self._lock = asyncio.Lock()

    async def on_block(self, code: int):
        async with self._lock:
            self._blocks += 1
            self._successes = 0
            # تأخير أسي محدود: 0.5 → 1 → 2 → 4 → 8 → max 30s
            self._current = min(self._base * (2 ** min(self._blocks, 5)), 30.0)
            wait = self._current + random.uniform(0, self._current * 0.5)
            log.warning(f"  🚧 Block {code} (×{self._blocks}) — next wait: {self._current:.1f}s")
            return wait

    async def on_success(self):
        async with self._lock:
            self._successes += 1
            self._blocks = 0
            # بعد 10 نجاحات متتالية، انخفض تدريجياً نحو القاعدة
            if self._successes >= 10 and self._current > self._base:
                self._current = max(self._current * 0.8, self._base)

    def sleep_time(self) -> float:
        return self._current + random.uniform(0, self._current * 0.3)

    def next_tls(self, domain: str) -> str:
        """اختر TLS profile — يُفضّل الأخيرة الناجحة إن وُجدت."""
        if domain in self._last_tls and random.random() < 0.7:
            return self._last_tls[domain]
        return random.choice(TLS_IMP)

    def record_tls(self, domain: str, profile: str):
        self._last_tls[domain] = profile


# ═══════════════════════════════════════════════════════════
# 🔹 SmartUpdater — المحدّث الرئيسي
# ═══════════════════════════════════════════════════════════
class SmartUpdater:
    def __init__(self, start_url, out, use_curl=True, proxy_url=None,
                 deleted_check=True, full=False, mode="fast-changed"):
        mode_cfg = MODES.get(mode, MODES["fast-changed"])
        self.mode = mode
        self._sem = asyncio.Semaphore(mode_cfg["sem"])
        self.full = full or mode_cfg["full"]
        self.proxy_url = proxy_url if mode_cfg["proxy"] else None
        self.dynamic_only = mode_cfg.get("dynamic_only", False)

        self.start_url = _norm(start_url)
        self.out = out
        self.domain = urlparse(start_url).netloc
        self.scheme = urlparse(start_url).scheme or "https"
        self.use_curl = use_curl and HAS_CURL_CFFI
        self.deleted_check = deleted_check

        self.dl_p=set(); self.dl_d=set(); self.dl_i=set()
        self._dl_sem = asyncio.Semaphore(2)
        self.state = UpdateState(out, self.scheme)
        self._cffi: Optional[CffiSession] = None
        self._tls_profile = random.choice(TLS_IMP)   # profile الحالي
        self._block_adapter = BlockAdapter(mode_cfg["sleep"])
        self.stats = {
            "new":0,"changed":0,"unchanged":0,"deleted":0,"failed":0,
            "304":0,"files_dl":0,"retries":0,"noise_skip":0,
        }
        for s in ("pages","tables","pdfs","docs","images"): (out/s).mkdir(parents=True,exist_ok=True)

    def n(self,u): return _norm(u,force_scheme=self.scheme)

    async def _gcffi(self) -> CffiSession:
        """أعد بناء الـ session إذا تغيّر الـ TLS profile."""
        if self._cffi is None:
            self._cffi = CffiSession(impersonate=self._tls_profile)
        return self._cffi

    async def _rotate_tls(self):
        """دوّر TLS profile — يُستدعى بعد الحظر."""
        if self._cffi:
            try: await self._cffi.close()
            except: pass
            self._cffi = None
        old = self._tls_profile
        candidates = [p for p in TLS_IMP if p != old]
        self._tls_profile = random.choice(candidates)
        log.info(f"  🔄 TLS rotated: {old} → {self._tls_profile}")

    async def _stealth_sleep(self):
        await asyncio.sleep(self._block_adapter.sleep_time())

    async def _head_req(self, sess, url, etag="", lm=""):
        h = _pgh(self.start_url)
        if etag: h["If-None-Match"] = etag
        elif lm: h["If-Modified-Since"] = lm
        for attempt in range(1, 4):
            try:
                await self._stealth_sleep()
                async with self._sem:
                    if self.use_curl:
                        cffi = await self._gcffi()
                        kw = {"headers":h,"timeout":15,"verify":False}
                        if self.proxy_url: kw["proxy"] = self.proxy_url
                        r = await cffi.head(url,**kw)
                        st = r.status_code
                        if st in _RETRY_CODES:
                            wait = await self._block_adapter.on_block(st)
                            await asyncio.sleep(wait)
                            await self._rotate_tls()
                            self.stats["retries"] += 1
                            continue
                        await self._block_adapter.on_success()
                        return st, r.headers.get("ETag",""), r.headers.get("Last-Modified","")
                    else:
                        async with sess.head(url,headers=h,ssl=False,timeout=ClientTimeout(total=15),proxy=self.proxy_url) as r:
                            st = r.status
                            if st in _RETRY_CODES:
                                wait = await self._block_adapter.on_block(st)
                                await asyncio.sleep(wait)
                                self.stats["retries"] += 1
                                continue
                            await self._block_adapter.on_success()
                            return st, r.headers.get("ETag",""), r.headers.get("Last-Modified","")
            except Exception:
                await asyncio.sleep(2 * attempt)
        return 0,"",""

    async def _get_req(self, sess, url):
        ref = self.start_url if random.random() < 0.5 else ""
        for attempt in range(1, 5):   # زيادة المحاولات من 3 إلى 4
            h = _pgh(ref)             # header جديد في كل محاولة (UA مختلف)
            try:
                await self._stealth_sleep()
                async with self._sem:
                    if self.use_curl:
                        cffi = await self._gcffi()
                        kw = {"headers":h,"allow_redirects":True,"timeout":30,"verify":False}
                        if self.proxy_url: kw["proxy"] = self.proxy_url
                        r = await cffi.get(url,**kw)
                        last = r.status_code
                        if last == 200 and r.text:
                            if _is_err(r.text): return None, None, 0
                            fu = self.n(str(r.real_url if hasattr(r,'real_url') else url))
                            await self._block_adapter.on_success()
                            self._block_adapter.record_tls(self.domain, self._tls_profile)
                            return r.text, fu, 200, r.headers.get("ETag",""), r.headers.get("Last-Modified","")
                        if last in _RETRY_CODES:
                            wait = await self._block_adapter.on_block(last)
                            await asyncio.sleep(wait)
                            await self._rotate_tls()
                            self.stats["retries"] += 1
                            continue
                        return None, None, last, "", ""
                    else:
                        async with sess.get(url,headers=h,allow_redirects=True,ssl=False,timeout=ClientTimeout(total=30),proxy=self.proxy_url) as r:
                            last = r.status
                            if last == 200:
                                ct = r.headers.get("Content-Type","").lower()
                                if any(x in ct for x in ("text/html","xhtml","text/plain")) or not ct:
                                    html = await r.text(errors="replace")
                                    if _is_err(html): return None, None, 0, "", ""
                                    await self._block_adapter.on_success()
                                    return html, self.n(str(r.url)), 200, r.headers.get("ETag",""), r.headers.get("Last-Modified","")
                            if last in _RETRY_CODES:
                                wait = await self._block_adapter.on_block(last)
                                await asyncio.sleep(wait)
                                self.stats["retries"] += 1
                                continue
                            return None, None, last, "", ""
            except Exception:
                await asyncio.sleep(2 * attempt)
        return None, None, 0, "", ""

    async def _dl(self, sess, u, dest, ref=""):
        if dest.exists(): return True
        try:
            h = {"User-Agent":_ua(),"Accept":"application/pdf,application/octet-stream,*/*","Accept-Language":_al(),**({"Referer":ref} if ref else {})}
            async with sess.get(u,headers=h,ssl=False,timeout=ClientTimeout(total=120),proxy=self.proxy_url) as br:
                if br.status == 200:
                    d = await br.read()
                    if len(d) < 100: return False
                    async with aiofiles.open(dest,"wb") as f: await f.write(d)
                    return True
                return False
        except: return False

    def _doc_links(self, s, bu):
        seen=set(); docs=[]
        for a in s.find_all("a",href=True):
            f=urljoin(bu,a["href"].strip()).split("#")[0].split("?")[0]
            for ext,sub in _DOC_EXT.items():
                if f.lower().endswith(ext) and f not in seen:
                    seen.add(f); t=a.get_text().strip() or Path(urlparse(f).path).stem.replace("_"," ").replace("-"," ")
                    docs.append({"title":t[:150],"url":f,"type":ext.lstrip("."),"sub":sub}); break
        for tag,attr in [("iframe","src"),("embed","src"),("source","src")]:
            for el in s.find_all(tag,**{attr:True}):
                f=urljoin(bu,el[attr].strip()).split("#")[0].split("?")[0]
                for ext,sub in _DOC_EXT.items():
                    if f.lower().endswith(ext) and f not in seen:
                        seen.add(f); t=el.get("title","").strip() or Path(urlparse(f).path).stem.replace("_"," ").replace("-"," ")
                        docs.append({"title":t[:150],"url":f,"type":ext.lstrip("."),"sub":sub}); break
        for obj in s.find_all("object"):
            for attr in ("data","data-src"):
                f=urljoin(bu,obj.get(attr,"").strip()).split("#")[0].split("?")[0]
                if not f: continue
                for ext,sub in _DOC_EXT.items():
                    if f.lower().endswith(ext) and f not in seen:
                        seen.add(f); t=obj.get("title","").strip() or Path(urlparse(f).path).stem.replace("_"," ").replace("-"," ")
                        docs.append({"title":t[:150],"url":f,"type":ext.lstrip("."),"sub":sub}); break
        return docs

    def _img_links(self, s, bu):
        seen=set(); imgs=[]
        for img in s.find_all("img",src=True):
            f=urljoin(bu,img["src"].strip()).split("#")[0]
            if f in seen: continue
            seen.add(f); p=urlparse(f).path; ext=Path(p).suffix.lower()
            if ext not in _IMAGE_EXT: ext=".jpg"
            alt=img.get("alt","").strip(); desc=img.get("title","").strip() or alt or Path(p).stem.replace("_"," ").replace("-"," ")
            imgs.append({"desc":desc[:150],"url":f,"alt":alt[:100],"ext":ext})
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

    async def _dl_files(self, sess, url, pid, docs, imgs):
        pdfs_out, docs_out, imgs_out = [], [], []
        if self.full:
            async def _dld(d):
                async with self._dl_sem:
                    gs=self.dl_p if d["sub"]=="pdfs" else self.dl_d
                    fn=f"{pid}_{_safe(d['title'],40)}.{d['type']}"; dest=self.out/d["sub"]/fn
                if d["url"] not in gs:
                    if await self._dl(sess,d["url"],dest,url):
                        gs.add(d["url"]); log.info(f"      ↓ {d['sub']}/{fn}")
                return {"title":d["title"],"url":d["url"],"local_file":f"{d['sub']}/{fn}" if dest.exists() else None}
            async def _dli(img):
                async with self._dl_sem:
                    fn=f"{pid}_{_safe(img['desc'],40)}{img['ext']}"; dest=self.out/"images"/fn
                    if img["url"] not in self.dl_i:
                        if await self._dl(sess,img["url"],dest,url):
                            self.dl_i.add(img["url"]); log.info(f"      ↓ images/{fn}")
                return {"url":img["url"],"local_file":f"images/{fn}" if dest.exists() else None}
            res=await asyncio.gather(*[_dld(d) for d in docs],*[_dli(i) for i in imgs],return_exceptions=True)
            nd=len(docs)
            for i,d in enumerate(docs):
                e=res[i]; (pdfs_out if d["sub"]=="pdfs" else docs_out).append(e if not isinstance(e,Exception) else {"title":d["title"],"url":d["url"],"local_file":None})
            for i,img in enumerate(imgs):
                e=res[nd+i] if nd+i<len(res) else None
                imgs_out.append(e if e and not isinstance(e,Exception) else {"url":img["url"],"local_file":None})
            dl_count = sum(1 for x in res if not isinstance(x,Exception) and isinstance(x,dict) and x.get("local_file"))
            if dl_count > 0: self.stats["files_dl"] += dl_count
        else:
            for d in docs: (pdfs_out if d["sub"]=="pdfs" else docs_out).append({"title":d["title"],"url":d["url"],"local_file":None})
            imgs_out=[{"url":i["url"],"local_file":None} for i in imgs]
        return pdfs_out, docs_out, imgs_out

    async def _save_page(self, sess, soup, url, update_num, html_raw=""):
        if not isinstance(soup, BeautifulSoup): return "", "", "", 0
        pid=_page_id(url); title=_title_from_soup(soup,url)
        text,_=extract_text(soup)
        chash   = _content_hash(text)
        html_sig = _html_structure_hash(html_raw) if html_raw else ""
        _,_,cont_lk=_extract_links(soup,url,self.domain,self.scheme)
        docs = self._doc_links(soup, url)
        imgs = self._img_links(soup, url)
        tbls=_tables_from_soup(soup,pid,url)
        for old_f in (self.out/"pages").glob(f"{pid}_*.json"): old_f.unlink()
        old_fname=self.state.url_to_file.get(url)
        if old_fname:
            op=self.out/"pages"/old_fname
            if op.exists():
                try: op.unlink()
                except: pass
        for tbl in tbls:
            async with aiofiles.open(self.out/"tables"/tbl["filename"],"w",encoding="utf-8") as f:
                await f.write(json.dumps(tbl["data"],ensure_ascii=False,indent=2))
        pdfs_out, docs_out, imgs_out = await self._dl_files(sess, url, pid, docs, imgs)
        pd_data={
            "metadata":{
                "page":{"url":url,"title":title,"id":pid},
                "update":{
                    "fetched_at":datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    "stable_hash":chash,   # الاسم الجديد
                    "hash":chash,          # للتوافق مع الكراولر القديم
                    "html_sig":html_sig,
                    "update_num":update_num,
                }
            },
            "content":{"text":text},
            "resources":{
                "links":[{"text":l["text"],"url":l["url"],"type":"internal" if l["is_internal"] else "external"} for l in cont_lk],
                "images":[{"url":i["url"],"alt":i.get("alt",""),"desc":i.get("desc",""), **({"local_file":i["local_file"]} if i.get("local_file") else {})} for i in imgs_out],
                "documents":[{"title":d["title"],"url":d["url"], **({"local_file":d["local_file"]} if d.get("local_file") else {})} for d in pdfs_out+docs_out],
                "tables":[{"file":f"tables/{t['filename']}","rows":t["row_count"]} for t in tbls],
            }
        }
        fname=f"{pid}_{_safe(title)}.json"
        async with aiofiles.open(self.out/"pages"/fname,"w",encoding="utf-8") as f:
            await f.write(json.dumps(pd_data,ensure_ascii=False,indent=2))
        self.state.register_file(url,fname)
        return chash, html_sig, title, len(text)

    # ─────────────────────────────────────────────────────
    # Phase 1: مسح الصفحات الفهرسية
    # ─────────────────────────────────────────────────────
    async def _phase1_scan_indexes(self, sess, update_num):
        current_all: Set[str] = set()
        idx_list = self.state.index_pages_list
        log.info(f"\n── Phase 1: Scanning {len(idx_list)} index page(s) ──")
        tasks = [self._process_index(sess, idx_url, update_num) for idx_url in idx_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, set): current_all |= res
            elif isinstance(res, Exception): log.error(f"Index error: {str(res)[:50]}")
        return current_all

    async def _process_index(self, sess, idx_url, update_num):
        meta = self.state.page_meta.get(idx_url, {})
        old_links = self.state.index_links.get(idx_url, set())

        res = await self._get_req(sess, idx_url)
        html, final_url, st = res[0], res[1], res[2]
        new_etag, new_lm = res[3] if len(res)>3 else "", res[4] if len(res)>4 else ""

        if not html or st != 200:
            if st in (404, 410):
                log.info(f"  🗑️ Index deleted ({st}): {idx_url[:60]}")
                self.state.mark_deleted(idx_url); self.stats["deleted"] += 1
                return set()
            return old_links

        soup = BeautifulSoup(html, "html.parser")
        crawl, _, _ = _extract_links(soup, final_url, self.domain, self.scheme)
        current = {u for u in crawl if u != final_url}

        text, _ = extract_text(soup)
        new_hash = _content_hash(text)
        new_sig  = _html_structure_hash(html)
        old_hash = meta.get("hash","")
        old_sig  = meta.get("html_sig","")
        old_text = meta.get("text","")

        changed = _is_real_change(old_text, text, old_sig, new_sig, page_type="index") if old_hash else True

        if changed and old_hash:
            log.info(f"  📝 Index changed: {idx_url[:55]}")
            await self._save_page(sess, soup, final_url, update_num, html)
            self.stats["changed"] += 1
        elif not changed and old_hash:
            self.stats["unchanged"] += 1
        else:
            self.stats["unchanged"] += 1

        self.state.mark_checked(final_url, new_hash, new_sig, changed=changed,
                                 etag=new_etag, last_modified=new_lm, new_text=text)
        self.state.index_links[final_url] = current
        self.state.visited.add(final_url)
        log.info(f"  ✓ {idx_url[:55]} → {len(current)} links")
        return current

    # ─────────────────────────────────────────────────────
    # Phase 2: المقارنة
    # ─────────────────────────────────────────────────────
    def _phase2_diff(self, current_all: Set[str]) -> Tuple[Set[str], Set[str]]:
        truly_new = current_all - self.state.visited
        content_urls = {u for u, m in self.state.page_meta.items() if m["type"] == "content"}
        disappeared = (self.state.index_discovered - current_all) & self.state.visited & content_urls
        disappeared -= current_all
        log.info(f"\n── Phase 2: Diff ──")
        log.info(f"  ✨ NEW: {len(truly_new)} | 🗑️ GONE: {len(disappeared)}")
        return truly_new, disappeared

    # ─────────────────────────────────────────────────────
    # Phase 3: الزحف للصفحات الجديدة
    # ─────────────────────────────────────────────────────
    async def _phase3_crawl_new(self, sess, new_urls: Set[str], update_num):
        if not new_urls: return
        log.info(f"\n── Phase 3: Crawling {len(new_urls)} new page(s) ──")
        tasks = [self._crawl_one(sess, u, update_num, i+1) for i, u in enumerate(new_urls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        extra_new = set()
        for r in results:
            if isinstance(r, set): extra_new |= r
        if extra_new: await self._phase3_crawl_new(sess, extra_new, update_num)

    async def _crawl_one(self, sess, url, update_num, idx):
        if url in self.state.visited: return set()
        log.info(f"  [NEW {idx:03d}] {url[:75]}")
        res = await self._get_req(sess, url)
        html, final_url, st = res[0], res[1], res[2]
        self.state.visited.add(url)
        if final_url and final_url != url: self.state.visited.add(final_url)
        if not html or st != 200:
            self.stats["failed"] += 1; return set()
        soup = BeautifulSoup(html, "html.parser")
        crawl, _, _ = _extract_links(soup, final_url, self.domain, self.scheme)
        chash, html_sig, title, tl = await self._save_page(sess, soup, final_url, update_num, html)
        self.state.mark_checked(final_url, chash, html_sig, changed=True, new_text="")
        pt = _classify(final_url)
        if pt == "index":
            il = {u for u in crawl if u != final_url}
            self.state.index_links[final_url] = il; self.state.index_discovered |= il
        log.info(f"    ✓ '{title[:40]}' | {tl:,}c | [{pt}]")
        self.stats["new"] += 1
        return crawl

    # ─────────────────────────────────────────────────────
    # Phase 4: تحقق من المحذوفات
    # ─────────────────────────────────────────────────────
    async def _phase4_check_deleted(self, sess, disappeared: Set[str]):
        if not disappeared or not self.deleted_check: return
        log.info(f"\n── Phase 4: Checking {len(disappeared)} disappeared ──")
        tasks = [
            self._head_req(sess, u,
                           self.state.page_meta.get(u,{}).get("etag",""),
                           self.state.page_meta.get(u,{}).get("last_modified",""))
            for u in disappeared
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, res in zip(disappeared, results):
            if isinstance(res, Exception): continue
            status, _, _ = res
            if status in (404, 410):
                log.info(f"  🗑️ Deleted ({status}): {url[:65]}")
                self.state.mark_deleted(url); self.stats["deleted"] += 1

    # ─────────────────────────────────────────────────────
    # Phase 5: فحص صفحات المحتوى
    # ─────────────────────────────────────────────────────
    async def _phase5_check_all_content(self, sess, update_num):
        candidates = [url for url, meta in self.state.page_meta.items()
                      if meta.get("type") not in ("index","static","dynamic")]

        if self.dynamic_only:
            dynamic_candidates = []
            skipped_static = skipped_stable = skipped_no_data = 0
            for url, meta in self.state.page_meta.items():
                mt = meta.get("type", "content")
                if mt in ("index","static"): skipped_static += 1; continue
                ck = meta.get("check_count",0); cc = meta.get("change_count",0)
                if mt == "dynamic": dynamic_candidates.append(url)
                elif ck >= 2 and cc >= 2 and (cc/ck) >= 0.4: dynamic_candidates.append(url)
                elif ck < 2: skipped_no_data += 1
                else: skipped_stable += 1
            if not dynamic_candidates:
                log.info(f"\n── Phase 5: ⏭️ SKIPPED (dynamic-only mode) ──")
                log.info(f"  📊 Tracked: {len(self.state.page_meta)} | No dynamic pages yet")
                log.info(f"     (needs ≥2 update runs to identify dynamic pages)")
                return
            candidates = dynamic_candidates
            log.info(f"\n── Phase 5: 🚀 Turbo-checking {len(candidates)} dynamic page(s) ──")
            log.info(f"  ⏭️ Skipped: {skipped_stable} stable | {skipped_no_data} unknown | {skipped_static} index/static")
        else:
            if not candidates: return
            log.info(f"\n── Phase 5: Deep-checking {len(candidates)} content pages ──")

        # HEAD requests بالجملة
        head_tasks = [
            self._head_req(sess, u,
                           self.state.page_meta.get(u,{}).get("etag",""),
                           self.state.page_meta.get(u,{}).get("last_modified",""))
            for u in candidates
        ]
        head_results = await asyncio.gather(*head_tasks, return_exceptions=True)

        need_get = []
        for url, res in zip(candidates, head_results):
            if isinstance(res, Exception): continue
            status, ne, nl = res; meta = self.state.page_meta.get(url,{})
            if status == 304:
                self.stats["304"] += 1; self.stats["unchanged"] += 1
                self.state.mark_checked(url, meta.get("hash",""), meta.get("html_sig",""),
                                         etag=ne, last_modified=nl)
                continue
            if status in (404, 410):
                self.state.mark_deleted(url); self.stats["deleted"] += 1; continue
            if status == 0: continue
            need_get.append(url)

        if not need_get:
            label = "dynamic" if self.dynamic_only else "content"
            log.info(f"  ✓ All {len(candidates)} {label} pages unchanged (304)")
            return

        log.info(f"  ⚡ {len(need_get)} pages need content verification...")
        get_tasks = [self._get_req(sess, u) for u in need_get]
        get_results = await asyncio.gather(*get_tasks, return_exceptions=True)

        extra_new = set()
        for url, res in zip(need_get, get_results):
            if isinstance(res, Exception): self.stats["failed"] += 1; continue
            html, final_url, gst = res[0], res[1], res[2]
            new_etag = res[3] if len(res)>3 else ""
            new_lm   = res[4] if len(res)>4 else ""
            if not html or gst != 200: continue

            soup = BeautifulSoup(html, "html.parser")
            text, _ = extract_text(soup)
            new_hash = _content_hash(text)
            new_sig  = _html_structure_hash(html)
            meta     = self.state.page_meta.get(url,{})
            old_hash = meta.get("hash","")
            old_sig  = meta.get("html_sig","")
            old_text = meta.get("text","")

            # المقارنة الذكية المضادة للضجيج
            if old_hash == new_hash:
                # هاش متطابق → لا تغيير قطعاً
                self.state.mark_checked(url, new_hash, new_sig,
                                         etag=new_etag, last_modified=new_lm)
                self.stats["unchanged"] += 1
            elif not _is_real_change(old_text, text, old_sig, new_sig, page_type=meta.get("type","content")):
                # هاش مختلف لكن المقارنة الذكية تقول: ضجيج فقط
                log.debug(f"  ~ Noise skipped: {url[:65]}")
                self.stats["noise_skip"] += 1
                self.stats["unchanged"] += 1
                # نحدّث الهاش إلى الجديد حتى لا نعيد المقارنة في المرة القادمة
                self.state.mark_checked(url, new_hash, new_sig,
                                         etag=new_etag, last_modified=new_lm, new_text=text)
            else:
                log.info(f"  📝 Changed: {url[:65]}")
                await self._save_page(sess, soup, final_url, update_num, html)
                self.state.mark_checked(final_url, new_hash, new_sig, changed=True,
                                         etag=new_etag, last_modified=new_lm, new_text=text)
                self.stats["changed"] += 1
                crawl, _, _ = _extract_links(soup, final_url, self.domain, self.scheme)
                for u in crawl:
                    if u not in self.state.visited and u != final_url: extra_new.add(u)

        if extra_new:
            log.info(f"  ✨ {len(extra_new)} extra new link(s) discovered inside content")
            await self._phase3_crawl_new(sess, extra_new, update_num)

    # ─────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────
    async def run(self) -> Dict:
        t0 = time.monotonic()
        if not self.state.load():
            log.error("No previous data! Run fast extractor first."); return {}

        update_num = len(self.state.update_history) + 1
        log.info("=" * 70)
        log.info(f"SMART UPDATER v3 — Update #{update_num} | Mode: {self.mode}")
        log.info(f"Profile: {MODES[self.mode]['desc']}")
        if self.dynamic_only:
            log.info(f"🔥 DYNAMIC-ONLY: Checking only high-change-rate pages + new pages")
        log.info(f"Proxy: {'ON' if self.proxy_url else 'OFF (Direct)'} | Full Downloads: {'ON' if self.full else 'OFF'}")
        log.info(f"TLS: {self._tls_profile} | Anti-noise hash: ON")
        log.info("=" * 70)

        ssl_c = ssl.create_default_context()
        ssl_c.check_hostname = False; ssl_c.verify_mode = ssl.CERT_NONE
        conn = TCPConnector(
            ssl=ssl_c,
            limit=MODES[self.mode]["sem"]+20,
            limit_per_host=MODES[self.mode]["sem"]//2,
            keepalive_timeout=30,
        )
        try:
            async with aiohttp.ClientSession(connector=conn, timeout=ClientTimeout(total=30, connect=10)) as sess:
                current_all = await self._phase1_scan_indexes(sess, update_num)
                truly_new, disappeared = self._phase2_diff(current_all)
                await self._phase3_crawl_new(sess, truly_new, update_num)
                await self._phase4_check_deleted(sess, disappeared)
                await self._phase5_check_all_content(sess, update_num)
                self.state.index_discovered = current_all
        finally:
            if self._cffi:
                try: await self._cffi.close()
                except: pass

        elapsed = time.monotonic() - t0
        record = {
            "update_num": update_num,
            "timestamp":  datetime.now().isoformat(),
            "elapsed_s":  round(elapsed, 1),
            "mode":       self.mode,
            "dynamic_only": self.dynamic_only,
            "stats":      dict(self.stats),
        }
        self.state.update_history.append(record)
        self.state.save()

        hf = self.out / "_update_history.json"
        hist = []
        if hf.exists():
            try:
                with open(hf, encoding="utf-8") as f: hist = json.load(f)
            except: pass
        hist.append(record)
        with open(hf,"w",encoding="utf-8") as f: json.dump(hist[-100:], f, indent=2)

        log.info(f"\n{'='*70}")
        log.info(f"UPDATE #{update_num} DONE — {elapsed:.0f}s | Retries: {self.stats['retries']}")
        log.info(f"  ✨ New:     {self.stats['new']}")
        log.info(f"  📝 Changed: {self.stats['changed']}")
        log.info(f"  ✓ Unchanged:{self.stats['unchanged']}  (noise_skip: {self.stats['noise_skip']})")
        log.info(f"  🗑️ Deleted: {self.stats['deleted']}  | 📁 Files: {self.stats['files_dl']}")
        log.info(f"{'='*70}")
        return record


def main():
    ap = ArgumentParser(description="Smart Updater v3 — Stable Hash + Anti-Block")
    ap.add_argument("--url",required=True)
    ap.add_argument("--out",default="./extracted_data")
    ap.add_argument("--proxy",default="http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=MODES.keys(), default="fast-changed",
        help=(
            "\n  Standard modes:"
            "\n    fast-changed   ⚡ Quick daily — no proxy, no downloads"
            "\n    fast-full      🔍 Weekly — no proxy, download all files"
            "\n    proxy-changed  🛡️ Daily with proxy — no downloads"
            "\n    proxy-full     🛡️🔍 Weekly with proxy — download all files"
            "\n  Turbo dynamic modes:"
            "\n    fast-dynamic   🚀 Turbo — no proxy, only dynamic pages"
            "\n    proxy-dynamic  🚀🛡️ Turbo with proxy — only dynamic pages"
        ))
    ap.add_argument("--no-curl-cffi",action="store_true")
    ap.add_argument("--no-deleted-check",action="store_true")
    ap.add_argument("--full",action="store_true")
    a=ap.parse_args()

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    effective_proxy = a.proxy if MODES[a.mode]["proxy"] else None
    updater=SmartUpdater(a.url, out, use_curl=not a.no_curl_cffi,
                         proxy_url=effective_proxy,
                         deleted_check=not a.no_deleted_check,
                         full=a.full, mode=a.mode)
    try: asyncio.run(updater.run())
    except KeyboardInterrupt:
        log.info("\nStopped."); updater.state.save()

if __name__=="__main__": main()

# it was 1000 line 