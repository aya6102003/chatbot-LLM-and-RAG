from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*transformers.*")

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False

try:
    import google.generativeai as genai
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

try:
    import pypdf
    _PYPDF_OK = True
except ImportError:
    try:
        import PyPDF2 as pypdf  # type: ignore
        _PYPDF_OK = True
    except ImportError:
        _PYPDF_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ROOT_FOLDER    = "./test/university_farhat_abaas"
METADATA_PATH  = "./metadata.json"
PROGRESS_DB    = "./pipeline_progress.db"
STRUCTURE_FILE = "./structure_sciences.json"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
EMBED_MODEL    = "BAAI/bge-m3"
EMBED_DIM      = 1024

NEO4J_BATCH    = 50
PROGRESS_FLUSH = 50

CHUNK_TOKENS_BASE = 500
OVERLAP_TOKENS    = 100
MIN_CHUNK_CHARS   = 80
MIN_DOC_CHARS     = 100

EMBED_MIN_CONFIDENCE      = 0.65
LLM_CONFIDENCE_THRESHOLD  = 0.55
LLM_CONTENT_EXCERPT       = 500
CONTENT_SIGNAL_WINDOW     = 300
GEMINI_RATE_LIMIT_SECONDS = 1.0

_AMBIGUITY_PENALTY  = 0.1
_AMBIGUITY_MAX_MATCHES = 1

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="DEBUG",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

FACULTY_LABELS: Dict[str, str] = {
    "farhat_abbas_university": "Farhat Abbas University Sétif 1",
    "ftechnologie":            "Faculty of Technology",
    "fsciences":               "Faculty of Science",
    "fsnv":                    "Faculty of Nature and Life Sciences",
    "feco":                    "Faculty of Economics, Business and Management Sciences",
    "fmed":                    "Faculty of Medicine",
    "iomp":                    "Institute of Optics and Precision Mechanics",
    "iast":                    "Institute of Architecture and Earth Sciences",
    "istm":                    "Institute of Materials Science and Techniques",
}

_DEPTH_ORDER: Dict[str, int] = {
    "Faculty":        0,
    "Department":     1,
    "Level":          2,
    "Category":       3,
    "Program":        3,
    "Specialization": 4,
    "Year":           5,
    "General":        -1,
}

_SPECIFIC_LABELS: frozenset = frozenset({
    "Department", "Category", "Program", "Specialization", "Year", "Level"
})
_GENERAL_LABELS: frozenset = frozenset({"Faculty"})

_HIERARCHY_RELS = (
    "HAS_DEPARTMENT|HAS_LEVEL|HAS_PROGRAM|"
    "HAS_CATEGORY|HAS_SPECIALIZATION|HAS_YEAR|HAS_GENERAL"
)
_HIERARCHY_RELS_NO_GENERAL = (
    "HAS_DEPARTMENT|HAS_LEVEL|HAS_PROGRAM|"
    "HAS_CATEGORY|HAS_SPECIALIZATION|HAS_YEAR"
)

_REL_HAS_CONTENT = "HAS_CONTENT"
_REL_HAS_FILE    = "HAS_FILE"
_REL_HAS_CHUNK   = "HAS_CHUNK"
_REL_NEXT_CHUNK  = "NEXT_CHUNK"

_STOP_WORDS: frozenset = frozenset({
    "http", "https", "www", "html", "htm", "php", "asp", "jsp",
    "com", "org", "net", "edu", "fr", "dz", "gp",
    "savoirplus", "page", "uploads", "documents", "index", "accueil", "home",
    "de", "des", "les", "aux", "est", "sur", "dans", "une", "pour",
    "et", "la", "le", "en", "un", "au", "par", "avec", "du",
})

_ABBREV_MAP: Dict[str, str] = {
    "IDTW":  "Ingénierie des Données et Technologie Web",
    "PEER":  "Physique Energétique et Energies Renouvelables",
    "FIII":  "Fondements et Ingénierie de l'Information et de l'Image",
    "RSD":   "Réseaux et Systèmes Distribués",
    "ISI":   "Ingénierie des Systèmes d'Information",
    "GL":    "Génie Logiciel",
    "IA":    "Intelligence Artificielle",
    "AI":    "Intelligence Artificielle",
    "CS":    "Cyber Security",
    "IQ":    "Informatique Quantique",
    "SD":    "Sciences de données",
    "PM":    "Physique des Matériaux",
    "PMRE":  "Physique Médicale",
    "MA":    "Mathématiques Appliquées",
}

_LABEL_MAP: Dict[str, str] = {
    "Faculty": "Faculty", "Department": "Department",
    "Level": "Level", "Category": "Category",
    "Program": "Program", "Specialization": "Specialization",
    "Year": "Year", "General": "General",
    "Faculte": "Faculty", "Departement": "Department",
    "Filiere": "Category", "Niveau": "Level",
    "Specialite": "Specialization", "Annee": "Year",
}

_SRC_WEIGHT: Dict[str, int] = {
    "url":      5,
    "title":    3,
    "dept":     2,
    "content":  1,
    "pdf_name": 4,  # FIX C: PDF filename is a strong signal
}

_MIN_SCORE = 3.0

# ══════════════════════════════════════════════════════════════════════════════
# COMPILED REGEX
# ══════════════════════════════════════════════════════════════════════════════

_RE_CONTROL      = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_REPEATS      = re.compile(r"(.)\1{2,}")
_RE_APOSTROPHE   = re.compile(r"[''`´]")
_RE_QUOTES       = re.compile(r"[«»\u201c\u201d\u201e]")
_RE_PAGE_NUM     = re.compile(r"(?m)^\s*\d{1,4}\s*$")
_RE_MULTI_NL     = re.compile(r"\n{3,}")
_RE_SPACES       = re.compile(r"[ \t]+")
_RE_SENT_BOUND   = re.compile(r"(?<=[.!?؟])\s+")
_RE_YEAR_CAL     = re.compile(r"\b(20[12]\d)\b")
_RE_AR_CHARS     = re.compile(r"[\u0600-\u06FF]")
_RE_FR_WORDS     = re.compile(
    r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
    r"cours|semestre|licence|master|doctorat|fili.re)\b", re.IGNORECASE)
_RE_URL_GENERAL  = re.compile(r"https?://[^\s<>\"')\]]+")
_RE_PDF_URL      = re.compile(
    r"https?://[^\s<>\"')\]]+\.pdf(?:[?#][^\s<>\"')\]]*)?", re.IGNORECASE)
_RE_REL_PDF      = re.compile(r"[\"']([^\"']+\.pdf)[\"']", re.IGNORECASE)
_RE_PARA_BREAK   = re.compile(r"\n{2,}")
_RE_URL_DOMAIN   = re.compile(r"https?://[^/]+")
_RE_URL_TOKENS   = re.compile(r"[a-zA-ZÀ-ÿ\u0600-\u06FF]{2,}|\d+")
_RE_HEADING      = re.compile(
    r"^(?:#{1,4}\s+|(?:CHAPITRE|CHAPTER|SECTION|PARTIE|PART)\s+[\w\d]+"
    r"|(?:\d+\.){1,3}\s+\w"
    r"|[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{4,}(?:\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]+)*$)",
    re.MULTILINE | re.UNICODE,
)
_RE_TABLE_MARKER = re.compile(r"(?m)^\|.+\|$")
_RE_LIST_ITEM    = re.compile(r"(?m)^[-•*▶]\s+\S")
_RE_LEVEL_TOKEN  = re.compile(
    r"\b(L[123]|M[12]|ING[1-5])\b", re.IGNORECASE)

_LEVEL_PHRASE_MAP: Dict[str, str] = {
    "master 1":           "M1", "master1":              "M1",
    "master 2":           "M2", "master2":              "M2",
    "licence 1":          "L1", "licence1":             "L1",
    "licence 2":          "L2", "licence2":             "L2",
    "licence 3":          "L3", "licence3":             "L3",
    "1ere annee master":  "M1", "premiere annee master": "M1",
    "2eme annee master":  "M2", "deuxieme annee master": "M2",
    "premiere annee":     "L1", "1ere annee":            "L1",
    "deuxieme annee":     "L2", "2eme annee":            "L2",
    "troisieme annee":    "L3", "3eme annee":            "L3",
}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TextNormalizer:
    _AR_DIAC = re.compile(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC"
        r"\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]"
    )

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = _RE_CONTROL.sub("", text)
        text = cls._AR_DIAC.sub("", text)
        text = _RE_REPEATS.sub(r"\1\1", text)
        text = text.replace("œ", "oe").replace("æ", "ae")
        text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
        text = _RE_APOSTROPHE.sub("'", text)
        text = _RE_QUOTES.sub('"', text)
        text = _RE_PAGE_NUM.sub("", text)
        text = _RE_MULTI_NL.sub("\n\n", text)
        text = _RE_SPACES.sub(" ", text)
        return text.strip()

    @classmethod
    def for_match(cls, text: str) -> str:
        if not text:
            return ""
        text = cls.normalize(text)
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.lower().strip()


def normalize_text(text: str) -> str:
    return TextNormalizer.normalize(text)


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "en"
    sz     = min(500, len(text) // 4)
    sample = text[:sz] + text[len(text) // 3: len(text) // 3 + sz]
    ar     = len(_RE_AR_CHARS.findall(sample))
    fr_w   = len(_RE_FR_WORDS.findall(sample))
    fr_acc = len(re.findall(r"[éèêëàâäùûüîïôöçÉÈÊËÀÂÄÙÛÜÎÏÔÖÇ]", sample))
    if ar > len(sample) * 0.12:
        return "ar"
    if fr_w > 4 or fr_acc > 3:
        return "fr"
    return "en"


def extract_year_calendar(text: str) -> Optional[int]:
    m = _RE_YEAR_CAL.search(text[:500])
    return int(m.group(1)) if m else None


def extract_links(text: str) -> List[str]:
    return list(dict.fromkeys(_RE_URL_GENERAL.findall(text)))


def extract_pdf_urls(text: str, base_url: str = "") -> List[str]:
    urls = list(dict.fromkeys(_RE_PDF_URL.findall(text)))
    if base_url:
        base = base_url.rstrip("/")
        for rel in _RE_REL_PDF.findall(text):
            if not rel.startswith("http"):
                full = f"{base}/{rel.lstrip('/')}"
                if full not in urls:
                    urls.append(full)
    return urls


def doc_fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def url_to_id(url: str, fallback_fp: str) -> str:
    if url and url.strip():
        return hashlib.md5(url.strip().encode()).hexdigest()[:16]
    return f"url_{fallback_fp}"


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURE INDEX
# ══════════════════════════════════════════════════════════════════════════════

class StructureIndex:
    """
    Loads structure_sciences.json and indexes all entries for fast lookup.

    Key addition for FIX B: exposes `word_to_node_ids()` so callers can check
    how many DISTINCT nodes (not entries) a word maps to, and whether they all
    share the same parent — enabling "collapse to parent" behavior.
    """

    def __init__(self, path: str = STRUCTURE_FILE):
        self.entries: List[str]              = []
        self._full_to_orig:    Dict[str, str]       = {}
        self._word_to_orig:    Dict[str, str]       = {}
        self._bigram_to_orig:  Dict[str, str]       = {}
        self._trigram_to_orig: Dict[str, str]       = {}
        self._year_tokens:     Set[str]             = set()
        self._word_ambiguity:  Dict[str, int]       = {}
        # FIX B: word → set of entry names that contain it (for parent-collapse)
        self._word_to_entries: Dict[str, Set[str]]  = {}
        self._load(path)

    def _load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠  Structure file not found: {path}")
            return
        except Exception as exc:
            logger.warning(f"⚠  StructureIndex load error: {exc}")
            return

        sorted_entries = sorted(self.entries, key=len, reverse=True)
        word_entry_count: Dict[str, int]      = {}
        word_to_entries:  Dict[str, Set[str]] = {}

        for name in sorted_entries:
            norm = TextNormalizer.for_match(name)
            self._full_to_orig[norm] = name

            if re.match(r'^(m[12]|l[123]|ing[1-5])$', norm, re.IGNORECASE):
                self._year_tokens.add(norm)
                continue

            words = [w for w in norm.split() if len(w) >= 2 and w not in _STOP_WORDS]

            for w in words:
                word_entry_count[w] = word_entry_count.get(w, 0) + 1
                word_to_entries.setdefault(w, set()).add(name)
                if w not in self._word_to_orig:
                    self._word_to_orig[w] = name

            for i in range(len(words) - 1):
                bg = f"{words[i]} {words[i+1]}"
                if bg not in self._bigram_to_orig:
                    self._bigram_to_orig[bg] = name

            for i in range(len(words) - 2):
                tg = f"{words[i]} {words[i+1]} {words[i+2]}"
                if tg not in self._trigram_to_orig:
                    self._trigram_to_orig[tg] = name

        self._word_ambiguity  = word_entry_count
        self._word_to_entries = word_to_entries

        for abbrev, expansion in _ABBREV_MAP.items():
            norm_exp  = TextNormalizer.for_match(expansion)
            abbrev_lc = abbrev.lower()
            if norm_exp in self._full_to_orig:
                self._full_to_orig[abbrev_lc] = self._full_to_orig[norm_exp]
            if norm_exp in self._word_to_orig:
                self._word_to_orig[abbrev_lc] = self._word_to_orig[norm_exp]

        logger.info(
            f"✅ StructureIndex: {len(self.entries)} entries | "
            f"{len(self._full_to_orig)} full-names | "
            f"{len(self._bigram_to_orig)} bigrams | "
            f"{len(self._trigram_to_orig)} trigrams | "
            f"{len(self._word_to_orig)} words | "
            f"{len(self._year_tokens)} year-tokens"
        )

    def has_full(self, norm: str) -> bool:
        return norm in self._full_to_orig

    def has_word(self, norm_word: str) -> bool:
        return norm_word in self._word_to_orig

    def has_bigram(self, norm_bigram: str) -> bool:
        return norm_bigram in self._bigram_to_orig

    def has_trigram(self, norm_trigram: str) -> bool:
        return norm_trigram in self._trigram_to_orig

    def is_year_token(self, norm: str) -> bool:
        return norm in self._year_tokens

    def lookup_full(self, norm: str) -> Optional[str]:
        return self._full_to_orig.get(norm)

    def lookup_word(self, norm_word: str) -> Optional[str]:
        return self._word_to_orig.get(norm_word)

    def word_ambiguity(self, norm_word: str) -> int:
        return self._word_ambiguity.get(norm_word, 0)

    def word_entry_set(self, norm_word: str) -> Set[str]:
        """FIX B: Return the set of entry names containing this word."""
        return self._word_to_entries.get(norm_word, set())

    def all_full_names_sorted_desc(self) -> List[str]:
        return sorted(self._full_to_orig.values(), key=len, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN EXTRACTOR  (Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

class TokenExtractor:
    """
    Extracts and weights classification tokens.

    Each token now carries an `_explicit` flag indicating whether it was
    matched as a full-name / bigram / trigram (True) or only as a single
    ambiguous word (False).

    Returns: Dict[token_str, {"weight": float, "explicit": bool}]
    But to maintain backward compatibility with callers expecting
    Dict[str, float], the public `.extract()` returns Dict[str, float]
    AND stores explicitness in self._last_explicit: Dict[str, bool].
    Callers that need explicitness call `.extract_with_explicit()`.
    """

    def __init__(self, structure: StructureIndex):
        self.structure = structure
        self._sorted_names = self.structure.all_full_names_sorted_desc()
        self._last_explicit: Dict[str, bool] = {}

    def extract(
        self,
        url:             str,
        title:           str,
        content_excerpt: str,
        department_hint: str,
        faculty_hint:    str,
        pdf_filenames:   Optional[List[str]] = None,
    ) -> Dict[str, float]:
        result, explicit = self._extract_internal(
            url, title, content_excerpt, department_hint, faculty_hint, pdf_filenames
        )
        self._last_explicit = explicit
        return result

    def extract_with_explicit(
        self,
        url:             str,
        title:           str,
        content_excerpt: str,
        department_hint: str,
        faculty_hint:    str,
        pdf_filenames:   Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        result, explicit = self._extract_internal(
            url, title, content_excerpt, department_hint, faculty_hint, pdf_filenames
        )
        self._last_explicit = explicit
        return result, explicit

    def _extract_internal(
        self,
        url:             str,
        title:           str,
        content_excerpt: str,
        department_hint: str,
        faculty_hint:    str,
        pdf_filenames:   Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        merged:   Dict[str, float] = {}
        explicit: Dict[str, bool]  = {}

        def _absorb(tw: Dict[str, float], te: Dict[str, bool]):
            for tok, w in tw.items():
                merged[tok] = merged.get(tok, 0.0) + w
                # A token is explicit if it was explicit in ANY source
                explicit[tok] = explicit.get(tok, False) or te.get(tok, False)

        url_path = _RE_URL_DOMAIN.sub("", url) if url else ""

        sources = [
            (url_path,        _SRC_WEIGHT["url"]),
            (title,           _SRC_WEIGHT["title"]),
            (department_hint, _SRC_WEIGHT["dept"]),
            (faculty_hint,    _SRC_WEIGHT["content"]),
            (content_excerpt, _SRC_WEIGHT["content"]),
        ]
        # FIX C: add PDF filenames as a signal source
        if pdf_filenames:
            for fname in pdf_filenames:
                stem = Path(fname).stem if fname else ""
                if stem:
                    sources.append((stem, _SRC_WEIGHT["pdf_name"]))

        for text, weight in sources:
            if not text:
                continue
            tw, te = self._extract_from_source(text, weight)
            _absorb(tw, te)

        # Level phrase detection (unchanged)
        all_text_lower = TextNormalizer.for_match(
            " ".join(filter(None, [url_path, title, content_excerpt, department_hint]))
        )
        if pdf_filenames:
            all_text_lower += " " + " ".join(
                TextNormalizer.for_match(Path(f).stem) for f in pdf_filenames if f
            )

        for phrase, level_tok in _LEVEL_PHRASE_MAP.items():
            if phrase in all_text_lower:
                norm_lt = level_tok.lower()
                if self.structure.is_year_token(norm_lt):
                    merged[norm_lt]  = merged.get(norm_lt, 0.0) + _SRC_WEIGHT["content"]
                    explicit[norm_lt] = True  # level tokens are always explicit

        for source_text, source_w in [(url_path, _SRC_WEIGHT["url"]),
                                       (title,    _SRC_WEIGHT["title"])]:
            if not source_text:
                continue
            for m in _RE_LEVEL_TOKEN.finditer(source_text):
                norm_lt = m.group(0).lower()
                if self.structure.is_year_token(norm_lt):
                    merged[norm_lt]   = merged.get(norm_lt, 0.0) + source_w
                    explicit[norm_lt] = True

        return merged, explicit

    def _extract_from_source(
        self, text: str, weight: int
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """
        Returns (weighted_tokens, explicit_flags).

        explicit = True  → matched as full-name, bigram, or trigram
        explicit = False → matched only as an ambiguous single word
        """
        result:   Dict[str, float] = {}
        explicit: Dict[str, bool]  = {}
        norm_text = TextNormalizer.for_match(text)
        matched_spans: List[Tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            for ms, me in matched_spans:
                if start < me and end > ms:
                    return True
            return False

        # ── Step A: greedy full-name matching ────────────────────────────────
        for entry_name in self._sorted_names:
            norm_entry = TextNormalizer.for_match(entry_name)
            if not norm_entry:
                continue
            start_pos = 0
            while True:
                idx = norm_text.find(norm_entry, start_pos)
                if idx == -1:
                    break
                end_pos = idx + len(norm_entry)
                before_ok = (idx == 0 or not norm_text[idx - 1].isalpha())
                after_ok  = (end_pos == len(norm_text) or not norm_text[end_pos].isalpha())
                if before_ok and after_ok and not _overlaps(idx, end_pos):
                    longer_follows = self._longer_name_follows(
                        norm_entry, norm_text, idx, end_pos
                    )
                    if longer_follows:
                        start_pos = end_pos
                        continue
                    matched_spans.append((idx, end_pos))
                    result[norm_entry]   = result.get(norm_entry, 0.0) + weight
                    explicit[norm_entry] = True   # full-name match → explicit
                start_pos = end_pos

        # ── Build word position map ──────────────────────────────────────────
        words_all = norm_text.split()
        word_positions: Dict[str, List[Tuple[int, int]]] = {}
        cursor = 0
        for w in words_all:
            idx = norm_text.find(w, cursor)
            if idx != -1:
                word_positions.setdefault(w, []).append((idx, idx + len(w)))
                cursor = idx + len(w)

        def _first_uncovered(w: str) -> Optional[Tuple[int, int]]:
            for pos in word_positions.get(w, []):
                if not _overlaps(*pos):
                    return pos
            return None

        stop_filtered = [w for w in words_all if len(w) >= 2 and w not in _STOP_WORDS]

        # ── Step B: bigram/trigram on uncovered words ─────────────────────────
        for n in (3, 2):
            for i in range(len(stop_filtered) - n + 1):
                phrase_words = stop_filtered[i:i+n]
                phrase = " ".join(phrase_words)
                if not (self.structure.has_bigram(phrase) or
                        self.structure.has_trigram(phrase)):
                    continue
                if phrase in result:
                    continue
                word_uncovered = [_first_uncovered(w) for w in phrase_words]
                if any(pos is None for pos in word_uncovered):
                    continue
                result[phrase]   = result.get(phrase, 0.0) + weight
                explicit[phrase] = True   # phrase match → explicit
                for pos in word_uncovered:
                    if pos:
                        matched_spans.append(pos)

        # ── Step C: single word — only for genuinely uncovered words ──────────
        for w in stop_filtered:
            if _first_uncovered(w) is None:
                continue
            if any(w in existing for existing in result):
                continue
            if len(w) < 4:
                continue
            if self.structure.has_word(w):
                ambiguity = self.structure.word_ambiguity(w)
                pts = weight
                if ambiguity > _AMBIGUITY_MAX_MATCHES:
                    pts *= _AMBIGUITY_PENALTY
                    # FIX D: ambiguous single word → NOT explicit
                    explicit[w] = explicit.get(w, False)  # keep False
                else:
                    explicit[w] = True  # unambiguous single word → explicit
                result[w] = result.get(w, 0.0) + pts

        # ── Step D: abbreviation expansion ────────────────────────────────────
        extra:   Dict[str, float] = {}
        extra_e: Dict[str, bool]  = {}
        for abbrev, expansion in _ABBREV_MAP.items():
            abbrev_lc = abbrev.lower()
            norm_exp  = TextNormalizer.for_match(expansion)
            if re.search(r'\b' + re.escape(abbrev_lc) + r'\b', norm_text):
                if self.structure.has_full(norm_exp):
                    extra[norm_exp]   = extra.get(norm_exp, 0.0) + weight
                    extra_e[norm_exp] = True
            if norm_exp in norm_text:
                if self.structure.has_full(abbrev_lc) or self.structure.has_word(abbrev_lc):
                    extra[abbrev_lc]   = extra.get(abbrev_lc, 0.0) + weight
                    extra_e[abbrev_lc] = True
        for tok, w in extra.items():
            result[tok]  = result.get(tok, 0.0) + w
            explicit[tok] = explicit.get(tok, False) or extra_e.get(tok, False)

        return result, explicit

    def _longer_name_follows(
        self, norm_entry: str, norm_text: str, match_start: int, match_end: int
    ) -> Optional[str]:
        window = norm_text[match_end:match_end + 60].lstrip()
        for longer_name in self._sorted_names:
            norm_longer = TextNormalizer.for_match(longer_name)
            if len(norm_longer) <= len(norm_entry):
                continue
            if not norm_longer.startswith(norm_entry):
                continue
            suffix = norm_longer[len(norm_entry):].lstrip()
            if not suffix:
                continue
            suffix_words = suffix.split()
            if not suffix_words:
                continue
            next_word = suffix_words[0]
            if len(next_word) < 2:
                continue
            if re.search(r'\b' + re.escape(next_word) + r'\b', window):
                return norm_longer
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SHARED NODE INDEX
# ══════════════════════════════════════════════════════════════════════════════

class SharedNodeIndex:
    def __init__(self):
        self.token_idx:   Dict[str, List[Dict]] = {}
        self.bigram_idx:  Dict[str, List[Dict]] = {}
        self.trigram_idx: Dict[str, List[Dict]] = {}
        self._built = False

    def build(self, candidates: Dict[str, List[Dict]]):
        if self._built:
            return
        for label, nodes in candidates.items():
            for node in nodes:
                self._index_node(node, label)
        self._built = True
        logger.info(
            f"✅ SharedNodeIndex built: "
            f"{len(self.token_idx)} token entries, "
            f"{len(self.bigram_idx)} bigram entries, "
            f"{len(self.trigram_idx)} trigram entries"
        )

    def _index_node(self, node: Dict, label: str):
        name = node.get("name", "")
        nid  = node.get("id",   "")
        if not name or not nid:
            return
        norm  = TextNormalizer.for_match(name)
        depth = _DEPTH_ORDER.get(label, 0)
        entry = {"id": nid, "name": name, "label": label, "_depth": depth}

        def _add(d: Dict, key: str):
            if key not in d:
                d[key] = []
            if not any(e["id"] == entry["id"] and e["label"] == entry["label"]
                       for e in d[key]):
                d[key].append(entry)

        _add(self.token_idx, norm)
        words = [w for w in norm.split() if len(w) >= 3 and w not in _STOP_WORDS]

        if label not in ("Year", "Level"):
            for w in words:
                _add(self.token_idx, w)
            for i in range(len(words) - 1):
                _add(self.bigram_idx, f"{words[i]} {words[i+1]}")
            for i in range(len(words) - 2):
                _add(self.trigram_idx, f"{words[i]} {words[i+1]} {words[i+2]}")

        for abbrev, expansion in _ABBREV_MAP.items():
            if TextNormalizer.for_match(expansion) == norm:
                _add(self.token_idx, abbrev.lower())

    def lookup_token(self, norm: str) -> List[Dict]:
        return list(self.token_idx.get(norm, []))

    def lookup_bigram(self, norm: str) -> List[Dict]:
        return list(self.bigram_idx.get(norm, []))

    def lookup_trigram(self, norm: str) -> List[Dict]:
        return list(self.trigram_idx.get(norm, []))


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jClient:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._node_cache:           Optional[Dict[str, List[Dict]]] = None
        self._ancestor_cache:       Dict[Tuple[str, str], bool]     = {}
        self._hierarchy_path_cache: Dict[Tuple[str, str], str]      = {}
        self._node_exists_cache:    Dict[Tuple[str, str], bool]     = {}
        # FIX A: cache parent ID for each node (for parent-collapse logic)
        self._parent_cache:         Dict[str, Optional[str]]        = {}
        self._init_constraints()

    def close(self):
        self.driver.close()

    def _init_constraints(self):
        with self.driver.session() as s:
            for label in ("URL", "Chunk"):
                try:
                    s.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    )
                except Exception:
                    pass

    def get_all_candidates(self) -> Dict[str, List[Dict]]:
        if self._node_cache is not None:
            return self._node_cache

        result: Dict[str, List[Dict]] = {
            "Faculty": [], "Department": [], "Level": [],
            "Category": [], "Program": [], "Specialization": [], "Year": [],
        }

        query = """
        CALL () {
            MATCH (n:Faculty)
            RETURN 'Faculty' AS lbl, n.id AS id, n.name AS name,
                   null AS parent, null AS parent_id, n.name AS faculty,
                   null AS level, null AS dept
        UNION ALL
            MATCH (f:Faculty)-[:HAS_DEPARTMENT]->(d:Department)
            RETURN 'Department' AS lbl, d.id AS id, d.name AS name,
                   f.name AS parent, f.id AS parent_id, f.name AS faculty,
                   null AS level, null AS dept
        UNION ALL
            MATCH (d:Department)-[:HAS_LEVEL]->(l:Level)
            RETURN 'Level' AS lbl, l.id AS id, l.name AS name,
                   d.name AS parent, d.id AS parent_id, null AS faculty,
                   null AS level, d.name AS dept
        UNION ALL
            MATCH (l:Level)-[:HAS_CATEGORY]->(c:Category)
            OPTIONAL MATCH (d2:Department)-[:HAS_LEVEL]->(l)
            RETURN 'Category' AS lbl, c.id AS id, c.name AS name,
                   l.name AS parent, l.id AS parent_id, null AS faculty,
                   l.name AS level, d2.name AS dept
        UNION ALL
            MATCH (l2:Level)-[:HAS_PROGRAM]->(p:Program)
            OPTIONAL MATCH (d3:Department)-[:HAS_LEVEL]->(l2)
            RETURN 'Program' AS lbl, p.id AS id, p.name AS name,
                   l2.name AS parent, l2.id AS parent_id, null AS faculty,
                   l2.name AS level, d3.name AS dept
        UNION ALL
            MATCH (sp:Specialization)
            OPTIONAL MATCH (c2:Category)-[:HAS_SPECIALIZATION]->(sp)
            OPTIONAL MATCH (lsp:Level)-[:HAS_SPECIALIZATION]->(sp)
            OPTIONAL MATCH (psp:Program)-[:HAS_SPECIALIZATION]->(sp)
            RETURN 'Specialization' AS lbl, sp.id AS id, sp.name AS name,
                   COALESCE(c2.name, lsp.name, psp.name) AS parent,
                   COALESCE(c2.id, lsp.id, psp.id) AS parent_id,
                   null AS faculty, null AS level, null AS dept
        UNION ALL
            MATCH (y:Year)
            OPTIONAL MATCH (spy:Specialization)-[:HAS_YEAR]->(y)
            OPTIONAL MATCH (py:Program)-[:HAS_YEAR]->(y)
            RETURN 'Year' AS lbl, y.id AS id, y.name AS name,
                   COALESCE(spy.name, py.name) AS parent,
                   COALESCE(spy.id, py.id) AS parent_id,
                   null AS faculty, null AS level, null AS dept
        }
        RETURN lbl, id, name, parent, parent_id, faculty, level, dept
        """
        with self.driver.session() as s:
            for r in s.run(query):
                lbl = r["lbl"]
                if not r["name"] or lbl not in result:
                    continue
                result[lbl].append({
                    "id":        r["id"],
                    "name":      r["name"],
                    "parent":    r["parent"],
                    "parent_id": r["parent_id"],
                    "faculty":   r["faculty"],
                    "level":     r["level"],
                    "dept":      r["dept"],
                    "label":     lbl,
                })
                # populate parent cache
                if r["parent_id"]:
                    self._parent_cache[r["id"]] = r["parent_id"]

        self._node_cache = result
        total = sum(len(v) for v in result.values())
        logger.info(f"✅ Neo4j candidates loaded: {total} nodes (cached)")
        return result

    def get_parent_id(self, node_id: str) -> Optional[str]:
        """Return the direct parent's ID, or None if root."""
        if node_id in self._parent_cache:
            return self._parent_cache[node_id]
        # Lazy load via graph query if not in cache yet
        with self.driver.session() as s:
            r = s.run(
                f"""
                MATCH (p)-[:{_HIERARCHY_RELS_NO_GENERAL}]->(n {{id: $id}})
                RETURN p.id AS pid LIMIT 1
                """,
                id=node_id,
            ).single()
            pid = r["pid"] if r else None
        self._parent_cache[node_id] = pid
        return pid

    def batch_ancestor_check(
        self, pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], bool]:
        uncached = [p for p in pairs if p not in self._ancestor_cache]
        if not uncached:
            return {p: self._ancestor_cache[p] for p in pairs}

        query = f"""
        UNWIND $pairs AS pair
        MATCH (a {{id: pair[0]}})
        MATCH (d {{id: pair[1]}})
        RETURN pair[0] AS ancestor_id,
               pair[1] AS descendant_id,
               CASE
                 WHEN shortestPath(
                   (a)-[:{_HIERARCHY_RELS_NO_GENERAL}*1..8]->(d)
                 ) IS NOT NULL THEN true
                 ELSE false
               END AS is_anc
        """
        try:
            with self.driver.session() as s:
                for r in s.run(query, pairs=[[a, d] for a, d in uncached]):
                    key = (r["ancestor_id"], r["descendant_id"])
                    self._ancestor_cache[key] = bool(r["is_anc"])
        except Exception as exc:
            logger.warning(f"  shortestPath failed, fallback: {exc}")
            fallback = f"""
            UNWIND $pairs AS pair
            MATCH (a {{id: pair[0]}})
            MATCH (d {{id: pair[1]}})
            OPTIONAL MATCH path = (a)-[:{_HIERARCHY_RELS_NO_GENERAL}*1..8]->(d)
            RETURN pair[0] AS ancestor_id, pair[1] AS descendant_id,
                   count(path) > 0 AS is_anc
            """
            with self.driver.session() as s:
                for r in s.run(fallback, pairs=[[a, d] for a, d in uncached]):
                    key = (r["ancestor_id"], r["descendant_id"])
                    self._ancestor_cache[key] = bool(r["is_anc"])

        for p in uncached:
            if p not in self._ancestor_cache:
                self._ancestor_cache[p] = False

        return {p: self._ancestor_cache[p] for p in pairs}

    def is_ancestor_of(self, ancestor_id: str, descendant_id: str) -> bool:
        key = (ancestor_id, descendant_id)
        if key not in self._ancestor_cache:
            self.batch_ancestor_check([key])
        return self._ancestor_cache.get(key, False)

    def node_exists(self, label: str, node_id: str) -> bool:
        actual = _LABEL_MAP.get(label, label)
        key = (actual, node_id)
        if key in self._node_exists_cache:
            return self._node_exists_cache[key]
        with self.driver.session() as s:
            r = s.run(
                f"MATCH (n:{actual} {{id: $id}}) RETURN count(n) AS cnt",
                id=node_id,
            ).single()
            exists = bool(r and r["cnt"] > 0)
        self._node_exists_cache[key] = exists
        return exists

    def get_node_by_id(self, label: str, node_id: str) -> Optional[Dict]:
        actual = _LABEL_MAP.get(label, label)
        with self.driver.session() as s:
            for r in s.run(
                f"MATCH (n:{actual} {{id: $id}}) "
                f"RETURN n.id AS id, n.name AS name LIMIT 1",
                id=node_id,
            ):
                return {"id": r["id"], "name": r["name"]}
        return None

    def get_hierarchy_path(self, label: str, node_id: str) -> str:
        cache_key = (label, node_id)
        if cache_key in self._hierarchy_path_cache:
            return self._hierarchy_path_cache[cache_key]
        actual = _LABEL_MAP.get(label, label)
        with self.driver.session() as s:
            r = s.run(
                f"""
                MATCH path = (root)-[:{_HIERARCHY_RELS_NO_GENERAL}*0..8]->(n {{id: $id}})
                WHERE $lbl IN labels(n)
                WITH path ORDER BY length(path) DESC LIMIT 1
                RETURN [node IN nodes(path) | node.name] AS names
                """,
                id=node_id, lbl=actual,
            ).single()
            if r and r["names"]:
                result = " > ".join(n for n in r["names"] if n)
                self._hierarchy_path_cache[cache_key] = result
                return result
        fallback = label
        self._hierarchy_path_cache[cache_key] = fallback
        return fallback

    def validate_classification(self, label: str, node_id: str) -> Tuple[bool, str]:
        actual = _LABEL_MAP.get(label, label)
        if actual == "General":
            return True, "general fallback"
        if not self.node_exists(actual, node_id):
            return False, f"Node {actual}:{node_id} does not exist"
        return True, "exists"

    def get_year_nodes_under(
        self, parent_label: str, parent_id: str, year_name: str
    ) -> List[Dict]:
        actual  = _LABEL_MAP.get(parent_label, parent_label)
        norm_yn = TextNormalizer.for_match(year_name)
        with self.driver.session() as s:
            rows = s.run(
                f"""
                MATCH (p:{actual} {{id: $pid}})-[:{_HIERARCHY_RELS_NO_GENERAL}*1..6]->(y:Year)
                WHERE toLower(y.name) = $yn OR toLower(y.name) = $ynn
                OPTIONAL MATCH (sp:Specialization)-[:HAS_YEAR]->(y)
                OPTIONAL MATCH (pg:Program)-[:HAS_YEAR]->(y)
                RETURN DISTINCT y.id AS id, y.name AS name,
                       COALESCE(sp.name, pg.name) AS parent_name
                """,
                pid=parent_id, yn=year_name.lower(), ynn=norm_yn,
            )
            result = []
            for r in rows:
                hp = self.get_hierarchy_path("Year", r["id"])
                result.append({
                    "id":             r["id"],
                    "name":           r["name"],
                    "label":          "Year",
                    "parent_name":    r["parent_name"],
                    "hierarchy_path": hp,
                })
            return result

    def get_general_under_faculty(
        self, faculty_id: str, faculty_name: str
    ) -> Optional[Dict]:
        with self.driver.session() as s:
            r = s.run(
                "MATCH (f:Faculty {id: $fid})-[:HAS_GENERAL]->(g:General) "
                "RETURN g.id AS id, g.name AS name LIMIT 1",
                fid=faculty_id,
            ).single()
            if r:
                return {
                    "id":             r["id"],
                    "name":           r["name"],
                    "label":          "General",
                    "hierarchy_path": f"{faculty_name} > General",
                }
        with self.driver.session() as s:
            r = s.run(
                "MATCH (g:General) RETURN g.id AS id, g.name AS name LIMIT 1"
            ).single()
            if r:
                return {
                    "id":             r["id"],
                    "name":           r["name"],
                    "label":          "General",
                    "hierarchy_path": f"{faculty_name} > General",
                }
        return None

    def get_general_id(self) -> str:
        with self.driver.session() as s:
            r = s.run(
                "MATCH (g:General) RETURN g.id AS id LIMIT 1"
            ).single()
            return r["id"] if r else "general"

    def get_ancestor_ids(self, node_id: str) -> Set[str]:
        with self.driver.session() as s:
            rows = s.run(
                f"""
                MATCH (anc)-[:{_HIERARCHY_RELS_NO_GENERAL}*1..8]->(n {{id: $id}})
                RETURN DISTINCT anc.id AS anc_id
                """,
                id=node_id,
            )
            return {r["anc_id"] for r in rows if r["anc_id"]}

    # ── FIX B helper: find the common parent of a set of sibling nodes ────

    def get_common_parent(self, node_ids: List[str]) -> Optional[Dict]:
        """
        FIX B: Given a list of node IDs that are all children of the same parent,
        return that parent node dict. Used when ambiguous single words match
        multiple siblings — we link to the parent instead.
        """
        if not node_ids:
            return None
        parent_ids: List[Optional[str]] = [self.get_parent_id(nid) for nid in node_ids]
        # All must have the same non-None parent
        if not all(pid is not None for pid in parent_ids):
            return None
        unique_parents = set(p for p in parent_ids if p)
        if len(unique_parents) != 1:
            return None
        parent_id = unique_parents.pop()
        # Find the parent node in cache
        candidates = self.get_all_candidates()
        for label, nodes in candidates.items():
            for node in nodes:
                if node["id"] == parent_id:
                    return {**node, "label": label}
        return None

    # ── Write operations ─────────────────────────────────────────────────

    def upsert_url_node(
        self,
        url_id:                str,
        url:                   str,
        title:                 str,
        source_type:           str,
        target_label:          str,
        target_id:             str,
        hierarchy_path:        str   = "",
        classification_method: str   = "none",
        confidence:            float = 0.0,
        parent_url_id:         Optional[str] = None,
    ):
        actual = _LABEL_MAP.get(target_label, target_label)
        with self.driver.session() as s:
            s.run(
                "MERGE (u:URL {id: $id}) "
                "SET u.url=$url, u.title=$title, u.source_type=$st, "
                "    u.hierarchy_path=$hp, u.classification_method=$cm, u.confidence=$conf",
                id=url_id, url=url, title=title, st=source_type,
                hp=hierarchy_path, cm=classification_method, conf=confidence,
            )
            if parent_url_id is None:
                s.run(
                    f"MATCH (n:{actual} {{id: $nid}}) "
                    f"MATCH (u:URL {{id: $uid}}) "
                    f"MERGE (n)-[:{_REL_HAS_CONTENT}]->(u)",
                    nid=target_id, uid=url_id,
                )
            else:
                s.run(
                    f"MATCH (p:URL {{id: $pid}}) "
                    f"MATCH (f:URL {{id: $fid}}) "
                    f"MERGE (p)-[:{_REL_HAS_FILE}]->(f)",
                    pid=parent_url_id, fid=url_id,
                )

    def link_url_to_extra_targets(self, url_id: str, targets: List[Dict]):
        with self.driver.session() as s:
            for t in targets:
                actual = _LABEL_MAP.get(t["label"], t["label"])
                if not self.node_exists(actual, t["id"]):
                    continue
                s.run(
                    f"MATCH (n:{actual} {{id: $nid}}) "
                    f"MATCH (u:URL {{id: $uid}}) "
                    f"MERGE (n)-[:{_REL_HAS_CONTENT}]->(u)",
                    nid=t["id"], uid=url_id,
                )

    def create_chunks(self, url_id: str, chunks: List[Dict], classification: Dict):
        label  = classification.get("label",          "General")
        nid    = classification.get("id",             "general")
        hp     = classification.get("hierarchy_path", "")
        method = classification.get("match_method",   "none")
        conf   = float(classification.get("confidence", 0.0))

        with self.driver.session() as s:
            for i in range(0, len(chunks), NEO4J_BATCH):
                batch = chunks[i:i+NEO4J_BATCH]
                s.run(
                    """
                    UNWIND $batch AS ch
                    MERGE (c:Chunk {id: ch.id})
                    SET c.text                 = ch.text,
                        c.chunk_index          = ch.chunk_index,
                        c.token_count          = ch.token_count,
                        c.language             = ch.language,
                        c.classification_id    = ch.cid,
                        c.classification_label = ch.clabel,
                        c.hierarchy_path       = ch.hp,
                        c.match_method         = ch.method,
                        c.confidence           = ch.conf,
                        c.embedding            = ch.embedding
                    WITH c, ch
                    MATCH (u:URL {id: $uid})
                    MERGE (u)-[:HAS_CHUNK {order: ch.chunk_index}]->(c)
                    """,
                    batch=[{
                        "id":          ck["id"],
                        "text":        ck["text"][:4000],
                        "chunk_index": ck["chunk_index"],
                        "token_count": ck.get("token_count", 0),
                        "language":    ck.get("language", ""),
                        "cid":         nid,
                        "clabel":      label,
                        "hp":          hp,
                        "method":      method,
                        "conf":        conf,
                        "embedding": (
                            ck["embedding"]
                            if isinstance(ck.get("embedding"), list)
                               and len(ck["embedding"]) == EMBED_DIM
                            else []
                        ),
                    } for ck in batch],
                    uid=url_id,
                )
            if len(chunks) > 1:
                s.run(
                    """
                    MATCH (u:URL {id: $uid})-[:HAS_CHUNK]->(c:Chunk)
                    WITH c ORDER BY c.chunk_index
                    WITH collect(c) AS ordered
                    UNWIND range(0, size(ordered)-2) AS i
                    WITH ordered[i] AS cur, ordered[i+1] AS nxt
                    MERGE (cur)-[:NEXT_CHUNK]->(nxt)
                    """,
                    uid=url_id,
                )


# ══════════════════════════════════════════════════════════════════════════════
# NODE MATCHER  — FIX A + FIX B + FIX D
# ══════════════════════════════════════════════════════════════════════════════

class NodeMatcher:
    """
    Scores tokens against SharedNodeIndex.

    KEY CHANGE (FIX A + FIX D):
    Each qualified node is now tagged with `_explicit: bool`.
    
    A node is EXPLICIT if it was reached via:
      - An exact full-name token match (the token IS the normalized node name)
      - A bigram/trigram match flagged explicit by TokenExtractor
      - An abbreviation expansion that maps to this node
    
    A node is NON-EXPLICIT if it was reached ONLY via ambiguous single-word
    matches (e.g. "physique" matching both Department:Physique AND
    Specialization:PEER, Specialization:PM, etc.)

    FIX B — PARENT COLLAPSE:
    After scoring, if a set of sibling nodes are all non-explicit and they all
    share the same parent, replace them with that parent node (which IS the
    level that was explicitly mentioned — e.g. "physique" → Department:Physique,
    NOT all its specialization children).
    """

    def __init__(self, neo4j: Neo4jClient, shared_idx: SharedNodeIndex):
        self.neo4j = neo4j
        self.idx   = shared_idx

    def match(
        self,
        weighted_tokens: Dict[str, float],
        explicit_flags:  Optional[Dict[str, bool]] = None,
    ) -> List[Dict]:
        if explicit_flags is None:
            explicit_flags = {}

        scored: Dict[str, Tuple[Dict, float, bool]] = {}
        # scored[key] = (entry, total_score, is_explicit)

        def _accum(entry: Dict, pts: float, is_exp: bool):
            key = f"{entry['id']}|{entry['label']}"
            if key in scored:
                old_entry, old_score, old_exp = scored[key]
                scored[key] = (old_entry, old_score + pts, old_exp or is_exp)
            else:
                scored[key] = (entry, pts, is_exp)

        token_list = sorted(
            weighted_tokens.keys(),
            key=lambda t: weighted_tokens[t],
            reverse=True,
        )

        for tok in token_list:
            sw      = weighted_tokens[tok]
            is_exp  = explicit_flags.get(tok, False)
            matches = self.idx.lookup_token(tok)
            if not matches:
                continue
            is_ambiguous = len(matches) > 1
            for entry in matches:
                norm_name = TextNormalizer.for_match(entry["name"])
                depth     = entry.get("_depth", 0)
                if norm_name == tok:
                    pts      = 20.0 * sw
                    node_exp = True   # exact full-name token match → always explicit
                else:
                    pts      = (1.0 + depth * 0.2) * sw
                    if is_ambiguous:
                        pts *= 0.5
                    # The node is explicit only if the token itself was explicit
                    node_exp = is_exp
                _accum(entry, pts, node_exp)

        for n in (3, 2):
            for i in range(len(token_list) - n + 1):
                gram = " ".join(token_list[i:i+n])
                if n == 3:
                    matches = self.idx.lookup_trigram(gram)
                else:
                    matches = self.idx.lookup_bigram(gram)
                if not matches:
                    continue
                gram_w  = [weighted_tokens.get(token_list[i+j], 1.0) for j in range(n)]
                avg_sw  = sum(gram_w) / n
                gram_exp = any(explicit_flags.get(token_list[i+j], False) for j in range(n))
                for entry in matches:
                    norm_name = TextNormalizer.for_match(entry["name"])
                    depth     = entry.get("_depth", 0)
                    if norm_name == gram:
                        pts      = 20.0 * avg_sw
                        node_exp = True
                    else:
                        pts      = (2.0 * n + depth * 0.2) * avg_sw
                        node_exp = gram_exp
                    _accum(entry, pts, node_exp)

        # Filter by threshold + existence
        qualified = []
        for _, (entry, total_score, is_exp) in scored.items():
            if total_score < _MIN_SCORE:
                continue
            if not self.neo4j.node_exists(entry["label"], entry["id"]):
                continue
            qualified.append({**entry, "_score": total_score, "_explicit": is_exp})

        qualified.sort(key=lambda e: e["_score"], reverse=True)

        # FIX B: replace groups of non-explicit siblings with their common parent
        qualified = self._collapse_non_explicit_siblings(qualified)

        # FIX 8: filter orphan year nodes (unchanged)
        qualified = self._filter_orphan_year_nodes(qualified)

        return qualified

    def _collapse_non_explicit_siblings(self, qualified: List[Dict]) -> List[Dict]:
        """
        FIX B: If multiple non-explicit nodes share the same parent, replace
        them with that parent (if the parent isn't already in the list).

        Example:
          non-explicit: [Specialization:PEER, Specialization:PM, Specialization:PF]
          all have parent: Department:Physique
          → replace all three with Department:Physique (explicit=True, since
            the department was the level actually mentioned)

        This is the core of "link to what was mentioned, not its children."
        """
        if not qualified:
            return qualified

        non_explicit = [n for n in qualified if not n.get("_explicit", True)]
        explicit     = [n for n in qualified if n.get("_explicit", True)]

        if not non_explicit:
            return qualified

        # Group non-explicit nodes by their parent ID
        parent_groups: Dict[str, List[Dict]] = {}
        ungroupable: List[Dict] = []

        for node in non_explicit:
            pid = self.neo4j.get_parent_id(node["id"])
            if pid:
                parent_groups.setdefault(pid, []).append(node)
            else:
                ungroupable.append(node)

        result = list(explicit)

        for parent_id, siblings in parent_groups.items():
            # Check if parent is already explicitly in the list
            already_present = any(
                n["id"] == parent_id for n in result
            )
            if already_present:
                # Parent already there — just discard the non-explicit siblings
                logger.debug(
                    f"    [FIX B] {len(siblings)} non-explicit siblings discarded "
                    f"(parent already present): "
                    + ", ".join(s["name"] for s in siblings)
                )
                continue

            # Find the parent node
            parent_node = self.neo4j.get_common_parent([s["id"] for s in siblings])
            if parent_node:
                avg_score = sum(s["_score"] for s in siblings) / len(siblings)
                result.append({
                    **parent_node,
                    "_score":    avg_score,
                    "_explicit": True,   # parent is what was actually mentioned
                })
                logger.debug(
                    f"    [FIX B] {len(siblings)} non-explicit siblings → "
                    f"parent {parent_node['label']}:{parent_node['name']}: "
                    + ", ".join(s["name"] for s in siblings)
                )
            else:
                # Can't find common parent → keep siblings with heavy penalty
                # (they will likely lose to other explicit nodes in PARCOURS)
                for sib in siblings:
                    result.append({**sib, "_score": sib["_score"] * 0.1})
                logger.debug(
                    f"    [FIX B] {len(siblings)} non-explicit nodes with no "
                    f"common parent → kept with ×0.1 penalty"
                )

        # Add ungroupable non-explicit nodes with penalty
        for node in ungroupable:
            result.append({**node, "_score": node["_score"] * 0.1})

        result.sort(key=lambda e: e["_score"], reverse=True)
        return result

    def _filter_orphan_year_nodes(self, qualified: List[Dict]) -> List[Dict]:
        """FIX 8: unchanged from v18."""
        non_year  = [n for n in qualified if n.get("label") not in ("Year", "Level")]
        year_nodes = [n for n in qualified if n.get("label") in ("Year", "Level")]

        if not year_nodes:
            return qualified
        if not non_year:
            if len(year_nodes) == 1:
                return year_nodes
            else:
                logger.debug(
                    f"    [FIX 8] {len(year_nodes)} Year nodes but NO parent → all dropped"
                )
                return []

        pairs = [(ny["id"], y["id"]) for ny in non_year for y in year_nodes]
        ancestor_results = self.neo4j.batch_ancestor_check(pairs)

        kept_years = []
        for y in year_nodes:
            has_ancestor = any(
                ancestor_results.get((ny["id"], y["id"]), False)
                for ny in non_year
            )
            if has_ancestor:
                kept_years.append(y)
            else:
                logger.debug(
                    f"    [FIX 8] Year '{y['name']}' dropped — no qualified ancestor"
                )

        return non_year + kept_years


# ══════════════════════════════════════════════════════════════════════════════
# PARCOURS + SPOTLIGHT  — FIX A
# ══════════════════════════════════════════════════════════════════════════════

class ParcoursResolver:
    """
    PARCOURS with the NO-DESCENT rule (FIX A):

    RULE: Only nodes that are EXPLICITLY mentioned (tagged _explicit=True by
    NodeMatcher) are fed into PARCOURS. Non-explicit nodes have already been
    either collapsed to their parent (FIX B) or penalized.

    The lit-subtree logic (FIX 3+5 from v18) is preserved: if two explicit
    nodes share a branch (root is ancestor of child), we descend.
    But we NEVER descend to a child that wasn't explicitly mentioned.

    This means:
    - "physique" alone → Department:Physique (not its specialization children)
    - "physique energetique" → whatever node matches that phrase directly
    - "physique energetique M1" → Specialization:PEER + Year:M1 under PEER
    """

    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    def resolve(self, url: str, signals: Optional[Dict] = None) -> Optional[Dict]:
        url_path = _RE_URL_DOMAIN.sub("", url) if url else ""
        if not url_path:
            return None
    
        # ===== FIX v21: Try DIRECT match first (bypasses NodeMatcher) =====
        direct_result = self._try_direct_match(url_path, signals)
        if direct_result:
            logger.debug(f"  [L1/direct] → {self._fmt(direct_result)}")
            return direct_result
        # ================================================================
    
        # Fallback: use token extraction + NodeMatcher
        weighted, explicit = self._extract_from_url(url_path)
    
        if signals:
            # Process PDF filenames
            pdf_fnames = signals.get("pdf_filenames", [])
            for fname in pdf_fnames:
                fw, fe = self._extract_from_url(fname)
                for tok, w in fw.items():
                    weighted[tok] = weighted.get(tok, 0.0) + w * (_SRC_WEIGHT["pdf_name"] / _SRC_WEIGHT["url"])
                    explicit[tok] = explicit.get(tok, False) or fe.get(tok, False)
    
            for hint_key, hint_w in (
                ("department_hint", _SRC_WEIGHT["dept"]),
                ("faculty_hint",    _SRC_WEIGHT["content"]),
            ):
                hint_text = signals.get(hint_key, "") or ""
                if hint_text:
                    for tok, w in self._greedy_extract(hint_text, hint_w):
                        weighted[tok] = weighted.get(tok, 0.0) + w
                        explicit[tok] = explicit.get(tok, False) or True
    
        if not weighted:
            return None
    
        qualified = self.matcher.match(weighted, explicit_flags=explicit)
        if not qualified:
            return None
    
        if _all_faculty_only(qualified):
            return None
    
        targets = self.resolver.resolve(qualified)
        if not targets:
            return None
    
        return self._build_result(targets, "url")
    
    def _handle_single(self, node: Dict) -> List[Dict]:
        label = node.get("label", "")
        if label == "Faculty":
            general = self.neo4j.get_general_under_faculty(node["id"], node["name"])
            if general:
                return [general]
        hp = self.neo4j.get_hierarchy_path(label, node["id"])
        return [{**node, "hierarchy_path": hp}]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTOR  — FIX C
# ══════════════════════════════════════════════════════════════════════════════

class SignalExtractor:
    """
    FIX C: Now collects PDF filenames from ext_docs and passes them as an
    additional signal to the token extractor. PDF filenames like
    "M1_Energ_tiques.pdf" carry strong classification signals.
    """

    def extract(
        self,
        url:             str,
        title:           str,
        content:         str,
        file_path:       str,
        faculty_hint:    str,
        department_hint: str,
        ext_docs:        Optional[List[Dict]] = None,   # FIX C
    ) -> Dict[str, Any]:
        excerpt = content[:CONTENT_SIGNAL_WINDOW].strip() if content else ""

        # FIX C: extract PDF filenames from ext_docs
        pdf_filenames: List[str] = []
        if ext_docs:
            for doc in ext_docs:
                fname = doc.get("title", "") or ""
                lfile = doc.get("local_file", "") or ""
                url_f = doc.get("url", "") or ""
                for candidate in [lfile, url_f, fname]:
                    if candidate:
                        stem = Path(candidate).stem if "." in Path(candidate).name else candidate
                        clean = re.sub(r"[_\-\s]+", " ", stem).strip()
                        if clean:
                            pdf_filenames.append(clean)

        combined = " ".join(filter(None, [
            url, title, excerpt,
            Path(file_path).stem if file_path else "",
            faculty_hint, department_hint,
            " ".join(pdf_filenames),
        ]))

        level_tokens: List[str] = []
        text_low = TextNormalizer.for_match(combined)
        for m in _RE_LEVEL_TOKEN.finditer(combined):
            tok = m.group(0).upper()
            if tok not in level_tokens:
                level_tokens.append(tok)
        for phrase, abbr in _LEVEL_PHRASE_MAP.items():
            if phrase in text_low and abbr not in level_tokens:
                level_tokens.append(abbr)

        found_abbrevs: Dict[str, str] = {}
        for abbrev, expansion in _ABBREV_MAP.items():
            if re.search(r'\b' + re.escape(abbrev.lower()) + r'\b', text_low):
                found_abbrevs[abbrev] = expansion

        return {
            "url":              url,
            "title":            title,
            "content_excerpt":  excerpt,
            "combined_signal":  combined,
            "file_path":        file_path,
            "faculty_hint":     faculty_hint,
            "department_hint":  department_hint,
            "level_tokens":     level_tokens,
            "found_abbrevs":    found_abbrevs,
            "pdf_filenames":    pdf_filenames,   # FIX C
        }
# ══════════════════════════════════════════════════════════════════════════════
# L1: URL RESOLVER — FIX v21: DEPTH-FIRST DIRECT MATCHING
# ══════════════════════════════════════════════════════════════════════════════

class URLResolver:
    """
    L1: URL-based classification.

    FIX v21 — DEPTH-FIRST DIRECT MATCHING:
    ────────────────────────────────────────
    Core philosophy: text-to-base matching from deepest to shallowest.
    
    1. Try DIRECT match first: search ALL graph nodes ordered by depth
       (Specialization → Program → Category → Department → Faculty).
    2. For each node, check if the URL segment partially matches the node's
       name (with prefix stripping: "Licence en", "Master", etc.).
    3. First deep match with score >= 0.5 → link DIRECTLY to that node.
    4. Bypasses NodeMatcher entirely when a direct match is found.
    5. Removes ancestor nodes when a deeper child is matched.
    6. Falls back to token extraction + NodeMatcher only if no direct match.
    """

    def __init__(
        self,
        neo4j:      "Neo4jClient",
        shared_idx: "SharedNodeIndex",
        resolver:   "ParcoursResolver",
        matcher:    "NodeMatcher",
        structure:  "StructureIndex",
    ):
        self.neo4j     = neo4j
        self.idx       = shared_idx
        self.resolver  = resolver
        self.matcher   = matcher
        self.structure = structure
        self._sorted_names: List[str] = []
        self._nodes_by_depth: Optional[List[Dict]] = None

    # ══════════════════════════════════════════════════════════════════════
    # FIX v21: NEW METHODS
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_nodes_by_depth(self):
        """
        Build list of all graph nodes ordered by depth (deepest first).
        Also creates a stripped version of each name for better matching.
        """
        if self._nodes_by_depth is not None:
            return

        candidates = self.neo4j.get_all_candidates()
        nodes: List[Dict] = []

        _prefixes_to_strip = [
            "licence en ", "licence de ", "licence ",
            "master en ", "master de ", "master ",
            "doctorat en ", "doctorat de ", "doctorat ",
            "ingenieur en ", "ingenieur de ", "ingenieur ",
            "ingénieur en ", "ingénieur de ", "ingénieur ",
            "tronc commun ", "1ère année ", "1ere annee ",
        ]

        for label, node_list in candidates.items():
            for node in node_list:
                depth = _DEPTH_ORDER.get(label, 0)
                name = node.get("name", "")
                name_stripped = name
                for prefix in _prefixes_to_strip:
                    if name_stripped.lower().startswith(prefix):
                        name_stripped = name_stripped[len(prefix):]
                        break
                nodes.append({
                    "id":             node["id"],
                    "name":           name,
                    "name_stripped":  name_stripped,
                    "label":          label,
                    "_depth":         depth,
                })

        # Sort deepest first, then by name length descending (longer names first)
        nodes.sort(key=lambda n: (-n["_depth"], -len(n["name"])))
        self._nodes_by_depth = nodes
        logger.debug(
            f"    [URL v21] Depth-ordered index built: {len(nodes)} nodes "
            f"(deepest depth={nodes[0]['_depth'] if nodes else 'N/A'})"
        )

    def _find_best_match_in_segment(self, norm_seg: str) -> Optional[Dict]:
        """
        FIX v21: Search ALL nodes by depth (deepest first) for a partial match.

        Matching criteria:
        - Tries both original and stripped node names.
        - Calculates overlap score between segment words and node words.
        - Applies depth bonus (deeper nodes preferred).
        - Returns the single best matching node, or None.

        Returns a dict with node info + "_match_score" if found.
        """
        self._ensure_nodes_by_depth()
        if not self._nodes_by_depth:
            return None

        seg_words = set(norm_seg.split())
        seg_words_clean = {w for w in seg_words if w not in _STOP_WORDS and len(w) >= 2}
        if not seg_words_clean:
            seg_words_clean = seg_words

        best_match = None
        best_score = 0.0

        for node in self._nodes_by_depth:
            # Try stripped name first, then original name
            for name_to_try in (node.get("name_stripped", node["name"]), node["name"]):
                node_norm = TextNormalizer.for_match(name_to_try)
                if not node_norm:
                    continue

                node_words = set(node_norm.split())
                node_words_clean = {
                    w for w in node_words
                    if w not in _STOP_WORDS and len(w) >= 2
                }
                if not node_words_clean:
                    node_words_clean = node_words

                common = seg_words_clean & node_words_clean
                if not common:
                    continue

                # Score 1: How much of the node's name is covered by the segment?
                node_coverage = len(common) / len(node_words_clean) if node_words_clean else 0

                # Score 2: How much of the segment is covered by the node's name?
                seg_coverage = len(common) / len(seg_words_clean) if seg_words_clean else 0

                # Combined score (node coverage weighted more)
                score = (node_coverage * 0.6) + (seg_coverage * 0.4)

                # Exact match bonuses
                if node_norm == norm_seg:
                    score = 1.0
                elif TextNormalizer.for_match(
                    node.get("name_stripped", node["name"])
                ) == norm_seg:
                    score = 0.95

                # Bonus for deeper nodes
                depth_bonus = node.get("_depth", 0) * 0.05
                adjusted_score = min(score + depth_bonus, 1.0)

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match = dict(node)
                    best_match["_match_score"] = adjusted_score

        # Only return if score meets minimum threshold
        if best_match and best_score >= 0.5:
            logger.debug(
                f"    [URL v21 best] '{norm_seg}' → "
                f"{best_match['label']}:{best_match['name']} "
                f"(score={best_score:.2f}, depth={best_match['_depth']})"
            )
            return best_match

        return None

    def _filter_deepest_only(self, nodes: List[Dict]) -> List[Dict]:
        """
        FIX v21: Keep only the deepest nodes.
        
        If a node is an ancestor of another matched node, remove it.
        This is the core of "don't go back up to parent when child is found."
        
        Example:
          matched: [Department:Physique, Specialization:PEER]
          → keep only Specialization:PEER (Department:Physique is its ancestor)
        """
        if len(nodes) <= 1:
            return nodes

        # Sort deepest first
        sorted_nodes = sorted(
            nodes,
            key=lambda n: _DEPTH_ORDER.get(n.get("label", ""), 0),
            reverse=True,
        )

        # Batch-check ancestor relationships
        ids = [n["id"] for n in sorted_nodes]
        pairs = [(a, d) for a in ids for d in ids if a != d]
        if pairs:
            self.neo4j.batch_ancestor_check(pairs)

        kept = []
        for i, node in enumerate(sorted_nodes):
            is_ancestor_of_another = False
            for j, other in enumerate(sorted_nodes):
                if i == j:
                    continue
                # Is this node an ancestor of another matched (deeper) node?
                if self.neo4j.is_ancestor_of(node["id"], other["id"]):
                    is_ancestor_of_another = True
                    logger.debug(
                        f"    [L1 filter] Removing '{node['label']}:{node['name']}' "
                        f"— ancestor of '{other['label']}:{other['name']}'"
                    )
                    break
            if not is_ancestor_of_another:
                kept.append(node)

        return kept

    def _try_direct_match(
        self, url_path: str, signals: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        FIX v21: Try to find a SINGLE direct match from the URL without going
        through NodeMatcher. If found, build the result directly.

        This prevents the problem where finding "physique energetique" also
        pulls in all other "physique" siblings through the index lookup.

        Process:
        1. Split URL into segments.
        2. For each segment, try _find_best_match_in_segment.
        3. Collect all matched nodes + year tokens.
        4. Filter to keep only deepest nodes (remove ancestors).
        5. Build final targets with year nodes under matched nodes.
        6. Return result directly (bypasses NodeMatcher and PARCOURS).
        """
        self._ensure_nodes_by_depth()

        raw_path = url_path.split("?")[0].split("#")[0]
        segments = [s for s in raw_path.split("/") if s]

        # Words that should NEVER be considered as standalone matches
        _FORBIDDEN: frozenset = frozenset({
            "licence", "master", "doctorat", "ingenieur", "ingénieur",
            "l1", "l2", "l3", "m1", "m2",
            "ing1", "ing2", "ing3", "ing4", "ing5",
            "licence 1", "licence 2", "licence 3",
            "master 1", "master 2",
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",  # semester codes
        })

        all_matched_nodes: List[Dict] = []
        all_year_tokens: List[str] = []

        for seg in segments:
            # Remove file extension
            if "." in seg:
                seg = seg.rsplit(".", 1)[0]
            if not seg:
                continue

            # Normalize: replace _ and - with spaces
            seg_normalised = re.sub(r"[_\-]+", " ", seg).strip()
            norm_seg = TextNormalizer.for_match(seg_normalised)
            if not norm_seg or norm_seg in _FORBIDDEN:
                continue

            # Step 0: Extract Year/Level tokens (L1, M1, ING1, etc.)
            for m in _RE_LEVEL_TOKEN.finditer(seg_normalised):
                year_tok = m.group(0).lower()
                if self.structure.is_year_token(year_tok):
                    if year_tok not in all_year_tokens:
                        all_year_tokens.append(year_tok)
                    logger.debug(f"    [URL year] '{year_tok}' ← segment '{seg}'")

            # Step 1: Depth-first partial matching
            matched = self._find_best_match_in_segment(norm_seg)
            if matched and self.neo4j.node_exists(matched["label"], matched["id"]):
                # Avoid duplicates
                if not any(
                    n["id"] == matched["id"] and n["label"] == matched["label"]
                    for n in all_matched_nodes
                ):
                    all_matched_nodes.append(matched)
                    logger.debug(
                        f"    [L1 direct] '{norm_seg}' → "
                        f"{matched['label']}:{matched['name']} "
                        f"(score={matched.get('_match_score', 0):.2f})"
                    )

        if not all_matched_nodes:
            logger.debug("    [L1 direct] No direct match found → falling back to token extraction")
            return None

        # FIX v21: Keep only the deepest nodes
        # If we matched both Department:Physique and Specialization:PEER,
        # keep only PEER (the child), discard Department (the parent)
        final_nodes = self._filter_deepest_only(all_matched_nodes)

        # Build final targets
        final_targets: List[Dict] = []
        seen: Set[Tuple[str, str]] = set()

        for node in final_nodes:
            # Get the full hierarchy path
            hp = self.neo4j.get_hierarchy_path(node["label"], node["id"])
            target = {
                "id":             node["id"],
                "name":           node["name"],
                "label":          node["label"],
                "hierarchy_path": hp,
            }
            key = (node["id"], node["label"])
            if key not in seen:
                final_targets.append(target)
                seen.add(key)

            # Add Year nodes under this matched node
            for yt in all_year_tokens:
                year_nodes = self.neo4j.get_year_nodes_under(
                    node["label"], node["id"], yt
                )
                for yn in year_nodes:
                    yn_key = (yn["id"], yn["label"])
                    if yn_key not in seen:
                        final_targets.append(yn)
                        seen.add(yn_key)
                        logger.debug(
                            f"    [L1 direct] Year '{yn['name']}' under "
                            f"{node['label']}:{node['name']}"
                        )

        if not final_targets:
            return None

        # Build result
        primary = max(
            final_targets,
            key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0)
        )
        paths = list(dict.fromkeys(
            t.get("hierarchy_path", "")
            for t in final_targets
            if t.get("hierarchy_path")
        ))

        return {
            "targets":        final_targets,
            "label":          primary["label"],
            "id":             primary["id"],
            "name":           primary["name"],
            "match_method":   "url_direct",
            "confidence":     1.0,
            "hierarchy_path": " | ".join(paths[:3]),
            "source":         "url_direct",
            "reason":         f"direct → {len(final_targets)} target(s)",
        }

    def _fmt(self, result: Dict) -> str:
        """Helper to format result for logging."""
        tgts = result.get("targets", [])
        return f"{len(tgts)} target(s): " + ", ".join(
            f"{t.get('label','?')}:{t.get('name','?')}" for t in tgts[:4]
        )

    # ══════════════════════════════════════════════════════════════════════
    # MAIN RESOLVE METHOD
    # ══════════════════════════════════════════════════════════════════════

    def resolve(self, url: str, signals: Optional[Dict] = None) -> Optional[Dict]:
        url_path = _RE_URL_DOMAIN.sub("", url) if url else ""
        if not url_path:
            return None

        # ===== FIX v21: Try DIRECT match first (bypasses NodeMatcher) =====
        direct_result = self._try_direct_match(url_path, signals)
        if direct_result:
            logger.debug(f"  [L1/direct] → {self._fmt(direct_result)}")
            return direct_result
        # ==================================================================

        # Fallback: use token extraction + NodeMatcher
        weighted, explicit = self._extract_from_url(url_path)

        if signals:
            # FIX C: also process PDF filenames through URL-style extraction
            pdf_fnames = signals.get("pdf_filenames", [])
            for fname in pdf_fnames:
                fw, fe = self._extract_from_url(fname)
                for tok, w in fw.items():
                    weighted[tok] = weighted.get(tok, 0.0) + w * (
                        _SRC_WEIGHT["pdf_name"] / _SRC_WEIGHT["url"]
                    )
                    explicit[tok] = explicit.get(tok, False) or fe.get(tok, False)

            for hint_key, hint_w in (
                ("department_hint", _SRC_WEIGHT["dept"]),
                ("faculty_hint",    _SRC_WEIGHT["content"]),
            ):
                hint_text = signals.get(hint_key, "") or ""
                if hint_text:
                    for tok, w in self._greedy_extract(hint_text, hint_w):
                        weighted[tok] = weighted.get(tok, 0.0) + w
                        explicit[tok] = explicit.get(tok, False) or True

        if not weighted:
            return None

        qualified = self.matcher.match(weighted, explicit_flags=explicit)
        if not qualified:
            return None

        # FIX 7: if only Faculty-level nodes qualified → skip
        if _all_faculty_only(qualified):
            logger.debug("    L1 qualified ONLY Faculty nodes → skip to General")
            return None

        logger.debug(f"    L1 qualified: {[q['name'] for q in qualified[:5]]}")
        targets = self.resolver.resolve(qualified)
        if not targets:
            return None

        return self._build_result(targets, "url")

    # ══════════════════════════════════════════════════════════════════════
    # FALLBACK: TOKEN EXTRACTION (unchanged from v19, kept as fallback)
    # ══════════════════════════════════════════════════════════════════════

    def _ensure_sorted_names(self):
        if not self._sorted_names:
            self._sorted_names = self.structure.all_full_names_sorted_desc()

    def _extract_from_url(
        self, url_path: str
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """Returns (weighted_tokens, explicit_flags)."""
        self._ensure_sorted_names()
        weighted: Dict[str, float] = {}
        explicit: Dict[str, bool]  = {}
        w = _SRC_WEIGHT["url"]

        raw_path = url_path.split("?")[0].split("#")[0]
        segments = [s for s in raw_path.split("/") if s]

        for seg in segments:
            if "." in seg:
                seg = seg.rsplit(".", 1)[0]
            if not seg:
                continue

            seg_normalised = re.sub(r"[_\-]+", " ", seg).strip()
            norm_seg = TextNormalizer.for_match(seg_normalised)
            if not norm_seg:
                continue

            # Step 0: Year/Level tokens
            for m in _RE_LEVEL_TOKEN.finditer(seg_normalised):
                year_tok = m.group(0).lower()
                if self.structure.is_year_token(year_tok):
                    weighted[year_tok] = weighted.get(year_tok, 0.0) + w
                    explicit[year_tok] = True
                    logger.debug(f"    [URL year] '{year_tok}' ← segment '{seg}'")

            matched_spans: List[Tuple[int, int]] = []

            def _overlaps_seg(s: int, e: int) -> bool:
                return any(s < me and e > ms for ms, me in matched_spans)

            # Step 1: greedy full-name with defer
            for entry_name in self._sorted_names:
                norm_entry = TextNormalizer.for_match(entry_name)
                if not norm_entry or len(norm_entry) < 3:
                    continue
                start_pos = 0
                while True:
                    idx = norm_seg.find(norm_entry, start_pos)
                    if idx == -1:
                        break
                    end_pos = idx + len(norm_entry)
                    before_ok = (idx == 0 or not norm_seg[idx - 1].isalpha())
                    after_ok  = (end_pos == len(norm_seg) or not norm_seg[end_pos].isalpha())
                    if before_ok and after_ok and not _overlaps_seg(idx, end_pos):
                        longer = self._longer_name_follows_in(
                            norm_entry, norm_seg, end_pos
                        )
                        if longer:
                            logger.debug(
                                f"    [URL defer] '{norm_entry}' → "
                                f"longer possible: '{longer}' in '{seg}'"
                            )
                            start_pos = end_pos
                            continue
                        matched_spans.append((idx, end_pos))
                        weighted[norm_entry] = weighted.get(norm_entry, 0.0) + w
                        explicit[norm_entry] = True
                        logger.debug(
                            f"    [URL greedy] '{norm_entry}' ← segment '{seg}'"
                        )
                    start_pos = end_pos

            # Build word positions
            all_seg_words = norm_seg.split()
            seg_word_pos: Dict[str, Tuple[int, int]] = {}
            cur = 0
            for ww in all_seg_words:
                fidx = norm_seg.find(ww, cur)
                if fidx != -1 and ww not in seg_word_pos:
                    seg_word_pos[ww] = (fidx, fidx + len(ww))
                if fidx != -1:
                    cur = fidx + len(ww)

            uncovered_words = [
                ww for ww in all_seg_words
                if len(ww) >= 2
                and ww not in _STOP_WORDS
                and not _overlaps_seg(*seg_word_pos.get(ww, (0, 0)))
            ]

            # Step 2: phrase match on uncovered words
            for n in (3, 2):
                for i in range(len(uncovered_words) - n + 1):
                    phrase_words = uncovered_words[i:i+n]
                    phrase = " ".join(phrase_words)
                    if not (self.structure.has_bigram(phrase) or
                            self.structure.has_trigram(phrase)):
                        continue
                    if phrase in weighted:
                        continue
                    word_spans = [seg_word_pos.get(pw) for pw in phrase_words]
                    if any(sp is None or _overlaps_seg(*sp) for sp in word_spans):
                        continue
                    weighted[phrase] = weighted.get(phrase, 0.0) + w
                    explicit[phrase] = True
                    for sp in word_spans:
                        if sp:
                            matched_spans.append(sp)
                    logger.debug(f"    [URL phrase] '{phrase}' ← segment '{seg}'")

            # Step 3: individual word fallback — FIX B logic
            for ww in uncovered_words:
                wp = seg_word_pos.get(ww)
                if wp and _overlaps_seg(*wp):
                    continue
                if len(ww) < 4:
                    continue
                if any(ww in existing for existing in weighted):
                    continue
                if not self.structure.has_word(ww):
                    continue

                ambiguity = self.structure.word_ambiguity(ww)

                if ambiguity == 1:
                    weighted[ww] = weighted.get(ww, 0.0) + w
                    explicit[ww] = True
                else:
                    entry_set = self.structure.word_entry_set(ww)
                    matching_nodes = self.idx.lookup_token(ww)
                    matching_ids = [n["id"] for n in matching_nodes]

                    if len(matching_ids) > 1:
                        common_parent = self.neo4j.get_common_parent(matching_ids)
                        if common_parent:
                            pnorm = TextNormalizer.for_match(common_parent["name"])
                            weighted[pnorm] = weighted.get(pnorm, 0.0) + w
                            explicit[pnorm] = True
                            logger.debug(
                                f"    [URL FIX B] '{ww}' → parent "
                                f"{common_parent['label']}:{common_parent['name']}"
                            )
                        else:
                            weighted[ww] = weighted.get(ww, 0.0) + w * _AMBIGUITY_PENALTY
                            explicit[ww] = False
                    else:
                        weighted[ww] = weighted.get(ww, 0.0) + w
                        explicit[ww] = True

        return weighted, explicit

    def _longer_name_follows_in(
        self, norm_entry: str, norm_seg: str, match_end: int
    ) -> Optional[str]:
        window = norm_seg[match_end:match_end + 60].lstrip()
        for longer_name in self._sorted_names:
            norm_longer = TextNormalizer.for_match(longer_name)
            if len(norm_longer) <= len(norm_entry):
                continue
            if not norm_longer.startswith(norm_entry):
                continue
            suffix = norm_longer[len(norm_entry):].lstrip()
            if not suffix:
                continue
            suffix_words = suffix.split()
            if not suffix_words:
                continue
            next_word = suffix_words[0]
            if len(next_word) < 2:
                continue
            if re.search(r'\b' + re.escape(next_word) + r'\b', window):
                return norm_longer
        return None

    def _greedy_extract(
        self, text: str, weight: float
    ) -> List[Tuple[str, float]]:
        """Greedy full-name matching on arbitrary text (used for hints)."""
        self._ensure_sorted_names()
        result: List[Tuple[str, float]] = []
        norm_text = TextNormalizer.for_match(text)
        matched_spans: List[Tuple[int, int]] = []

        def _overlaps(s: int, e: int) -> bool:
            for ms, me in matched_spans:
                if s < me and e > ms:
                    return True
            return False

        for entry_name in self._sorted_names:
            norm_entry = TextNormalizer.for_match(entry_name)
            if not norm_entry:
                continue
            start_pos = 0
            while True:
                idx = norm_text.find(norm_entry, start_pos)
                if idx == -1:
                    break
                end_pos = idx + len(norm_entry)
                before_ok = (idx == 0 or not norm_text[idx - 1].isalpha())
                after_ok  = (end_pos == len(norm_text) or not norm_text[end_pos].isalpha())
                if before_ok and after_ok and not _overlaps(idx, end_pos):
                    matched_spans.append((idx, end_pos))
                    result.append((norm_entry, weight))
                start_pos = end_pos
        return result

    @staticmethod
    def _build_result(targets: List[Dict], method: str) -> Dict:
        primary = max(targets, key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0))
        paths   = list(dict.fromkeys(
            t.get("hierarchy_path", "") for t in targets if t.get("hierarchy_path")
        ))
        return {
            "targets":        targets,
            "label":          primary["label"],
            "id":             primary["id"],
            "name":           primary["name"],
            "match_method":   method,
            "confidence":     1.0,
            "hierarchy_path": " | ".join(paths[:3]),
            "source":         method,
            "reason":         f"{method} → {len(targets)} target(s)",
        }

   
# ══════════════════════════════════════════════════════════════════════════════
# L2: KEYWORD CLASSIFIER  — FIX A + FIX C
# ══════════════════════════════════════════════════════════════════════════════

class KeywordClassifier:
    def __init__(
        self,
        neo4j:      Neo4jClient,
        structure:  StructureIndex,
        shared_idx: SharedNodeIndex,
        resolver:   ParcoursResolver,
        matcher:    NodeMatcher,
    ):
        self.neo4j     = neo4j
        self.resolver  = resolver
        self.matcher   = matcher
        self.extractor = TokenExtractor(structure)

    def classify(self, signals: Dict[str, Any]) -> Optional[Dict]:
        # FIX C: pass pdf_filenames to token extractor
        weighted, explicit = self.extractor.extract_with_explicit(
            url             = signals.get("url",             ""),
            title           = signals.get("title",           ""),
            content_excerpt = signals.get("content_excerpt", ""),
            department_hint = signals.get("department_hint", ""),
            faculty_hint    = signals.get("faculty_hint",    ""),
            pdf_filenames   = signals.get("pdf_filenames",   []),
        )
        if not weighted:
            return None

        logger.debug(
            f"    L2 tokens: "
            + str(dict(sorted(weighted.items(), key=lambda x: -x[1])[:8]))
        )

        # FIX A: pass explicit flags to NodeMatcher
        qualified = self.matcher.match(weighted, explicit_flags=explicit)
        if not qualified:
            return None

        logger.debug(
            f"    L2 qualified: "
            + str([(q["name"], round(q["_score"], 1), q.get("_explicit")) for q in qualified[:6]])
        )

        if _all_faculty_only(qualified):
            logger.debug("    [FIX 2/7] L2 Faculty-only → skip to L5")
            return None

        targets = self.resolver.resolve(qualified)
        if not targets:
            return None

        primary = max(targets, key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0))
        paths   = list(dict.fromkeys(
            t.get("hierarchy_path", "") for t in targets if t.get("hierarchy_path")
        ))
        return {
            "targets":        targets,
            "label":          primary["label"],
            "id":             primary["id"],
            "name":           primary["name"],
            "match_method":   "keyword",
            "confidence":     1.0,
            "hierarchy_path": " | ".join(paths[:3]),
            "source":         "keyword",
            "reason":         f"keyword+parcours → {len(targets)} target(s)",
        }


# ══════════════════════════════════════════════════════════════════════════════
# SPECIFICITY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _all_faculty_only(qualified: List[Dict]) -> bool:
    if not qualified:
        return False
    return all(n.get("label") in _GENERAL_LABELS for n in qualified)


# ══════════════════════════════════════════════════════════════════════════════
# L3: GEMINI CLASSIFIER  (unchanged from v18)
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM = """You are a precise document classifier for Farhat Abbas University Sétif 1.
Your output is used directly by a Neo4j ingestion pipeline.

NEO4J HIERARCHY: Faculty → Department → Level → Category/Program → Specialization → Year

RULES:
1. Match ONLY to nodes listed in the candidates below.
2. Choose the DEEPEST possible node (Year > Specialization > Category > Level > Department).
3. If Year can be determined → return Year, not Level.
4. Non-academic content (news, admin pages, profiles) → return null.
5. Output ONLY valid JSON. No markdown fences.

LEVEL INFERENCE:
- "1ere annee" / "premiere annee" → L1
- "Master 1" / "1ere annee master" → M1
- "Master 2" / "2eme annee master" → M2
- S1/S2 → L1, S3/S4 → L2, S5/S6 → L3, S7/S8 → M1

OUTPUT FORMAT:
{"target":{"label":"<label>","id":"<exact_id>","name":"<exact_name>","reason":"<one sentence>"},"match_method":"llm","confidence":<0.0-1.0>,"reasoning":"<brief>"}
If no match: {"target":null,"match_method":"general","confidence":0.0,"reasoning":"no match"}"""


def _extract_json_from_text(text: str) -> Optional[str]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",       "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0; in_str = False; escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:                escape = False; continue
        if ch == "\\" and in_str: escape = True;  continue
        if ch == '"' and not escape: in_str = not in_str; continue
        if in_str:                continue
        if ch == "{":             depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


class GeminiClassifier:
    def __init__(self, neo4j: Neo4jClient, resolver: ParcoursResolver,
                 api_key: str = GEMINI_API_KEY):
        self.neo4j           = neo4j
        self.resolver        = resolver
        self._model: Any     = None
        self._cand_text: Optional[str] = None
        self._last_call      = 0.0
        self._quota_exceeded = False

        if _GEMINI_OK and api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=_GEMINI_SYSTEM,
                generation_config=genai.GenerationConfig(
                    temperature=0.0, max_output_tokens=512,
                ),
            )
            logger.info(f"✅ Gemini ready ({GEMINI_MODEL})")
        else:
            logger.warning("⚠  Gemini disabled")

    @property
    def available(self) -> bool:
        return self._model is not None and not self._quota_exceeded

    def _candidates_text(self) -> str:
        if self._cand_text:
            return self._cand_text
        c = self.neo4j.get_all_candidates()
        lines = ["## Specializations"]
        for sp in c.get("Specialization", []):
            p = f" (parent: {sp['parent']})" if sp.get("parent") else ""
            lines.append(f"  id={sp['id']} name={sp['name']}{p}")
        lines.append("\n## Years")
        for y in c.get("Year", []):
            p = f" (parent: {y['parent']})" if y.get("parent") else ""
            lines.append(f"  id={y['id']} name={y['name']}{p}")
        lines.append("\n## Categories")
        for cat in c.get("Category", []):
            lines.append(f"  id={cat['id']} name={cat['name']}")
        lines.append("\n## Programs")
        for pg in c.get("Program", []):
            lines.append(f"  id={pg['id']} name={pg['name']}")
        lines.append("\n## Departments")
        for d in c.get("Department", []):
            lines.append(f"  id={d['id']} name={d['name']}")
        lines.append("\n## Levels")
        for lv in c.get("Level", []):
            lines.append(f"  id={lv['id']} name={lv['name']}")
        self._cand_text = "\n".join(lines)
        return self._cand_text

    def _rate_limit(self):
        wait = GEMINI_RATE_LIMIT_SECONDS - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def classify(self, signals: Dict, url: str, title: str, content: str) -> Optional[Dict]:
        if not self.available:
            return None
        prompt = (
            f"{self._candidates_text()}\n\n"
            f"## DOCUMENT\nURL: {url}\nTitle: {title}\n"
            f"Faculty: {signals.get('faculty_hint','')}\n"
            f"Department: {signals.get('department_hint','')}\n"
            f"Level tokens: {', '.join(signals.get('level_tokens',[]))}\n"
            f"Abbreviations: {json.dumps(signals.get('found_abbrevs',{}))}\n"
            f"PDF files: {', '.join(signals.get('pdf_filenames',[]))}\n\n"
            f"## CONTENT EXCERPT:\n{content[:LLM_CONTENT_EXCERPT]}\n\n"
            "Classify to DEEPEST valid node. Return ONLY valid JSON."
        )
        self._rate_limit()
        for attempt in range(2):
            try:
                resp = self._model.generate_content(prompt)
                raw  = ""
                if resp.candidates:
                    for part in resp.candidates[0].content.parts:
                        if getattr(part, "thought", False):
                            continue
                        if hasattr(part, "text") and part.text:
                            raw += part.text
                if not raw:
                    raw = getattr(resp, "text", "") or ""
                json_str = _extract_json_from_text(raw)
                if not json_str:
                    return None
                parsed = json.loads(json_str)
                return self._validate_and_resolve(parsed)
            except json.JSONDecodeError:
                return None
            except Exception as exc:
                err = str(exc)
                if "429" in err or "quota" in err.lower():
                    self._quota_exceeded = True
                    return None
                if attempt < 1:
                    time.sleep(2.0)
                    continue
                logger.warning(f"  Gemini error: {exc}")
                return None
        return None

    def _validate_and_resolve(self, parsed: Dict) -> Optional[Dict]:
        raw = parsed.get("target")
        if not raw:
            return None
        lbl = raw.get("label", "")
        nid = raw.get("id",    "")
        if not lbl or not nid:
            return None
        actual = _LABEL_MAP.get(lbl)
        if not actual:
            return None
        valid, _ = self.neo4j.validate_classification(actual, nid)
        if not valid:
            return None
        confidence = float(parsed.get("confidence", 0.0))
        if confidence < LLM_CONFIDENCE_THRESHOLD:
            return None
        matched = [{"id": nid, "name": raw.get("name", ""), "label": actual,
                    "_explicit": True}]
        targets = self.resolver.resolve(matched)
        if not targets:
            hp = self.neo4j.get_hierarchy_path(actual, nid)
            targets = [{"id": nid, "name": raw.get("name", ""),
                        "label": actual, "hierarchy_path": hp}]
        primary = max(targets, key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0))
        paths   = list(dict.fromkeys(
            t.get("hierarchy_path", "") for t in targets if t.get("hierarchy_path")
        ))
        return {
            "targets":        targets,
            "label":          primary["label"],
            "id":             primary["id"],
            "name":           primary["name"],
            "match_method":   "llm",
            "confidence":     confidence,
            "hierarchy_path": " | ".join(paths[:3]),
            "source":         "llm",
            "reason":         raw.get("reason", ""),
        }


# ══════════════════════════════════════════════════════════════════════════════
# L4: EMBEDDING CLASSIFIER  (unchanged from v18)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingClassifier:
    def __init__(self, neo4j: Neo4jClient, model: SentenceTransformer,
                 resolver: ParcoursResolver):
        self.neo4j    = neo4j
        self.model    = model
        self.resolver = resolver
        self._index:  Optional[Tuple[List[Dict], np.ndarray]] = None

    def build_index(self):
        if self._index is not None:
            return
        candidates = self.neo4j.get_all_candidates()
        pairs: List[Tuple[Dict, str]] = []

        def _sig(node: Dict, label: str) -> str:
            name   = node.get("name", "")
            parent = node.get("parent", "") or ""
            lines  = [name]
            if parent:
                lines.append(f"{parent} > {name}")
            for lvl in ("L1", "L2", "L3", "M1", "M2"):
                lines.append(f"{lvl} {name}")
            for abbrev, expansion in _ABBREV_MAP.items():
                if TextNormalizer.for_match(expansion) == TextNormalizer.for_match(name):
                    lines.append(abbrev)
                    for lvl in ("M1", "M2", "L1", "L2", "L3"):
                        lines.append(f"{lvl} {abbrev}")
            return "\n".join(dict.fromkeys(l.strip() for l in lines if l.strip()))

        for label in ("Year", "Specialization", "Category", "Program"):
            for node in candidates.get(label, []):
                pairs.append(({**node, "label": label}, _sig(node, label)))

        if not pairs:
            self._index = ([], np.array([]))
            return

        nodes = [p[0] for p in pairs]
        sigs  = [p[1] for p in pairs]
        logger.info(f"  Building BGE-M3 index for {len(nodes)} nodes …")
        vecs = self.model.encode(sigs, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=64)
        self._index = (nodes, np.array(vecs, dtype=np.float32))
        logger.info("  ✅ BGE-M3 index ready")

    def encode_chunks(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=32)
        return [v.tolist() if len(v) == EMBED_DIM else [] for v in vecs]

    def classify(self, signals: Dict[str, Any]) -> Optional[Dict]:
        if self._index is None:
            self.build_index()
        nodes, node_vecs = self._index
        if len(nodes) == 0 or node_vecs.size == 0:
            return None
        doc_text = self._doc_text(signals)
        doc_vec  = self.model.encode([doc_text], normalize_embeddings=True,
                                     show_progress_bar=False)[0].astype(np.float32)
        sims     = np.dot(node_vecs, doc_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < EMBED_MIN_CONFIDENCE:
            logger.debug(f"    [FIX 6] L4 best sim={best_sim:.3f} < {EMBED_MIN_CONFIDENCE} → skip")
            return None

        best  = nodes[best_idx]
        label = best["label"]
        node  = self.neo4j.get_node_by_id(label, best["id"])
        if not node:
            return None
        valid, _ = self.neo4j.validate_classification(label, node["id"])
        if not valid:
            return None
        matched = [{"id": node["id"], "name": node["name"], "label": label,
                    "_explicit": True}]
        targets = self.resolver.resolve(matched)
        if not targets:
            hp = self.neo4j.get_hierarchy_path(label, node["id"])
            targets = [{**node, "label": label, "hierarchy_path": hp}]
        primary = max(targets, key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0))
        paths   = list(dict.fromkeys(
            t.get("hierarchy_path", "") for t in targets if t.get("hierarchy_path")
        ))
        return {
            "targets":        targets,
            "label":          primary["label"],
            "id":             primary["id"],
            "name":           primary["name"],
            "match_method":   "embedding",
            "confidence":     best_sim,
            "hierarchy_path": " | ".join(paths[:3]),
            "source":         "embedding",
            "reason":         f"BGE-M3 cosine={best_sim:.3f}",
        }

    @staticmethod
    def _doc_text(signals: Dict[str, Any]) -> str:
        parts = []
        for key in ("title", "content_excerpt", "faculty_hint", "department_hint"):
            v = signals.get(key)
            if v:
                parts.append(v)
        for abbrev, expansion in signals.get("found_abbrevs", {}).items():
            parts.append(f"{abbrev} {expansion}")
        if signals.get("level_tokens"):
            parts.append(" ".join(signals["level_tokens"]))
        # FIX C: include PDF filenames in embedding query
        if signals.get("pdf_filenames"):
            parts.append(" ".join(signals["pdf_filenames"]))
        return " | ".join(filter(None, parts))


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationOrchestrator:
    def __init__(
        self,
        neo4j:    Neo4jClient,
        url_res:  URLResolver,
        keyword:  KeywordClassifier,
        embedder: EmbeddingClassifier,
        gemini:   GeminiClassifier,
    ):
        self.neo4j    = neo4j
        self.url_res  = url_res
        self.keyword  = keyword
        self.embedder = embedder
        self.gemini   = gemini
        self.signals  = SignalExtractor()

    def classify(
        self,
        url:             str,
        title:           str,
        text:            str,
        language:        str,
        faculty_hint:    str,
        department_hint: str,
        ext_docs:        Optional[List[Dict]] = None,   # FIX C
    ) -> Dict[str, Any]:

        # FIX C: pass ext_docs for PDF filename extraction
        sig = self.signals.extract(
            url=url, title=title, content=text, file_path="",
            faculty_hint=faculty_hint, department_hint=department_hint,
            ext_docs=ext_docs,
        )

        result = self.url_res.resolve(url, signals=sig)
        if result:
            logger.debug(f"  [L1/url] → {self._fmt(result)}")
            return self._ensure_targets(result)

        result = self.keyword.classify(sig)
        if result:
            logger.debug(f"  [L2/keyword] → {self._fmt(result)}")
            return self._ensure_targets(result)

        if self._is_general_page_signal(sig, text, title):
            logger.debug("  [FIX 2/7] General faculty page → L5")
            return self._make_general_result(sig)

        if self.gemini.available:
            result = self.gemini.classify(signals=sig, url=url, title=title, content=text)
            if result and result.get("confidence", 0.0) >= LLM_CONFIDENCE_THRESHOLD:
                logger.debug(f"  [L3/llm] → {self._fmt(result)}")
                return self._ensure_targets(result)

        result = self.embedder.classify(sig)
        if result and result.get("confidence", 0.0) >= EMBED_MIN_CONFIDENCE:
            logger.debug(f"  [L4/embed] → {self._fmt(result)}")
            return self._ensure_targets(result)

        logger.debug("  [L5/general]")
        return self._make_general_result(sig)

    @staticmethod
    def _is_general_page_signal(sig: Dict, text: str, title: str) -> bool:
        title_lower = title.lower() if title else ""
        text_lower  = text.lower()  if text  else ""
        faculty_title_words = {
            "faculté", "faculte", "faculty", "université", "universite",
            "university", "accueil", "home", "présentation", "presentation",
        }
        specific_words = {
            "département", "departement", "department", "spécialité", "specialite",
            "specialization", "master", "licence", "doctorat", "programme",
            "informatique", "physique", "chimie", "mathématique", "biologie",
            "technologie", "médecine",
        }
        title_words = set(re.findall(r'\w+', title_lower))
        has_faculty_title  = bool(title_words & faculty_title_words)
        has_specific_title = bool(title_words & specific_words)
        text_len = len(text.strip())
        if has_faculty_title and not has_specific_title and text_len < 300:
            return True
        return False

    def _make_general_result(self, sig: Dict) -> Dict[str, Any]:
        gid = self.neo4j.get_general_id()
        faculty_hint = sig.get("faculty_hint", "")
        if faculty_hint:
            candidates = self.neo4j.get_all_candidates()
            for fac_node in candidates.get("Faculty", []):
                if (TextNormalizer.for_match(fac_node["name"]) ==
                        TextNormalizer.for_match(faculty_hint)):
                    general = self.neo4j.get_general_under_faculty(
                        fac_node["id"], fac_node["name"]
                    )
                    if general:
                        return {
                            "targets":        [general],
                            "label":          "General",
                            "id":             general["id"],
                            "name":           general["name"],
                            "match_method":   "general",
                            "confidence":     0.0,
                            "hierarchy_path": general["hierarchy_path"],
                            "source":         "fallback_general",
                            "reason":         "General faculty page",
                        }
        return {
            "targets": [{
                "label":          "General",
                "id":             gid,
                "name":           "General",
                "reason":         "No confident match",
                "hierarchy_path": "General",
            }],
            "label":          "General",
            "id":             gid,
            "name":           "General",
            "match_method":   "general",
            "confidence":     0.0,
            "hierarchy_path": "General",
            "source":         "fallback",
            "reason":         "No confident match",
        }

    @staticmethod
    def _fmt(result: Dict) -> str:
        tgts = result.get("targets", [])
        return f"{len(tgts)} target(s): " + ", ".join(
            f"{t.get('label','?')}:{t.get('name','?')}" for t in tgts[:4]
        )

    def _ensure_targets(self, result: Dict) -> Dict:
        if not result.get("targets"):
            result["targets"] = [{
                "label":          result.get("label",  "General"),
                "id":             result.get("id",     self.neo4j.get_general_id()),
                "name":           result.get("name",   "General"),
                "reason":         result.get("reason", ""),
                "hierarchy_path": result.get("hierarchy_path", ""),
            }]
        return result


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL CHUNKER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalChunker:
    def __init__(self, chunk_tokens=CHUNK_TOKENS_BASE, overlap=OVERLAP_TOKENS,
                 min_chars=MIN_CHUNK_CHARS):
        self.chunk_tokens = chunk_tokens
        self.overlap      = overlap
        self.min_chars    = min_chars
        self._enc = tiktoken.get_encoding("cl100k_base") if _TIKTOKEN_OK else None

    def split(self, text: str, title: str = "") -> List[Dict]:
        text = normalize_text(text)
        if not text:
            return []
        segments = self._segment(text)
        if not segments:
            return []

        chunks:   List[Dict] = []
        cur:      List[str]  = []
        cur_tok:  int        = 0
        cur_sec:  str        = ""
        cur_type: str        = "paragraph"
        idx:      int        = 0

        for seg in segments:
            seg_tok = self._tok(seg["content"])
            if seg["type"] == "heading":
                if cur:
                    c = self._make(cur, idx, title, cur_sec, cur_type)
                    if len(c["clean_body"]) >= self.min_chars:
                        chunks.append(c); idx += 1
                cur, cur_tok = [], 0
                cur_sec = seg["content"]; cur_type = "section"
                continue
            if seg_tok > self.chunk_tokens:
                if cur:
                    c = self._make(cur, idx, title, cur_sec, cur_type)
                    if len(c["clean_body"]) >= self.min_chars:
                        chunks.append(c); idx += 1
                    cur, cur_tok = [], 0
                sub = self._split_long(seg["content"], title, cur_sec, idx)
                chunks.extend(sub); idx += len(sub); continue
            if cur_tok + seg_tok > self.chunk_tokens and cur:
                c = self._make(cur, idx, title, cur_sec, cur_type)
                if len(c["clean_body"]) >= self.min_chars:
                    chunks.append(c); idx += 1
                over: List[str] = []; ot = 0
                for prev in reversed(cur):
                    pt = self._tok(prev)
                    if ot + pt > self.overlap: break
                    over.insert(0, prev); ot += pt
                cur = over + [seg["content"]]; cur_tok = ot + seg_tok
            else:
                cur.append(seg["content"]); cur_tok += seg_tok; cur_type = seg["type"]

        if cur:
            c = self._make(cur, idx, title, cur_sec, cur_type)
            if len(c["clean_body"]) >= self.min_chars:
                chunks.append(c)
        return chunks

    def _segment(self, text: str) -> List[Dict]:
        segs = []
        for block in _RE_PARA_BREAK.split(text):
            block = block.strip()
            if not block: continue
            if _RE_HEADING.match(block):          segs.append({"type": "heading",   "content": block})
            elif _RE_TABLE_MARKER.search(block):  segs.append({"type": "table",     "content": block})
            elif _RE_LIST_ITEM.search(block):     segs.append({"type": "list",      "content": block})
            else:                                 segs.append({"type": "paragraph", "content": block})
        return segs

    def _split_long(self, para: str, title: str, section: str, start: int) -> List[Dict]:
        sents  = [s.strip() for s in _RE_SENT_BOUND.split(para) if s.strip()]
        chunks: List[Dict] = []; cur: List[str] = []; tok = 0; idx = start
        for sent in sents:
            st = self._tok(sent)
            if cur and tok + st > self.chunk_tokens:
                c = self._make(cur, idx, title, section, "paragraph")
                if len(c["clean_body"]) >= self.min_chars:
                    chunks.append(c); idx += 1
                cur, tok = [sent], st
            else:
                cur.append(sent); tok += st
        if cur:
            c = self._make(cur, idx, title, section, "paragraph")
            if len(c["clean_body"]) >= self.min_chars:
                chunks.append(c)
        return chunks

    def _make(self, parts, idx, title, section, chunk_type) -> Dict:
        body       = " ".join(parts)
        embed_text = "\n".join(p for p in [title, section, body] if p)
        return {
            "embed_text":  embed_text,
            "text":        embed_text,
            "clean_body":  body,
            "section":     section,
            "chunk_type":  chunk_type,
            "token_count": self._tok(body),
            "chunk_index": idx,
        }

    def _tok(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        return len(text) // 4


# ══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTOR  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(file_path: str) -> str:
    if not file_path or not Path(file_path).exists() or not _PYPDF_OK:
        return ""
    try:
        reader = pypdf.PdfReader(file_path)
        parts  = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t: parts.append(t)
            except Exception:
                pass
        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug(f"  PDF extraction failed: {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# JSON DOCUMENT PARSER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_doc(data: dict) -> dict:
    meta      = data.get("metadata", {})
    content   = data.get("content",  {})
    resources = data.get("resources", {})
    ext_docs  = resources.get("documents", []) if isinstance(resources, dict) else []

    if "page" in meta:
        page     = meta["page"]
        page_url = page.get("url", "")
        parts    = [content.get("text", "")]
        for sec in content.get("sections", []):
            if isinstance(sec, dict):
                parts.extend([sec.get("text", ""), sec.get("title", "")])
        raw_text = "\n\n".join(filter(None, parts))
        combined = normalize_text(raw_text)
        return dict(
            text=combined, title=page.get("title", ""), url=page_url,
            file_path="", file_type="web", source="page",
            links=extract_links(raw_text),
            pdf_urls=extract_pdf_urls(raw_text, page_url),
            ext_docs=ext_docs,
        )

    file_info = meta.get("file", {})
    file_path = file_info.get("path", "")
    file_type = file_info.get("type", "")
    src = "pdf" if (
        file_type.lower() == "pdf" or file_path.lower().endswith(".pdf")
    ) else "extracted"

    parts: List[str] = []
    if content.get("text"): parts.append(content["text"])
    for pg in content.get("pages", []):
        if isinstance(pg, dict) and pg.get("text"): parts.append(pg["text"])
    for sec in content.get("sections", []):
        if isinstance(sec, dict):
            if sec.get("title"): parts.append(sec["title"])
            if sec.get("text"):  parts.append(sec["text"])

    raw_text = "\n\n".join(filter(None, parts))
    combined = normalize_text(raw_text)
    title    = file_info.get("name", "") or Path(file_path).stem

    return dict(
        text=combined, title=title, url=file_info.get("url", ""),
        file_path=file_path, file_type=file_type, source=src,
        links=extract_links(raw_text), pdf_urls=[],
        ext_docs=ext_docs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FILE COLLECTION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def collect_json_files(root: Path) -> List[Tuple[Path, str, str]]:
    results: List[Tuple[Path, str, str]] = []
    for faculty_dir in sorted(root.iterdir()):
        if not faculty_dir.is_dir():
            continue
        fl     = FACULTY_LABELS.get(faculty_dir.name.lower(), faculty_dir.name.upper())
        before = len(results)
        for sub in ("pages", "extracted", "tables"):
            sfolder = faculty_dir / sub
            if not sfolder.exists():
                continue
            base_str = str(sfolder)
            base_len = len(base_str) + 1
            for dirpath, _, filenames in os.walk(base_str):
                for fname in filenames:
                    if not fname.endswith(".json"):
                        continue
                    jf   = Path(dirpath) / fname
                    rem  = str(jf)[base_len:]
                    sp   = rem.find(os.sep)
                    dept = (
                        rem[:sp] if sp != -1 else "General"
                    ).replace("_", " ").replace("-", " ").title()
                    results.append((jf, fl, dept))
        logger.info(f"📂 {faculty_dir.name} → {len(results)-before} files")
    logger.info(f"🔎 TOTAL: {len(results)} JSON files")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS DATABASE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class ProgressDB:
    def __init__(self, db_path: str = PROGRESS_DB):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS processed "
            "(file_key TEXT PRIMARY KEY, chunks INTEGER, "
            "doc_type TEXT, language TEXT, processed_at TEXT)"
        )
        self.conn.commit()
        self._done: Set[str] = {
            row[0] for row in self.conn.execute("SELECT file_key FROM processed")
        }
        self._pending: List[Tuple] = []
        logger.info(f"ProgressDB: {len(self._done)} already processed")

    def is_done(self, fk: str) -> bool:
        return fk in self._done

    def mark_done(self, fk: str, n: int, dt: str, lang: str):
        self._done.add(fk)
        self._pending.append((fk, n, dt, lang, datetime.utcnow().isoformat()))
        if len(self._pending) >= PROGRESS_FLUSH:
            self._flush()

    def _flush(self):
        if self._pending:
            self.conn.executemany(
                "INSERT OR REPLACE INTO processed VALUES (?,?,?,?,?)", self._pending)
            self.conn.commit()
            self._pending.clear()

    def flush_final(self):
        self._flush()

    def reset(self):
        self.conn.execute("DELETE FROM processed")
        self.conn.commit()
        self._done.clear()
        self._pending.clear()

    def close(self):
        self.flush_final()
        self.conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE  — FIX C integrated
# ══════════════════════════════════════════════════════════════════════════════

class IngestionPipeline:
    def __init__(self):
        logger.info("Loading BGE-M3 model …")
        self.model = SentenceTransformer(EMBED_MODEL)
        logger.info("✅ BGE-M3 ready")

        self.neo4j     = Neo4jClient()
        self.structure = StructureIndex(STRUCTURE_FILE)

        logger.info("Pre-loading graph candidates …")
        candidates = self.neo4j.get_all_candidates()

        self.shared_idx = SharedNodeIndex()
        self.shared_idx.build(candidates)

        self.resolver = ParcoursResolver(self.neo4j)
        self.matcher  = NodeMatcher(self.neo4j, self.shared_idx)

        self.url_res  = URLResolver(
            self.neo4j, self.shared_idx, self.resolver, self.matcher, self.structure
        )
        self.keyword  = KeywordClassifier(
            self.neo4j, self.structure, self.shared_idx, self.resolver, self.matcher
        )
        self.embedder = EmbeddingClassifier(self.neo4j, self.model, self.resolver)
        self.gemini   = GeminiClassifier(self.neo4j, self.resolver)

        self.orchestrator = ClassificationOrchestrator(
            neo4j    = self.neo4j,
            url_res  = self.url_res,
            keyword  = self.keyword,
            embedder = self.embedder,
            gemini   = self.gemini,
        )
        self.chunker  = HierarchicalChunker()
        self.progress = ProgressDB()
        self.embedder.build_index()

    def run(self, resume: bool = True):
        if not resume:
            self.progress.reset()
            logger.info("🔥 Fresh start — progress cleared")

        root       = Path(ROOT_FOLDER)
        all_files  = collect_json_files(root)
        to_process = [
            (jf, f, d) for jf, f, d in all_files
            if not (resume and self.progress.is_done(f"{f}/{d}/{jf.name}"))
        ]
        logger.info(f"📂 {len(to_process)} / {len(all_files)} to process")

        ok = skip = fail = total_chunks = 0
        meta_store: List[Dict]     = []
        cls_stats:  Dict[str, int] = {}

        for jf, faculty, department in to_process:
            try:
                doc = self._process_file(jf, faculty, department)
                if doc is None:
                    skip += 1; continue

                self._store(doc)
                meta_store.extend(doc["metadata_rows"])

                method = doc["classification"].get("match_method", "none")
                cls_stats[method] = cls_stats.get(method, 0) + 1
                self.progress.mark_done(
                    doc["file_key"], len(doc["chunks"]), "document", doc["language"]
                )
                total_chunks += len(doc["chunks"])
                ok += 1

                n_pdf    = len(doc.get("pdf_docs", []))
                n_tgt    = len(doc["classification"].get("targets", []))
                tgt_names = ", ".join(
                    f"{t.get('label','?')}:{t.get('name','?')}"
                    for t in doc["classification"].get("targets", [])[:4]
                )
                logger.info(
                    f"✅ {jf.name} → {len(doc['chunks'])} chunks "
                    f"[{method}] → {n_tgt} target(s): {tgt_names}"
                    + (f" + {n_pdf} PDF(s)" if n_pdf else "")
                )

            except Exception as exc:
                import traceback
                logger.error(f"❌ {jf.name}: {exc}\n{traceback.format_exc()}")
                fail += 1

        with open(METADATA_PATH, "w", encoding="utf-8") as fh:
            json.dump(meta_store, fh, ensure_ascii=False, indent=2)

        self.progress.close()
        self.neo4j.close()

        logger.info("\n" + "─" * 70)
        logger.info("  PIPELINE COMPLETE")
        logger.info(f"   SUCCESS: {ok}  SKIPPED: {skip}  FAILED: {fail}")
        logger.info(f"   TOTAL CHUNKS: {total_chunks}")
        for m, c in sorted(cls_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {m:40s}: {c}")
        logger.info("─" * 70)

    def _process_file(self, jf: Path, faculty: str, department: str) -> Optional[Dict]:
        file_key = f"{faculty}/{department}/{jf.name}"

        with open(jf, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)

        parsed = parse_json_doc(raw_data)
        if len(parsed["text"].strip()) < MIN_DOC_CHARS:
            return None

        title    = parsed["title"] or jf.stem
        language = detect_language(parsed["text"])
        year_cal = extract_year_calendar(parsed["text"])
        url      = parsed.get("url", "")
        fp       = doc_fingerprint(parsed["text"])
        url_id   = url_to_id(url, fp)

        # FIX C: pass ext_docs to orchestrator
        classification = self.orchestrator.classify(
            url=url, title=title, text=parsed["text"],
            language=language, faculty_hint=faculty, department_hint=department,
            ext_docs=parsed.get("ext_docs", []),
        )

        raw_chunks = self.chunker.split(parsed["text"], title=title)
        if not raw_chunks:
            return None

        chunk_texts = [c["embed_text"] for c in raw_chunks]
        embeddings  = self.embedder.encode_chunks(chunk_texts)

        chunks:    List[Dict] = []
        meta_rows: List[Dict] = []
        pdf_urls = parsed.get("pdf_urls", [])

        for i, (cd, emb) in enumerate(zip(raw_chunks, embeddings)):
            cid = f"{fp}_c{i}"
            chunks.append({
                "id":          cid,
                "text":        cd["embed_text"],
                "chunk_index": i,
                "token_count": cd.get("token_count", 0),
                "language":    language,
                "section":     cd.get("section", ""),
                "chunk_type":  cd.get("chunk_type", "paragraph"),
                "embedding":   emb,
            })
            meta_rows.append({
                "chunk_id":       cid,
                "doc_id":         fp,
                "file":           jf.name,
                "url_id":         url_id,
                "faculty":        faculty,
                "department":     department,
                "title":          title,
                "url":            url,
                "file_path":      parsed.get("file_path", ""),
                "file_type":      parsed.get("file_type", ""),
                "source":         parsed["source"],
                "language":       language,
                "chunk_index":    i,
                "section":        cd.get("section", ""),
                "chunk_type":     cd.get("chunk_type", "paragraph"),
                "text_preview":   cd["embed_text"][:200],
                "year":           year_cal,
                "classification": classification,
                "pdf_urls":       pdf_urls,
                "links":          parsed.get("links", []),
            })

        child_url_records: List[Dict] = []
        for pu in pdf_urls:
            child_url_records.append({
                "url_id":         url_to_id(pu, doc_fingerprint(pu)),
                "url":            pu,
                "title":          f"PDF from {title}",
                "source_type":    "pdf",
                "parent_url_id":  url_id,
                "classification": classification,
                "text":           None,
            })

        pdf_docs: List[Dict] = []
        for ext_doc in parsed.get("ext_docs", []):
            pdf_url   = ext_doc.get("url", "")
            pdf_title = ext_doc.get("title", "") or f"PDF from {title}"
            local_f   = ext_doc.get("local_file", "")
            if not pdf_url:
                continue
            pdf_fp     = doc_fingerprint(pdf_url)
            pdf_url_id = url_to_id(pdf_url, pdf_fp)
            pdf_text   = extract_pdf_text(str(local_f)) if local_f else ""
            child_url_records.append({
                "url_id":         pdf_url_id,
                "url":            pdf_url,
                "title":          pdf_title,
                "source_type":    "pdf",
                "parent_url_id":  url_id,
                "classification": classification,
                "text":           pdf_text if pdf_text else None,
            })
            if pdf_text and len(pdf_text.strip()) >= MIN_DOC_CHARS:
                pdf_docs.append({
                    "url_id":         pdf_url_id,
                    "url":            pdf_url,
                    "title":          pdf_title,
                    "text":           normalize_text(pdf_text),
                    "classification": classification,
                    "language":       detect_language(pdf_text),
                    "fp":             doc_fingerprint(pdf_text),
                })

        return {
            "file_key":          file_key,
            "fp":                fp,
            "faculty":           faculty,
            "department":        department,
            "title":             title,
            "language":          language,
            "url":               url,
            "url_id":            url_id,
            "source_type":       parsed["source"],
            "year":              year_cal,
            "classification":    classification,
            "chunks":            chunks,
            "metadata_rows":     meta_rows,
            "child_url_records": child_url_records,
            "pdf_docs":          pdf_docs,
        }

    def _store(self, doc: Dict):
        cls     = doc["classification"]
        targets = cls.get("targets") or [{
            "label":          cls["label"],
            "id":             cls["id"],
            "name":           cls["name"],
            "reason":         "",
            "hierarchy_path": cls.get("hierarchy_path", ""),
        }]

        primary = max(targets, key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0))

        self.neo4j.upsert_url_node(
            url_id                = doc["url_id"],
            url                   = doc["url"],
            title                 = doc["title"],
            source_type           = doc["source_type"],
            target_label          = primary["label"],
            target_id             = primary["id"],
            hierarchy_path        = primary.get("hierarchy_path",
                                                cls.get("hierarchy_path", "")),
            classification_method = cls.get("match_method", "none"),
            confidence            = float(cls.get("confidence", 0.0)),
            parent_url_id         = None,
        )

        extra = [
            t for t in targets
            if (t["id"], t["label"]) != (primary["id"], primary["label"])
        ]
        if extra:
            self.neo4j.link_url_to_extra_targets(doc["url_id"], extra)

        self.neo4j.create_chunks(doc["url_id"], doc["chunks"], cls)

        gid = self.neo4j.get_general_id()
        for child in doc.get("child_url_records", []):
            child_cls = child["classification"]
            if not child.get("text") or len((child.get("text") or "").strip()) < MIN_DOC_CHARS:
                child_cls = {
                    "label":          "General",
                    "id":             gid,
                    "name":           "General",
                    "match_method":   "inherited_general",
                    "confidence":     0.0,
                    "hierarchy_path": "General",
                    "targets": [{
                        "label":          "General",
                        "id":             gid,
                        "name":           "General",
                        "reason":         "no content",
                        "hierarchy_path": "General",
                    }],
                }
            ctargets = child_cls.get("targets") or [child_cls]
            cprimary = max(ctargets, key=lambda t: _DEPTH_ORDER.get(t.get("label", ""), 0))
            self.neo4j.upsert_url_node(
                url_id                = child["url_id"],
                url                   = child["url"],
                title                 = child["title"],
                source_type           = child["source_type"],
                target_label          = cprimary.get("label", "General"),
                target_id             = cprimary.get("id",    gid),
                hierarchy_path        = child_cls.get("hierarchy_path", ""),
                classification_method = child_cls.get("match_method", "none"),
                confidence            = float(child_cls.get("confidence", 0.0)),
                parent_url_id         = child["parent_url_id"],
            )

        for pdf_doc in doc.get("pdf_docs", []):
            pdf_cls    = pdf_doc["classification"]
            raw_pdf    = self.chunker.split(pdf_doc["text"], title=pdf_doc["title"])
            if not raw_pdf:
                continue
            pdf_embeds = self.embedder.encode_chunks([c["embed_text"] for c in raw_pdf])
            pdf_chunks = [
                {
                    "id":          f"{pdf_doc['fp']}_c{i}",
                    "text":        cd["embed_text"],
                    "chunk_index": i,
                    "token_count": cd.get("token_count", 0),
                    "language":    pdf_doc.get("language", "fr"),
                    "section":     cd.get("section", ""),
                    "chunk_type":  cd.get("chunk_type", "paragraph"),
                    "embedding":   emb,
                }
                for i, (cd, emb) in enumerate(zip(raw_pdf, pdf_embeds))
            ]
            self.neo4j.create_chunks(pdf_doc["url_id"], pdf_chunks, pdf_cls)


# ══════════════════════════════════════════════════════════════════════════════
# INTENT ROUTER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class IntentRouter:
    _STRUCTURE = re.compile(
        r"\b(structure|hierarchy|programs?|departments?|levels?|"
        r"specializ|what exists|list|which facult|how many)\b", re.IGNORECASE)
    _CONTENT = re.compile(
        r"\b(explain|what is|define|cours|lecture|chapter|topic|content|"
        r"about|describe|summary|résumé|quel est|c'est quoi)\b", re.IGNORECASE)

    @classmethod
    def route(cls, query: str) -> str:
        hs = bool(cls._STRUCTURE.search(query))
        hc = bool(cls._CONTENT.search(query))
        if hs and not hc: return "STRUCTURE"
        if hc and not hs: return "CONTENT"
        return "HYBRID"

    @classmethod
    def describe(cls, query: str) -> Dict[str, Any]:
        s = cls.route(query)
        d = {
            "STRUCTURE": "Neo4j graph traversal — hierarchy/schema questions.",
            "CONTENT":   "Chunk vector search — semantic content search.",
            "HYBRID":    "Graph structure narrowing + chunk semantic search.",
        }
        return {"strategy": s, "description": d[s]}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    fresh = "--fresh" in sys.argv
    if fresh:
        logger.info("🔥 --fresh flag detected: resetting progress DB")
    IngestionPipeline().run(resume=not fresh)


if __name__ == "__main__":
    main()