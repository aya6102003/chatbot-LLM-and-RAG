#!/usr/bin/env python3
"""
RAG Retrieval Pipeline v12.0 — Farhat Abbas University Sétif 1
==============================================================
Precision-first hybrid retrieval with multilingual semantic expansion.

Changes over v11.0
------------------
  🔴 FIX A — E5 passage prefix asymmetry (CRITICAL correctness bug).
      SemanticRetriever now exposes encode_passage() using "passage: "
      prefix. _title_sim() and _neighbour_is_relevant() now call
      encode_passage() instead of encode(), fixing the silent
      query-vs-query similarity computation that systematically
      underestimated title and neighbour relevance scores.

  🔴 FIX B — Neighbour expansion batched encoding (CRITICAL perf bug).
      _neighbour_is_relevant() was called in a tight loop with a
      separate encode() call per candidate. Now all candidate neighbour
      texts are batch-encoded once before the filter loop, eliminating
      up to 12 serial model inference calls per query.  The semantic
      gate now operates on pre-computed vectors passed in as an argument.

  🟡 FIX C — Fuzzy-only chunk score inconsistency.
      In fuse_scores(), chunks found only via fuzzy (not in pool yet)
      received the raw fuzzy score as their bm25 contribution, while
      chunks already in pool got score*0.8.  Now all fuzzy contributions
      are uniformly multiplied by 0.8 regardless of whether the chunk
      was already pooled.

  🟡 FIX D — Embed cache off-by-one eviction.
      The cache size check used `> EMBED_CACHE_SIZE` (evicts at size+1).
      Changed to `>= EMBED_CACHE_SIZE` so the cache never exceeds the
      configured limit.

  🟡 FIX E — FuzzyRetriever cid_text dict rebuilt every search call.
      The per-call dict comprehension `{e[1]: e[2] for e in self._entries}`
      was O(N) on every query.  Moved to __init__ as self._cid_text.

  🟢 FIX F — Dead variable `lst` removed from _reconstruct_windows.
      The loop variable was assigned but never used; removed to prevent
      confusion.

  🟢 FIX G — GraphExpander __init__ wrapped in try/except.
      A hard Neo4j connection failure at startup no longer crashes the
      entire RAGRetriever init.  get_neighbors() already had protection;
      now the driver creation is equally safe.

All other components (fusion logic, reranker calibration, dynamic top-k,
entity filtering, answerability gate, query expansion) are unchanged.

Install
-------
  pip install sentence-transformers chromadb neo4j numpy rank-bm25 rapidfuzz
"""

# ─────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from neo4j import GraphDatabase

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    logging.warning("rank_bm25 not installed — BM25 disabled.  pip install rank-bm25")

try:
    from rapidfuzz import process as fuzz_process, fuzz as _fuzz
    _FUZZ_OK = True
except ImportError:
    _FUZZ_OK = False
    logging.warning("rapidfuzz not installed — fuzzy title search disabled.")

# ── Add this import at the top of the file (with other imports) ──
try:
    import argostranslate.translate as _argos
    _ARGOS_OK = True
except ImportError:
    _ARGOS_OK = False
    logging.warning("argostranslate not installed — translation fallback disabled. "
                    "Run: pip install argostranslate")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
METADATA_PATH = "./metadata.json"
COLLECTION    = "university_data"

EMBED_MODEL  = "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Retrieval pool sizes ──────────────────────────────────────
TOP_K_VECTOR = 40
TOP_K_BM25   = 20
TOP_K_RERANK = 15
TOP_K_FINAL  = 8

# ── Fuzzy title search ────────────────────────────────────────
TOP_K_FUZZY     = 10
FUZZY_MIN_SCORE = 65

# ── ChromaDB distance space ───────────────────────────────────
CHROMA_SPACE = "cosine"

# ── Fusion weights (semantic-first) ──────────────────────────
W_SEM   = 0.70
W_BM25  = 0.10
W_TITLE = 0.20

# ── Final blend after reranker calibration ────────────────────
W_FUSED  = 0.50
W_RERANK = 0.50

# ── Reranker score calibration ────────────────────────────────
RERANK_POWER   = 2.5

# ── Hard floor on calibrated rerank score ────────────────────
RERANK_MIN_CAL = 0.25

# ── Dynamic top-k ─────────────────────────────────────────────
DYN_TOP_K_MIN        = 3
DYN_TOP_K_MAX        = 8
DYNAMIC_SCORE_MARGIN = 0.25

# ── Additive boosts ───────────────────────────────────────────
LANG_MATCH_BOOST    = 0.04
ACADEMIC_BOOST_CAP  = 0.04
AUTHORITY_BOOST_CAP = 0.03

# ── Answerability gate ────────────────────────────────────────
ANSWER_THRESHOLD = 0.35

# ── Deduplication ─────────────────────────────────────────────
DEDUP_CHARS = 200

# ── Neighbour expansion ───────────────────────────────────────
NEIGHBOR_COUNT          = 3
NEIGHBOR_WINDOW         = 2
NEIGHBOR_SEED_MIN_SCORE = 0.40
NEIGHBOR_SCORE_INHERIT  = 0.80

# ── Semantic + keyword dual gate for neighbour expansion ──────
NBR_SEM_FLOOR = 0.30
NBR_KW_FLOOR  = 0.15

# ── Context window reconstruction ─────────────────────────────
CONTEXT_WINDOW_SIZE = 1

# ── Entity filter ─────────────────────────────────────────────
ENTITY_FUZZ_THRESHOLD = 72
ENTITY_MIN_TOKEN_LEN  = 3

# ── Embedding cache ───────────────────────────────────────────
EMBED_CACHE_SIZE = 256

# ── University abbreviation expansion ────────────────────────
# Maps short codes → full terms in multiple languages
# Add your own abbreviations here as needed
_UNI_ABBREVS: Dict[str, List[str]] = {
    # Semester codes
    "s1": ["semestre 1", "semester 1", "الفصل 1"],
    "s2": ["semestre 2", "semester 2", "الفصل 2"],
    "s3": ["semestre 3", "semester 3", "الفصل 3"],
    "s4": ["semestre 4", "semester 4", "الفصل 4"],
    "s5": ["semestre 5", "semester 5", "الفصل 5"],
    "s6": ["semestre 6", "semester 6", "الفصل 6"],
    # Licence levels
    "l1": ["licence 1", "première année licence", "السنة الأولى ليسانس"],
    "l2": ["licence 2", "deuxième année licence", "السنة الثانية ليسانس"],
    "l3": ["licence 3", "troisième année licence", "السنة الثالثة ليسانس"],
    # Master levels
    "m1": ["master 1", "première année master", "السنة الأولى ماستر"],
    "m2": ["master 2", "deuxième année master", "السنة الثانية ماستر"],
    # Doctorat
    "d":  ["doctorat", "doctorate", "دكتوراه"],
    # Specialty abbreviations — add yours here
    "mi": ["mathématiques et informatique", "mathematics and computer science",
           "رياضيات وإعلام آلي"],
    "st": ["sciences et technologie", "science and technology",
           "علوم وتكنولوجيا"],
    "sm": ["sciences de la matière", "material sciences", "علوم المادة"],
    "sv": ["sciences de la vie", "life sciences", "علوم الحياة"],
    "sn": ["sciences de la nature", "natural sciences", "علوم الطبيعة"],
    "gl": ["génie logiciel", "software engineering", "هندسة البرمجيات"],
    "rsd": ["réseaux et systèmes distribués", "networks and distributed systems",
            "شبكات وأنظمة موزعة"],
    "tp":  ["travaux pratiques", "practical work", "أعمال تطبيقية"],
    "td":  ["travaux dirigés", "tutorial", "أعمال موجهة"],
    "cc":  ["contrôle continu", "continuous assessment", "تقييم مستمر"],
    "em":  ["examen final", "final exam", "امتحان نهائي"],
}

# ── Arabic → Latin transliteration map (for proper names) ────
_AR_TRANSLIT: Dict[str, str] = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a",
    "ب": "b", "ت": "t", "ث": "th",
    "ج": "dj", "ح": "h", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "ch", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "dh", "ع": "a", "غ": "gh",
    "ف": "f", "ق": "k", "ك": "k", "ل": "l",
    "م": "m", "ن": "n", "ه": "h", "و": "ou",
    "ي": "i", "ى": "a", "ة": "e",
    "ّ": "", "َ": "", "ُ": "", "ِ": "",
    "ً": "", "ٌ": "", "ٍ": "", "ْ": "",
}

# ── Trigger words that signal a proper name follows ───────────
_NAME_TRIGGERS = re.compile(
    r"\b(dr|pr|prof|professeur|docteur|mr|mme|mlle|"
    r"أستاذ|دكتور|أ\.د|د\.)\s*\.?\s*",
    re.IGNORECASE | re.UNICODE,
)


# ─────────────────────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────────────────────
_STOPWORDS: Set[str] = {
    "le","la","les","de","du","des","et","en","un","une","pour","avec",
    "dans","sur","par","est","ce","qui","que","quoi","comment","quel",
    "je","tu","il","elle","nous","vous","ils","elles","mon","ton","son",
    "mes","tes","ses","nos","vos","leurs","au","aux","ou","si","ne","pas",
    "the","a","an","and","or","of","in","to","for","is","are","was","were",
    "it","its","this","that","with","by","on","at","from","be","been","has",
    "have","had","do","does","did","will","would","could","should","may",
    "في","من","إلى","على","عن","مع","هو","هي","هم","هن","أنا","نحن",
    "أنت","أنتم","كان","كانت","يكون","التي","الذي","الذين","اللتي","ما",
    "لا","هذا","هذه","ذلك","تلك","قد","لقد","إن","أن","لكن","أو",
    "who","what","where","when","why","how","which",
    "qui","que","quoi","comment","où","quand","quel","quelle",
    "من","ماذا","أين","متى","كيف","أي",
}

# ─────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:     str
    text:         str
    score:        float
    metadata:     Dict  = field(default_factory=dict)
    is_neighbor:  bool  = False
    sem_score:    float = 0.0
    bm25_score:   float = 0.0
    title_score:  float = 0.0
    fused_score:  float = 0.0
    rerank_raw:   float = 0.0
    rerank_cal:   float = 0.0


# ══════════════════════════════════════════════════════════════
# LAYER 1 — QUERY UNDERSTANDING
# ══════════════════════════════════════════════════════════════

class QueryUnderstanding:
    """Extended intent detection (unchanged from v11)."""

    _RE_AR = re.compile(r"[\u0600-\u06FF]")
    _RE_FR = re.compile(
        r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
        r"comment|quoi|qui|quel|quelle|mon|ton|son|leur|nos|vos)\b",
        re.IGNORECASE,
    )
    _BARE_NAME_RE = re.compile(
        r"^[A-ZÀ-Ö][a-zà-ö]+$|^[\u0600-\u06FF]{2,}$", re.UNICODE
    )
    _SIGNALS: Dict[str, re.Pattern] = {
        "translation": re.compile(
            r"translat|traduir|ترجم|translate to|en arabe|in french|in arabic|"
            r"بالفرنسية|بالعربية|بالإنجليزية|en anglais|in english",
            re.IGNORECASE),
        "person_lookup": re.compile(
            r"mail|email|contact|phone|téléphone|bureau|office|"
            r"prof\b|professeur|docteur|dr\.|mr\.|mme\.|"
            r"بريد|هاتف|مكتب|أستاذ|دكتور|"
            r"\bwho\s+is\b|\bwho's\b|\bc'est\s+qui\b|\bqui\s+est\b|"
            r"\bمن\s+هو\b|\bمن\s+هي\b|\bمن\s+هم\b|"
            r"\bprofil\b|\bfiche\b|\bbiographie\b",
            re.IGNORECASE),
        "table_query": re.compile(
            r"liste|list|tableau|table|tous les|all|كل|قائمة|جدول",
            re.IGNORECASE),
        "course_query": re.compile(
            r"cours|course|module|matière|syllabus|td\b|tp\b|"
            r"semestre|semester|programme|filière|licence|master|doctorat|"
            r"\b[SsLlMm][1-6]\b|"
            r"مقياس|فصل|برنامج|تخصص",
            re.IGNORECASE),
        "admin_query": re.compile(
            r"inscription|registration|calendrier|deadline|scolarité|"
            r"examen|exam|résultat|result|تسجيل|امتحان|نتيجة|إدارة",
            re.IGNORECASE),
    }

    def detect_language(self, text: str) -> str:
        s = text[:300]
        ar_chars = len(self._RE_AR.findall(s))
        # A single Arabic word has fewer than 5 chars but is clearly Arabic.
        # Use >5 for longer text, but for short queries (≤15 chars) lower to ≥1.
        ar_threshold = 1 if len(s.strip()) <= 15 else 5
        if ar_chars >= ar_threshold: return "ar"
        if len(self._RE_FR.findall(s)) > 2: return "fr"
        return "en"

    def detect_intent(self, query: str) -> str:
        q = query.lower()
        for intent, pat in self._SIGNALS.items():
            if pat.search(q):
                return intent
        if self._is_bare_name(query):
            return "person_lookup"
        return "general_info"

    def _is_bare_name(self, query: str) -> bool:
        tokens = query.strip().split()
        if not (1 <= len(tokens) <= 3): return False
        return sum(
            1 for t in tokens if self._BARE_NAME_RE.match(t)
        ) >= max(1, len(tokens) - 1)

    def normalize_for_bm25(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        text = text.replace("ة","ه").replace("ى","ي")
        text = (text.replace("é","e").replace("è","e").replace("ê","e")
                    .replace("à","a").replace("â","a").replace("ù","u")
                    .replace("û","u").replace("î","i").replace("ô","o")
                    .replace("ç","c"))
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", text).strip()

    def extract_keywords(self, norm_query: str) -> List[str]:
        tokens = norm_query.split()
        kws: List[str] = []
        for t in tokens:
            if re.match(r"^[a-zA-Z\u0600-\u06FF]{1,3}\d+$", t):
                kws.append(t)
                continue
            if len(t) >= 3 and t not in _STOPWORDS:
                kws.append(t)
        return kws if kws else tokens

    def extract_entity_tokens(self, query: str) -> List[str]:
        clean = re.sub(
            r"(?i)^(who\s+is|qui\s+est|c'est\s+qui|من\s+هو|من\s+هي|"
            r"quel\s+est|what\s+is|tell\s+me\s+about|parle[rz]?\s+moi\s+de|"
            r"give\s+me\s+info\s+(on|about)|informations?\s+sur)\s+",
            "", query.strip()
        )
        tokens = clean.split()
        entity = [
            t for t in tokens
            if (len(t) >= ENTITY_MIN_TOKEN_LEN and
                t not in _STOPWORDS and
                (t[0].isupper() or bool(re.search(r"[\u0600-\u06FF]", t))))
        ]
        return entity if entity else []

    def analyze(self, query: str) -> Dict:
        norm = self.normalize_for_bm25(query)
        return {
            "query":         query,
            "language":      self.detect_language(query),
            "intent":        self.detect_intent(query),
            "norm":          norm,
            "keywords":      self.extract_keywords(norm),
            "entity_tokens": self.extract_entity_tokens(query),
        }


# ── Add this standalone function (before QueryExpander class) ──

def _argos_translate(text: str, from_code: str, to_code: str) -> Optional[str]:
    """
    Translate text using ArgosTranslate (offline, no API key).
    Returns None if the language pair is not installed or translation fails.
    """
    if not _ARGOS_OK:
        return None
    try:
        result = _argos.translate(text, from_code, to_code)
        if result and result.strip() and result.strip() != text.strip():
            return result.strip()
    except Exception as exc:
        log.debug("ArgosTranslate %s→%s failed: %s", from_code, to_code, exc)
    return None


# ══════════════════════════════════════════════════════════════
# LAYER 2 — QUERY EXPANDER
# ══════════════════════════════════════════════════════════════

def _arabic_to_latin(text: str) -> str:
    """
    Transliterate Arabic characters to Latin approximation.
    Used for proper names — حراق → harrag, بوزيد → bouzid.
    Not a translation — preserves the name phonetically.
    """
    result = []
    for ch in text:
        result.append(_AR_TRANSLIT.get(ch, ch))
    # Clean up: lowercase, collapse spaces, strip punctuation
    latin = "".join(result).lower().strip()
    latin = re.sub(r"\s+", "", latin)          # names have no spaces
    latin = re.sub(r"[^\w]", "", latin)        # remove stray punctuation
    return latin if latin else text


def _is_proper_name_query(query: str, intent: str) -> bool:
    """
    Returns True if the query looks like a proper name search:
      - intent is person_lookup
      - OR query contains a name-trigger word (dr, prof, أستاذ …)
      - OR query is 1–2 tokens of pure Arabic/uppercase Latin with no verbs
    """
    if intent == "person_lookup":
        return True
    if _NAME_TRIGGERS.search(query):
        return True
    return False


def _expand_abbreviations(query: str) -> List[str]:
    """
    Detects university abbreviations in the query and returns
    expanded forms in all languages.

    "s2 mi" → ["semestre 2 mathématiques et informatique",
                "semester 2 mathematics and computer science",
                "الفصل 2 رياضيات وإعلام آلي"]
    """
    tokens = query.lower().strip().split()
    expansions_per_lang: List[List[str]] = [[], [], []]  # fr, en, ar

    found_any = False
    for token in tokens:
        # Strip trailing punctuation from token
        clean = re.sub(r"[^\w]", "", token)
        if clean in _UNI_ABBREVS:
            found_any = True
            for i, expanded in enumerate(_UNI_ABBREVS[clean]):
                expansions_per_lang[i].append(expanded)
        else:
            # Keep the original token for all languages
            for i in range(3):
                expansions_per_lang[i].append(token)

    if not found_any:
        return []

    return [
        " ".join(parts).strip()
        for parts in expansions_per_lang
        if any(p.strip() for p in parts)
    ]


class QueryExpander:
    """
    Always-on multilingual semantic expansion (v11 design, unchanged).

    LLM prompt requests 5 variants: FR / EN / AR paraphrases +
    same-language alternative + more specific version.
    Structural fallback prepends language-instruction prefixes so the
    E5 multilingual model produces cross-lingual embedding coverage
    even without an LLM.
    """

    _SPLIT_RE = re.compile(r"[.!?؟،,;:\n]+")

    _SYS = (
        "You are a multilingual search query generator for a university "
        "knowledge base that contains documents in Arabic, French, and English.\n"
        "Given a user query, output ONLY a JSON array of exactly 5 search queries:\n"
        "  [0] Semantically equivalent query in FRENCH (rephrase, not word-for-word)\n"
        "  [1] Semantically equivalent query in ENGLISH (rephrase, not word-for-word)\n"
        "  [2] Semantically equivalent query in ARABIC (rephrase, not word-for-word)\n"
        "  [3] Alternative phrasing in the SAME language as the input query\n"
        "  [4] A more specific or detailed version of the query (any language)\n"
        "Rules:\n"
        "  - Output ONLY the JSON array. No explanation. No markdown.\n"
        "  - Each element must be a non-empty string.\n"
        "  - Use different vocabulary/phrasing, not just literal translation.\n"
        "  - Focus on semantic meaning, not surface form.\n"
    )

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self._url          = base_url.rstrip("/") + "/api/chat"
        self._model        = model
        self._last_intent  = "general_info"   # ← ADD THIS LINE
        self._ok           = self._ping()

    def _ping(self) -> bool:
        try:
            r = requests.get(self._url.replace("/api/chat", "/api/tags"), timeout=3)
            return r.status_code == 200
        except Exception:
            log.debug("Ollama unavailable — structural multilingual fallback only")
            return False

    def expand(self, analysis: Dict) -> List[str]:
        query    = analysis["query"]
        self._last_intent = analysis.get("intent", "general_info")   # ← ADD THIS LINE
        variants = [query]
    
        if self._ok:
            for v in self._llm_expand(query):
                if v and v.strip() and v.strip() not in variants:
                    variants.append(v.strip())
    
        for v in self._struct_expand(query, analysis.get("language", "en")):
            if v not in variants:
                variants.append(v)
    
        return variants[:8]

    def _llm_expand(self, query: str) -> List[str]:
        payload = {
            "model":    self._model,
            "messages": [
                {"role": "system", "content": self._SYS},
                {"role": "user",   "content": f'Query: "{query}"'},
            ],
            "stream":  False,
            "options": {
                "temperature":    0.3,
                "num_predict":    300,
                "repeat_penalty": 1.2,
            },
        }
        try:
            r = requests.post(self._url, json=payload, timeout=12)
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            m = re.search(r"\[.*?\]", content, re.DOTALL)
            if not m:
                return []
            vs = json.loads(m.group())
            return [str(v).strip() for v in vs if v and str(v).strip()]
        except Exception as exc:
            log.debug("LLM expansion failed: %s", exc)
        return []

# ── Replace _struct_expand() inside QueryExpander ──

    def _struct_expand(self, query: str, lang: str) -> List[str]:
        """
        Smart structural expansion:
          1. Abbreviation expansion → full terms in FR / EN / AR
          2. If proper name query → transliterate Arabic to Latin
          3. Otherwise → prefix-instruction fallback for E5 cross-lingual coverage
        """
        vs: List[str] = []
    
        # ── Sub-query splitting (unchanged) ──────────────────────
        parts = [p.strip() for p in self._SPLIT_RE.split(query) if p.strip()]
        if len(parts) > 1:
            vs += [p for p in parts if len(p.split()) >= 3]
    
        clean = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE).strip()
        if clean != query:
            vs.append(clean)
    
        # ── 1. Abbreviation expansion ─────────────────────────────
        abbrev_expansions = _expand_abbreviations(query)
        for exp in abbrev_expansions:
            if exp and exp not in vs:
                vs.append(exp)
    
        # ── 2. Proper name → transliterate, do NOT translate ─────
        intent = self._last_intent  # set in expand() below
        if _is_proper_name_query(query, intent):
            # Extract the name part (strip trigger words like "dr", "prof")
            name_part = _NAME_TRIGGERS.sub("", query).strip()
            # Check if it contains Arabic characters
            if re.search(r"[\u0600-\u06FF]", name_part):
                latin = _arabic_to_latin(name_part)
                if latin and latin != name_part:
                    vs.append(latin)                       # e.g. "harrag"
                    vs.append(f"dr {latin}")               # e.g. "dr harrag"
                    vs.append(f"prof {latin}")             # e.g. "prof harrag"
                    log.debug("Name transliteration: %r → %r", name_part, latin)
            # Also keep the original Arabic for Arabic-indexed chunks
            # (already in variants[0], no need to re-add)
            return vs   # skip generic language-prefix fallback for names
    
# ── 3. General query → real translation via ArgosTranslate ─
        targets = [code for code in ("ar", "fr", "en") if code != lang]
        for tgt in targets:
            translated = _argos_translate(query, lang, tgt)
            if translated and translated not in vs:
                vs.append(translated)
                log.debug("Translated %s→%s: %r → %r", lang, tgt, query, translated)
            else:
                # Fallback: prefix-instruction string if ArgosTranslate fails
                fallback_map = {
                    "fr": f"en français : {query}",
                    "en": f"in English: {query}",
                    "ar": f"بالعربية: {query}",
                }
                vs.append(fallback_map[tgt])
    
        return vs
# ══════════════════════════════════════════════════════════════
# LAYER 3A — SEMANTIC RETRIEVER
# ══════════════════════════════════════════════════════════════

def _chroma_dist_to_sim(dist: float) -> float:
    if CHROMA_SPACE == "cosine":
        return float(np.clip(1.0 - dist, 0.0, 1.0))
    return float(np.clip(1.0 - dist / 4.0, 0.0, 1.0))


class SemanticRetriever:
    """
    FIX A — Two separate prefixes for query vs passage text.

    The intfloat/multilingual-e5 family is trained with asymmetric prefixes:
      "query: <text>"   — for queries / search strings
      "passage: <text>" — for documents / titles / chunks

    Using "query: " for document text at similarity time (as v11 did for
    _title_sim and _neighbour_is_relevant) computes query-query similarity
    instead of query-passage similarity, systematically underestimating
    how relevant a document is relative to a query.

    encode()         — uses "query: " prefix  → for query strings
    encode_passage() — uses "passage: " prefix → for titles, neighbour chunks
    Both share the same LRU cache (keyed by the full prefixed string).
    """

    _QUERY_PREFIX   = "query: "
    _PASSAGE_PREFIX = "passage: "

    # Keep _PREFIX for backward compat (encode() is unchanged externally)
    _PREFIX = _QUERY_PREFIX

    def __init__(self, model_name: str = EMBED_MODEL):
        log.info("Loading embedding model: %s", model_name)
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
        log.info("Embedder device: %s", dev)
        self._model  = SentenceTransformer(model_name, device=dev)
        self._cache: Dict[str, np.ndarray] = {}

    def _encode_with_prefix(self, texts: List[str], prefix: str) -> np.ndarray:
        """
        Shared implementation for encode() and encode_passage().
        FIX D — cache eviction uses >= so size never exceeds EMBED_CACHE_SIZE.
        """
        results:    List[Optional[np.ndarray]] = [None] * len(texts)
        miss_idx:   List[int]  = []
        miss_texts: List[str]  = []

        for i, t in enumerate(texts):
            key = prefix + t
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                miss_idx.append(i)
                miss_texts.append(t)

        if miss_texts:
            embs = self._model.encode(
                [prefix + t for t in miss_texts],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for local_i, global_i in enumerate(miss_idx):
                vec = embs[local_i]
                key = prefix + miss_texts[local_i]
                # FIX D: evict BEFORE inserting so cache never exceeds limit
                if len(self._cache) >= EMBED_CACHE_SIZE:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = vec
                results[global_i] = vec

        return np.vstack(results)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode query strings with 'query: ' prefix."""
        return self._encode_with_prefix(texts, self._QUERY_PREFIX)

    def encode_passage(self, texts: List[str]) -> np.ndarray:
        """
        FIX A — Encode document/title/chunk text with 'passage: ' prefix.
        Use this for anything that is NOT a search query.
        """
        return self._encode_with_prefix(texts, self._PASSAGE_PREFIX)

    def search(
        self,
        variants:     List[str],
        collection,
        top_k:        int,
        where_filter: Optional[Dict] = None,
    ) -> Dict[str, RetrievedChunk]:
        if not variants:
            return {}

        # Variants are query strings → use encode() with "query: " prefix
        embeddings = self.encode(variants)
        candidates: Dict[str, RetrievedChunk] = {}

        for vec in embeddings:
            kwargs: Dict = dict(
                query_embeddings=[vec.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            if where_filter:
                kwargs["where"] = where_filter

            try:
                res = collection.query(**kwargs)
            except Exception as exc:
                log.warning("ChromaDB query failed: %s", exc)
                continue

            if not res["ids"] or not res["ids"][0]:
                continue

            for i, cid in enumerate(res["ids"][0]):
                sim  = _chroma_dist_to_sim(float(res["distances"][0][i]))
                meta = res["metadatas"][0][i] or {}

                if cid not in candidates or sim > candidates[cid].sem_score:
                    candidates[cid] = RetrievedChunk(
                        chunk_id=cid,
                        text=res["documents"][0][i],
                        score=sim,
                        metadata=meta,
                        sem_score=sim,
                    )

        return candidates


# ══════════════════════════════════════════════════════════════
# LAYER 3B — BM25 RETRIEVER
# ══════════════════════════════════════════════════════════════

class BM25Retriever:
    def __init__(self, metadata_path: str = METADATA_PATH):
        self._chunk_ids: List[str] = []
        self._texts:     List[str] = []
        self._bm25 = None

        if not _BM25_OK:
            return

        p = Path(metadata_path)
        if not p.exists():
            log.warning("metadata.json not found — BM25 disabled")
            return

        qu        = QueryUnderstanding()
        tokenized: List[List[str]] = []

        with open(p, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        for r in records:
            cid  = r.get("chunk_id", "")
            text = r.get("chunk", "") or r.get("text", "")
            if not cid or not text:
                continue
            self._chunk_ids.append(cid)
            self._texts.append(text)

            tok = r.get("tokenized_text", "")
            if isinstance(tok, str) and tok.strip():
                tokens = tok.split()
            elif isinstance(tok, list) and tok:
                tokens = [str(t) for t in tok]
            else:
                tokens = qu.normalize_for_bm25(text).split()
            tokenized.append(tokens)

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            log.info("BM25 index built: %d documents", len(tokenized))

    def search(
        self, keywords: List[str], top_k: int = TOP_K_BM25
    ) -> List[Tuple[str, str, float]]:
        if self._bm25 is None or not keywords:
            return []

        raw = self._bm25.get_scores(keywords)
        mx  = raw.max()
        if mx <= 0:
            return []

        norm    = raw / mx
        top_idx = np.argsort(norm)[::-1][:top_k]
        return [
            (self._chunk_ids[i], self._texts[i], float(norm[i]))
            for i in top_idx if norm[i] > 0
        ]


# ══════════════════════════════════════════════════════════════
# LAYER 3C — FUZZY TITLE RETRIEVER
# ══════════════════════════════════════════════════════════════

class FuzzyRetriever:
    """
    FIX E — cid_text lookup dict built once in __init__ instead of
    being reconstructed on every search() call (was O(N) per query).
    """

    def __init__(self, metadata_path: str = METADATA_PATH):
        self._entries:  List[Tuple[str, str, str]] = []
        self._cid_text: Dict[str, str] = {}   # FIX E

        if not _FUZZ_OK:
            return

        p = Path(metadata_path)
        if not p.exists():
            return

        with open(p, "r", encoding="utf-8") as fh:
            records = json.load(fh)

        for r in records:
            cid  = r.get("chunk_id", "")
            text = r.get("chunk", "") or r.get("text", "")
            if not cid: continue

            primary = (r.get("title","") or r.get("file","") or "").lower()
            if primary:
                self._entries.append((primary, cid, text))

            for alt in (r.get("alternate_titles") or []):
                if alt and alt.lower() != primary:
                    self._entries.append((alt.lower(), cid, text))

        # FIX E — build lookup once
        self._cid_text = {e[1]: e[2] for e in self._entries}
        log.info("Fuzzy index: %d title variants", len(self._entries))

    def search(
        self, query: str, top_k: int = TOP_K_FUZZY
    ) -> List[Tuple[str, str, float]]:
        if not _FUZZ_OK or not self._entries:
            return []

        q      = query.lower().strip()
        titles = [e[0] for e in self._entries]
        hits   = fuzz_process.extract(
            q, titles, scorer=_fuzz.token_set_ratio, limit=top_k * 2
        )

        seen: Dict[str, float] = {}
        for _, raw, idx in hits:
            if raw < FUZZY_MIN_SCORE: continue
            _, cid, _ = self._entries[idx]
            norm = raw / 100.0
            if cid not in seen or norm > seen[cid]:
                seen[cid] = norm

        # FIX E — use pre-built dict instead of rebuilding per call
        return sorted(
            [(cid, self._cid_text.get(cid, ""), s) for cid, s in seen.items()],
            key=lambda x: x[2], reverse=True,
        )[:top_k]


# ══════════════════════════════════════════════════════════════
# TITLE SIMILARITY
# ══════════════════════════════════════════════════════════════

def _title_sim(
    query_vec: np.ndarray,
    meta:      Dict,
    retriever: SemanticRetriever,
) -> float:
    """
    FIX A — Title text is a passage, not a query.
    encode_passage() uses 'passage: ' prefix so the E5 model computes
    proper asymmetric query-vs-passage similarity instead of
    query-vs-query similarity.
    """
    titles: List[str] = []
    t = meta.get("title", "")
    if t: titles.append(t)
    for alt in (meta.get("alternate_titles") or []):
        if alt and alt not in titles:
            titles.append(alt)
    if not titles:
        return 0.0

    # FIX A: was retriever.encode(titles) — wrong prefix for passages
    title_vecs = retriever.encode_passage(titles)
    sims       = title_vecs @ query_vec
    return float(sims.max())


# ══════════════════════════════════════════════════════════════
# FINGERPRINT (near-duplicate filter)
# ══════════════════════════════════════════════════════════════

def _fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text[:DEDUP_CHARS]).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════
# LAYER 4 — SCORE FUSION
# ══════════════════════════════════════════════════════════════

def fuse_scores(
    semantic:   Dict[str, RetrievedChunk],
    bm25_hits:  List[Tuple[str, str, float]],
    fuzz_hits:  List[Tuple[str, str, float]],
    meta_store: "MetadataStore",
    query_vec:  np.ndarray,
    retriever:  SemanticRetriever,
    query_lang: str,
    intent:     str,
) -> List[RetrievedChunk]:
    """
    FIX 1  — Zero-signal chunks (sem=0 AND bm25=0) dropped immediately.
    FIX C  — Fuzzy-only chunks now consistently use score*0.8 for their
              bm25 contribution, matching the treatment of fuzzy scores
              applied to already-pooled chunks.  Previously fuzzy-only
              chunks received the raw fuzzy score (no 0.8 discount),
              making them appear more BM25-confident than pool members.
    v11    — Weights: W_SEM=0.70, W_BM25=0.10, W_TITLE=0.20.
    """
    pool: Dict[str, Dict] = {}

    for cid, chunk in semantic.items():
        pool[cid] = {
            "sem":  chunk.sem_score,
            "bm25": 0.0,
            "text": chunk.text,
            "meta": chunk.metadata,
        }

    for cid, text, score in bm25_hits:
        if cid not in pool:
            rec = meta_store.get(cid)
            if rec is None: continue
            pool[cid] = {"sem": 0.0, "bm25": score,
                         "text": rec.get("chunk", text), "meta": rec}
        else:
            pool[cid]["bm25"] = max(pool[cid]["bm25"], score)

    for cid, text, score in fuzz_hits:
        # FIX C — uniform 0.8 discount for fuzzy scores in all cases
        fuzzy_contrib = score * 0.8
        if cid not in pool:
            rec = meta_store.get(cid)
            if rec is None: continue
            pool[cid] = {"sem": 0.0, "bm25": fuzzy_contrib,
                         "text": rec.get("chunk", text), "meta": rec}
        else:
            pool[cid]["bm25"] = max(pool[cid]["bm25"], fuzzy_contrib)

    fused_chunks: List[RetrievedChunk] = []
    seen_fp: Dict[str, float] = {}

    for cid, d in pool.items():
        sem  = float(d["sem"])
        bm25 = float(d["bm25"])
        text = d["text"]
        meta = d["meta"] if isinstance(d["meta"], dict) else {}

        # Drop zero-signal chunks
        if sem == 0.0 and bm25 == 0.0:
            log.debug("FUSE DROP %s — zero signal (sem=0, bm25=0)", cid[-15:])
            continue

        t_sim = _title_sim(query_vec, meta, retriever)
        fused = W_SEM * sem + W_BM25 * bm25 + W_TITLE * t_sim

        # Additive boosts
        if meta.get("language") == query_lang:
            fused += LANG_MATCH_BOOST
        acad = float(meta.get("academic_score", 0.0) or 0.0)
        fused += min(ACADEMIC_BOOST_CAP, acad * ACADEMIC_BOOST_CAP)
        if intent == "admin_query":
            auth = float(meta.get("authority_score", 0.0) or 0.0)
            fused += min(AUTHORITY_BOOST_CAP, auth * AUTHORITY_BOOST_CAP)

        fp = _fingerprint(text)
        if fp in seen_fp:
            if fused <= seen_fp[fp]:
                log.debug("FUSE DROP %s — near-duplicate", cid[-15:])
                continue
        seen_fp[fp] = fused

        fused_chunks.append(RetrievedChunk(
            chunk_id=cid,
            text=text,
            score=fused,
            metadata=meta,
            sem_score=sem,
            bm25_score=bm25,
            title_score=t_sim,
            fused_score=fused,
        ))

    fused_chunks.sort(key=lambda c: c.score, reverse=True)
    return fused_chunks


# ══════════════════════════════════════════════════════════════
# ANSWERABILITY GATE
# ══════════════════════════════════════════════════════════════

def _passes_answerability(chunks: List[RetrievedChunk]) -> bool:
    return bool(chunks and chunks[0].score >= ANSWER_THRESHOLD)


# ══════════════════════════════════════════════════════════════
# LAYER 5 — CROSS-ENCODER RERANKER
# ══════════════════════════════════════════════════════════════

class Reranker:
    """
    Unchanged from v11.  BGE-reranker-v2-m3 natively handles AR/FR/EN.
    Original query (not expanded variants) is always used for reranking
    to preserve a clean relevance signal.
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        log.info("Loading reranker: %s", model_name)
        self._model = CrossEncoder(model_name)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-float(x))))

    @staticmethod
    def _calibrate(raw: float) -> float:
        return raw ** RERANK_POWER

    def rerank(
        self,
        query:   str,
        chunks:  List[RetrievedChunk],
        top_k:   int,
        min_cal: float = RERANK_MIN_CAL,   # ← add this
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
    
        pairs  = [(query, c.text) for c in chunks]
        logits = self._model.predict(pairs)
    
        kept: List[RetrievedChunk] = []
        for chunk, logit in zip(chunks, logits):
            raw = self._sigmoid(logit)
            cal = self._calibrate(raw)
            chunk.rerank_raw = raw
            chunk.rerank_cal = cal
    
            if cal < min_cal:   # ← use min_cal instead of RERANK_MIN_CAL
                log.debug("RERANK DROP %s — cal=%.3f < %.2f",
                          chunk.chunk_id[-15:], cal, min_cal)
                continue
    
            chunk.score = W_FUSED * chunk.fused_score + W_RERANK * cal
            kept.append(chunk)
    
        kept.sort(key=lambda c: c.score, reverse=True)
        return kept[:top_k]

# ══════════════════════════════════════════════════════════════
# DYNAMIC TOP-K SELECTION
# ══════════════════════════════════════════════════════════════

def _apply_dynamic_topk(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    if not chunks:
        return []

    best  = chunks[0].score
    floor = best - DYNAMIC_SCORE_MARGIN
    kept  = [c for c in chunks if c.score >= floor]

    result = kept[:DYN_TOP_K_MAX]
    if len(result) < DYN_TOP_K_MIN:
        result = chunks[:DYN_TOP_K_MIN]

    if len(result) < len(chunks):
        log.debug("DYN TOP-K: kept %d / %d (best=%.3f floor=%.3f)",
                  len(result), len(chunks), best, floor)
    return result


# ══════════════════════════════════════════════════════════════
# ENTITY-AWARE POST-FILTER
# ══════════════════════════════════════════════════════════════

def _entity_filter(
    chunks:        List[RetrievedChunk],
    entity_tokens: List[str],
    intent:        str,
) -> List[RetrievedChunk]:
    if intent != "person_lookup" or not entity_tokens or not _FUZZ_OK:
        return chunks

    def _chunk_contains_entity(text: str) -> bool:
        text_low = text.lower()
        for token in entity_tokens:
            if len(token) < ENTITY_MIN_TOKEN_LEN:
                continue
            if token.lower() in text_low:
                return True
            score = _fuzz.partial_ratio(token.lower(), text_low)
            if score >= ENTITY_FUZZ_THRESHOLD:
                return True
        return False

    kept:    List[RetrievedChunk] = []
    dropped: List[RetrievedChunk] = []

    for c in chunks:
        if _chunk_contains_entity(c.text):
            kept.append(c)
        else:
            log.debug("ENTITY DROP %s — tokens %s not found",
                      c.chunk_id[-15:], entity_tokens)
            dropped.append(c)

    if not kept and chunks:
        log.debug("ENTITY FILTER kept top-1 as anchor")
        kept = [chunks[0]]

    return kept


# ══════════════════════════════════════════════════════════════
# SMART NEIGHBOUR EXPANSION
# ══════════════════════════════════════════════════════════════

def _keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of query keywords present in the neighbour text."""
    if not keywords:
        return 0.0
    tl = text.lower()
    return sum(1 for kw in keywords if kw in tl) / len(keywords)


def _filter_neighbours_by_relevance(
    candidates:  List[Tuple[str, str]],   # (chunk_id, chunk_text)
    query_vec:   np.ndarray,
    retriever:   SemanticRetriever,
    keywords:    List[str],
) -> List[str]:
    """
    FIX A + FIX B — Batched neighbour relevance filtering.

    FIX B: All candidate neighbour texts are batch-encoded in a single
    model call instead of one encode() per candidate.  For up to 12
    candidates this eliminates up to 11 serial GPU/CPU round-trips.

    FIX A: Neighbour chunks are document passages, not queries.
    encode_passage() is used so the E5 model applies the correct
    "passage: " prefix, giving proper asymmetric similarity scores.

    A neighbour passes if:
      (a) cosine(query_vec, passage_vec) >= NBR_SEM_FLOOR  [primary]
      OR
      (b) keyword overlap >= NBR_KW_FLOOR                  [secondary]
    """
    if not candidates:
        return []

    accepted_ids: List[str] = []
    texts = [text for _, text in candidates]
    ids   = [cid  for cid, _ in candidates]

    # Fast lexical pre-filter (no model call needed)
    lexical_pass = {
        cid for cid, text in candidates
        if keywords and _keyword_overlap(text, keywords) >= NBR_KW_FLOOR
    }
    # Collect IDs that need semantic evaluation (didn't pass lexical gate)
    sem_needed_idx  = [i for i, cid in enumerate(ids) if cid not in lexical_pass]

    if sem_needed_idx:
        # FIX B — single batch encode for all remaining candidates
        # FIX A — use encode_passage() for document text
        try:
            sem_texts = [texts[i] for i in sem_needed_idx]
            # FIX A: encode_passage, not encode
            passage_vecs = retriever.encode_passage(sem_texts)
            sims = passage_vecs @ query_vec   # shape: (N,)
        except Exception as exc:
            log.debug("Batch neighbour encoding failed: %s", exc)
            passage_vecs = None
            sims         = None

        for local_i, global_i in enumerate(sem_needed_idx):
            cid = ids[global_i]
            if sims is not None and float(sims[local_i]) >= NBR_SEM_FLOOR:
                log.debug("NBR ACCEPT %s sem_sim=%.3f", cid[-15:], sims[local_i])
                lexical_pass.add(cid)

    for cid in ids:
        if cid in lexical_pass:
            accepted_ids.append(cid)
        else:
            log.debug("NBR SKIP %s — low sem+kw relevance", cid[-15:])

    return accepted_ids


def _same_doc_prefix(cid_a: str, cid_b: str) -> bool:
    prefix_a = cid_a.rsplit("_c", 1)[0] if "_c" in cid_a else cid_a
    prefix_b = cid_b.rsplit("_c", 1)[0] if "_c" in cid_b else cid_b
    return prefix_a == prefix_b


# ══════════════════════════════════════════════════════════════
# LAYER 6 — GRAPH EXPANDER
# ══════════════════════════════════════════════════════════════

class GraphExpander:
    """
    FIX G — __init__ wrapped in try/except so a Neo4j connection
    failure at startup does not crash RAGRetriever initialisation.
    get_neighbors() already had protection; now the driver creation
    is equally resilient.
    """

    def __init__(self):
        self._driver = None
        try:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
        except Exception as exc:
            log.warning("Neo4j driver init failed: %s — neighbour expansion disabled", exc)

    def get_neighbors(
        self,
        chunk_ids: List[str],
        window:    int = NEIGHBOR_WINDOW,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        result: Dict[str, Tuple[List[str], List[str]]] = {
            cid: ([], []) for cid in chunk_ids
        }
        if not chunk_ids or self._driver is None:
            return result

        try:
            with self._driver.session() as sess:
                records = sess.run(
                    """
                    UNWIND $ids AS cid
                    MATCH (c:Chunk {id: cid})
                    OPTIONAL MATCH (p)-[:NEXT_CHUNK*1..%d]->(c)
                    OPTIONAL MATCH (c)-[:NEXT_CHUNK*1..%d]->(n)
                    RETURN cid,
                           collect(DISTINCT p.id) AS prev_ids,
                           collect(DISTINCT n.id) AS next_ids
                    """ % (window, window),
                    ids=chunk_ids,
                )
                for rec in records:
                    cid = rec["cid"]
                    result[cid] = (
                        [x for x in (rec["prev_ids"] or []) if x],
                        [x for x in (rec["next_ids"] or []) if x],
                    )
        except Exception as exc:
            log.warning("Neo4j expansion failed: %s", exc)

        return result

    def close(self):
        if self._driver is not None:
            self._driver.close()


# ══════════════════════════════════════════════════════════════
# METADATA STORE
# ══════════════════════════════════════════════════════════════

class MetadataStore:
    def __init__(self, path: str = METADATA_PATH):
        self._data: Dict[str, Dict] = {}
        p = Path(path)
        if not p.exists():
            log.warning("metadata.json not found at %s", path)
            return
        with open(p, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        self._data = {r["chunk_id"]: r for r in records if "chunk_id" in r}
        log.info("MetadataStore: %d chunks", len(self._data))

    def get(self, chunk_id: str) -> Optional[Dict]:
        return self._data.get(chunk_id)


# ══════════════════════════════════════════════════════════════
# BOILERPLATE FILTER
# ══════════════════════════════════════════════════════════════

_BOILERPLATE = frozenset([
    "call for papers","قراءة المزيد","skip to content","back to top",
    "accueil | contact","home | about | contact","print page",
    "se connecter","login",
])

def _is_boilerplate(text: str, meta: Dict) -> bool:
    if len(text.strip()) < 30:
        return True
    low = text.lower()
    if any(p in low for p in _BOILERPLATE):
        return True
    title = (meta.get("title","") or "").lower()
    if title and len(text.replace(title,"").strip()) < 40:
        return True
    return False


# ══════════════════════════════════════════════════════════════
# CONTEXT WINDOW RECONSTRUCTION
# ══════════════════════════════════════════════════════════════

def _reconstruct_windows(
    chunks:     List[RetrievedChunk],
    meta_store: MetadataStore,
    window:     int = CONTEXT_WINDOW_SIZE,
) -> List[RetrievedChunk]:
    """
    FIX F — Removed dead loop variable `lst` that was assigned in the
    for-loop but never used (direct references to prev_parts/next_parts
    were used instead).
    """
    if window == 0:
        return chunks

    selected_ids = {c.chunk_id for c in chunks}

    for c in chunks:
        meta       = c.metadata or {}
        chunk_idx  = meta.get("chunk_index")
        doc_prefix = c.chunk_id.rsplit("_c", 1)[0] if "_c" in c.chunk_id else None

        if chunk_idx is None or doc_prefix is None:
            continue

        prev_parts: List[str] = []
        next_parts: List[str] = []

        for delta in range(1, window + 1):
            # FIX F — removed dead `lst` variable; use sign directly
            for sign in (-delta, +delta):
                nid = f"{doc_prefix}_c{chunk_idx + sign}"
                rec = meta_store.get(nid)
                if rec and nid not in selected_ids:
                    t = (rec.get("chunk","") or rec.get("text","")).strip()
                    if t:
                        if sign < 0:
                            prev_parts.insert(0, t)
                        else:
                            next_parts.append(t)

        parts  = prev_parts + [c.text.strip()] + next_parts
        c.text = "\n\n".join(p for p in parts if p)

    return chunks


# ══════════════════════════════════════════════════════════════
# LLM CONTEXT FORMATTER
# ══════════════════════════════════════════════════════════════

def _format_llm_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context available."

    sep    = "\n" + "─" * 55 + "\n"
    blocks: List[str] = []

    for i, c in enumerate(chunks, 1):
        meta     = c.metadata or {}
        title    = meta.get("title", "Unknown")
        faculty  = meta.get("faculty", "")
        doc_type = meta.get("doc_type", "")
        lang     = meta.get("language", "")
        url      = meta.get("url","") or meta.get("pdf_url","")

        header = f"[{i}] {title}"
        if faculty or doc_type:
            header += f"  ({' / '.join(filter(None, [faculty, doc_type]))})"
        if lang:
            header += f"  [{lang.upper()}]"

        lines = [header, c.text.strip()]
        if url:
            lines.append(f"Source: {url}")

        blocks.append("\n".join(lines))

    return sep.join(blocks)


# ══════════════════════════════════════════════════════════════
# MAIN RAG RETRIEVER
# ══════════════════════════════════════════════════════════════

class RAGRetriever:
    """
    v12.0 — All v11 improvements retained; surgical bug fixes applied.

    Steps
    -----
    1.  Query analysis + entity extraction
    2.  Multilingual query expansion
    3A. Semantic search — ALL multilingual variants, encode() with query prefix
    3B. BM25 search
    3C. Fuzzy title search
    4.  Fusion — semantic-first weights; uniform fuzzy discount (FIX C)
    4B. Boilerplate filter
    5.  Cross-encoder rerank — calibration + floor
    6.  Smart neighbour expansion — batched semantic gate (FIX A, FIX B)
    7.  Second rerank pass
    8.  Entity filter
    9.  Dynamic top-k
    10. Context window reconstruction (dead variable removed, FIX F)
    11. Answerability gate
    """

    def __init__(self):
        log.info("Initializing RAGRetriever v12.0")
        self._qu       = QueryUnderstanding()
        self._expander = QueryExpander(OLLAMA_URL, OLLAMA_MODEL)
        self._semantic = SemanticRetriever(EMBED_MODEL)
        self._bm25     = BM25Retriever(METADATA_PATH)
        self._fuzzy    = FuzzyRetriever(METADATA_PATH)
        self._reranker = Reranker()
        self._graph    = GraphExpander()          # FIX G — no longer crashes on Neo4j down
        self._meta     = MetadataStore(METADATA_PATH)
        self._chroma   = chromadb.PersistentClient(path=CHROMA_PATH)
        self._col      = self._chroma.get_collection(COLLECTION)
        log.info("RAGRetriever v12.0 ready")

    def retrieve(
        self,
        query:      str,
        top_k:      int           = TOP_K_FINAL,
        faculty:    Optional[str] = None,
        department: Optional[str] = None,
    ) -> List[RetrievedChunk]:

        # ── Step 1 ────────────────────────────────────────────
        analysis       = self._qu.analyze(query)
        intent         = analysis["intent"]
        lang           = analysis["language"]
        keywords       = analysis["keywords"]
        entity_tokens  = analysis["entity_tokens"]
        log.info("Query lang=%s intent=%s keywords=%s entities=%s",
                 lang, intent, keywords[:5], entity_tokens)

        # ── Step 2 — Multilingual expansion ───────────────────
        variants  = self._expander.expand(analysis)
        # query_vec uses "query: " prefix (correct for query strings)
        query_vec = self._semantic.encode([query])[0]
        log.info("Expanded to %d multilingual variants: %s",
                 len(variants), [v[:40] for v in variants])

        # ── Steps 3A–3C ───────────────────────────────────────
        where         = self._build_where(faculty, department)
        semantic_hits = self._semantic.search(variants, self._col, TOP_K_VECTOR, where)
        bm25_hits     = self._bm25.search(keywords, TOP_K_BM25)
        fuzz_hits     = self._fuzzy.search(query, TOP_K_FUZZY)

        log.info("Candidates — semantic:%d  BM25:%d  fuzzy:%d",
                 len(semantic_hits), len(bm25_hits), len(fuzz_hits))

        # ── Step 4: fusion ────────────────────────────────────
        fused = fuse_scores(
            semantic_hits, bm25_hits, fuzz_hits,
            self._meta, query_vec, self._semantic,
            lang, intent,
        )
        fused = [c for c in fused if not _is_boilerplate(c.text, c.metadata)]

        if not fused:
            log.warning("All candidates filtered — using raw semantic fallback")
            fused = sorted(semantic_hits.values(),
                           key=lambda c: c.sem_score, reverse=True)
            for c in fused:
                c.fused_score = c.sem_score

        if not fused:
            return []

        # ── Step 5: rerank ─────────────────────────────────────────
        # For short/name queries, the cross-encoder gives lower raw scores —
        # lower the floor so valid results aren't all dropped.
        _rerank_floor = RERANK_MIN_CAL
        if intent == "person_lookup" or len(query.strip().split()) <= 2:
            _rerank_floor = 0.10   # relaxed for bare-name queries
        
        reranked = self._reranker.rerank(query, fused, top_k=TOP_K_RERANK,
                                          min_cal=_rerank_floor)

        if not reranked:
            log.warning("Reranker dropped all chunks — no results")
            return []

        # ── Step 6: smart neighbour expansion ─────────────────
        if intent != "translation":
            w = NEIGHBOR_WINDOW + (1 if intent == "course_query" else 0)
            seed_ids = [
                c.chunk_id for c in reranked[:NEIGHBOR_COUNT]
                if c.score >= NEIGHBOR_SEED_MIN_SCORE
            ]
            nbr_map  = self._graph.get_neighbors(seed_ids, w)
            expanded = list(reranked)
            seen_ids = {c.chunk_id for c in expanded}

            # Collect all valid neighbour candidates first (FIX B — batch)
            nbr_candidates: List[Tuple[str, str]] = []  # (nid, text)
            for seed in reranked[:NEIGHBOR_COUNT]:
                if seed.score < NEIGHBOR_SEED_MIN_SCORE: continue
                prev_ids, next_ids = nbr_map.get(seed.chunk_id, ([], []))
                for nid in (prev_ids[:w] + next_ids[:w]):
                    if nid in seen_ids: continue
                    if not _same_doc_prefix(seed.chunk_id, nid):
                        log.debug("NBR SKIP %s — cross-document", nid[-15:])
                        continue
                    rec = self._meta.get(nid)
                    if not rec: continue
                    nbr_text = rec.get("chunk","")
                    if not nbr_text or len(nbr_text.strip()) < 30: continue
                    nbr_candidates.append((nid, nbr_text))

            # FIX B — single batched semantic + keyword gate for all candidates
            if nbr_candidates:
                accepted_ids = _filter_neighbours_by_relevance(
                    nbr_candidates, query_vec, self._semantic, keywords
                )
                accepted_set = set(accepted_ids)
                # Build a seed-score lookup for score inheritance
                seed_score_map: Dict[str, float] = {}
                for seed in reranked[:NEIGHBOR_COUNT]:
                    if seed.score < NEIGHBOR_SEED_MIN_SCORE: continue
                    prev_ids, next_ids = nbr_map.get(seed.chunk_id, ([], []))
                    for nid in (prev_ids[:w] + next_ids[:w]):
                        if nid in accepted_set and nid not in seed_score_map:
                            seed_score_map[nid] = seed.fused_score

                for nid, nbr_text in nbr_candidates:
                    if nid not in accepted_set: continue
                    if nid in seen_ids: continue
                    rec       = self._meta.get(nid)
                    fused_nbr = seed_score_map.get(nid, 0.0) * NEIGHBOR_SCORE_INHERIT
                    seen_ids.add(nid)
                    expanded.append(RetrievedChunk(
                        chunk_id=nid,
                        text=nbr_text,
                        score=fused_nbr,
                        metadata=rec or {},
                        is_neighbor=True,
                        fused_score=fused_nbr,
                    ))
        else:
            expanded = list(reranked)

        # ── Step 7: second rerank ─────────────────────────────────
        final = self._reranker.rerank(query, expanded, top_k=top_k,
                               min_cal=_rerank_floor)
        
        # ── Step 8: entity filter ─────────────────────────────
        final = _entity_filter(final, entity_tokens, intent)

        # ── Step 9: dynamic top-k ─────────────────────────────
        final = _apply_dynamic_topk(final)

        # ── Step 10: context windows ──────────────────────────
        final = _reconstruct_windows(final, self._meta, CONTEXT_WINDOW_SIZE)

        # ── Step 11: answerability gate ───────────────────────
        _ans_threshold = (
            0.20 if (intent == "person_lookup" or len(query.strip().split()) <= 2)
            else ANSWER_THRESHOLD
        )
        if not (final and final[0].score >= _ans_threshold):
            log.warning(
                "Answerability gate: best=%.3f < %.2f — NO_ANSWER",
                final[0].score if final else 0.0,
                _ans_threshold,
            )
            return []
        log.info(
            "Final: %d chunks  scores=%s",
            len(final),
            [round(c.score, 3) for c in final],
        )
        return final

    def close(self):
        self._graph.close()

    @staticmethod
    def _build_where(
        faculty:    Optional[str],
        department: Optional[str],
    ) -> Optional[Dict]:
        clauses = []
        if faculty:    clauses.append({"faculty":    {"$eq": faculty}})
        if department: clauses.append({"department": {"$eq": department}})
        if not clauses: return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ══════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ══════════════════════════════════════════════════════════════

_retriever: Optional[RAGRetriever] = None

def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


# ── PUBLIC API ─────────────────────────────────────────────────

def retrieve_for_llm(
    query:      str,
    top_k:      int           = TOP_K_FINAL,
    faculty:    Optional[str] = None,
    department: Optional[str] = None,
) -> List[Dict]:
    """
    Primary entry point for LLM.py and api.py.

    Returns [] when answerability gate fires.

    Each dict
    ---------
    chunk_id, text, score, is_neighbor,
    sem_score, bm25_score, title_score, fused_score,
    rerank_raw, rerank_cal,
    url, pdf_url, file_path, source, title, language, chunk_index,
    metadata,
    llm_context  (pre-formatted numbered blocks, first result only)
    """
    chunks = _get_retriever().retrieve(
        query, top_k=top_k, faculty=faculty, department=department,
    )

    result = [
        {
            "chunk_id":    c.chunk_id,
            "text":        c.text,
            "score":       round(c.score,        4),
            "is_neighbor": c.is_neighbor,
            "sem_score":   round(c.sem_score,    4),
            "bm25_score":  round(c.bm25_score,   4),
            "title_score": round(c.title_score,  4),
            "fused_score": round(c.fused_score,  4),
            "rerank_raw":  round(c.rerank_raw,   4),
            "rerank_cal":  round(c.rerank_cal,   4),
            "url":         c.metadata.get("url",         ""),
            "pdf_url":     c.metadata.get("pdf_url",     ""),
            "file_path":   c.metadata.get("file_path",   ""),
            "source":      c.metadata.get("source",      ""),
            "title":       c.metadata.get("title",       ""),
            "language":    c.metadata.get("language",    ""),
            "chunk_index": c.metadata.get("chunk_index",
                           c.metadata.get("index",
                           c.metadata.get("order", None))),
            "metadata":    c.metadata,
        }
        for c in chunks
    ]

    if result:
        result[0]["llm_context"] = _format_llm_context(chunks)

    return result


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag.py "query" '
              '[--faculty FAC] [--department DEPT] [--top-k N] [--debug]')
        sys.exit(1)

    query      = sys.argv[1]
    faculty    = None
    department = None
    top_k      = TOP_K_FINAL

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--faculty" and i + 1 < len(sys.argv):
            faculty = sys.argv[i + 1]; i += 2
        elif arg == "--department" and i + 1 < len(sys.argv):
            department = sys.argv[i + 1]; i += 2
        elif arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1]); i += 2
        elif arg == "--debug":
            logging.getLogger().setLevel(logging.DEBUG); i += 1
        else:
            i += 1

    results = retrieve_for_llm(query, top_k=top_k,
                                faculty=faculty, department=department)

    print("\n" + "=" * 64)
    print(f"QUERY  : {query}")
    print(f"FILTER : faculty={faculty!r}  department={department!r}")
    print(f"RESULTS: {len(results)}")
    print("=" * 64)

    if not results:
        print(f"\n⚠  NO_ANSWER (best score < {ANSWER_THRESHOLD}).")
        sys.exit(0)

    for rank, r in enumerate(results, 1):
        tag  = " [NBR]" if r["is_neighbor"] else ""
        meta = r["metadata"]
        print(f"\n{rank}. {r['chunk_id']}{tag}")
        print(f"   Final  : {r['score']:.4f}  "
              f"(fused={r['fused_score']:.3f}  "
              f"rerank_raw={r['rerank_raw']:.3f}  "
              f"rerank_cal={r['rerank_cal']:.3f})")
        print(f"   Signals: sem={r['sem_score']:.3f}  "
              f"bm25={r['bm25_score']:.3f}  "
              f"title={r['title_score']:.3f}")
        print(f"   Faculty: {meta.get('faculty','N/A')}")
        print(f"   Dept   : {meta.get('department','N/A')}")
        print(f"   Lang   : {r['language'] or 'N/A'}")
        print(f"   Title  : {(r['title'] or 'N/A')[:60]}")
        print(f"   URL    : {r['url'] or r['pdf_url'] or 'N/A'}")
        print(f"   Text   : {r['text'][:280]}…")

    if results and results[0].get("llm_context"):
        print("\n" + "═" * 64)
        print("LLM CONTEXT PREVIEW")
        print("═" * 64)
        print(results[0]["llm_context"][:1400])