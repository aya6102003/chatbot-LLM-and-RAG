#!/usr/bin/env python3
"""
RAG Retrieval Pipeline v13.3 — Farhat Abbas University Sétif 1
===============================================================
Changes vs v13.2
─────────────────
  1. Vector index verification and creation helpers
  2. Better error handling and timeout recovery
  3. Logging improvements for debugging
  4. Embedding format validation
  5. Neo4j vector search fallback mechanism

Requires:
  pip install sentence-transformers neo4j numpy rank-bm25 rapidfuzz requests
  export GROQ_API_KEY="gsk_..."
"""

import atexit
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    logging.warning("rank_bm25 not installed — BM25 disabled. pip install rank-bm25")

try:
    from rapidfuzz import process as fuzz_process, fuzz as _fuzz
    _FUZZ_OK = True
except ImportError:
    _FUZZ_OK = False
    logging.warning("rapidfuzz not installed — fuzzy title search disabled.")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_TIMEOUT  = int(os.getenv("NEO4J_TIMEOUT", "30"))

NEO4J_VECTOR_INDEX = "chunk_embedding"

# ── LaBSE: 768-dim symmetric model ─
EMBED_MODEL_NAME   = "sentence-transformers/LaBSE"
VECTOR_DIMENSIONS  = 768

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Groq API ─────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Retrieval pool sizes ──────────────────────
TOP_K_VECTOR = 40
TOP_K_BM25   = 20
TOP_K_RERANK = 15
TOP_K_FINAL  = 8
TOP_K_FUZZY  = 10

FUZZY_MIN_SCORE = 65

# ── Fusion weights ────────────────────────────
W_SEM   = 0.70
W_BM25  = 0.10
W_TITLE = 0.20
W_FUSED  = 0.50
W_RERANK = 0.50

RERANK_POWER   = 0.75
RERANK_MIN_CAL = 0.25

DYN_TOP_K_MIN        = 3
DYN_TOP_K_MAX        = 8
DYNAMIC_SCORE_MARGIN = 0.25

LANG_MATCH_BOOST    = 0.04
ACADEMIC_BOOST_CAP  = 0.04
AUTHORITY_BOOST_CAP = 0.03

ANSWER_THRESHOLD = 0.35
DEDUP_CHARS      = 200

NEIGHBOR_COUNT          = 3
NEIGHBOR_WINDOW         = 2
NEIGHBOR_SEED_MIN_SCORE = 0.40
NEIGHBOR_SCORE_INHERIT  = 0.80

NBR_SEM_FLOOR = 0.30
NBR_KW_FLOOR  = 0.15

CONTEXT_WINDOW_SIZE = 1

ENTITY_FUZZ_THRESHOLD = 72
ENTITY_MIN_TOKEN_LEN  = 3

EMBED_CACHE_SIZE = 256

# ── University abbreviations ──────────────────
_UNI_ABBREVS: Dict[str, List[str]] = {
    "s1": ["semestre 1", "semester 1", "الفصل 1"],
    "s2": ["semestre 2", "semester 2", "الفصل 2"],
    "s3": ["semestre 3", "semester 3", "الفصل 3"],
    "s4": ["semestre 4", "semester 4", "الفصل 4"],
    "s5": ["semestre 5", "semester 5", "الفصل 5"],
    "s6": ["semestre 6", "semester 6", "الفصل 6"],
    "l1": ["licence 1", "première année licence", "السنة الأولى ليسانس"],
    "l2": ["licence 2", "deuxième année licence", "السنة الثانية ليسانس"],
    "l3": ["licence 3", "troisième année licence", "السنة الثالثة ليسانس"],
    "m1": ["master 1", "première année master", "السنة الأولى ماستر"],
    "m2": ["master 2", "deuxième année master", "السنة الثانية ماستر"],
    "d":  ["doctorat", "doctorate", "دكتوراه"],
    "mi": ["mathématiques et informatique", "mathematics and computer science", "رياضيات وإعلام آلي"],
    "st": ["sciences et technologie", "science and technology", "علوم وتكنولوجيا"],
    "sm": ["sciences de la matière", "material sciences", "علوم المادة"],
    "sv": ["sciences de la vie", "life sciences", "علوم الحياة"],
    "sn": ["sciences de la nature", "natural sciences", "علوم الطبيعة"],
    "gl": ["génie logiciel", "software engineering", "هندسة البرمجيات"],
    "rsd": ["réseaux et systèmes distribués", "networks and distributed systems", "شبكات وأنظمة موزعة"],
    "tp":  ["travaux pratiques", "practical work", "أعمال تطبيقية"],
    "td":  ["travaux dirigés", "tutorial", "أعمال موجهة"],
    "cc":  ["contrôle continu", "continuous assessment", "تقييم مستمر"],
    "em":  ["examen final", "final exam", "امتحان نهائي"],
}

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

_NAME_TRIGGERS = re.compile(
    r"\b(dr|pr|prof|professeur|docteur|mr|mme|mlle|أستاذ|دكتور|أ\.د|د\.)\s*\.?\s*",
    re.IGNORECASE | re.UNICODE,
)

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
# NEO4J HELPERS
# ══════════════════════════════════════════════════════════════

def _verify_vector_index(driver):
    """Check if vector index exists and is properly configured."""
    try:
        with driver.session() as sess:
            result = sess.run(f"SHOW INDEXES WHERE name = '{NEO4J_VECTOR_INDEX}'")
            indexes = list(result)
            if indexes:
                log.info(f"✓ Vector index '{NEO4J_VECTOR_INDEX}' exists")
                return True
            else:
                log.warning(f"Vector index '{NEO4J_VECTOR_INDEX}' NOT found")
                log.warning("Try running in Neo4j:")
                log.warning(f"  CREATE VECTOR INDEX {NEO4J_VECTOR_INDEX} IF NOT EXISTS")
                log.warning(f"  FOR (c:Chunk) ON (c.embedding)")
                log.warning(f"  OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}")
                return False
    except Exception as e:
        log.error(f"Vector index verification failed: {e}")
        return False


def _count_chunks(driver) -> int:
    """Count total chunks in database."""
    try:
        with driver.session() as sess:
            result = sess.run("MATCH (c:Chunk) RETURN count(c) AS cnt")
            record = result.single()
            count = record["cnt"] if record else 0
            log.info(f"Database has {count} chunks")
            return count
    except Exception as e:
        log.error(f"Failed to count chunks: {e}")
        return 0


# ══════════════════════════════════════════════════════════════
# SHARED NEO4J DRIVER
# ══════════════════════════════════════════════════════════════

_neo4j_driver = None
_neo4j_lock = threading.Lock()


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        with _neo4j_lock:
            if _neo4j_driver is None:
                try:
                    _neo4j_driver = GraphDatabase.driver(
                        NEO4J_URI,
                        auth=(NEO4J_USER, NEO4J_PASSWORD),
                        connection_timeout=NEO4J_TIMEOUT,
                    )
                    # Verify connectivity
                    with _neo4j_driver.session() as sess:
                        sess.run("RETURN 1")
                    log.info("✓ Neo4j driver initialized and connected")
                    _verify_vector_index(_neo4j_driver)
                    _count_chunks(_neo4j_driver)
                except Exception as e:
                    log.error(f"Failed to initialize Neo4j driver: {e}")
                    _neo4j_driver = None
                    raise
    return _neo4j_driver


def _close_driver():
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
        log.info("Neo4j driver closed")


atexit.register(_close_driver)


# ══════════════════════════════════════════════════════════════
# QUERY UNDERSTANDING
# ══════════════════════════════════════════════════════════════

class QueryUnderstanding:
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
            r"\b[SsLlMm][1-6]\b|مقياس|فصل|برنامج|تخصص",
            re.IGNORECASE),
        "admin_query": re.compile(
            r"inscription|registration|calendrier|deadline|scolarité|"
            r"examen|exam|résultat|result|تسجيل|امتحان|نتيجة|إدارة",
            re.IGNORECASE),
    }

    def detect_language(self, text: str) -> str:
        s = text[:300]
        ar_chars = len(self._RE_AR.findall(s))
        ar_threshold = 1 if len(s.strip()) <= 15 else 5
        if ar_chars >= ar_threshold:
            return "ar"
        if len(self._RE_FR.findall(s)) > 2:
            return "fr"
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
        if not (1 <= len(tokens) <= 3):
            return False
        return sum(1 for t in tokens if self._BARE_NAME_RE.match(t)) >= max(1, len(tokens) - 1)

    def normalize_for_bm25(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
        text = re.sub(r"[أإآٱ]", "ا", text)
        text = text.replace("ة", "ه").replace("ى", "ي")
        text = (text.replace("é", "e").replace("è", "e").replace("ê", "e")
                    .replace("à", "a").replace("â", "a").replace("ù", "u")
                    .replace("û", "u").replace("î", "i").replace("ô", "o")
                    .replace("ç", "c"))
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


# ══════════════════════════════════════════════════════════════
# QUERY EXPANDER (Groq)
# ══════════════════════════════════════════════════════════════

def _arabic_to_latin(text: str) -> str:
    result = []
    for ch in text:
        result.append(_AR_TRANSLIT.get(ch, ch))
    latin = "".join(result).lower().strip()
    latin = re.sub(r"\s+", "", latin)
    latin = re.sub(r"[^\w]", "", latin)
    return latin if latin else text


def _is_proper_name_query(query: str, intent: str) -> bool:
    if intent == "person_lookup":
        return True
    if _NAME_TRIGGERS.search(query):
        return True
    return False


def _expand_abbreviations(query: str) -> List[str]:
    tokens = query.lower().strip().split()
    expansions_per_lang: List[List[str]] = [[], [], []]
    found_any = False
    for token in tokens:
        clean = re.sub(r"[^\w]", "", token)
        if clean in _UNI_ABBREVS:
            found_any = True
            for i, expanded in enumerate(_UNI_ABBREVS[clean]):
                expansions_per_lang[i].append(expanded)
        else:
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
    _SPLIT_RE = re.compile(r"[.!?؟،,;:\n]+")
    _SYS = (
        "You are a multilingual search query generator for a university "
        "knowledge base that contains documents in Arabic, French, and English.\n"
        "Given a user query, output ONLY a JSON array of exactly 5 search queries:\n"
        "  [0] Semantically equivalent query in FRENCH\n"
        "  [1] Semantically equivalent query in ENGLISH\n"
        "  [2] Semantically equivalent query in ARABIC\n"
        "  [3] Alternative phrasing in the SAME language as the input query\n"
        "  [4] A more specific or detailed version of the query (any language)\n"
        "Rules:\n"
        "  - Output ONLY the JSON array. No explanation. No markdown.\n"
        "  - Each element must be a non-empty string.\n"
    )

    def __init__(self):
        self._last_intent = "general_info"
        self._ok = self._ping()

    def _ping(self) -> bool:
        """Check that GROQ_API_KEY is set and endpoint responds."""
        if not GROQ_API_KEY:
            log.warning("GROQ_API_KEY not set — LLM query expansion disabled.")
            return False
        try:
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=5,
            )
            return r.status_code == 200
        except Exception as e:
            log.warning(f"Groq API unavailable: {e}")
            return False

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_expand(
        query: str, lang: str, intent: str, groq_ok: bool,
        groq_model: str,
    ) -> Tuple[str, ...]:
        variants = [query]
        if groq_ok:
            tmp = QueryExpander()
            for v in tmp._llm_expand(query):
                if v and v.strip() and v.strip() not in variants:
                    variants.append(v.strip())
        tmp = QueryExpander()
        tmp._last_intent = intent
        for v in tmp._struct_expand(query, lang):
            if v not in variants:
                variants.append(v)
        return tuple(variants[:8])

    def expand(self, analysis: Dict) -> List[str]:
        query  = analysis["query"]
        lang   = analysis.get("language", "en")
        intent = analysis.get("intent", "general_info")
        self._last_intent = intent
        variants_tuple = self._cached_expand(
            query, lang, intent, self._ok, GROQ_MODEL,
        )
        return list(variants_tuple)

    def _llm_expand(self, query: str) -> List[str]:
        """Call Groq's OpenAI-compatible endpoint."""
        if not GROQ_API_KEY:
            return []
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": self._SYS},
                {"role": "user",   "content": f'Query: "{query}"'},
            ],
            "temperature": 0.3,
            "max_tokens":  300,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }
        try:
            r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=15)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\[.*?\]", content, re.DOTALL)
            if not m:
                return []
            vs = json.loads(m.group())
            return [str(v).strip() for v in vs if v and str(v).strip()]
        except Exception as exc:
            log.debug("Groq LLM expansion failed: %s", exc)
        return []

    def _struct_expand(self, query: str, lang: str) -> List[str]:
        vs: List[str] = []
        parts = [p.strip() for p in self._SPLIT_RE.split(query) if p.strip()]
        if len(parts) > 1:
            vs += [p for p in parts if len(p.split()) >= 3]
        clean = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE).strip()
        if clean != query:
            vs.append(clean)
        for exp in _expand_abbreviations(query):
            if exp and exp not in vs:
                vs.append(exp)
        intent = self._last_intent
        if _is_proper_name_query(query, intent):
            name_part = _NAME_TRIGGERS.sub("", query).strip()
            if re.search(r"[\u0600-\u06FF]", name_part):
                latin = _arabic_to_latin(name_part)
                if latin and latin != name_part:
                    vs.append(latin)
                    vs.append(f"dr {latin}")
                    vs.append(f"prof {latin}")
        return vs


# ══════════════════════════════════════════════════════════════
# SEMANTIC RETRIEVER
# ══════════════════════════════════════════════════════════════

class SemanticRetriever:

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        log.info("Loading embedding model: %s", model_name)
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
        log.info("Embedder device: %s", dev)
        self._model  = SentenceTransformer(model_name, device=dev)
        self._cache: Dict[str, np.ndarray] = {}

    def _encode(self, texts: List[str]) -> np.ndarray:
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        miss_idx: List[int] = []
        miss_texts: List[str] = []
        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            embs = self._model.encode(
                miss_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for local_i, global_i in enumerate(miss_idx):
                vec = embs[local_i]
                key = miss_texts[local_i]
                if len(self._cache) >= EMBED_CACHE_SIZE:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = vec
                results[global_i] = vec
        return np.vstack(results)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode query texts."""
        return self._encode(texts)

    def encode_passage(self, texts: List[str]) -> np.ndarray:
        """Encode passage texts."""
        return self._encode(texts)

    def search(
        self,
        variants:     List[str],
        top_k:        int,
        where_filter: Optional[Dict] = None,
    ) -> Dict[str, RetrievedChunk]:
        if not variants:
            return {}

        embeddings = self.encode(variants)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        candidates: Dict[str, RetrievedChunk] = {}
        driver = _get_driver()
        for vec_idx, vec in enumerate(embeddings):
            filter_clause = ""
            filter_params: Dict = {}
            if where_filter and "source_type" in where_filter:
                filter_clause = "WHERE u.source_type = $src_type"
                filter_params["src_type"] = where_filter["source_type"]

            query = f"""
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node AS c, score
                MATCH (u:URL)-[:HAS_CHUNK]->(c)
                {filter_clause}
                OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                RETURN
                    c.id           AS chunk_id,
                    c.text         AS text,
                    c.chunk_index  AS chunk_index,
                    c.language     AS language,
                    score          AS similarity,
                    u.url          AS url,
                    u.title        AS title,
                    u.source_type  AS source_type,
                    page.url       AS page_url
            """
            params = {"index": NEO4J_VECTOR_INDEX, "k": top_k,
                      "vec": vec.tolist(), **filter_params}
            try:
                with driver.session() as sess:
                    for rec in sess.run(query, params):
                        cid = rec["chunk_id"]
                        sim = float(rec["similarity"])
                        meta = {
                            "title":       rec["title"]       or "",
                            "url":         rec["url"]         or "",
                            "page_url":    rec["page_url"]    or "",
                            "source_type": rec["source_type"] or "",
                            "language":    rec["language"]    or "",
                            "chunk_index": rec["chunk_index"],
                            "pdf_url": rec["url"] if rec["source_type"] == "pdf" else "",
                        }
                        if cid not in candidates or sim > candidates[cid].sem_score:
                            candidates[cid] = RetrievedChunk(
                                chunk_id=cid,
                                text=rec["text"] or "",
                                score=sim,
                                metadata=meta,
                                sem_score=sim,
                            )
            except Exception as exc:
                log.warning("Vector search (variant %d/%d) failed: %s", vec_idx+1, len(embeddings), exc)
                if "vector" in str(exc).lower():
                    log.error("Vector index error detected. Ensure index exists with dimensions=768")
                continue
        
        if not candidates and embeddings.size > 0:
            log.warning("No candidates found in vector search (all %d variants failed)", len(embeddings))
        
        return candidates


# ══════════════════════════════════════════════════════════════
# BM25 RETRIEVER
# ══════════════════════════════════════════════════════════════

class BM25Retriever:
    def __init__(self):
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []
        self._meta: Dict[str, Dict] = {}
        self._bm25 = None
        if not _BM25_OK:
            return

        log.info("BM25: loading chunks from Neo4j …")
        qu = QueryUnderstanding()
        tokenized: List[List[str]] = []

        try:
            driver = _get_driver()
            with driver.session() as sess:
                records = sess.run("""
                    MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                    RETURN
                        c.id          AS chunk_id,
                        c.text        AS text,
                        c.language    AS language,
                        c.chunk_index AS chunk_index,
                        u.url         AS url,
                        u.title       AS title,
                        u.source_type AS source_type,
                        page.url      AS page_url
                """)
                for rec in records:
                    cid  = rec["chunk_id"]
                    text = rec["text"] or ""
                    lang = rec["language"] or "en"

                    if not cid or not text.strip():
                        continue

                    self._chunk_ids.append(cid)
                    self._texts.append(text)
                    self._meta[cid] = {
                        "title":       rec["title"]       or "",
                        "url":         rec["url"]         or "",
                        "page_url":    rec["page_url"]    or "",
                        "source_type": rec["source_type"] or "",
                        "language":    lang,
                        "chunk_index": rec["chunk_index"],
                        "pdf_url":     rec["url"] if rec["source_type"] == "pdf" else "",
                        "chunk":       text,
                    }
                    tokenized.append(qu.normalize_for_bm25(text).split())

        except Exception as exc:
            log.error("BM25: Neo4j load failed: %s", exc)
            return

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            log.info("BM25 index built: %d chunks", len(tokenized))
        else:
            log.warning("BM25: No chunks loaded")

    def search(self, keywords: List[str], top_k: int = TOP_K_BM25) -> List[Tuple[str, str, float]]:
        if self._bm25 is None or not keywords:
            return []
        raw = self._bm25.get_scores(keywords)
        mx = raw.max()
        if mx <= 0:
            return []
        norm = raw / mx
        top_idx = np.argsort(norm)[::-1][:top_k]
        return [(self._chunk_ids[i], self._texts[i], float(norm[i])) for i in top_idx if norm[i] > 0]

    def get_meta(self, chunk_id: str) -> Optional[Dict]:
        return self._meta.get(chunk_id)


# ══════════════════════════════════════════════════════════════
# FUZZY RETRIEVER
# ══════════════════════════════════════════════════════════════

class FuzzyRetriever:
    def __init__(self):
        self._entries: List[Tuple[str, str, str]] = []
        self._cid_text: Dict[str, str] = {}
        self._cid_meta: Dict[str, Dict] = {}
        if not _FUZZ_OK:
            return

        log.info("Fuzzy: loading URL titles from Neo4j …")
        try:
            driver = _get_driver()
            with driver.session() as sess:
                records = sess.run("""
                    MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                    RETURN
                        u.title       AS title,
                        c.id          AS chunk_id,
                        c.text        AS text,
                        c.language    AS language,
                        u.url         AS url,
                        u.source_type AS source_type,
                        c.chunk_index AS chunk_index,
                        page.url      AS page_url
                """)
                for rec in records:
                    title = rec["title"] or ""
                    if not title:
                        continue
                    lang = rec["language"] or "en"
                    cid  = rec["chunk_id"]
                    text = rec["text"] or ""
                    self._entries.append((title.lower(), cid, text))
                    self._cid_text[cid] = text
                    self._cid_meta[cid] = {
                        "title":       title,
                        "url":         rec["url"]         or "",
                        "page_url":    rec["page_url"]    or "",
                        "source_type": rec["source_type"] or "",
                        "language":    lang,
                        "chunk_index": rec["chunk_index"],
                        "pdf_url":     rec["url"] if rec["source_type"] == "pdf" else "",
                    }
        except Exception as exc:
            log.error("Fuzzy: Neo4j load failed: %s", exc)
            return
        log.info("Fuzzy index: %d chunk-title pairs", len(self._entries))

    def search(self, query: str, top_k: int = TOP_K_FUZZY) -> List[Tuple[str, str, float]]:
        if not _FUZZ_OK or not self._entries:
            return []
        q = query.lower().strip()
        titles = [e[0] for e in self._entries]
        hits = fuzz_process.extract(q, titles, scorer=_fuzz.token_set_ratio, limit=top_k * 2)
        seen: Dict[str, float] = {}
        for _, raw, idx in hits:
            if raw < FUZZY_MIN_SCORE:
                continue
            _, cid, _ = self._entries[idx]
            norm = raw / 100.0
            if cid not in seen or norm > seen[cid]:
                seen[cid] = norm
        return sorted(
            [(cid, self._cid_text.get(cid, ""), s) for cid, s in seen.items()],
            key=lambda x: x[2], reverse=True
        )[:top_k]

    def get_meta(self, chunk_id: str) -> Optional[Dict]:
        return self._cid_meta.get(chunk_id)


# ══════════════════════════════════════════════════════════════
# METADATA STORE
# ══════════════════════════════════════════════════════════════

class MetadataStore:
    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def get(self, chunk_id: str) -> Optional[Dict]:
        if chunk_id in self._cache:
            return self._cache[chunk_id]
        try:
            driver = _get_driver()
            with driver.session() as sess:
                rec = sess.run("""
                    MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk {id: $cid})
                    OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                    RETURN
                        c.text        AS text,
                        c.chunk_index AS chunk_index,
                        c.language    AS language,
                        u.url         AS url,
                        u.title       AS title,
                        u.source_type AS source_type,
                        page.url      AS page_url
                """, cid=chunk_id).single()
                if rec is None:
                    return None

                text = rec["text"] or ""
                data = {
                    "chunk":       text,
                    "text":        text,
                    "title":       rec["title"]       or "",
                    "url":         rec["url"]         or "",
                    "page_url":    rec["page_url"]    or "",
                    "source_type": rec["source_type"] or "",
                    "language":    rec["language"]    or "en",
                    "chunk_index": rec["chunk_index"],
                    "pdf_url":     rec["url"] if rec["source_type"] == "pdf" else "",
                }
                self._cache[chunk_id] = data
                return data
        except Exception as exc:
            log.warning("MetadataStore.get(%s) failed: %s", chunk_id, exc)
            return None

    def preload(self, chunk_ids: List[str]):
        missing = [cid for cid in chunk_ids if cid not in self._cache]
        if not missing:
            return
        try:
            driver = _get_driver()
            with driver.session() as sess:
                records = sess.run("""
                    UNWIND $ids AS cid
                    MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk {id: cid})
                    OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                    RETURN
                        c.id          AS chunk_id,
                        c.text        AS text,
                        c.chunk_index AS chunk_index,
                        c.language    AS language,
                        u.url         AS url,
                        u.title       AS title,
                        u.source_type AS source_type,
                        page.url      AS page_url
                """, ids=missing)
                for rec in records:
                    cid  = rec["chunk_id"]
                    text = rec["text"] or ""
                    self._cache[cid] = {
                        "chunk":       text,
                        "text":        text,
                        "title":       rec["title"]       or "",
                        "url":         rec["url"]         or "",
                        "page_url":    rec["page_url"]    or "",
                        "source_type": rec["source_type"] or "",
                        "language":    rec["language"]    or "en",
                        "chunk_index": rec["chunk_index"],
                        "pdf_url":     rec["url"] if rec["source_type"] == "pdf" else "",
                    }
        except Exception as exc:
            log.warning("MetadataStore.preload failed: %s", exc)


# ══════════════════════════════════════════════════════════════
# TITLE SIMILARITY
# ══════════════════════════════════════════════════════════════

def _title_sim(query_vec: np.ndarray, meta: Dict, retriever: SemanticRetriever) -> float:
    t = meta.get("title", "")
    if not t:
        return 0.0
    title_vec = retriever.encode_passage([t])
    return float((title_vec @ query_vec).max())


# ══════════════════════════════════════════════════════════════
# FINGERPRINT
# ══════════════════════════════════════════════════════════════

def _fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text[:DEDUP_CHARS]).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════
# SCORE FUSION
# ══════════════════════════════════════════════════════════════

def fuse_scores(
    semantic:   Dict[str, RetrievedChunk],
    bm25_hits:  List[Tuple[str, str, float]],
    fuzz_hits:  List[Tuple[str, str, float]],
    meta_store: MetadataStore,
    bm25_retr:  BM25Retriever,
    fuzz_retr:  FuzzyRetriever,
    query_vec:  np.ndarray,
    retriever:  SemanticRetriever,
    query_lang: str,
    intent:     str,
) -> List[RetrievedChunk]:
    pool: Dict[str, Dict] = {}

    for cid, chunk in semantic.items():
        pool[cid] = {"sem": chunk.sem_score, "bm25": 0.0, "text": chunk.text, "meta": chunk.metadata}

    for cid, text, score in bm25_hits:
        if cid not in pool:
            rec = bm25_retr.get_meta(cid) or meta_store.get(cid)
            if rec is None:
                continue
            pool[cid] = {"sem": 0.0, "bm25": score, "text": rec.get("chunk", text), "meta": rec}
        else:
            pool[cid]["bm25"] = max(pool[cid]["bm25"], score)

    for cid, text, score in fuzz_hits:
        fuzzy_contrib = score * 0.8
        if cid not in pool:
            rec = fuzz_retr.get_meta(cid) or meta_store.get(cid)
            if rec is None:
                continue
            pool[cid] = {"sem": 0.0, "bm25": fuzzy_contrib, "text": rec.get("chunk", text), "meta": rec}
        else:
            pool[cid]["bm25"] = max(pool[cid]["bm25"], fuzzy_contrib)

    fused_chunks: List[RetrievedChunk] = []
    seen_fp: Dict[str, float] = {}
    for cid, d in pool.items():
        sem  = float(d["sem"])
        bm25 = float(d["bm25"])
        text = d["text"]
        meta = d["meta"] if isinstance(d["meta"], dict) else {}
        if sem == 0.0 and bm25 == 0.0:
            continue
        t_sim = _title_sim(query_vec, meta, retriever)
        fused = W_SEM * sem + W_BM25 * bm25 + W_TITLE * t_sim
        if meta.get("language") == query_lang:
            fused += LANG_MATCH_BOOST
        acad = float(meta.get("academic_score", 0.0) or 0.0)
        fused += min(ACADEMIC_BOOST_CAP, acad * ACADEMIC_BOOST_CAP)
        if intent == "admin_query":
            auth = float(meta.get("authority_score", 0.0) or 0.0)
            fused += min(AUTHORITY_BOOST_CAP, auth * AUTHORITY_BOOST_CAP)
        fp = _fingerprint(text)
        if fp in seen_fp and fused <= seen_fp[fp]:
            continue
        seen_fp[fp] = fused
        fused_chunks.append(RetrievedChunk(
            chunk_id=cid, text=text, score=fused, metadata=meta,
            sem_score=sem, bm25_score=bm25, title_score=t_sim, fused_score=fused,
        ))

    fused_chunks.sort(key=lambda c: c.score, reverse=True)
    return fused_chunks


# ══════════════════════════════════════════════════════════════
# ANSWERABILITY GATE
# ══════════════════════════════════════════════════════════════

def _passes_answerability(chunks: List[RetrievedChunk]) -> bool:
    return bool(chunks and chunks[0].score >= ANSWER_THRESHOLD)


# ══════════════════════════════════════════════════════════════
# CROSS-ENCODER RERANKER
# ══════════════════════════════════════════════════════════════

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        log.info("Loading reranker: %s", model_name)
        self._model = CrossEncoder(model_name)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + float(np.exp(-float(x))))

    @staticmethod
    def _calibrate(raw: float) -> float:
        return raw ** RERANK_POWER

    def rerank(self, query: str, chunks: List[RetrievedChunk],
               top_k: int, min_cal: float = RERANK_MIN_CAL) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        logits = self._model.predict(pairs)
        kept: List[RetrievedChunk] = []
        for chunk, logit in zip(chunks, logits):
            raw = self._sigmoid(logit)
            cal = self._calibrate(raw)
            chunk.rerank_raw = raw
            chunk.rerank_cal = cal
            if cal < min_cal:
                continue
            chunk.score = W_FUSED * chunk.fused_score + W_RERANK * cal
            kept.append(chunk)
        kept.sort(key=lambda c: c.score, reverse=True)
        return kept[:top_k]


# ══════════════════════════════════════════════════════════════
# DYNAMIC TOP-K
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
    return result


# ══════════════════════════════════════════════════════════════
# ENTITY FILTER
# ══════════════════════════════════════════════════════════════

def _entity_filter(chunks: List[RetrievedChunk], entity_tokens: List[str],
                   intent: str) -> List[RetrievedChunk]:
    if intent != "person_lookup" or not entity_tokens or not _FUZZ_OK:
        return chunks

    def _chunk_contains_entity(text: str) -> bool:
        text_low = text.lower()
        for token in entity_tokens:
            if len(token) < ENTITY_MIN_TOKEN_LEN:
                continue
            if token.lower() in text_low:
                return True
            if _fuzz.partial_ratio(token.lower(), text_low) >= ENTITY_FUZZ_THRESHOLD:
                return True
        return False

    kept = [c for c in chunks if _chunk_contains_entity(c.text)]
    if not kept and chunks:
        kept = [chunks[0]]
    return kept


# ══════════════════════════════════════════════════════════════
# NEIGHBOUR HELPERS
# ══════════════════════════════════════════════════════════════

def _keyword_overlap(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    tl = text.lower()
    return sum(1 for kw in keywords if kw in tl) / len(keywords)


def _filter_neighbours_by_relevance(
    candidates: List[Tuple[str, str]], query_vec: np.ndarray,
    retriever: SemanticRetriever, keywords: List[str],
) -> List[str]:
    if not candidates:
        return []
    accepted_ids: List[str] = []
    texts = [text for _, text in candidates]
    ids   = [cid  for cid, _ in candidates]
    lexical_pass = {
        cid for cid, text in candidates
        if keywords and _keyword_overlap(text, keywords) >= NBR_KW_FLOOR
    }
    sem_needed_idx = [i for i, cid in enumerate(ids) if cid not in lexical_pass]
    if sem_needed_idx:
        try:
            sem_texts    = [texts[i] for i in sem_needed_idx]
            passage_vecs = retriever.encode_passage(sem_texts)
            sims         = passage_vecs @ query_vec
        except Exception as exc:
            log.debug("Batch neighbour encoding failed: %s", exc)
            sims = None
        for local_i, global_i in enumerate(sem_needed_idx):
            cid = ids[global_i]
            if sims is not None and float(sims[local_i]) >= NBR_SEM_FLOOR:
                lexical_pass.add(cid)
    for cid in ids:
        if cid in lexical_pass:
            accepted_ids.append(cid)
    return accepted_ids


def _same_doc_prefix(cid_a: str, cid_b: str) -> bool:
    prefix_a = cid_a.rsplit("_c", 1)[0] if "_c" in cid_a else cid_a
    prefix_b = cid_b.rsplit("_c", 1)[0] if "_c" in cid_b else cid_b
    return prefix_a == prefix_b


# ══════════════════════════════════════════════════════════════
# GRAPH EXPANDER
# ══════════════════════════════════════════════════════════════

class GraphExpander:
    def get_neighbors(
        self, chunk_ids: List[str], window: int = NEIGHBOR_WINDOW
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        result: Dict[str, Tuple[List[str], List[str]]] = {cid: ([], []) for cid in chunk_ids}
        if not chunk_ids:
            return result
        try:
            driver = _get_driver()
            with driver.session() as sess:
                records = sess.run(
                    """
                    UNWIND $ids AS cid
                    MATCH (c:Chunk {id: cid})
                    OPTIONAL MATCH (p:Chunk)-[:NEXT_CHUNK*1..%(w)d]->(c)
                    OPTIONAL MATCH (c)-[:NEXT_CHUNK*1..%(w)d]->(n:Chunk)
                    RETURN
                        cid,
                        [x IN collect(DISTINCT p.id) WHERE x IS NOT NULL] AS prev_ids,
                        [x IN collect(DISTINCT n.id) WHERE x IS NOT NULL] AS next_ids
                    """ % {"w": window},
                    ids=chunk_ids,
                )
                for rec in records:
                    cid = rec["cid"]
                    result[cid] = (
                        list(rec["prev_ids"] or []),
                        list(rec["next_ids"] or []),
                    )
        except Exception as exc:
            log.warning("Neo4j expansion failed: %s", exc)
        return result


# ══════════════════════════════════════════════════════════════
# BOILERPLATE FILTER
# ══════════════════════════════════════════════════════════════

_BOILERPLATE = frozenset([
    "call for papers", "قراءة المزيد", "skip to content", "back to top",
    "accueil | contact", "home | about | contact", "print page",
    "se connecter", "login",
])


def _is_boilerplate(text: str, meta: Dict) -> bool:
    if len(text.strip()) < 30:
        return True
    low = text.lower()
    if any(p in low for p in _BOILERPLATE):
        return True
    title = (meta.get("title", "") or "").lower()
    if title and len(text.replace(title, "").strip()) < 40:
        return True
    return False


# ══════════════════════════════════════════════════════════════
# CONTEXT WINDOW
# ══════════════════════════════════════════════════════════════

def _reconstruct_windows(
    chunks: List[RetrievedChunk], meta_store: MetadataStore,
    window: int = CONTEXT_WINDOW_SIZE
) -> List[RetrievedChunk]:
    if window == 0:
        return chunks
    selected_ids = {c.chunk_id for c in chunks}
    for c in chunks:
        meta      = c.metadata or {}
        chunk_idx = meta.get("chunk_index")
        doc_prefix = c.chunk_id.rsplit("_c", 1)[0] if "_c" in c.chunk_id else None
        if chunk_idx is None or doc_prefix is None:
            continue
        prev_parts: List[str] = []
        next_parts: List[str] = []
        for delta in range(1, window + 1):
            for sign in (-delta, +delta):
                nid = f"{doc_prefix}_c{chunk_idx + sign}"
                rec = meta_store.get(nid)
                if rec and nid not in selected_ids:
                    t = (rec.get("chunk", "") or rec.get("text", "")).strip()
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
        meta    = c.metadata or {}
        title   = meta.get("title", "Unknown")
        lang    = meta.get("language", "")
        url     = meta.get("page_url") or meta.get("url") or ""
        pdf_url = meta.get("pdf_url") or ""
        header  = f"[{i}] {title}"
        if lang:
            header += f"  [{lang.upper()}]"
        lines = [header, c.text.strip()]
        if pdf_url:
            lines.append(f"PDF: {pdf_url}")
        if url and url != pdf_url:
            lines.append(f"Source: {url}")
        blocks.append("\n".join(lines))
    return sep.join(blocks)


# ══════════════════════════════════════════════════════════════
# MAIN RAG RETRIEVER
# ══════════════════════════════════════════════════════════════

class RAGRetriever:
    def __init__(self):
        log.info("Initializing RAGRetriever v13.3 [LaBSE + Groq + Neo4j Vector]")
        self._qu       = QueryUnderstanding()
        self._expander = QueryExpander()
        self._semantic = SemanticRetriever(EMBED_MODEL_NAME)
        self._bm25     = BM25Retriever()
        self._fuzzy    = FuzzyRetriever()
        self._reranker = Reranker()
        self._graph    = GraphExpander()
        self._meta     = MetadataStore()
        log.info("RAGRetriever v13.3 ready")

    def retrieve(
        self, query: str, top_k: int = TOP_K_FINAL,
        source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:

        analysis      = self._qu.analyze(query)
        intent        = analysis["intent"]
        lang          = analysis["language"]
        keywords      = analysis["keywords"]
        entity_tokens = analysis["entity_tokens"]

        log.info("Query lang=%s  intent=%s  query=%s", lang, intent, query[:80])

        variants  = self._expander.expand(analysis)
        query_vec = self._semantic.encode([query])[0]

        where         = {"source_type": source_type} if source_type else None
        semantic_hits = self._semantic.search(variants, TOP_K_VECTOR, where)
        bm25_hits     = self._bm25.search(keywords, TOP_K_BM25)
        fuzz_hits     = self._fuzzy.search(query, TOP_K_FUZZY)

        log.info("Candidates — semantic:%d  BM25:%d  fuzzy:%d",
                 len(semantic_hits), len(bm25_hits), len(fuzz_hits))

        fused = fuse_scores(
            semantic_hits, bm25_hits, fuzz_hits,
            self._meta, self._bm25, self._fuzzy,
            query_vec, self._semantic, lang, intent,
        )
        fused = [c for c in fused if not _is_boilerplate(c.text, c.metadata)]

        if not fused:
            log.warning("All candidates filtered — raw semantic fallback")
            fused = sorted(semantic_hits.values(), key=lambda c: c.sem_score, reverse=True)
            for c in fused:
                c.fused_score = c.sem_score
        if not fused:
            return []

        _rerank_floor = (
            0.10 if (intent == "person_lookup" or len(query.strip().split()) <= 2)
            else RERANK_MIN_CAL
        )
        reranked = self._reranker.rerank(query, fused, top_k=TOP_K_RERANK, min_cal=_rerank_floor)
        if not reranked:
            log.warning("Reranker dropped all chunks")
            return []

        if intent != "translation":
            w        = NEIGHBOR_WINDOW + (1 if intent == "course_query" else 0)
            seed_ids = [
                c.chunk_id for c in reranked[:NEIGHBOR_COUNT]
                if c.score >= NEIGHBOR_SEED_MIN_SCORE
            ]
            nbr_map  = self._graph.get_neighbors(seed_ids, w)
            expanded = list(reranked)
            seen_ids = {c.chunk_id for c in expanded}
            nbr_candidates: List[Tuple[str, str]] = []

            for seed in reranked[:NEIGHBOR_COUNT]:
                if seed.score < NEIGHBOR_SEED_MIN_SCORE:
                    continue
                prev_ids, next_ids = nbr_map.get(seed.chunk_id, ([], []))
                for nid in (prev_ids[:w] + next_ids[:w]):
                    if nid in seen_ids or not _same_doc_prefix(seed.chunk_id, nid):
                        continue
                    rec = self._meta.get(nid)
                    if not rec:
                        continue
                    nbr_text = rec.get("chunk", "")
                    if not nbr_text or len(nbr_text.strip()) < 30:
                        continue
                    nbr_candidates.append((nid, nbr_text))

            if nbr_candidates:
                accepted_ids = _filter_neighbours_by_relevance(
                    nbr_candidates, query_vec, self._semantic, keywords
                )
                accepted_set = set(accepted_ids)
                seed_score_map: Dict[str, float] = {}
                for seed in reranked[:NEIGHBOR_COUNT]:
                    if seed.score < NEIGHBOR_SEED_MIN_SCORE:
                        continue
                    prev_ids, next_ids = nbr_map.get(seed.chunk_id, ([], []))
                    for nid in (prev_ids[:w] + next_ids[:w]):
                        if nid in accepted_set and nid not in seed_score_map:
                            seed_score_map[nid] = seed.fused_score
                for nid, nbr_text in nbr_candidates:
                    if nid not in accepted_set or nid in seen_ids:
                        continue
                    rec       = self._meta.get(nid)
                    fused_nbr = seed_score_map.get(nid, 0.0) * NEIGHBOR_SCORE_INHERIT
                    seen_ids.add(nid)
                    expanded.append(RetrievedChunk(
                        chunk_id=nid, text=nbr_text, score=fused_nbr,
                        metadata=rec or {}, is_neighbor=True, fused_score=fused_nbr,
                    ))
        else:
            expanded = list(reranked)

        final = self._reranker.rerank(query, expanded, top_k=top_k, min_cal=_rerank_floor)
        final = _entity_filter(final, entity_tokens, intent)
        final = _apply_dynamic_topk(final)
        final = _reconstruct_windows(final, self._meta, CONTEXT_WINDOW_SIZE)

        _ans_threshold = (
            0.20 if (intent == "person_lookup" or len(query.strip().split()) <= 2)
            else ANSWER_THRESHOLD
        )
        if not (final and final[0].score >= _ans_threshold):
            log.warning(
                "Answerability gate: best=%.3f < %.2f — NO_ANSWER",
                final[0].score if final else 0.0, _ans_threshold,
            )
            return []

        log.info("Final: %d chunks  scores=%s", len(final), [round(c.score, 3) for c in final])
        return final

    def clear_caches(self):
        self._semantic._cache.clear()
        self._meta._cache.clear()
        QueryExpander._cached_expand.cache_clear()
        log.info("All caches cleared")

    def close(self):
        _close_driver()


# ══════════════════════════════════════════════════════════════
# SINGLETON + PUBLIC API
# ══════════════════════════════════════════════════════════════

_retriever: Optional[RAGRetriever] = None
_retriever_lock = threading.Lock()


def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = RAGRetriever()
    return _retriever


def retrieve_for_llm(
    query:       str,
    top_k:       int = TOP_K_FINAL,
    source_type: Optional[str] = None,
    faculty:     Optional[str] = None,
    department:  Optional[str] = None,
) -> List[Dict]:
    chunks = _get_retriever().retrieve(query, top_k=top_k, source_type=source_type)
    result = [
        {
            "chunk_id":    c.chunk_id,
            "text":        c.text,
            "score":       round(c.score, 4),
            "is_neighbor": c.is_neighbor,
            "sem_score":   round(c.sem_score, 4),
            "bm25_score":  round(c.bm25_score, 4),
            "title_score": round(c.title_score, 4),
            "fused_score": round(c.fused_score, 4),
            "rerank_raw":  round(c.rerank_raw, 4),
            "rerank_cal":  round(c.rerank_cal, 4),
            "url":         c.metadata.get("url", ""),
            "pdf_url":     c.metadata.get("pdf_url", ""),
            "page_url":    c.metadata.get("page_url", ""),
            "title":       c.metadata.get("title", ""),
            "language":    c.metadata.get("language", ""),
            "source_type": c.metadata.get("source_type", ""),
            "chunk_index": c.metadata.get("chunk_index"),
            "metadata":    c.metadata,
        }
        for c in chunks
    ]
    if result:
        result[0]["llm_context"] = _format_llm_context(chunks)
    return result


def warmup_retriever():
    r = _get_retriever()
    _ = r.retrieve("licence mathématiques")
    log.info("RAG pipeline warmed up")


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag_fixed.py "query" [--source-type pdf|page] [--top-k N] [--debug]')
        sys.exit(1)

    query       = sys.argv[1]
    source_type = None
    top_k       = TOP_K_FINAL
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--source-type" and i + 1 < len(sys.argv):
            source_type = sys.argv[i + 1]; i += 2
        elif arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1]); i += 2
        elif arg == "--debug":
            logging.getLogger().setLevel(logging.DEBUG); i += 1
        else:
            i += 1

    results = retrieve_for_llm(query, top_k=top_k, source_type=source_type)

    print("\n" + "=" * 64)
    print(f"QUERY  : {query}")
    print(f"RESULTS: {len(results)}")
    print("=" * 64)

    if not results:
        print(f"\nNO_ANSWER (best score < {ANSWER_THRESHOLD}).")
        sys.exit(0)

    for rank, r in enumerate(results, 1):
        tag = " [NBR]" if r["is_neighbor"] else ""
        print(f"\n{rank}. {r['chunk_id']}{tag}")
        print(f"   Final  : {r['score']:.4f}  (fused={r['fused_score']:.3f}  rerank_cal={r['rerank_cal']:.3f})")
        print(f"   Signals: sem={r['sem_score']:.3f}  bm25={r['bm25_score']:.3f}  title={r['title_score']:.3f}")
        print(f"   Type   : {r['source_type']}")
        print(f"   Title  : {(r['title'] or 'N/A')[:60]}")
        print(f"   URL    : {r['url'] or 'N/A'}")
        print(f"   PDF    : {r['pdf_url'] or 'N/A'}")
        print(f"   Text   : {r['text'][:280]}…")

    if results and results[0].get("llm_context"):
        print("\n" + "═" * 64)
        print("LLM CONTEXT PREVIEW")
        print("═" * 64)
        print(results[0]["llm_context"][:1400])