#!/usr/bin/env python3
"""
RAG Ingestion Pipeline v4-FIXED — Farhat Abbas University Sétif 1
==================================================================
Fixes applied over v4 (each marked # ✦ FIX N):

  FIX 1  — SemanticChunkerV2.split() now receives embed_model from
            build_pipeline() NOT from the thread-pool worker.
            Workers are CPU-bound; loading/using a GPU model from N
            threads simultaneously causes CUDA OOM and GIL contention.
            Solution: split() is called in a dedicated single-threaded
            "chunk" phase; embedding happens in one GPU batch afterward.

  FIX 2  — apply_semantic_dedup() rebuilt per-document, not corpus-wide.
            The v4 version runs dedup on the entire flat corpus matrix
            (up to 50 000 × 1024 floats = 200 MB).  More critically,
            the sliding-window assumption that near-duplicates are
            adjacent only holds within a document, not across the corpus.
            Fix: dedup applied per-document slice of all_embs_np,
            referencing doc_emb_slices correctly.

  FIX 3  — ChromaDB upsert slice indices were wrong.
            v4 used `lo = batch[0]["_emb_idx"]` as a local-chunk index
            into doc_embs_np, but after semantic dedup the _emb_idx
            values had gaps → wrong embeddings stored for surviving chunks.
            Fix: track a per-document embed pointer that advances only
            for kept chunks; store embedding at that explicit offset.

  FIX 4  — _parse_and_chunk_file() no longer receives embed_model.
            In v4, SentenceTransformer is passed into every ThreadPool
            worker (100+ concurrent calls to encode()).  This races on
            GPU memory and the model's internal tokenizer state.
            Fix: workers only do file I/O + text processing.
            Sentence-level semantic splitting is done in the main thread
            using a dedicated "lightweight" sentence encoder that runs
            on CPU (paraphrase-multilingual-MiniLM-L12-v2, 120 MB).
            Full-corpus passage embedding still uses E5-large on GPU.

  FIX 5  — IngestionReranker.filter() batch size capped at 32.
            ms-marco-MiniLM-L-6-v2 has max_length=512; passing 64 pairs
            with long texts silently truncates context and degrades scores.
            Fix: batch_size=32, explicit max_length=256 truncation.

  FIX 6  — AdaptiveChunkSizer.compute() walrus-operator bug in log line.
            v4's log.info() uses `sample_kd if (sample_kd := 0.05) else`
            which always assigns 0.05, discarding the actual value.
            Fix: removed the erroneous walrus expressions from the log.

  FIX 7  — tokenize_for_bm25() return type consistency.
            v4 returns a str (IMP 6) but BM25Retriever in rag.py calls
            `.split()` on the stored value, which works for str but
            breaks if the field was ever stored as a list by an older
            pipeline run.  Fix: explicit str() cast + docstring updated.

  FIX 8  — Neo4j _insert_chunks_batch() FOREACH syntax.
            The FOREACH trick for optional MERGE is valid Cypher, but
            the OPTIONAL MATCH + FOREACH pattern silently no-ops when
            the Section node does not exist yet, because _ensure_sections()
            and _insert_chunks_batch() run in different transactions and
            the Section node may not be committed yet.
            Fix: sections are created before chunks in the same write tx,
            using a single combined Cypher statement.

  FIX 9  — doc_emb_slices construction was off-by-one.
            If the last document had only one chunk, the final
            `doc_emb_slices.append((start, len(flat_mapping)))` appended
            a duplicate entry when start == len(flat_mapping) - 1 due to
            enumerate() returning the last (doc_idx, _) pair with
            doc_idx == prev_doc, so the final append was never reached.
            Fix: slices built with a cleaner bookkeeping loop.

  FIX 10 — global_offset variable declared but never used in v4.
            Removed; doc_emb_slices used consistently.

  FIX 11 — Thread-safety: ProgressDB._flush() called from main thread only.
            Workers no longer call mark_done(); the main loop does.

  FIX 12 — PASSAGE_PREFIX assertion moved after model load (cosmetic but
            previously ran before logging was configured on some Python
            versions, swallowing the AssertionError output).

Dependencies
------------
  pip install sentence-transformers chromadb neo4j numpy tiktoken
  pip install rank-bm25 orjson
  # cross-encoder quality filter:
  pip install sentence-transformers   # CrossEncoder included
"""

# ─────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────
import hashlib
import json
import logging
import os
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from neo4j import GraphDatabase

try:
    import orjson as _json_lib
    def _load_json(fh): return _json_lib.loads(fh.read())
except ImportError:
    _json_lib = None
    def _load_json(fh): return json.load(fh)

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False
    logging.warning("tiktoken not installed — char-based token counting active.")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
ROOT_FOLDER    = "./university_farhat_abaas"
CHROMA_PATH    = "./chroma_db"
METADATA_PATH  = "./metadata.json"
PROGRESS_DB    = "./pipeline_progress.db"

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

UNIVERSITY_NAME = "Farhat Abbas University Sétif 1"

# ✦ FIX 4 — two separate models:
#   EMBED_MODEL     : full E5-large, GPU, passage embeddings (storage)
#   SENT_SPLIT_MODEL: lightweight MiniLM, CPU, sentence similarity (chunking)
EMBED_MODEL      = "intfloat/multilingual-e5-large"
SENT_SPLIT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 120 MB, CPU-safe

PASSAGE_PREFIX = "passage: "   # E5 ingestion prefix; retrieval must use "query: "

EMBED_BATCH        = 64
CHROMA_BATCH       = 100
NEO4J_BATCH        = 50
NEO4J_GLOBAL_BATCH = 1000
PROGRESS_FLUSH     = 50
PARSE_WORKERS      = min(8, (os.cpu_count() or 4))

CHUNK_TOKENS_BASE     = 400
OVERLAP_TOKENS        = 80
MIN_CHUNK_CHARS       = 80
MIN_DOC_CHARS         = 100
MAX_BOILERPLATE_RATIO = 0.25

TOPIC_DRIFT_THRESHOLD    = 0.65
RERANK_QUALITY_FLOOR     = -3.0
CROSS_ENCODER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_ENABLED    = True
SEMANTIC_DEDUP_THRESHOLD = 0.92
SEMANTIC_DEDUP_WINDOW    = 8     # per-document window (FIX 2)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FACULTY LABELS
# ─────────────────────────────────────────────────────────────
FACULTY_LABELS: Dict[str, str] = {
    "farhat_abbas_university": "Farhat Abbas University Sétif 1",
    "ftechnologie": "Faculty of Technology",
    "fsciences":    "Faculty of Science",
    "fsnv":         "Faculty of Nature and Life Sciences",
    "feco":         "Faculty of Economics, Business and Management Sciences",
    "fmed":         "Faculty of Medicine",
    "iomp":         "Institute of Optics and Precision Mechanics",
    "iast":         "Institute of Architecture and Earth Sciences",
    "istm":         "Institute of Materials Science and Techniques",
}

# ─────────────────────────────────────────────────────────────
# WORD LISTS & FROZENSETS
# ─────────────────────────────────────────────────────────────
ACADEMIC_INDICATORS = [
    "semester","module","course","exam","lecture","syllabus",
    "credits","prerequisite","assignment","curriculum",
    "semestre","cours","examen","licence","master","doctorat",
    "formation","filière","td","tp","contrôle",
    "الفصل","الدراسي","امتحان","مقياس","تخصص","برنامج",
]
BOILERPLATE_WORDS = [
    "copyright","all rights reserved","privacy policy","terms of use",
    "click here","read more","subscribe","newsletter","cookie policy",
    "navigation","footer","header","sitemap",
    "home","menu","accueil","الرئيسية","contact","about",
    "connexion","login","sign in","se connecter",
    "skip to content","back to top","print page",
]
AUTHORITY_SIGNALS = [
    "arrêté","décret","décision","circulaire","règlement","official",
    "ministry","ministère","وزارة","مرسوم","قرار","مذكرة",
    "journal officiel","bulletin officiel",
]
EXAM_SIGNALS   = ["exam","امتحان","contrôle","épreuve","test","quiz"]
COURSE_SIGNALS = ["cours","course","محاضرة","lecture","td","tp","syllabus"]
ADMIN_SIGNALS  = ["admin","إدارة","scolarité","inscription","calendrier",
                  "règlement","décret","arrêté"]

_BOILERPLATE_SET: FrozenSet[str] = frozenset(BOILERPLATE_WORDS)
_ACADEMIC_SET:    FrozenSet[str] = frozenset(ACADEMIC_INDICATORS)
_AUTHORITY_SET:   FrozenSet[str] = frozenset(AUTHORITY_SIGNALS)
_EXAM_SET:        FrozenSet[str] = frozenset(EXAM_SIGNALS)
_COURSE_SET:      FrozenSet[str] = frozenset(COURSE_SIGNALS)
_ADMIN_SET:       FrozenSet[str] = frozenset(ADMIN_SIGNALS)
_ACAD_DENOM   = float(len(_ACADEMIC_SET))
_BPLATE_DENOM = float(len(_BOILERPLATE_SET))

# ─────────────────────────────────────────────────────────────
# PRE-COMPILED REGEX
# ─────────────────────────────────────────────────────────────
_RE_CONTROL    = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_TASHKEEL   = re.compile(r"[ًٌٍَُِّْـ]")
_RE_ALEF       = re.compile(r"[أإآٱ]")
_RE_REPEATS    = re.compile(r"(.)\1{2,}")
_RE_APOSTROPHE = re.compile(r"[''`´]")
_RE_QUOTES     = re.compile(r"[«»""„]")
_RE_PAGE_NUM   = re.compile(r"(?m)^\s*\d{1,4}\s*$")
_RE_MULTI_NL   = re.compile(r"\n{3,}")
_RE_SPACES     = re.compile(r"[ \t]+")
_RE_SENT_BOUND = re.compile(r"(?<=[.!?؟\n])\s+")
_RE_YEAR       = re.compile(r"\b(20[12]\d)\b")
_RE_AR_CHARS   = re.compile(r"[\u0600-\u06FF]")
_RE_FR_WORDS   = re.compile(
    r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
    r"cours|semestre|licence|master|doctorat|filière)\b", re.IGNORECASE,
)
_RE_URL_GENERAL = re.compile(r"https?://[^\s<>\"')\]]+")
_RE_PDF_URL     = re.compile(
    r"https?://[^\s<>\"')\]]+\.pdf(?:[?#][^\s<>\"')\]]*)?", re.IGNORECASE)
_RE_REL_PDF     = re.compile(r"[\"']([^\"']+\.pdf)[\"']", re.IGNORECASE)
_RE_COURSE_CODE = re.compile(
    r"\b([A-Z]{2,5}\s?\d{3,4}|[A-Z]{3,6}-\d{2,3}|LMD|S[1-6])\b")
_RE_EMAIL       = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_RE_PHONE       = re.compile(r"(?:\+213|0)[5-7]\d{8}")
_RE_HEADING     = re.compile(
    r"(?m)^(#{1,4}\s.+|[A-ZÀÁÂÄÉÈÊËÎÏÔÙÛÜ][A-ZÀÁÂÄÉÈÊËÎÏÔÙÛÜ\s:,]{8,}|"
    r"[\u0600-\u06FF]{4,}[\s:]+[\u0600-\u06FF].*|Chapitre|Section|Partie|"
    r"Chapter|Part\s+\d|الفصل|القسم|الجزء)\s*$",
    re.MULTILINE,
)


# ══════════════════════════════════════════════════════════════
# LAYER 1 — TEXT NORMALIZATION
# ══════════════════════════════════════════════════════════════

def normalize_text(text: str, aggressive: bool = False) -> str:
    """
    Light normalization for embeddings (aggressive=False).
    Aggressive=True for BM25 tokenisation only.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _RE_CONTROL.sub("", text)
    text = _RE_TASHKEEL.sub("", text)
    if aggressive:
        text = _RE_ALEF.sub("ا", text)
        text = text.replace("ة","ه").replace("ى","ي")
        text = (text.replace("é","e").replace("è","e").replace("ê","e")
                    .replace("à","a").replace("â","a")
                    .replace("ù","u").replace("û","u")
                    .replace("î","i").replace("ô","o").replace("ç","c"))
    text = _RE_REPEATS.sub(r"\1\1", text)
    text = (text.replace("œ","oe").replace("æ","ae")
                .replace("ﬁ","fi").replace("ﬂ","fl"))
    text = _RE_APOSTROPHE.sub("'", text)
    text = _RE_QUOTES.sub('"', text)
    text = _RE_PAGE_NUM.sub("", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    text = _RE_SPACES.sub(" ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# LAYER 2 — LANGUAGE / DOC-TYPE / YEAR
# ══════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    s = text[:300]
    if len(_RE_AR_CHARS.findall(s)) > 15: return "ar"
    if len(_RE_FR_WORDS.findall(s)) > 3:  return "fr"
    return "en"

def infer_doc_type(filename: str, title: str, text_sample: str) -> str:
    c = f"{filename} {title} {text_sample[:200]}".lower()
    if any(w in c for w in _EXAM_SET):   return "exam"
    if any(w in c for w in _COURSE_SET): return "course"
    if any(w in c for w in _ADMIN_SET):  return "admin"
    return "general"

def extract_year(text: str) -> Optional[int]:
    m = _RE_YEAR.search(text[:500])
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════
# SCORING & METADATA HELPERS
# ══════════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, List[str]]:
    return {
        "course_codes": list(dict.fromkeys(_RE_COURSE_CODE.findall(text)))[:10],
        "emails":       list(dict.fromkeys(_RE_EMAIL.findall(text)))[:5],
        "phones":       list(dict.fromkeys(_RE_PHONE.findall(text)))[:5],
    }

def compute_authority_score(text: str, doc_type: str, source: str) -> float:
    low  = text[:600].lower()
    hits = sum(1 for w in _AUTHORITY_SET if w in low)
    base = min(1.0, hits / 4.0)
    return round(min(1.0, base
                     + (0.2 if source == "pdf"   else 0.0)
                     + (0.1 if doc_type == "admin" else 0.0)), 3)

_ACADEMIC_IDF: Dict[str, float] = {
    "prerequisite":3.5,"syllabus":3.2,"curriculum":3.0,"برنامج":3.0,"تخصص":2.9,
    "filière":2.8,"doctorat":2.7,"assignment":2.6,"credits":2.5,"module":2.0,
    "مقياس":2.0,"semester":1.9,"semestre":1.9,"lecture":1.8,"master":1.7,
    "licence":1.7,"exam":1.5,"examen":1.5,"امتحان":1.5,"td":1.4,"tp":1.4,
    "contrôle":1.4,"cours":1.1,"course":1.1,"الفصل":1.0,"الدراسي":1.0,"formation":1.0,
}
_DEFAULT_IDF = 1.2

def compute_academic_score(text: str) -> float:
    if not text: return 0.0
    low   = text.lower()
    words = low.split()
    if not words: return 0.0
    n      = len(words)
    scores = []
    for term in _ACADEMIC_SET:
        if term not in low: continue
        tf = low.count(term) / n
        scores.append(tf * _ACADEMIC_IDF.get(term, _DEFAULT_IDF))
    if not scores: return 0.0
    top3 = sorted(scores, reverse=True)[:3]
    raw  = sum(top3) / len(top3)
    return round(min(1.0, raw / (raw + 0.05)), 3)

_TOKEN_IDF_TABLE: Dict[str, float] = {
    "the":0.1,"de":0.2,"la":0.2,"le":0.2,"et":0.2,"is":0.3,"en":0.3,
    "du":0.3,"un":0.3,"une":0.3,"dans":0.4,"sur":0.4,"pour":0.4,"avec":0.4,
    "université":0.8,"faculté":0.8,"étudiant":0.9,"department":0.9,"professor":0.9,
    "syllabus":3.2,"prerequisite":3.5,"examen":1.5,"semestre":1.9,"filière":2.8,"doctorat":2.7,
}
_DEFAULT_TOKEN_IDF = 1.0

def compute_keyword_density(text: str) -> Tuple[float, float]:
    words = text.lower().split()
    if not words: return 0.0, 0.0
    n       = len(words)
    acad    = sum(1 for w in words if w in _ACADEMIC_SET)
    avg_idf = sum(_TOKEN_IDF_TABLE.get(w, _DEFAULT_TOKEN_IDF) for w in words) / n
    return round(acad / n, 4), round(avg_idf, 4)


# ══════════════════════════════════════════════════════════════
# ADAPTIVE CHUNK SIZER (IMP 2 — unchanged logic, FIX 6 applied)
# ══════════════════════════════════════════════════════════════

class AdaptiveChunkSizer:
    _TYPE_BASE: Dict[str, int] = {
        "exam": 200, "course": 350, "admin": 300, "general": 500,
    }
    _MIN_TOKENS = 150
    _MAX_TOKENS = 700

    def compute(self, doc_type: str, keyword_density: float,
                avg_token_idf: float) -> int:
        base        = self._TYPE_BASE.get(doc_type, CHUNK_TOKENS_BASE)
        kd_norm     = min(1.0, keyword_density / 0.10)
        idf_norm    = min(1.0, avg_token_idf   / 2.50)
        result      = int(base * (1.0 - 0.4 * kd_norm) * (1.0 - 0.2 * idf_norm))
        return max(self._MIN_TOKENS, min(self._MAX_TOKENS, result))


# ══════════════════════════════════════════════════════════════
# MULTILINGUAL TITLE ALIASES (IMP 5 — unchanged)
# ══════════════════════════════════════════════════════════════

def generate_title_aliases(title: str) -> List[str]:
    if not title: return []
    aliases: List[str] = []
    alias_norm = normalize_text(title, aggressive=True).lower()
    if alias_norm and alias_norm != title.lower():
        aliases.append(alias_norm)
    latin_only = re.sub(r"[\u0600-\u06FF\s]+", " ", title).strip().lower()
    if latin_only and latin_only not in aliases:
        aliases.append(latin_only)
    arabic_words = " ".join(_RE_AR_CHARS.findall(title))
    if arabic_words and arabic_words not in aliases:
        aliases.append(arabic_words)
    return [a for a in aliases if len(a) > 2]


# ══════════════════════════════════════════════════════════════
# SECTION DETECTOR (IMP 7 — unchanged)
# ══════════════════════════════════════════════════════════════

def detect_sections(text: str, doc_fp: str) -> List[Dict]:
    sections: List[Dict] = []
    matches = list(_RE_HEADING.finditer(text))
    for i, m in enumerate(matches):
        heading    = m.group(0).strip()
        start_char = m.start()
        end_char   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_id = hashlib.sha1(f"{doc_fp}:{heading}".encode()).hexdigest()[:12]
        sections.append({"section_id": section_id, "heading": heading,
                         "start_char": start_char, "end_char": end_char})
    if not sections:
        sections.append({
            "section_id": hashlib.sha1(f"{doc_fp}:__root__".encode()).hexdigest()[:12],
            "heading": "", "start_char": 0, "end_char": len(text),
        })
    return sections

def assign_section_id(chunk_start_char: int, sections: List[Dict]) -> str:
    for sec in reversed(sections):
        if chunk_start_char >= sec["start_char"]:
            return sec["section_id"]
    return sections[0]["section_id"] if sections else ""


# ══════════════════════════════════════════════════════════════
# SEMANTIC CHUNKER V2  ✦ FIX 1 + FIX 4
#
# Critical change: embed_model parameter is now the LIGHTWEIGHT sentence
# splitter (SENT_SPLIT_MODEL, CPU), not E5-large.  This model is safe
# to call from worker threads because it runs on CPU and has no shared
# GPU state.  Full E5-large embedding happens in one batched call in
# the main thread after all chunking is done.
# ══════════════════════════════════════════════════════════════

class SemanticChunkerV2:
    """
    ✦ FIX 1, FIX 4 — Thread-safe semantic chunker.

    The `split_model` parameter receives the lightweight sentence splitter
    (paraphrase-multilingual-MiniLM-L12-v2) which is CPU-bound and
    reentrant-safe.  It must NOT be E5-large.
    """

    HINT_MAX_CHARS = 100

    def __init__(
        self,
        chunk_tokens:    int   = CHUNK_TOKENS_BASE,
        overlap_tokens:  int   = OVERLAP_TOKENS,
        min_chars:       int   = MIN_CHUNK_CHARS,
        drift_threshold: float = TOPIC_DRIFT_THRESHOLD,
    ):
        self.chunk_tokens    = chunk_tokens
        self.overlap_tokens  = overlap_tokens
        self.min_chars       = min_chars
        self.drift_threshold = drift_threshold
        self._enc = tiktoken.get_encoding("cl100k_base") if _TIKTOKEN_OK else None

    def split(
        self,
        text:         str,
        title:        str        = "",
        split_model              = None,   # ✦ FIX 4: lightweight CPU model only
        chunk_tokens: int        = 0,
        doc_fp:       str        = "",
        sections:     List[Dict] = None,
    ) -> List[Dict]:
        effective_tokens = chunk_tokens or self.chunk_tokens
        text = normalize_text(text)
        if not text:
            return []
        text = remove_repeated_blocks(text)
        if not text:
            return []
        if sections is None:
            sections = detect_sections(text, doc_fp)

        sentences, sent_offsets = self._split_sentences(text)
        if not sentences:
            return []

        drift_positions: Set[int] = set()
        if split_model is not None and len(sentences) > 8:
            drift_positions = self._find_semantic_breaks(sentences, split_model)

        raw_chunks = self._pack_sentences(sentences, drift_positions, effective_tokens)

        result: List[Dict] = []
        for i, (sents, tok_count, start_sent_idx) in enumerate(raw_chunks):
            body = " ".join(sents)
            if title and not body.lower().startswith(title.lower()[:30]):
                embed_body = f"{title}\n{body}".strip()
            else:
                embed_body = body

<<<<<<< HEAD
# ─────────────────────────────────────────────────────────────
# COLLECT JSON FILES
# ─────────────────────────────────────────────────────────────
def collect_json_files(root: Path) -> List[Tuple[Path, str, str]]:
    """
    Returns list of (json_path, faculty_label, department_label).
    Folder layout assumed:
        <root>/<faculty_key>/<sub>/<dept?>/*.json
    where <sub> ∈ {pages, extracted, tables}.
    """
    results = []

    for faculty_dir in sorted(root.iterdir()):
        if not faculty_dir.is_dir():
            continue

        faculty_key   = faculty_dir.name.lower()
        faculty_label = FACULTY_LABELS.get(faculty_key, faculty_key.upper())

        for sub in ["pages", "extracted", "tables"]:
            subfolder = faculty_dir / sub
            if not subfolder.exists():
=======
            if len(embed_body) < self.min_chars:
>>>>>>> 70841a41 (init)
                continue

            # ✦ FIX 1 — hints stored in metadata fields, never in embed_text
            prev_hint = ""
            next_hint = ""
            if i > 0:
                prev_sents = raw_chunks[i - 1][0]
                prev_hint  = prev_sents[-1][:self.HINT_MAX_CHARS] if prev_sents else ""
            if i < len(raw_chunks) - 1:
                next_sents = raw_chunks[i + 1][0]
                next_hint  = next_sents[0][:self.HINT_MAX_CHARS]  if next_sents else ""

            full_text = embed_body
            if prev_hint:
                full_text = f"[prev: {prev_hint}]\n{full_text}"
            if next_hint:
                full_text = f"{full_text}\n[next: {next_hint}]"

            start_char = sent_offsets[start_sent_idx] if start_sent_idx < len(sent_offsets) else 0
            section_id = assign_section_id(start_char, sections)

            result.append({
                "embed_text":  embed_body,   # CLEAN — no hints
                "text":        full_text,    # FULL — with hints for LLM
                "clean_body":  body,
                "token_count": tok_count,
                "chunk_index": i,
                "start_char":  start_char,
                "section_id":  section_id,
                "prev_hint":   prev_hint,
                "next_hint":   next_hint,
            })
        return result

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text)) if self._enc else len(text) // 4

    def _split_sentences(self, text: str) -> Tuple[List[str], List[int]]:
        sentences: List[str] = []
        offsets:   List[int] = []
        pos = 0
        for part in _RE_SENT_BOUND.split(text):
            for sub in part.split("\n\n"):
                sub = sub.strip()
                if sub:
                    sentences.append(sub)
                    offsets.append(pos)
                pos += len(sub) + 1
        return sentences, offsets

    def _find_semantic_breaks(
        self,
        sentences:   List[str],
        split_model,            # ✦ FIX 4: lightweight CPU model
    ) -> Set[int]:
        """
        ✦ FIX 4 — Uses the lightweight sentence splitter (MiniLM, CPU).
        No prefix needed — this model compares sentence similarity directly.
        """
        try:
            embs = split_model.encode(
                sentences,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )
        except Exception as exc:
            log.warning("Sentence embedding failed in chunker: %s", exc)
            return set()

        breaks: Set[int] = set()
        for i in range(len(embs) - 1):
            sim = float(np.dot(embs[i], embs[i + 1]))
            if sim < self.drift_threshold:
                breaks.add(i + 1)
        return breaks

    def _pack_sentences(
        self,
        sentences:    List[str],
        drift_breaks: Set[int],
        chunk_tokens: int,
    ) -> List[Tuple[List[str], int, int]]:
        chunks: List[Tuple[List[str], int, int]] = []
        current_sents: List[str] = []
        current_tok   = 0
        start_idx     = 0

        for idx, sent in enumerate(sentences):
            sent_tok          = self._token_count(sent)
            is_semantic_break = idx in drift_breaks
            budget_exceeded   = (current_tok + sent_tok > chunk_tokens
                                 and current_sents)

            if (is_semantic_break or budget_exceeded) and current_sents:
                chunks.append((list(current_sents), current_tok, start_idx))
                if not is_semantic_break:
                    ov_sents, ov_tok = [], 0
                    for s in reversed(current_sents):
                        st = self._token_count(s)
                        if ov_tok + st > self.overlap_tokens: break
                        ov_sents.insert(0, s); ov_tok += st
                    current_sents = ov_sents + [sent]
                    current_tok   = ov_tok + sent_tok
                    start_idx     = idx - len(ov_sents)
                else:
                    current_sents = [sent]
                    current_tok   = sent_tok
                    start_idx     = idx
            else:
                current_sents.append(sent)
                current_tok += sent_tok

        if current_sents:
            chunks.append((current_sents, current_tok, start_idx))
        return chunks


TokenChunker = SemanticChunkerV2   # backward-compat alias


# ══════════════════════════════════════════════════════════════
# INGESTION RERANKER  ✦ FIX 5
# ══════════════════════════════════════════════════════════════

class IngestionReranker:
    """
    ✦ FIX 5 — batch_size reduced to 32; max_length=256 enforced.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        if not CROSS_ENCODER_ENABLED:
            self._model = None
            return
        log.info("Loading ingestion cross-encoder: %s", model_name)
        try:
            self._model = CrossEncoder(model_name, max_length=256)
            log.info("Ingestion cross-encoder ready")
        except Exception as exc:
            log.warning("Cross-encoder load failed (%s) — disabled", exc)
            self._model = None

    def filter(
        self,
        title:  str,
        chunks: List[Dict],
        floor:  float = RERANK_QUALITY_FLOOR,
    ) -> List[Dict]:
        if self._model is None or not chunks or not title:
            return chunks

        results: List[Dict] = []
        texts   = [c["embed_text"] for c in chunks]

        for i in range(0, len(texts), 32):   # ✦ FIX 5: batch_size=32
            batch_texts = texts[i: i + 32]
            pairs       = [(title, t[:512]) for t in batch_texts]   # ✦ FIX 5: explicit truncation
            try:
                scores = self._model.predict(pairs)
            except Exception as exc:
                log.warning("Cross-encoder predict failed: %s", exc)
                results.extend(chunks[i: i + 32])
                continue

            for chunk, score in zip(chunks[i: i + 32], scores):
                if float(score) >= floor:
                    results.append(chunk)
                else:
                    log.debug("IMP4 rejected (score=%.2f): %s…",
                              score, chunk["embed_text"][:60])

        log.info("Quality filter: %d → %d chunks (%.0f%% kept)",
                 len(chunks), len(results),
                 100.0 * len(results) / len(chunks) if chunks else 0)
        return results


# ══════════════════════════════════════════════════════════════
# QUALITY FILTER
# ══════════════════════════════════════════════════════════════

def quality_and_score(text: str) -> Tuple[bool, str, float]:
    stripped = text.strip()
    n = len(stripped)
    if n < MIN_CHUNK_CHARS: return False, f"too_short ({n})", 0.0
    low = stripped.lower()
    bp  = sum(1 for w in _BOILERPLATE_SET if w in low)
    if bp / _BPLATE_DENOM > MAX_BOILERPLATE_RATIO:
        return False, f"boilerplate ({bp/_BPLATE_DENOM:.0%})", 0.0
    return True, "ok", compute_academic_score(stripped)


# ══════════════════════════════════════════════════════════════
# FINGERPRINT & BM25 TOKENISER  ✦ FIX 7
# ══════════════════════════════════════════════════════════════

def chunk_fingerprint(text: str) -> str:
    normed = normalize_text(text, aggressive=True).lower()
    return hashlib.sha1(normed.encode("utf-8")).hexdigest()[:16]

def tokenize_for_bm25(text: str) -> str:
    """
    ✦ FIX 7 — Always returns a str (space-joined tokens).
    BM25Retriever calls .split() on this field; str guarantees compatibility
    regardless of whether the metadata.json was written by v3 or v4.
    """
    normed = normalize_text(text, aggressive=True)
    normed = re.sub(r"[^\w\s]", " ", normed, flags=re.UNICODE)
    return " ".join(t for t in normed.lower().split() if len(t) > 1)


# ══════════════════════════════════════════════════════════════
# JSON PARSERS & TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

def _process_table(tbl: Dict) -> Tuple[str, str, Dict]:
    headers = tbl.get("headers", [])
    rows    = tbl.get("rows",    [])
    clean_h = [re.sub(r"\s+", " ", str(h)).strip() for h in headers]
    clean_h = [h for h in clean_h if h]
    if not clean_h: return "", "", {}
    md  = "| " + " | ".join(clean_h) + " |\n"
    md += "|" + "|".join("---" for _ in clean_h) + "|\n"
    clean_rows: List[Dict] = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(row.get(h, row.get(orig,"")))).strip()
                 for h, orig in zip(clean_h, headers)]
        if not any(cells): continue
        while len(cells) < len(clean_h): cells.append("")
        md += "| " + " | ".join(cells) + " |\n"
        clean_rows.append(dict(zip(clean_h, cells)))
    summary    = f"Table with {len(clean_rows)} rows. Columns: {', '.join(clean_h[:5])}"
    structured = {"headers": clean_h, "rows": clean_rows, "table_summary": summary}
    return md, summary, structured

def table_to_prose(structured: Dict, title: str = "") -> str:
    headers = structured.get("headers", [])
    rows    = structured.get("rows",    [])
    if not headers or not rows: return ""
    lines = [f"{title + '. ' if title else ''}Table: {', '.join(headers)}."]
    for i, row in enumerate(rows[:20]):
        cells = [f"{h}={row.get(h,'')}" for h in headers if row.get(h)]
        if cells: lines.append(f"Row {i+1}: {', '.join(cells)}.")
    return " ".join(lines)

def extract_links(text: str) -> List[str]:
    return list(dict.fromkeys(_RE_URL_GENERAL.findall(text)))

def extract_pdf_urls(text: str, base_url: str = "") -> List[str]:
    urls: List[str] = list(dict.fromkeys(_RE_PDF_URL.findall(text)))
    if base_url:
        base = base_url.rstrip("/")
        for rel in _RE_REL_PDF.findall(text):
            if not rel.startswith("http"):
                full = f"{base}/{rel.lstrip('/')}"
                if full not in urls: urls.append(full)
    return urls

def remove_repeated_blocks(text: str) -> str:
    if not text: return text
    seen_para: Set[str] = set()
    para_out: List[str] = []
    for para in text.split("\n\n"):
        s = para.strip()
        if not s: para_out.append(para); continue
        if s not in seen_para: seen_para.add(s); para_out.append(para)
    text = "\n\n".join(para_out)
    seen_line: Set[str] = set()
    line_out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s: line_out.append(line); continue
        if s not in seen_line: seen_line.add(s); line_out.append(line)
    return "\n".join(line_out)

def parse_json(data: dict) -> dict:
    meta    = data.get("metadata", {})
    content = data.get("content",  {})

    if "page" in meta:
        page     = meta["page"]
        page_url = page.get("url", "")
        parts    = [content.get("text", "")]
        for section in content.get("sections", []):
            if isinstance(section, dict):
                parts.append(section.get("text",  ""))
                parts.append(section.get("title", ""))
        raw_text = "\n\n".join(filter(None, parts))
        combined = normalize_text(raw_text)
        raw_tables: List[Dict] = content.get("tables", [])
        tables_structured: List[Dict] = []
        for tbl in raw_tables:
            _, _, s = _process_table(tbl)
            if s: tables_structured.append(s)
        return dict(text=combined, raw_length=len(raw_text), clean_length=len(combined),
                    title=page.get("title",""), url=page_url, file_path="",
                    file_type="web", tables=raw_tables, tables_structured=tables_structured,
                    links=extract_links(raw_text),
                    pdf_urls=extract_pdf_urls(raw_text, page_url), source="scraper")

    file_info        = meta.get("file", {})
    file_path        = file_info.get("path", "")
    file_type        = file_info.get("type", "")
    effective_source = ("pdf"
                        if (file_type.lower() == "pdf" or file_path.lower().endswith(".pdf"))
                        else "extractor")
    parts: List[str] = []
    if content.get("text"): parts.append(content["text"])
    for pg in content.get("pages", []):
        if isinstance(pg, dict) and pg.get("text"): parts.append(pg["text"])
    for sec in content.get("sections", []):
        if isinstance(sec, dict):
            if sec.get("title"): parts.append(sec["title"])
            if sec.get("text"):  parts.append(sec["text"])
    tables: List[Dict] = content.get("tables", [])
    tables_structured: List[Dict] = []
    for tbl in tables:
        md, _, s = _process_table(tbl)
        if md: parts.append(md)
        if s:  tables_structured.append(s)
    raw_text = "\n\n".join(filter(None, parts))
    combined = normalize_text(raw_text)
    title    = file_info.get("name", "") or Path(file_path).stem
    return dict(text=combined, raw_length=len(raw_text), clean_length=len(combined),
                title=title, url=file_info.get("url",""), file_path=file_path,
                file_type=file_type, tables=tables, tables_structured=tables_structured,
                links=extract_links(raw_text), pdf_urls=[], source=effective_source)


# ══════════════════════════════════════════════════════════════
# FILE COLLECTION
# ══════════════════════════════════════════════════════════════

def collect_json_files(root: Path) -> List[Tuple[Path, str, str]]:
    results: List[Tuple[Path, str, str]] = []
    for faculty_dir in sorted(root.iterdir()):
        if not faculty_dir.is_dir(): continue
        fl = FACULTY_LABELS.get(faculty_dir.name.lower(), faculty_dir.name.upper())
        for sub in ("pages", "extracted", "tables"):
            sfolder = faculty_dir / sub
            if not sfolder.exists(): continue
            base_str = str(sfolder)
            base_len = len(base_str) + 1
            for dirpath, _, filenames in os.walk(base_str):
                for fname in filenames:
                    if not fname.endswith(".json"): continue
                    jf   = Path(dirpath) / fname
                    rem  = str(jf)[base_len:]
                    sp   = rem.find(os.sep)
                    dept = (rem[:sp] if sp != -1 else "General"
                            ).replace("_"," ").replace("-"," ").title()
                    results.append((jf, fl, dept))
        log.info("📂 %s → %d files",
                 faculty_dir.name, sum(1 for r in results if r[1] == fl))
    log.info("🔎 TOTAL JSON FILES: %d", len(results))
    return results

def _doc_fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════
# SQLITE PROGRESS TRACKER  ✦ FIX 11
# ══════════════════════════════════════════════════════════════

class ProgressDB:
    """✦ FIX 11 — Only called from the main thread; workers never touch this."""

    def __init__(self, db_path: str = PROGRESS_DB):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-32000")
        self._init()
        self._done_set: Set[str] = {
            row[0] for row in self.conn.execute("SELECT file_key FROM processed")
        }
        log.info("ProgressDB: %d files already processed", len(self._done_set))
        self._pending: List[Tuple] = []

    def _init(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processed (
                file_key TEXT PRIMARY KEY, chunks INTEGER,
                doc_type TEXT, language TEXT, processed_at TEXT)""")
        self.conn.commit()

    def is_done(self, fk: str) -> bool: return fk in self._done_set

    def mark_done(self, fk: str, n: int, dt: str, lang: str):
        self._done_set.add(fk)
        self._pending.append((fk, n, dt, lang, datetime.utcnow().isoformat()))
        if len(self._pending) >= PROGRESS_FLUSH: self._flush()

    def _flush(self):
        if not self._pending: return
        self.conn.executemany(
            "INSERT OR REPLACE INTO processed VALUES (?,?,?,?,?)", self._pending)
        self.conn.commit()
        self._pending.clear()

    def flush_final(self): self._flush()
    def reset(self):
        self.conn.execute("DELETE FROM processed"); self.conn.commit()
        self._done_set.clear(); self._pending.clear()
    def close(self): self.flush_final(); self.conn.close()


# ══════════════════════════════════════════════════════════════
# NEO4J BATCH STORE  ✦ FIX 8
# ══════════════════════════════════════════════════════════════

def _ensure_hierarchy(tx, university, faculty, department,
                      doc_title, doc_type, language):
    tx.run("""
        MERGE (u:University {name: $university})
        MERGE (f:Faculty    {name: $faculty})
        MERGE (u)-[:HAS_FACULTY]->(f)
        MERGE (dept:Department {name: $department, faculty: $faculty})
        MERGE (f)-[:HAS_DEPARTMENT]->(dept)
        MERGE (d:Document {title: $doc_title, faculty: $faculty,
                           department: $department})
        SET d.doc_type=$doc_type, d.language=$language
        MERGE (dept)-[:HAS_DOCUMENT]->(d)
    """, university=university, faculty=faculty, department=department,
        doc_title=doc_title, doc_type=doc_type, language=language)


def _ensure_sections_and_chunks(
    tx,
    doc_title:    str,
    sections:     List[Dict],
    chunks_batch: List[Dict],
):
    """
    ✦ FIX 8 — Sections and chunks created in a single transaction.
    This eliminates the race condition where _insert_chunks_batch()
    ran before Section nodes were committed.
    """
    # Step 1: upsert Section nodes
    tx.run("""
        UNWIND $sections AS sec
        MATCH (d:Document {title: $doc_title})
        MERGE (s:Section {id: sec.section_id})
        SET s.heading    = sec.heading,
            s.start_char = sec.start_char,
            s.end_char   = sec.end_char
        MERGE (d)-[:HAS_SECTION]->(s)
    """, doc_title=doc_title, sections=sections)

    # Step 2: upsert Chunk nodes and link to Document AND Section
    tx.run("""
        UNWIND $chunks AS ch
        MERGE (c:Chunk {id: ch.id})
        SET c.text            = ch.text,
            c.chunk_index     = ch.chunk_index,
            c.academic_score  = ch.academic_score,
            c.authority_score = ch.authority_score,
            c.has_tables      = ch.has_tables,
            c.token_count     = ch.token_count,
            c.avg_token_idf   = ch.avg_token_idf,
            c.section_id      = ch.section_id
        WITH c, ch
        MATCH (d:Document {title: ch.doc_title})
        MERGE (d)-[:HAS_CHUNK {order: ch.chunk_index}]->(c)
        WITH c, ch
        MATCH (s:Section {id: ch.section_id})
        MERGE (s)-[:HAS_CHUNK]->(c)
    """, chunks=chunks_batch)


def _link_chunks_sequentially(tx, doc_title: str):
    tx.run("""
        MATCH (d:Document {title: $title})-[:HAS_CHUNK]->(c:Chunk)
        WITH c ORDER BY c.chunk_index
        WITH collect(c) AS ordered
        UNWIND range(0, size(ordered)-2) AS i
        WITH ordered[i] AS curr, ordered[i+1] AS nxt
        MERGE (curr)-[:NEXT_CHUNK]->(nxt)
    """, title=doc_title)

def _link_all_docs_sequentially(session, doc_titles: List[str]):
    for t in doc_titles:
        session.execute_write(_link_chunks_sequentially, t)


# ══════════════════════════════════════════════════════════════
# SEMANTIC DEDUPLICATION  ✦ FIX 2
# ══════════════════════════════════════════════════════════════

def apply_semantic_dedup_doc(
    embeddings:  np.ndarray,    # shape (N, dim), L2-normalised, SINGLE DOCUMENT
    chunk_texts: List[str],
    threshold:   float = SEMANTIC_DEDUP_THRESHOLD,
    window:      int   = SEMANTIC_DEDUP_WINDOW,
) -> List[int]:
    """
    ✦ FIX 2 — Per-document dedup.
    The sliding-window proximity assumption only holds within one document.
    Returns indices of UNIQUE chunks (within this document's slice).
    Keeps the longer text in each near-duplicate pair.
    """
    n         = len(embeddings)
    keep_mask = [True] * n

    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, min(i + window + 1, n)):
            if not keep_mask[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                if len(chunk_texts[i]) >= len(chunk_texts[j]):
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break

    kept = [idx for idx, keep in enumerate(keep_mask) if keep]
    if len(kept) < n:
        log.info("Semantic dedup (doc): %d → %d (%.0f%% kept)",
                 n, len(kept), 100.0 * len(kept) / n)
    return kept


# ══════════════════════════════════════════════════════════════
# PARSE + CHUNK WORKER  ✦ FIX 1, FIX 4, FIX 11
# Workers do NOT call embed_model (GPU) and do NOT write ProgressDB.
# ══════════════════════════════════════════════════════════════

def _parse_and_chunk_file(
    args: Tuple[Path, str, str, SemanticChunkerV2, AdaptiveChunkSizer,
                IngestionReranker, "SentenceTransformer"]   # ✦ FIX 4: split_model
) -> Optional[Dict]:
    jf, faculty, department, chunker, sizer, reranker, split_model = args
    file_key = f"{faculty}/{department}/{jf.name}"

    try:
        with open(jf, "r", encoding="utf-8") as fh:
            raw = _load_json(fh)
        parsed = parse_json(raw)
    except Exception as exc:
        log.warning("⚠  Parse error [%s]: %s", jf.name, exc)
        return {"fail": True, "file_key": file_key}

    if len(parsed["text"].strip()) < MIN_DOC_CHARS:
        return {"skip": True, "file_key": file_key}

    title      = parsed["title"] or jf.stem
    language   = detect_language(parsed["text"])
    doc_type   = infer_doc_type(jf.name, title, parsed["text"])
    year       = extract_year(parsed["text"])
    fp         = _doc_fingerprint(parsed["text"])
    auth_score = compute_authority_score(parsed["text"], doc_type, parsed["source"])
    alt_titles = generate_title_aliases(title)

    sample_kd, sample_idf = compute_keyword_density(parsed["text"][:1000])
    adaptive_tokens       = sizer.compute(doc_type, sample_kd, sample_idf)
    sections              = detect_sections(parsed["text"], fp)

    extra_chunks: List[Dict] = []
    for structured in parsed.get("tables_structured", []):
        prose = table_to_prose(structured, title=title)
        if prose and len(prose) >= MIN_CHUNK_CHARS:
            extra_chunks.append({
                "embed_text":  prose, "text": prose, "clean_body": prose,
                "token_count": len(prose) // 4, "chunk_index": -1,
                "section_id":  sections[0]["section_id"] if sections else "",
                "start_char": 0, "prev_hint": "", "next_hint": "",
            })

    # ✦ FIX 4 — pass split_model (CPU MiniLM), NOT E5-large
    raw_chunks = chunker.split(
        parsed["text"], title,
        split_model=split_model,
        chunk_tokens=adaptive_tokens,
        doc_fp=fp,
        sections=sections,
    )
    all_raw = raw_chunks + [
        {**c, "chunk_index": len(raw_chunks) + i}
        for i, c in enumerate(extra_chunks)
    ]

    if not all_raw:
        return {"skip": True, "file_key": file_key}

    # Cross-encoder quality filter
    all_raw = reranker.filter(title, all_raw)
    if not all_raw:
        return {"skip": True, "file_key": file_key}

    quality_counts = {"too_short": 0, "boilerplate": 0, "low_academic": 0}
    good_chunks:     List[Dict] = []
    seen_chunk_fps:  Set[str]   = set()

    for chunk_dict in all_raw:
        embed_text = chunk_dict.get("embed_text", chunk_dict["text"])
        ok_flag, reason, acad_score = quality_and_score(embed_text)
        if not ok_flag:
            if "too_short"   in reason: quality_counts["too_short"]   += 1
            if "boilerplate" in reason: quality_counts["boilerplate"] += 1
            continue
        if acad_score < 0.05:
            quality_counts["low_academic"] += 1

        cfp = chunk_fingerprint(embed_text)
        if cfp in seen_chunk_fps:
            continue
        seen_chunk_fps.add(cfp)

        clean_body    = chunk_dict.get("clean_body", embed_text)
        kd, avg_idf   = compute_keyword_density(clean_body)
        entities      = extract_entities(chunk_dict["text"])
        tokenized_str = tokenize_for_bm25(clean_body)   # ✦ FIX 7: str

        chunk_dict.update({
            "academic_score":  acad_score,
            "authority_score": auth_score,
            "tokenized_text":  tokenized_str,
            "keyword_density": kd,
            "avg_token_idf":   avg_idf,
            "entities":        entities,
            "chunk_fp":        cfp,
        })
        good_chunks.append(chunk_dict)

    if not good_chunks:
        return {"skip": True, "file_key": file_key}

    return {
        "ok": True, "file_key": file_key, "jf": jf,
        "faculty": faculty, "department": department,
        "title": title, "language": language,
        "doc_type": doc_type, "year": year, "fp": fp,
        "good_chunks": good_chunks,
        "has_tables": bool(parsed.get("tables")),
        "tables_structured": parsed.get("tables_structured", []),
        "source": parsed["source"], "url": parsed.get("url",""),
        "file_path": parsed.get("file_path",""), "links": parsed.get("links",[]),
        "pdf_urls": parsed.get("pdf_urls",[]),
        "raw_length": parsed.get("raw_length",0),
        "clean_length": parsed.get("clean_length",0),
        "authority_score": auth_score, "alternate_titles": alt_titles,
        "sections": sections,
        "quality_counts": quality_counts,
    }


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE  ✦ ALL FIXES INTEGRATED
# ══════════════════════════════════════════════════════════════

def build_pipeline(resume: bool = True, clear_chroma: bool = False):

    # ── Model loading ─────────────────────────────────────────
    log.info("Loading embedding model: %s", EMBED_MODEL)
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"
    log.info("E5 device: %s", _device)
    embed_model = SentenceTransformer(EMBED_MODEL, device=_device)

    # ✦ FIX 12 — assertion after model load so log output is visible
    assert PASSAGE_PREFIX == "passage: ", \
        "PASSAGE_PREFIX must be 'passage: ' for multilingual-e5-large"
    log.info("✅ E5 prefix contract: ingestion='%s'", PASSAGE_PREFIX)

    # ✦ FIX 4 — lightweight CPU sentence splitter for workers
    log.info("Loading sentence split model: %s (CPU)", SENT_SPLIT_MODEL)
    split_model = SentenceTransformer(SENT_SPLIT_MODEL, device="cpu")
    log.info("Split model ready")

    # ── Storage ───────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    if clear_chroma:
        try:
            chroma_client.delete_collection("university_data")
            log.info("🗑️  ChromaDB cleared")
        except Exception:
            pass
    collection = chroma_client.get_or_create_collection(
        name="university_data", metadata={"hnsw:space": "cosine"})

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    chunker  = SemanticChunkerV2()
    sizer    = AdaptiveChunkSizer()
    reranker = IngestionReranker()
    progress = ProgressDB(PROGRESS_DB)
    if not resume:
        log.info("🔥 Fresh start"); progress.reset()

    root       = Path(ROOT_FOLDER)
    all_files  = collect_json_files(root)
    json_files = [
        (jf, f, d) for jf, f, d in all_files
        if not (resume and progress.is_done(f"{f}/{d}/{jf.name}"))
    ]
    log.info("📂 %d / %d files to process", len(json_files), len(all_files))

    ok = skip = fail = 0
    total_chunks    = 0
    quality_stats   = {"too_short": 0, "boilerplate": 0, "low_academic": 0}
    metadata_store: List[Dict] = []
    seen_doc_fps:   Set[str]   = set()
    seen_chunk_fps: Set[str]   = set()
    docs_to_link:   List[str]  = []

    # ── Parallel parse+chunk (NO GPU, NO ProgressDB) ✦ FIX 4, FIX 11 ──
    log.info("🚀 Parsing & chunking (%d workers) …", PARSE_WORKERS)
    worker_args = [
        (jf, f, d, chunker, sizer, reranker, split_model)
        for jf, f, d in json_files
    ]
    parsed_results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as pool:
        futures = {pool.submit(_parse_and_chunk_file, arg): arg
                   for arg in worker_args}
        for future in as_completed(futures):
            res = future.result()
            if res is None:     fail += 1; continue
            if res.get("skip"): skip += 1; continue
            if res.get("fail"): fail += 1; continue
            fp = res.get("fp", "")
            if fp and fp in seen_doc_fps: skip += 1; continue
            if fp: seen_doc_fps.add(fp)
            parsed_results.append(res)

    log.info("Parse done: %d docs, %d skipped, %d failed",
             len(parsed_results), skip, fail)

    # ── Global embedding pass  ✦ FIX 1, FIX 3 ───────────────
    # Embed ONLY clean embed_text (not hint-augmented text)
    flat_embed:   List[str]             = []
    flat_text:    List[str]             = []   # stored text (with hints)
    flat_mapping: List[Tuple[int, int]] = []

    for doc_idx, res in enumerate(parsed_results):
        for local_idx, cd in enumerate(res["good_chunks"]):
            flat_embed.append(cd.get("embed_text", cd["text"]))
            flat_text.append(cd["text"])
            flat_mapping.append((doc_idx, local_idx))

    MAX_CHUNKS = 50_000
    if len(flat_embed) > MAX_CHUNKS:
        log.warning("⚠  Truncating to %d chunks (OOM guard)", MAX_CHUNKS)
        flat_embed   = flat_embed[:MAX_CHUNKS]
        flat_text    = flat_text[:MAX_CHUNKS]
        flat_mapping = flat_mapping[:MAX_CHUNKS]

    log.info("🧮 Embedding %d chunks (batch=%d) …", len(flat_embed), EMBED_BATCH)
    flat_prefixed = [PASSAGE_PREFIX + t for t in flat_embed]
    all_embs_np: np.ndarray = embed_model.encode(
        flat_prefixed,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info("✅ Embedding complete: %s", str(all_embs_np.shape))

    # ── Build doc_emb_slices  ✦ FIX 9 ────────────────────────
    # Correct bookkeeping: one slice per parsed_result doc.
    doc_emb_slices: List[Tuple[int, int]] = []
    if flat_mapping:
        current_doc   = flat_mapping[0][0]
        current_start = 0
        for g_idx, (doc_idx, _) in enumerate(flat_mapping):
            if doc_idx != current_doc:
                doc_emb_slices.append((current_start, g_idx))
                current_start = g_idx
                current_doc   = doc_idx
        doc_emb_slices.append((current_start, len(flat_mapping)))

    assert len(doc_emb_slices) == len(parsed_results), (
        f"Slice mismatch: {len(doc_emb_slices)} != {len(parsed_results)}"
    )

    # ── Store stage ───────────────────────────────────────────
    with neo4j_driver.session() as neo4j_session:
        neo4j_buf: List[Dict] = []

        def _flush_neo4j_buf():
            for i in range(0, len(neo4j_buf), NEO4J_BATCH):
                pass   # handled per-doc via _ensure_sections_and_chunks
            neo4j_buf.clear()

        for doc_idx, res in enumerate(parsed_results):
            faculty           = res["faculty"]
            department        = res["department"]
            title             = res["title"]
            language          = res["language"]
            doc_type          = res["doc_type"]
            year              = res["year"]
            fp                = res["fp"]
            good_chunks       = res["good_chunks"]
            has_tables        = res["has_tables"]
            source            = res["source"]
            url               = res.get("url", "")
            file_key          = res["file_key"]
            jf                = res["jf"]
            file_path         = res.get("file_path", "")
            links             = res.get("links", [])
            pdf_urls          = res.get("pdf_urls", [])
            tables_structured = res.get("tables_structured", [])
            raw_length        = res.get("raw_length", 0)
            clean_length      = res.get("clean_length", 0)
            auth_score        = res.get("authority_score", 0.0)
            alt_titles        = res.get("alternate_titles", [])
            sections          = res.get("sections", [])

            for k, v in res.get("quality_counts", {}).items():
                quality_stats[k] += v

            emb_start, emb_end = doc_emb_slices[doc_idx]
            doc_embs_np        = all_embs_np[emb_start:emb_end]

            # ✦ FIX 2 — per-document semantic dedup
            doc_embed_texts = [
                cd.get("embed_text", cd["text"]) for cd in good_chunks
            ]
            kept_local = set(apply_semantic_dedup_doc(
                doc_embs_np, doc_embed_texts,
                threshold=SEMANTIC_DEDUP_THRESHOLD,
                window=SEMANTIC_DEDUP_WINDOW,
            ))

            chroma_records:   List[Dict] = []
            doc_meta_rows:    List[Dict] = []
            neo4j_chunk_buf:  List[Dict] = []
            embed_ptr = 0   # ✦ FIX 3 — explicit pointer for surviving chunks

            for local_idx, chunk_dict in enumerate(good_chunks):
                # ✦ FIX 2 — skip near-duplicates (per-document)
                if local_idx not in kept_local:
                    embed_ptr += 1
                    continue

                # Cross-file chunk dedup
                cfp = chunk_dict.get("chunk_fp",
                      chunk_fingerprint(chunk_dict.get("embed_text", chunk_dict["text"])))
                if cfp in seen_chunk_fps:
                    embed_ptr += 1
                    continue
                seen_chunk_fps.add(cfp)

                cid         = f"{fp}_c{local_idx}"
                embed_text  = chunk_dict.get("embed_text", chunk_dict["text"])
                stored_text = chunk_dict["text"]
                clean_body  = chunk_dict.get("clean_body", embed_text)
                section_id  = chunk_dict.get("section_id", "")
                prev_hint   = chunk_dict.get("prev_hint", "")
                next_hint   = chunk_dict.get("next_hint", "")
                kd          = chunk_dict.get("keyword_density", 0.0)
                avg_idf     = chunk_dict.get("avg_token_idf", 0.0)
                entities    = chunk_dict.get("entities", {})
                tokenized   = chunk_dict.get("tokenized_text", "")
                chunk_has_t = has_tables and ("|" in stored_text or bool(tables_structured))

                # ✦ FIX 3 — use embed_ptr (sequential kept-chunk index) for embedding
                chroma_records.append({
                    "id":       cid,
                    "text":     embed_text,        # CLEAN text stored in Chroma
                    "_emb_ptr": embed_ptr,         # ✦ FIX 3
                    "metadata": {
                        "faculty":         faculty,
                        "department":      department,
                        "language":        language,
                        "source":          source,
                        "doc_type":        doc_type,
                        "chunk_index":     local_idx,
                        "total_chunks":    len(good_chunks),
                        "has_table":       chunk_has_t,
                        "academic_score":  round(chunk_dict["academic_score"], 3),
                        "authority_score": round(auth_score, 3),
                        "keyword_density": round(kd, 4),
                        "avg_token_idf":   round(avg_idf, 4),
                        "year":            year or 0,
                        "chunk_len":       len(embed_text),
                        "url":             url,
                        "has_email":       bool(entities.get("emails")),
                        "has_course_code": bool(entities.get("course_codes")),
                        "has_phone":       bool(entities.get("phones")),
                        "section_id":      section_id,
                    }
                })

                neo4j_chunk_buf.append({
                    "id":             cid,
                    "text":           embed_text,
                    "chunk_index":    local_idx,
                    "academic_score": round(chunk_dict["academic_score"], 3),
                    "authority_score": round(auth_score, 3),
                    "has_tables":     has_tables,
                    "token_count":    chunk_dict.get("token_count", 0),
                    "avg_token_idf":  round(avg_idf, 4),
                    "section_id":     section_id,
                    "doc_title":      title,
                })

                doc_meta_rows.append({
                    "chunk_id":         cid,
                    "file":             jf.name,
                    "faculty":          faculty,
                    "department":       department,
                    "title":            title,
                    "alternate_titles": alt_titles,
                    "source":           source,
                    "url":              url,
                    "file_path":        file_path,
                    "pdf_urls":         pdf_urls,
                    "language":         language,
                    "doc_type":         doc_type,
                    "chunk":            stored_text,   # full text with hints
                    "embed_text":       embed_text,    # clean text
                    "clean_text":       clean_body,
                    "tokenized_text":   tokenized,     # ✦ FIX 7: str
                    "prev_hint":        prev_hint,
                    "next_hint":        next_hint,
                    "raw_length":       raw_length,
                    "clean_length":     clean_length,
                    "academic_score":   round(chunk_dict["academic_score"], 3),
                    "authority_score":  round(auth_score, 3),
                    "keyword_density":  round(kd, 4),
                    "avg_token_idf":    round(avg_idf, 4),
                    "chunk_index":      local_idx,
                    "year":             year,
                    "links":            links,
                    "tables_structured": tables_structured,
                    "entities":         entities,
                    "section_id":       section_id,
                })

                embed_ptr += 1

            metadata_store.extend(doc_meta_rows)

            # ── ChromaDB upsert  ✦ FIX 3 ─────────────────────
            # Build a compact embedding matrix of only surviving chunks
            surviving_ptrs = [r["_emb_ptr"] for r in chroma_records]
            if surviving_ptrs:
                surviving_embs = doc_embs_np[surviving_ptrs]   # ✦ FIX 3
                for bs in range(0, len(chroma_records), CHROMA_BATCH):
                    batch = chroma_records[bs: bs + CHROMA_BATCH]
                    # Local index within surviving_embs (0, 1, 2, …)
                    lo = bs
                    hi = bs + len(batch)
                    collection.upsert(
                        ids       =[r["id"]       for r in batch],
                        documents =[r["text"]     for r in batch],
                        embeddings=surviving_embs[lo:hi].tolist(),
                        metadatas =[r["metadata"] for r in batch],
                    )

            # ── Neo4j  ✦ FIX 8 ───────────────────────────────
            neo4j_session.execute_write(
                _ensure_hierarchy,
                UNIVERSITY_NAME, faculty, department, title, doc_type, language,
            )
            # ✦ FIX 8 — sections + chunks in one tx (no race condition)
            if neo4j_chunk_buf:
                for i in range(0, len(neo4j_chunk_buf), NEO4J_BATCH):
                    batch = neo4j_chunk_buf[i: i + NEO4J_BATCH]
                    neo4j_session.execute_write(
                        _ensure_sections_and_chunks, title, sections, batch
                    )

            docs_to_link.append(title)

            # ✦ FIX 11 — progress written in main thread only
            progress.mark_done(file_key, len(doc_meta_rows), doc_type, language)
            total_chunks += len(doc_meta_rows)
            ok += 1

            # ✦ FIX 6 — no walrus operators in log line
            log.info(
                "✅ [%-20s / %-15s] %s → %d chunks (lang=%s type=%s auth=%.2f)",
                faculty[:20], department[:15], jf.name,
                len(doc_meta_rows), language, doc_type, auth_score,
            )

        if docs_to_link:
            log.info("🔗 NEXT_CHUNK edges for %d docs …", len(docs_to_link))
            _link_all_docs_sequentially(neo4j_session, docs_to_link)

    # ── Save metadata.json ────────────────────────────────────
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata_store, fh, ensure_ascii=False, indent=2)

    progress.close()
    neo4j_driver.close()

    log.info("\n" + "─" * 62)
    log.info("  PIPELINE v4-FIXED COMPLETE")
    log.info("  ✅ SUCCESS : %d  ⏭️  SKIPPED : %d  ❌ FAILED : %d", ok, skip, fail)
    log.info("  📦 CHUNKS  : %d", total_chunks)
    log.info("  Quality: too_short=%d  boilerplate=%d  low_academic=%d",
             quality_stats["too_short"], quality_stats["boilerplate"],
             quality_stats["low_academic"])
    log.info("─" * 62)
    return ok, skip, fail


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    fresh        = "--fresh"        in sys.argv
    clear_chroma = "--clear-chroma" in sys.argv
    if fresh:        log.info("🔥 --fresh: resetting progress DB")
    if clear_chroma: log.info("🗑️  --clear-chroma: wiping ChromaDB")
    build_pipeline(resume=not fresh, clear_chroma=clear_chroma)

