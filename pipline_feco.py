"""
Ingestion Pipeline — Faculty of Economics (Farhat Abbas University Sétif 1)
Vector Database version using ChromaDB instead of Neo4j.

Install dependencies:
    pip install chromadb sentence-transformers loguru tiktoken pypdf

Optional:
    pip install google-generativeai   # for Gemini fallback classifier
"""

from __future__ import annotations

import hashlib, json, os, re, time, unicodedata, warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional deps ──────────────────────────────────────────────
try:
    import tiktoken; _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False

try:
    import google.generativeai as genai; _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

try:
    import pypdf; _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False
    raise ImportError("chromadb is required: pip install chromadb")

from sentence_transformers import SentenceTransformer

# ═══════════════════════════ CONFIG ═══════════════════════════

ROOT_FOLDER    = "./university_farhat_abaas"
STRUCTURE_FILE = "./structure_economics.json"   # ← your economics JSON
ALIASES_FILE   = "./aliases.json"

# ChromaDB — persists to disk so data survives restarts
CHROMA_PERSIST_DIR = "./chroma_economics_db"
CHROMA_COLLECTION  = "economics_chunks"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
EMBED_MODEL    = "sentence-transformers/LaBSE"
EMBED_DIM      = 768

CHUNK_TOKENS    = 500
OVERLAP_TOKENS  = 100
MIN_CHUNK_CHARS = 80
MIN_DOC_CHARS   = 50

CONTENT_SIGNAL_WINDOW    = 300
EMBED_MIN_CONFIDENCE     = 0.65
LLM_CONFIDENCE_THRESHOLD = 0.55
LLM_CONTENT_EXCERPT      = 500
GEMINI_RATE_LIMIT        = 1.0

UPSERT_BATCH = 100   # ChromaDB batch size

logger.remove()
logger.add(
    lambda msg: print(msg, end=""), level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

# Faculty label for the economics faculty folder name
FACULTY_LABELS = {
    "feco": "Faculty of Economics, Business and Management Sciences",
    "farhat_abbas_university": "Farhat Abbas University Sétif 1",
}

_LABEL_MAP = {
    "Faculty": "Faculty", "Department": "Department", "Level": "Level",
    "Category": "Category", "Program": "Program",
    "Specialization": "Specialization", "Year": "Year", "General": "General",
}

# ─── Load aliases ─────────────────────────────────────────────
_ALIASES: Dict[str, str] = {}
try:
    with open(ALIASES_FILE, "r", encoding="utf-8") as f:
        _ALIASES = json.load(f)
    logger.info(f"Loaded {len(_ALIASES)} aliases")
except FileNotFoundError:
    logger.warning(f"Aliases file not found: {ALIASES_FILE}")

# ═══════════════════════════ REGEX ═══════════════════════════

_RE_CONTROL   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_MULTI_NL  = re.compile(r"\n{3,}")
_RE_SPACES    = re.compile(r"[ \t]+")
_RE_SENT_BOUND = re.compile(r"(?<=[.!?؟])\s+")
_RE_PARA_BREAK = re.compile(r"\n{2,}")
_RE_HEADING   = re.compile(
    r"^(?:#{1,4}\s+|(?:CHAPITRE|CHAPTER|SECTION|PARTIE|PART)\s+[\w\d]+"
    r"|(?:\d+\.){1,3}\s+\w|[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{4,}(?:\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]+)*$)",
    re.MULTILINE | re.UNICODE)
_RE_TABLE_MARKER = re.compile(r"(?m)^\|.+\|$")
_RE_LIST_ITEM    = re.compile(r"(?m)^[-•*▶]\s+\S")

_YEAR_RE = [
    (re.compile(r"\b(M[12]|L[123]|ING[1-5]|D)\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bmaster\s*([12])\b",   re.I), lambda m: f"M{m.group(1)}"),
    (re.compile(r"\blicence\s*([123])\b", re.I), lambda m: f"L{m.group(1)}"),
    (re.compile(r"\b1[eè]re?\s+ann[eé]e\s+master\b", re.I), lambda _: "M1"),
    (re.compile(r"\b2[eè]me?\s+ann[eé]e\s+master\b", re.I), lambda _: "M2"),
    (re.compile(r"\b1[eè]re?\s+ann[eé]e\b", re.I), lambda _: "L1"),
    (re.compile(r"\b2[eè]me?\s+ann[eé]e\b", re.I), lambda _: "L2"),
    (re.compile(r"\b3[eè]me?\s+ann[eé]e\b", re.I), lambda _: "L3"),
]

_SEMESTER_RE     = re.compile(r"\bS(\d{1,2})\b", re.I)
_LICENCE_SEMESTER = {"1": "L1", "2": "L1", "3": "L2", "4": "L2", "5": "L3", "6": "L3"}
_MASTER_SEMESTER  = {"1": "M1", "2": "M1", "3": "M2", "4": "M2"}
_ING_SEMESTER     = {
    "1": "ING1", "2": "ING1", "3": "ING2", "4": "ING2",
    "5": "ING3", "6": "ING3", "7": "ING4", "8": "ING4",
    "9": "ING5", "10": "ING5",
}

# ═══════════════════════════ HELPERS ═══════════════════════════

def norm(text: str) -> str:
    if not text: return ""
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub(r"[_\-'`´''«»\u2019\u2018]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def normalize_text(text: str) -> str:
    if not text: return ""
    t = _RE_CONTROL.sub("", unicodedata.normalize("NFC", text))
    t = _RE_MULTI_NL.sub("\n\n", t)
    return _RE_SPACES.sub(" ", t).strip()

def extract_raw_years(text: str) -> List[str]:
    found, seen = [], set()
    for pat, fn in _YEAR_RE:
        for m in pat.finditer(text):
            v = fn(m)
            if v and v not in seen:
                found.append(v); seen.add(v)
    return found

def extract_semesters(text: str) -> List[str]:
    return [m.group(1) for m in _SEMESTER_RE.finditer(text)]

def resolve_semester_to_year(sem_num: str, level_context: str) -> Optional[str]:
    if level_context == "ingenieur":
        return _ING_SEMESTER.get(sem_num)
    elif level_context == "master":
        return _MASTER_SEMESTER.get(sem_num)
    return _LICENCE_SEMESTER.get(sem_num)

def fuzzy_match(phrase: str, text: str) -> bool:
    if not phrase: return False
    idx = text.find(phrase)
    if idx >= 0:
        before = text[idx - 1] if idx > 0 else " "
        after  = text[idx + len(phrase)] if idx + len(phrase) < len(text) else " "
        if not (before.isalpha() or after.isalpha()):
            return True
    pwords = phrase.split(); twords = text.split()
    for i in range(len(twords) - len(pwords) + 1):
        match = True
        for j, pw in enumerate(pwords):
            tw = twords[i + j]
            if pw == tw: continue
            if pw + "s" == tw: continue
            if pw == tw + "s": continue
            if pw.rstrip("s") == tw.rstrip("s") and len(pw.rstrip("s")) >= 3: continue
            match = False; break
        if match: return True
    return False

def detect_level_context(text: str) -> str:
    tl = text.lower()
    if re.search(r"\b(ing[_\s]|ingenieur|ingénieur|ingénierie|ingenierie)\b", tl):
        return "ingenieur"
    if re.search(r"\b(master|m[12])\b", tl):
        return "master"
    return "licence"

def get_level_from_spec(spec_node) -> str:
    n = spec_node
    while n:
        if n.label == "Level":
            ln = n.name.lower()
            if "licence" in ln: return "licence"
            if "master"  in ln: return "master"
            if any(x in ln for x in ("ingénieur", "ingenieur")): return "ingenieur"
        n = n.parent
    n = spec_node
    while n:
        nn = n.name.lower()
        if "licence" in nn: return "licence"
        if "master"  in nn: return "master"
        if any(x in nn for x in ("ingénieur", "ingenieur")): return "ingenieur"
        n = n.parent
    return "licence"

# ═══════════════════════════ NODE ═══════════════════════════

@dataclass
class Node:
    name: str; label: str; depth: int
    parent: Optional["Node"] = field(default=None, repr=False)
    children: List["Node"]   = field(default_factory=list, repr=False)
    years: List[str]         = field(default_factory=list)
    _norm: str               = field(default="", repr=False)
    _aliases: List[str]      = field(default_factory=list)

    def __post_init__(self): self._norm = norm(self.name)

    @property
    def path_str(self) -> str:
        parts, n = [], self
        while n: parts.append(n.name); n = n.parent
        return " > ".join(reversed(parts))

    def descendants(self) -> List["Node"]:
        r = []
        for c in self.children: r.append(c); r.extend(c.descendants())
        return r

    def is_ancestor_of(self, o: "Node") -> bool:
        n = o.parent
        while n:
            if n.label == self.label and n.name == self.name: return True
            n = n.parent
        return False

    def department_root(self) -> Optional["Node"]:
        n = self
        while n.parent:
            if n.parent.label == "Department": return n.parent
            if n.parent.label == "Faculty":    return None
            n = n.parent
        return None

# ═══════════════════════════ ACADEMIC TREE ═══════════════════════════

class AcademicTree:
    """
    Loads the faculty structure from a JSON file.

    Expected JSON shape (same as original, just scoped to economics):
    {
      "faculty": {
        "name": "Faculty of Economics, Business and Management Sciences",
        "departments": [
          {
            "name": "Department of Economics",
            "levels": [
              {
                "name": "Licence",
                "programs": [{"name": "Economics", "years": ["L1","L2","L3"]}],
                "specializations": [...]
              },
              ...
            ]
          },
          ...
        ]
      }
    }
    """

    def __init__(self, path: str = STRUCTURE_FILE):
        self.all_nodes: List[Node] = []
        self.faculty_name = ""
        self._load(path)
        self.specifics = sorted(
            [n for n in self.all_nodes if n.depth >= 3 and n.label != "Year"],
            key=lambda n: (-n.depth, -len(n._norm)),
        )
        self.departments = [n for n in self.all_nodes if n.label == "Department"]
        self.levels      = [n for n in self.all_nodes if n.label == "Level"]
        logger.info(
            f"Tree: {len(self.all_nodes)} nodes | {len(self.specifics)} specs | "
            f"{len(self.departments)} depts | {len(self.levels)} levels"
        )

    def _load(self, path):
        with open(path, encoding="utf-8") as f: data = json.load(f)
        fac_data = data.get("faculty", data)
        self.faculty_name = fac_data.get("name", "Faculty of Economics")
        root = Node(self.faculty_name, "Faculty", 0)
        self.all_nodes.append(root)
        for d in fac_data.get("departments", []):
            self._dept(d, root)
        # attach aliases
        for spec in self.all_nodes:
            if spec.depth >= 3 and spec.label != "Year":
                spec_norm = spec._norm
                for alias_key, alias_value in _ALIASES.items():
                    if alias_value and norm(alias_value) == spec_norm:
                        alias_norm = norm(alias_key.replace("_", " "))
                        if alias_norm and alias_norm not in spec._aliases and alias_norm != spec_norm:
                            spec._aliases.append(alias_norm)

    def _dept(self, d, parent):
        n = Node(d["name"], "Department", 1, parent)
        parent.children.append(n); self.all_nodes.append(n)
        for lv in d.get("levels", []): self._level(lv, n)

    def _level(self, d, parent):
        n = Node(d["name"], "Level", 2, parent)
        parent.children.append(n); self.all_nodes.append(n)
        for p in d.get("programs", []):       self._spec(p, n, "Program")
        for s in d.get("specializations", []): self._spec(s, n, "Specialization")
        for c in d.get("categories", []):      self._cat(c, n)

    def _cat(self, d, parent):
        nm = d.get("type", d.get("name", ""))
        n = Node(nm, "Category", 3, parent)
        parent.children.append(n); self.all_nodes.append(n)
        for s in d.get("specializations", []): self._spec(s, n, "Specialization")

    def _spec(self, d, parent, label="Specialization"):
        depth = 4 if label == "Specialization" else 3
        n = Node(d["name"], label, depth, parent)
        n.years = d.get("years", [])
        abbrev = d.get("abbrev", "")
        if abbrev:
            abbrev_norm = norm(abbrev)
            if abbrev_norm and abbrev_norm != n._norm:
                n._aliases.append(abbrev_norm)
        parent.children.append(n); self.all_nodes.append(n)
        for yr in n.years:
            yn = Node(yr, "Year", 5, n)
            n.children.append(yn); self.all_nodes.append(yn)

    # ── search helpers (identical logic to original) ─────────

    def find_specifics(self, text_norm: str) -> List[Tuple[Node, int]]:
        results, seen = [], set()
        for spec in self.specifics:
            for alias in spec._aliases:
                if fuzzy_match(alias, text_norm):
                    key = f"{spec.label}|{spec.name}"
                    if key not in seen:
                        results.append((spec, len(alias.split()))); seen.add(key)
        for spec in self.specifics:
            nm = spec._norm; words = nm.split()
            if len(words) < 2: continue
            key = f"{spec.label}|{spec.name}"
            if key in seen: continue
            matched = False
            if nm and len(nm) >= 5 and fuzzy_match(nm, text_norm):
                results.append((spec, len(words))); seen.add(key); continue
            for size in range(len(words), 1, -1):
                prefix = " ".join(words[:size])
                if len(prefix) < 8: continue
                if prefix in ("licence", "master", "licence en", "master en", "ingenieur", "ingénieur"): continue
                if fuzzy_match(prefix, text_norm):
                    results.append((spec, size)); seen.add(key); matched = True; break
            if matched: continue
            for n_gram in (3, 2):
                for i in range(len(words) - n_gram + 1):
                    gram = " ".join(words[i:i + n_gram])
                    if len(gram) < 8: continue
                    if gram in ("licence en", "master en", "licence", "master", "ingenieur", "ingénieur"): continue
                    if fuzzy_match(gram, text_norm):
                        results.append((spec, n_gram + 1)); seen.add(key); matched = True; break
                if matched: break
            if matched: continue
            stripped = re.sub(
                r"^(licence\s+(en\s+)?|master\s+(en\s+)?|option\s+|tronc\s+commun\s+|"
                r"premiere\s+annee\s+)", "", nm
            ).strip()
            if stripped and stripped != nm:
                s_words = stripped.split()
                for size in range(len(s_words), 1, -1):
                    prefix = " ".join(s_words[:size])
                    if len(prefix) < 8: continue
                    if prefix in ("licence", "master"): continue
                    if fuzzy_match(prefix, text_norm):
                        results.append((spec, size)); seen.add(key); break
        results.sort(key=lambda x: (-x[1], len(x[0]._norm.split())))
        return results

    def find_departments(self, text_norm: str) -> List[Node]:
        return [d for d in self.departments if fuzzy_match(d._norm, text_norm)]

    def find_year_node(self, spec: Node, yr: str) -> Optional[Node]:
        for c in spec.children:
            if c.label == "Year" and c.name.upper() == yr.upper(): return c
        return None

# ═══════════════════════════ VECTOR DB CLIENT ═══════════════════════════

class VectorDBClient:
    """
    Wraps ChromaDB.  Stores:
      - A 'chunks' collection  →  semantic search over text chunks.
      - A 'urls' collection    →  metadata store for page/document URLs.

    Metadata stored per chunk:
        url_id, url, title, source_type,
        chunk_index, token_count, language,
        classification_label, classification_name,
        hierarchy_path, classification_id,
        match_method, confidence
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._chunks = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._urls = self._client.get_or_create_collection(
            name=f"{CHROMA_COLLECTION}_urls",
        )
        logger.info(
            f"ChromaDB ready at '{persist_dir}' | "
            f"chunks={self._chunks.count()} | urls={self._urls.count()}"
        )

    # ── URL metadata ─────────────────────────────────────────

    def upsert_url(
        self,
        url_id: str,
        url: str,
        title: str,
        source_type: str,
        target_label: str,
        target_id: str,
        hierarchy_path: str = "",
        method: str = "none",
        confidence: float = 0.0,
        parent_url_id: Optional[str] = None,
    ):
        meta = {
            "url": url,
            "title": title,
            "source_type": source_type,
            "target_label": target_label,
            "target_id": target_id,
            "hierarchy_path": hierarchy_path,
            "classification_method": method,
            "confidence": confidence,
            "parent_url_id": parent_url_id or "",
        }
        self._urls.upsert(
            ids=[url_id],
            documents=[f"{title} {url}"],
            metadatas=[meta],
        )

    def link_extra_targets(self, url_id: str, targets: List[Dict]):
        """
        Chromadb doesn't have native multi-edge relationships.
        We store extra targets as a JSON string in the URL metadata.
        """
        existing = self._urls.get(ids=[url_id])
        if not existing["ids"]: return
        current_meta = existing["metadatas"][0]
        extra = json.loads(current_meta.get("extra_targets", "[]"))
        for t in targets:
            extra.append({"label": t.get("label",""), "id": t.get("id",""), "name": t.get("name","")})
        current_meta["extra_targets"] = json.dumps(extra)
        self._urls.upsert(
            ids=[url_id],
            documents=existing["documents"],
            metadatas=[current_meta],
        )

    # ── Chunks ───────────────────────────────────────────────

    def upsert_chunks(
        self,
        url_id: str,
        url: str,
        title: str,
        chunks: List[Dict],
        classification: Dict,
    ):
        """
        Upserts all chunks for a document in batches.
        Each chunk carries full metadata so you can filter at query time.
        """
        lbl    = classification.get("label", "General")
        cid    = classification.get("id", "general")
        cname  = classification.get("name", "General")
        hp     = classification.get("hierarchy_path", "")
        method = classification.get("method", "none")
        conf   = float(classification.get("confidence", 0.0))

        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk in chunks:
            ids.append(chunk["id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"][:4000])
            metadatas.append({
                "url_id":              url_id,
                "url":                 url,
                "title":               title,
                "source_type":         chunk.get("source_type", "page"),
                "chunk_index":         chunk["chunk_index"],
                "token_count":         chunk.get("token_count", 0),
                "language":            chunk.get("language", "fr"),
                "section":             chunk.get("section", ""),
                "chunk_type":          chunk.get("chunk_type", "paragraph"),
                "classification_label": lbl,
                "classification_name": cname,
                "classification_id":   cid,
                "hierarchy_path":      hp,
                "match_method":        method,
                "confidence":          conf,
            })

        # Upsert in batches
        for i in range(0, len(ids), UPSERT_BATCH):
            self._chunks.upsert(
                ids=ids[i:i + UPSERT_BATCH],
                embeddings=embeddings[i:i + UPSERT_BATCH],
                documents=documents[i:i + UPSERT_BATCH],
                metadatas=metadatas[i:i + UPSERT_BATCH],
            )

    # ── Query helpers ────────────────────────────────────────

    def query(
        self,
        embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search.  Returns list of dicts with keys:
            id, text, metadata, distance
        Optionally filter by metadata field, e.g.:
            where={"classification_label": "Specialization"}
            where={"hierarchy_path": {"$contains": "Economics"}}
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where: kwargs["where"] = where

        res = self._chunks.query(**kwargs)
        results = []
        for doc_id, text, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0]
        ):
            results.append({"id": doc_id, "text": text, "metadata": meta, "distance": dist})
        return results

    def count(self) -> int:
        return self._chunks.count()

    def close(self):
        pass  # PersistentClient flushes automatically

# ═══════════════════════════ SMART CLASSIFIER ═══════════════════════════

class SmartClassifier:
    """
    Keyword/rule-based classifier (same logic as original,
    but returns plain dicts instead of Neo4j node IDs).
    The 'id' field is a stable slug derived from the node path.
    """

    def __init__(self, tree: AcademicTree):
        self.tree = tree

    # ── public ───────────────────────────────────────────────

    def classify(self, url="", title="", content="") -> Dict:
        url_norm     = self._norm_url(url)
        url_expanded = self._expand_abbrev(url_norm)
        title_norm   = norm(title)
        content_norm = norm(content[:300]) if content else ""

        url_raw_years  = list(dict.fromkeys(extract_raw_years(url_expanded)))
        url_semesters  = list(dict.fromkeys(extract_semesters(url_expanded)))

        combined_all  = " ".join(filter(None, [url, title, content[:300]]))
        all_raw_years = list(dict.fromkeys(extract_raw_years(combined_all)))
        all_semesters = list(dict.fromkeys(extract_semesters(combined_all)))

        context = detect_level_context(url_expanded + " " + title_norm)
        logger.info(f"   Context: {context} | Raw years: {all_raw_years} | Semesters: {all_semesters}")

        lvl_words = {"master", "licence", "license", "doctorat", "doctorate"}
        def rmlvl(t): return " ".join(w for w in (t or "").split() if w not in lvl_words)

        # Layer 1 – URL only
        r = self._try_match(url_expanded, url_raw_years, url_semesters, context, "url")
        if r: return r

        # Layer 2 – URL + title
        if url_expanded and title_norm:
            comb = " ".join(filter(None, [url_expanded, rmlvl(title_norm)]))
            r = self._try_match(comb, all_raw_years, all_semesters, context, "keyword")
            if r: return r

        # Layer 3 – URL + title + content snippet
        comb3 = " ".join(filter(None, [url_expanded, rmlvl(title_norm), rmlvl(content_norm)]))
        if comb3:
            r = self._try_match(comb3, all_raw_years, all_semesters, context, "keyword")
            if r: return r

        return self._make_general()

    # ── internals ────────────────────────────────────────────

    def _try_match(self, text, raw_years, semesters, context, method):
        # 1. Specific programmes / specializations
        sp = self.tree.find_specifics(text)
        if sp:
            w = self._filter_all(sp)
            if w:
                spec_context = get_level_from_spec(w[0][0])
                t = self._build_targets(w, raw_years, semesters, spec_context)
                if t: return self._make_result(t, method)

        # 2. Departments
        depts = self.tree.find_departments(text)
        if depts:
            t = self._build_targets_dept(depts, raw_years, semesters, context)
            if t: return self._make_result(t, method)

        # 3. Level keywords
        level_kw = {
            "doctorat": "Doctorat", "doctorate": "Doctorat",
            "master": "Master", "licence": "Licence", "license": "Licence",
        }
        found_levels = []
        for kw, ln in level_kw.items():
            if kw in text.lower() and ln not in found_levels:
                found_levels.append(ln)

        if found_levels:
            detected_depts = self.tree.find_departments(text)
            targets = self._try_match_levels(found_levels, detected_depts)
            if targets: return self._make_result(targets, method)

        # 4. Year/semester only
        y2l = {"L1": "Licence", "L2": "Licence", "L3": "Licence",
               "M1": "Master", "M2": "Master", "D": "Doctorat"}
        all_years = list(raw_years)
        for s in semesters:
            yr = resolve_semester_to_year(s, context)
            if yr and yr not in all_years: all_years.append(yr)

        if all_years:
            detected_depts = self.tree.find_departments(text)
            targets, seen = [], set()
            for yr in all_years:
                yr_nodes = self._find_year_nodes(yr, detected_depts)
                for ny in yr_nodes:
                    if ny["id"] not in seen:
                        seen.add(ny["id"]); targets.append(ny)
            if targets: return self._make_result(targets, method)

        return None

    def _try_match_levels(self, found_levels, detected_depts):
        targets, seen = [], set()
        for ln in found_levels:
            for lv in self.tree.levels:
                if norm(lv.name) == norm(ln):
                    if detected_depts:
                        dept_names = {d.name for d in detected_depts}
                        dept = lv.department_root()
                        if dept and dept.name not in dept_names: continue
                    nid = _slug(lv)
                    if nid not in seen:
                        seen.add(nid)
                        targets.append({"label": lv.label, "name": lv.name, "path": lv.path_str, "id": nid})
        return targets

    def _find_year_nodes(self, yr: str, detected_depts=None) -> List[Dict]:
        results, seen = [], set()
        for n in self.tree.all_nodes:
            if n.label == "Year" and n.name.upper() == yr.upper():
                if detected_depts:
                    dept = n.parent.department_root() if n.parent else None
                    if dept and dept.name not in {d.name for d in detected_depts}: continue
                nid = _slug(n)
                if nid not in seen:
                    seen.add(nid)
                    results.append({"label": "Year", "name": n.name, "path": n.path_str, "id": nid})
        return results

    def _make_result(self, targets, method):
        p = targets[0]
        ps = list(dict.fromkeys(x.get("path", "") for x in targets if x.get("path")))
        return {
            "targets": targets, "method": method, "confidence": 1.0,
            "hierarchy_path": " | ".join(ps[:3]),
            "label": p.get("label", "General"),
            "id":    p.get("id", "general"),
            "name":  p.get("name", "General"),
        }

    def _make_general(self):
        return {
            "targets": [{"label": "General", "id": "general", "name": "General", "path": "General"}],
            "method": "general", "confidence": 0.0,
            "hierarchy_path": "General", "label": "General", "id": "general", "name": "General",
        }

    def _norm_url(self, url):
        if not url: return ""
        path = re.sub(r"https?://[^/]+", "", url)
        path = re.sub(r"\.\w{2,5}(\?.*)?$", "", path)
        path = path.replace("_", " ").replace("-", " ").replace("/", " ")
        return re.sub(r"\s+", " ", norm(path)).strip()

    def _expand_abbrev(self, text: str) -> str:
        if not text or not _ALIASES: return text
        words = text.split(); result = []
        for w in words:
            exp = _ALIASES.get(w, w)
            if exp and exp != w: result.append(w); result.append(exp)
            else: result.append(w)
        return " ".join(result)

    # ── filter pipeline (unchanged from original) ─────────────

    def _filter_all(self, matches):
        matches = self._filter_dominated(matches)
        matches = self._filter_prefix_overlap(matches)
        matches = self._filter_subset_specs(matches)
        matches = self._filter_by_best_score(matches)
        return matches

    def _filter_by_best_score(self, matches):
        if len(matches) <= 1: return matches
        by_dept = defaultdict(list)
        for nd, wc in matches:
            dept = nd.department_root()
            by_dept[dept.name if dept else "__none__"].append((nd, wc))
        result = []
        for dept_name, dm in by_dept.items():
            if len(dm) <= 1: result.extend(dm); continue
            max_s = max(wc for _, wc in dm); min_s = min(wc for _, wc in dm)
            result.extend([(nd, wc) for nd, wc in dm if wc == max_s] if max_s - min_s >= 2 else dm)
        return result

    def _filter_dominated(self, matches):
        if len(matches) <= 1: return matches
        dom = set(); nodes = [nd for nd, _ in matches]
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i == j: continue
                if ni.is_ancestor_of(nj): dom.add((ni.label, ni.name))
        return [(nd, wc) for nd, wc in matches if (nd.label, nd.name) not in dom]

    def _filter_prefix_overlap(self, matches):
        if len(matches) <= 1: return matches
        keep = []
        for i, (nd_i, wc_i) in enumerate(matches):
            dominated = False
            for j, (nd_j, wc_j) in enumerate(matches):
                if i == j: continue
                di = nd_i.department_root(); dj = nd_j.department_root()
                if not di or not dj or di.name != dj.name: continue
                if wc_j > wc_i:
                    pi = " ".join(nd_i._norm.split()[:wc_i])
                    pj = " ".join(nd_j._norm.split()[:wc_j])
                    if pi and pj and pi in pj: dominated = True; break
            if not dominated: keep.append((nd_i, wc_i))
        return keep

    def _filter_subset_specs(self, matches):
        if len(matches) <= 1: return matches
        to_remove = set()
        for i, (nd_i, wc_i) in enumerate(matches):
            for j, (nd_j, wc_j) in enumerate(matches):
                if i == j: continue
                di = nd_i.department_root(); dj = nd_j.department_root()
                if not di or not dj or di.name != dj.name: continue
                wi = set(nd_i._norm.split()); wj = set(nd_j._norm.split())
                if wi and wj and wi.issubset(wj) and wc_j >= wc_i: to_remove.add(i)
        return [m for idx, m in enumerate(matches) if idx not in to_remove]

    # ── target builders ───────────────────────────────────────

    def _build_targets(self, winners, raw_years, semesters, spec_context):
        targets, seen = [], set()
        for nd, _ in winners:
            level_ctx = spec_context if spec_context else get_level_from_spec(nd)
            all_years = list(raw_years)
            for s in semesters:
                yr = resolve_semester_to_year(s, level_ctx)
                if yr and yr not in all_years: all_years.append(yr)
            if all_years:
                added = False
                for yr in all_years:
                    yn = self.tree.find_year_node(nd, yr)
                    if yn:
                        nid = _slug(yn)
                        if nid not in seen:
                            seen.add(nid)
                            targets.append({"label": "Year", "name": yn.name, "path": yn.path_str, "id": nid})
                            added = True
                if not added:
                    nid = _slug(nd)
                    if nid not in seen:
                        seen.add(nid)
                        targets.append({"label": nd.label, "name": nd.name, "path": nd.path_str, "id": nid})
            else:
                nid = _slug(nd)
                if nid not in seen:
                    seen.add(nid)
                    targets.append({"label": nd.label, "name": nd.name, "path": nd.path_str, "id": nid})
        return targets

    def _build_targets_dept(self, depts, raw_years, semesters, context):
        targets, seen = [], set()
        y2l = {"L1": "Licence", "L2": "Licence", "L3": "Licence",
               "M1": "Master", "M2": "Master", "D": "Doctorat"}
        all_years = list(raw_years)
        for s in semesters:
            yr = resolve_semester_to_year(s, context)
            if yr and yr not in all_years: all_years.append(yr)
        for dept in depts:
            if not all_years:
                nid = _slug(dept)
                if nid not in seen:
                    seen.add(nid)
                    targets.append({"label": dept.label, "name": dept.name, "path": dept.path_str, "id": nid})
                continue
            added = False
            for yr in all_years:
                for ny in self._find_year_nodes(yr, [dept]):
                    if ny["id"] not in seen: seen.add(ny["id"]); targets.append(ny); added = True
            if not added:
                for yr in all_years:
                    ln = y2l.get(yr.upper(), "")
                    if ln:
                        for child in dept.children:
                            if child.label == "Level" and norm(child.name) == norm(ln):
                                nid = _slug(child)
                                if nid not in seen:
                                    seen.add(nid)
                                    targets.append({"label": "Level", "name": child.name,
                                                    "path": child.path_str, "id": nid})
                                    added = True
                if not added:
                    nid = _slug(dept)
                    if nid not in seen:
                        seen.add(nid)
                        targets.append({"label": dept.label, "name": dept.name, "path": dept.path_str, "id": nid})
        if not targets:
            for dept in depts:
                nid = _slug(dept)
                if nid not in seen:
                    seen.add(nid)
                    targets.append({"label": dept.label, "name": dept.name, "path": dept.path_str, "id": nid})
        return targets


def _slug(node: Node) -> str:
    """Deterministic ID from the node's full path (no Neo4j needed)."""
    return hashlib.md5(node.path_str.encode()).hexdigest()[:16]

# ═══════════════════════════ GEMINI CLASSIFIER ═══════════════════════════

GEMINI_PROMPT = """You are a document classifier for Farhat Abbas University Sétif 1 — Faculty of Economics.
Output ONLY valid JSON:
{"target":{"label":"<Faculty|Department|Level|Category|Program|Specialization|Year>","name":"<name>","reason":"<reason>"},"match_method":"llm","confidence":<0.0-1.0>}
If no match: {"target":null,"match_method":"general","confidence":0.0}"""

class GeminiClassifier:
    def __init__(self, api_key=GEMINI_API_KEY):
        self._model = None; self._last_call = 0.0
        if _GEMINI_OK and api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=GEMINI_PROMPT,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=256),
            )

    @property
    def available(self): return self._model is not None

    def classify(self, url, title, content, candidates_text):
        if not self.available: return None
        elapsed = time.time() - self._last_call
        if elapsed < GEMINI_RATE_LIMIT: time.sleep(GEMINI_RATE_LIMIT - elapsed)
        self._last_call = time.time()
        try:
            resp = self._model.generate_content(
                f"{candidates_text}\n\n## DOCUMENT\nURL:{url}\nTitle:{title}\n"
                f"Content:{content[:LLM_CONTENT_EXCERPT]}\n\nClassify to DEEPEST valid node."
            )
            raw = resp.text or ""
            js = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            parsed = json.loads(js); tgt = parsed.get("target")
            if not tgt: return None
            name = tgt.get("name", "")
            nid = hashlib.md5(name.encode()).hexdigest()[:16]
            return {
                "targets": [{"label": tgt.get("label",""), "id": nid, "name": name, "path": ""}],
                "method": "llm", "confidence": float(parsed.get("confidence", 0.55)),
                "hierarchy_path": "", "label": tgt.get("label",""), "id": nid, "name": name,
            }
        except: return None

# ═══════════════════════════ EMBEDDING CLASSIFIER ═══════════════════════════

class EmbeddingClassifier:
    def __init__(self, model):
        self.model = model; self._index = None

    def build_index(self, tree: AcademicTree):
        nodes, sigs = [], []
        for n in tree.all_nodes:
            if n.label in ("Year", "Specialization", "Program", "Category", "Department"):
                sig = n.name
                if n.parent: sig += f" | {n.path_str}"
                nodes.append(n); sigs.append(sig)
        if not nodes: self._index = ([], np.array([])); return
        vecs = np.array(self.model.encode(sigs, normalize_embeddings=True, show_progress_bar=False, batch_size=64))
        self._index = (nodes, vecs)
        logger.info(f"Embedding index: {len(nodes)} nodes")

    def encode_chunks(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        ).tolist()

    def classify(self, text: str) -> Optional[Dict]:
        if not self._index: return None
        nodes, vecs = self._index
        if len(nodes) == 0: return None
        dv   = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = np.dot(vecs, dv); bi = int(np.argmax(sims)); bs = float(sims[bi])
        if bs < EMBED_MIN_CONFIDENCE: return None
        n = nodes[bi]; nid = _slug(n)
        return {
            "targets": [{"label": n.label, "name": n.name, "path": n.path_str, "id": nid}],
            "method": "embedding", "confidence": bs,
            "hierarchy_path": n.path_str, "label": n.label, "id": nid, "name": n.name,
        }

# ═══════════════════════════ CHUNKER ═══════════════════════════

class HierarchicalChunker:
    def __init__(self):
        self._enc = tiktoken.get_encoding("cl100k_base") if _TIKTOKEN_OK else None

    def split(self, text: str, title: str = "") -> List[Dict]:
        text = normalize_text(text)
        if not text: return []
        segs = self._segment(text)
        if not segs: return []
        chunks, cur, cur_tok, cur_sec, cur_type, idx = [], [], 0, title or "", "paragraph", 0
        for seg in segs:
            st = self._tok(seg["content"])
            if seg["type"] == "heading":
                if cur:
                    c = self._make(cur, idx, title, cur_sec, cur_type)
                    if len(c["clean_body"]) >= MIN_CHUNK_CHARS: chunks.append(c); idx += 1
                cur, cur_tok = [], 0; cur_sec = seg["content"]; cur_type = "section"; continue
            if st > CHUNK_TOKENS:
                if cur:
                    c = self._make(cur, idx, title, cur_sec, cur_type)
                    if len(c["clean_body"]) >= MIN_CHUNK_CHARS: chunks.append(c); idx += 1
                    cur, cur_tok = [], 0
                sub = self._split_long(seg["content"], title, cur_sec, idx)
                chunks.extend(sub); idx += len(sub); continue
            if cur_tok + st > CHUNK_TOKENS and cur:
                c = self._make(cur, idx, title, cur_sec, cur_type)
                if len(c["clean_body"]) >= MIN_CHUNK_CHARS: chunks.append(c); idx += 1
                over, ot = [], 0
                for prev in reversed(cur):
                    pt = self._tok(prev)
                    if ot + pt > OVERLAP_TOKENS: break
                    over.insert(0, prev); ot += pt
                cur = over + [seg["content"]]; cur_tok = ot + st
            else:
                cur.append(seg["content"]); cur_tok += st; cur_type = seg["type"]
        if cur:
            c = self._make(cur, idx, title, cur_sec, cur_type)
            if len(c["clean_body"]) >= MIN_CHUNK_CHARS: chunks.append(c)
        if not chunks and text.strip():
            chunks = [self._make([text.strip()], 0, title, title, "paragraph")]
        return chunks

    def _segment(self, text):
        segs = []
        for block in _RE_PARA_BREAK.split(text):
            block = block.strip()
            if not block: continue
            if _RE_HEADING.match(block):       segs.append({"type": "heading",   "content": block})
            elif _RE_TABLE_MARKER.search(block): segs.append({"type": "table",   "content": block})
            elif _RE_LIST_ITEM.search(block):   segs.append({"type": "list",     "content": block})
            else:                               segs.append({"type": "paragraph","content": block})
        return segs

    def _split_long(self, para, title, section, start):
        sents = [s.strip() for s in _RE_SENT_BOUND.split(para) if s.strip()] or [para]
        chunks, cur, tok, idx = [], [], 0, start
        for sent in sents:
            st = self._tok(sent)
            if cur and tok + st > CHUNK_TOKENS:
                c = self._make(cur, idx, title, section, "paragraph")
                if len(c["clean_body"]) >= MIN_CHUNK_CHARS: chunks.append(c); idx += 1
                cur, tok = [sent], st
            else:
                cur.append(sent); tok += st
        if cur:
            c = self._make(cur, idx, title, section, "paragraph")
            if len(c["clean_body"]) >= MIN_CHUNK_CHARS: chunks.append(c)
        return chunks

    def _make(self, parts, idx, title, section, ctype):
        body = " ".join(parts)
        et   = "\n".join(p for p in [title, section, body] if p)
        return {"embed_text": et, "text": et, "clean_body": body,
                "section": section, "chunk_type": ctype,
                "token_count": self._tok(body), "chunk_index": idx}

    def _tok(self, text):
        if self._enc: return len(self._enc.encode(text))
        return len(text) // 4

# ═══════════════════════════ PARSER ═══════════════════════════

def parse_json_doc(data: dict) -> dict:
    meta      = data.get("metadata", {})
    content   = data.get("content", {})
    resources = data.get("resources", {})
    ext_docs  = resources.get("documents", []) if isinstance(resources, dict) else []
    if "page" in meta:
        page = meta["page"]
        parts = [content.get("text", "")]
        for sec in content.get("sections", []):
            if isinstance(sec, dict): parts.extend([sec.get("text", ""), sec.get("title", "")])
        return dict(text="\n\n".join(filter(None, parts)),
                    title=page.get("title", ""), url=page.get("url", ""), ext_docs=ext_docs)
    return dict(text=content.get("text", ""), title="", url="", ext_docs=ext_docs)


def collect_json_files(root: Path) -> List[Tuple[Path, str, str]]:
    results = []
    for faculty_dir in sorted(root.iterdir()):
        if not faculty_dir.is_dir(): continue
        fl = FACULTY_LABELS.get(faculty_dir.name.lower(), faculty_dir.name.upper())
        # Only process economics faculty folder (feco)
        if faculty_dir.name.lower() not in ("feco", "farhat_abbas_university"):
            logger.info(f"Skipping non-economics folder: {faculty_dir.name}")
            continue
        for sub in ("pages", "extracted", "tables"):
            sfolder = faculty_dir / sub
            if not sfolder.exists(): continue
            for dirpath, _, filenames in os.walk(str(sfolder)):
                for fname in filenames:
                    if not fname.endswith(".json"): continue
                    jf  = Path(dirpath) / fname
                    rem = str(jf)[len(str(sfolder)) + 1:]
                    sp  = rem.find(os.sep)
                    dept = (rem[:sp] if sp != -1 else "General").replace("_"," ").replace("-"," ").title()
                    results.append((jf, fl, dept))
    return results

# ═══════════════════════════ PIPELINE ═══════════════════════════

class IngestionPipeline:
    def __init__(self):
        logger.info("Loading LaBSE...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        logger.info("LaBSE ready")

        self.tree       = AcademicTree(STRUCTURE_FILE)
        self.vectordb   = VectorDBClient(CHROMA_PERSIST_DIR)
        self.classifier = SmartClassifier(self.tree)
        self.gemini     = GeminiClassifier()
        self.embed_clf  = EmbeddingClassifier(self.embed_model)
        self.embed_clf.build_index(self.tree)
        self.chunker    = HierarchicalChunker()
        self._candidates_text = self._build_candidates_text()

    def _build_candidates_text(self):
        lines = ["## Specializations"]
        for n in self.tree.all_nodes:
            if n.label == "Specialization": lines.append(f"  name={n.name}")
        lines.append("\n## Programs")
        for n in self.tree.all_nodes:
            if n.label == "Program": lines.append(f"  name={n.name}")
        lines.append("\n## Departments")
        for n in self.tree.departments: lines.append(f"  name={n.name}")
        return "\n".join(lines)

    def run(self):
        root      = Path(ROOT_FOLDER)
        all_files = collect_json_files(root)
        page_files = [(jf, f, d) for jf, f, d in all_files if "/pages/" in str(jf).replace("\\", "/")]
        logger.info(f"📂 {len(page_files)} page files to process")
        ok = skip = fail = 0
        for jf, faculty, department in page_files:
            try:
                doc = self._process_page(jf, faculty, department)
                if doc is None: skip += 1; continue
                self._store(doc); ok += 1
                n_tgt = len(doc["classification"].get("targets", []))
                logger.info(
                    f"✅ {jf.name} [{doc['classification'].get('method','?')}] → {n_tgt} target(s)"
                )
            except Exception as exc:
                import traceback
                logger.error(f"❌ {jf.name}: {exc}\n{traceback.format_exc()}")
                fail += 1
        self.vectordb.close()
        total = self.vectordb.count() if ok > 0 else "?"
        logger.info(f"\nCOMPLETE: ✅ {ok} ⏭ {skip} ❌ {fail} | total chunks in DB: {total}")

    # ── per-document processing ───────────────────────────────

    def _process_page(self, jf, faculty, department):
        with open(jf, "r", encoding="utf-8") as fh: raw = json.load(fh)
        parsed = parse_json_doc(raw)
        text   = parsed.get("text", "")
        if len(text.strip()) < MIN_DOC_CHARS: return None
        title = parsed.get("title", "") or jf.stem
        url   = parsed.get("url", "")
        fp     = hashlib.md5(text.encode()).hexdigest()[:16]
        url_id = hashlib.md5(url.encode()).hexdigest()[:16] if url else f"url_{fp}"

        logger.info(f"\n{'─'*60}\n📄 {title[:60]}\n   URL: {url[:80]}")

        # ── Classification cascade ────────────────────────────
        classification = self.classifier.classify(url=url, title=title, content=text)
        logger.info(f"   Tree: {classification.get('method')} → {len(classification.get('targets',[]))} targets")

        if not classification.get("targets") or classification.get("method") == "general":
            if self.gemini.available:
                logger.info("   Trying Gemini...")
                gr = self.gemini.classify(url, title, text, self._candidates_text)
                if gr and gr.get("targets"):
                    classification = gr
                    logger.info(f"   Gemini: {len(classification.get('targets',[]))} targets")

        if not classification.get("targets") or classification.get("method") == "general":
            logger.info("   Trying Embedding...")
            er = self.embed_clf.classify(text[:500])
            if er and er.get("targets"):
                classification = er
                logger.info(f"   Embedding: {len(classification.get('targets',[]))} targets")

        if not classification.get("targets"):
            classification = self.classifier._make_general()

        for t in classification.get("targets", [])[:5]:
            logger.info(f"     🎯 [{t.get('label','?')}] {t.get('name','?')} | path: {t.get('path','?')}")

        # ── Chunk + embed ─────────────────────────────────────
        raw_chunks  = self.chunker.split(text, title=title)
        chunk_texts = [c["embed_text"] for c in raw_chunks]
        embeddings  = self.embed_clf.encode_chunks(chunk_texts)

        chunks = []
        for i, (cd, emb) in enumerate(zip(raw_chunks, embeddings)):
            chunks.append({
                "id":          f"{fp}_c{i}",
                "text":        cd["embed_text"],
                "chunk_index": i,
                "token_count": cd.get("token_count", 0),
                "language":    "fr",
                "section":     cd.get("section", ""),
                "chunk_type":  cd.get("chunk_type", "paragraph"),
                "embedding":   emb,
                "source_type": "page",
            })
        logger.info(f"   Chunks: {len(chunks)}")

        # ── PDF attachments ───────────────────────────────────
        pdf_docs, child_urls = [], []
        for ext_doc in parsed.get("ext_docs", []):
            pdf_url = ext_doc.get("url", "")
            if not pdf_url: continue
            pdf_title      = ext_doc.get("title", "") or f"PDF from {title}"
            local_f        = ext_doc.get("local_file", "")
            child_url_id   = hashlib.md5(pdf_url.encode()).hexdigest()[:16]
            child_urls.append({"url_id": child_url_id, "url": pdf_url,
                                "title": pdf_title, "source_type": "pdf"})
            child_text = ""
            if local_f:
                local_name     = Path(local_f).name
                extracted_path = jf.parent.parent / "extracted" / local_name.replace(".pdf", ".json")
                if extracted_path.exists():
                    with open(extracted_path, "r", encoding="utf-8") as fh:
                        ext_data = json.load(fh)
                    child_text = ext_data.get("content", {}).get("text", "") or " ".join(
                        p.get("text", "") for p in ext_data.get("content", {}).get("pages", []))
                elif Path(local_f).exists() and _PYPDF_OK:
                    try:
                        reader     = pypdf.PdfReader(str(Path(local_f)))
                        child_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                    except: pass
            if child_text and len(child_text.strip()) >= MIN_DOC_CHARS:
                child_chunks = self.chunker.split(child_text, title=pdf_title)
                child_embs   = self.embed_clf.encode_chunks([c["embed_text"] for c in child_chunks])
                child_fp     = hashlib.md5(child_text.encode()).hexdigest()[:16]
                child_chunk_objs = []
                for i, (cd, emb) in enumerate(zip(child_chunks, child_embs)):
                    child_chunk_objs.append({
                        "id":          f"{child_fp}_c{i}",
                        "text":        cd["embed_text"],
                        "chunk_index": i,
                        "token_count": cd.get("token_count", 0),
                        "language":    "fr",
                        "section":     cd.get("section", ""),
                        "chunk_type":  cd.get("chunk_type", "paragraph"),
                        "embedding":   emb,
                        "source_type": "pdf",
                    })
                pdf_docs.append({
                    "url_id":         child_url_id,
                    "url":            pdf_url,
                    "title":          pdf_title,
                    "chunks":         child_chunk_objs,
                    "classification": classification,
                })
                logger.info(f"   PDF chunks: {len(child_chunk_objs)}")

        return {
            "url_id":         url_id,
            "url":            url,
            "title":          title,
            "source_type":    "page",
            "classification": classification,
            "chunks":         chunks,
            "pdf_docs":       pdf_docs,
            "child_urls":     child_urls,
        }

    def _store(self, doc):
        cls     = doc["classification"]
        targets = cls.get("targets", [])
        if not targets:
            targets = [{"label": "General", "id": "general", "name": "General", "path": "General"}]
        primary = targets[0]

        # ── Store URL metadata ────────────────────────────────
        self.vectordb.upsert_url(
            doc["url_id"], doc["url"], doc["title"], doc["source_type"],
            primary.get("label", "General"), primary.get("id", "general"),
            cls.get("hierarchy_path", ""), cls.get("method", "none"),
            float(cls.get("confidence", 0.0)),
        )
        if len(targets) > 1:
            self.vectordb.link_extra_targets(doc["url_id"], targets[1:])

        # ── Store page chunks ─────────────────────────────────
        self.vectordb.upsert_chunks(
            doc["url_id"], doc["url"], doc["title"],
            doc["chunks"], cls,
        )

        # ── Store child URL stubs ─────────────────────────────
        for child in doc.get("child_urls", []):
            self.vectordb.upsert_url(
                child["url_id"], child["url"], child["title"], child["source_type"],
                "General", "general", "", "inherited", 0.0,
                parent_url_id=doc["url_id"],
            )

        # ── Store PDF chunks ──────────────────────────────────
        for pdf in doc.get("pdf_docs", []):
            self.vectordb.upsert_chunks(
                pdf["url_id"], pdf["url"], pdf["title"],
                pdf["chunks"], pdf["classification"],
            )

# ═══════════════════════════ QUERY HELPER ═══════════════════════════

def query_economics(
    question: str,
    embed_model: SentenceTransformer,
    vectordb: VectorDBClient,
    n_results: int = 10,
    filter_label: Optional[str] = None,
    filter_hierarchy: Optional[str] = None,
) -> List[Dict]:
    """
    Convenience function: embed a question and retrieve the most relevant chunks.

    Example usage:
        model = SentenceTransformer("sentence-transformers/LaBSE")
        db    = VectorDBClient()
        hits  = query_economics("What is macroeconomics?", model, db)
        for h in hits:
            print(h["metadata"]["title"], h["distance"])
            print(h["text"][:200])
    """
    embedding = embed_model.encode([question], normalize_embeddings=True)[0].tolist()
    where = None
    if filter_label:
        where = {"classification_label": filter_label}
    elif filter_hierarchy:
        where = {"hierarchy_path": {"$contains": filter_hierarchy}}
    return vectordb.query(embedding, n_results=n_results, where=where)

# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    IngestionPipeline().run()

if __name__ == "__main__":
    main()