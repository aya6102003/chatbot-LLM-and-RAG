from __future__ import annotations

import hashlib, json, os, re, time, unicodedata, warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

try: import tiktoken; _TIKTOKEN_OK = True
except ImportError: _TIKTOKEN_OK = False

try: import google.generativeai as genai; _GEMINI_OK = True
except ImportError: _GEMINI_OK = False

try: import pypdf; _PYPDF_OK = True
except ImportError: _PYPDF_OK = False

# ═══════════════════════════ CONFIG ═══════════════════════════

ROOT_FOLDER    = "./university_farhat_abaas"
STRUCTURE_FILE = "./structure_sciences.json"
ALIASES_FILE   = "./aliases.json"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
EMBED_MODEL    = "gemini-embedding-001"
EMBED_DIM      = 3072

NEO4J_BATCH    = 50
NEO4J_VECTOR_INDEX = "chunk_embedding"
CHUNK_TOKENS   = 500
OVERLAP_TOKENS = 100
MIN_CHUNK_CHARS = 80
MIN_DOC_CHARS   = 50

CONTENT_SIGNAL_WINDOW = 300
EMBED_MIN_CONFIDENCE   = 0.65
LLM_CONFIDENCE_THRESHOLD = 0.55
LLM_CONTENT_EXCERPT     = 500
GEMINI_RATE_LIMIT       = 1.0

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", colorize=True)

FACULTY_LABELS = {
    "farhat_abbas_university": "Farhat Abbas University Sétif 1",
    "ftechnologie": "Faculty of Technology", "fsciences": "Faculty of Science",
    "fsnv": "Faculty of Nature and Life Sciences",
    "feco": "Faculty of Economics, Business and Management Sciences",
    "fmed": "Faculty of Medicine",
    "fsciences": "Faculty of Sciences",
}

_LABEL_MAP = {
    "Faculty":"Faculty","Department":"Department","Level":"Level",
    "Category":"Category","Program":"Program","Specialization":"Specialization",
    "Year":"Year","General":"General",
}

_HIERARCHY_RELS = "HAS_DEPARTMENT|HAS_LEVEL|HAS_PROGRAM|HAS_CATEGORY|HAS_SPECIALIZATION|HAS_YEAR"

_ALIASES: Dict[str, str] = {}
try:
    with open(ALIASES_FILE, "r", encoding="utf-8") as f:
        _ALIASES = json.load(f)
    logger.info(f"Loaded {len(_ALIASES)} aliases")
except FileNotFoundError:
    logger.warning(f"Aliases file not found: {ALIASES_FILE}")

# ═══════════════════════════ REGEX ═══════════════════════════

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_SPACES = re.compile(r"[ \t]+")
_RE_SENT_BOUND = re.compile(r"(?<=[.!?؟])\s+")
_RE_PARA_BREAK = re.compile(r"\n{2,}")
_RE_HEADING = re.compile(
    r"^(?:#{1,4}\s+|(?:CHAPITRE|CHAPTER|SECTION|PARTIE|PART)\s+[\w\d]+"
    r"|(?:\d+\.){1,3}\s+\w|[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{4,}(?:\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]+)*$)",
    re.MULTILINE|re.UNICODE)
_RE_TABLE_MARKER = re.compile(r"(?m)^\|.+\|$")
_RE_LIST_ITEM = re.compile(r"(?m)^[-•*▶]\s+\S")

_YEAR_RE = [
    (re.compile(r"\b(M[12]|L[123]|ING[1-5]|D)\b",re.I), lambda m:m.group(0).upper()),
    (re.compile(r"\bmaster\s*([12])\b",re.I), lambda m:f"M{m.group(1)}"),
    (re.compile(r"\blicence\s*([123])\b",re.I), lambda m:f"L{m.group(1)}"),
    (re.compile(r"\b1[eè]re?\s+ann[eé]e\s+master\b",re.I), lambda _:"M1"),
    (re.compile(r"\b2[eè]me?\s+ann[eé]e\s+master\b",re.I), lambda _:"M2"),
    (re.compile(r"\b1[eè]re?\s+ann[eé]e\b",re.I), lambda _:"L1"),
    (re.compile(r"\b2[eè]me?\s+ann[eé]e\b",re.I), lambda _:"L2"),
    (re.compile(r"\b3[eè]me?\s+ann[eé]e\b",re.I), lambda _:"L3"),
]

_SEMESTER_RE = re.compile(r"\bS(\d{1,2})\b", re.I)

_LICENCE_SEMESTER = {"1":"L1","2":"L1","3":"L2","4":"L2","5":"L3","6":"L3"}
_MASTER_SEMESTER  = {"1":"M1","2":"M1","3":"M2","4":"M2"}
_ING_SEMESTER     = {"1":"ING1","2":"ING1","3":"ING2","4":"ING2","5":"ING3","6":"ING3",
                     "7":"ING4","8":"ING4","9":"ING5","10":"ING5"}

# ═══════════════════════════ HELPERS ═══════════════════════════

def norm(text: str) -> str:
    if not text: return ""
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub(r"[_\-'`´''«»\u2019\u2018]"," ", t)
    return re.sub(r"\s+"," ", t).strip()

def normalize_text(text: str) -> str:
    if not text: return ""
    t = _RE_CONTROL.sub("", unicodedata.normalize("NFC", text))
    t = _RE_MULTI_NL.sub("\n\n", t); return _RE_SPACES.sub(" ", t).strip()

def extract_raw_years(text: str) -> List[str]:
    found, seen = [], set()
    for pat, fn in _YEAR_RE:
        for m in pat.finditer(text):
            v = fn(m)
            if v and v not in seen: found.append(v); seen.add(v)
    return found

def extract_semesters(text: str) -> List[str]:
    return [m.group(1) for m in _SEMESTER_RE.finditer(text)]

def resolve_semester_to_year(sem_num: str, level_context: str) -> Optional[str]:
    if level_context == 'ingenieur':
        return _ING_SEMESTER.get(sem_num)
    elif level_context == 'master':
        return _MASTER_SEMESTER.get(sem_num)
    else:
        return _LICENCE_SEMESTER.get(sem_num)

def fuzzy_match(phrase: str, text: str) -> bool:
    if not phrase: return False
    idx = text.find(phrase)
    if idx >= 0:
        before = text[idx-1] if idx>0 else " "
        after = text[idx+len(phrase)] if idx+len(phrase)<len(text) else " "
        if not (before.isalpha() or after.isalpha()): return True
    pwords = phrase.split(); twords = text.split()
    for i in range(len(twords)-len(pwords)+1):
        match = True
        for j, pw in enumerate(pwords):
            tw = twords[i+j]
            if pw == tw: continue
            if pw + "s" == tw: continue
            if pw == tw + "s": continue
            if pw.rstrip('s') == tw.rstrip('s') and len(pw.rstrip('s')) >= 3: continue
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
            if "master" in ln: return "master"
            if any(x in ln for x in ("ingénieur", "ingenieur")): return "ingenieur"
        n = n.parent
    n = spec_node
    while n:
        nn = n.name.lower()
        if "licence" in nn: return "licence"
        if "master" in nn: return "master"
        if any(x in nn for x in ("ingénieur", "ingenieur")): return "ingenieur"
        n = n.parent
    return "licence"

# ═══════════════════════════ NODE ═══════════════════════════

@dataclass
class Node:
    name: str; label: str; depth: int
    parent: Optional["Node"] = field(default=None, repr=False)
    children: List["Node"] = field(default_factory=list, repr=False)
    years: List[str] = field(default_factory=list)
    _norm: str = field(default="", repr=False)
    _aliases: List[str] = field(default_factory=list)
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
            if n.label==self.label and n.name==self.name: return True
            n = n.parent
        return False

    def department_root(self) -> Optional["Node"]:
        n = self
        while n.parent:
            if n.parent.label == "Department": return n.parent
            if n.parent.label == "Faculty": return None
            n = n.parent
        return None

# ═══════════════════════════ ACADEMIC TREE ═══════════════════════════

class AcademicTree:
    def __init__(self, path: str = STRUCTURE_FILE):
        self.all_nodes: List[Node] = []
        self.faculty_name = ""
        self._load(path)
        self.specifics = sorted(
            [n for n in self.all_nodes if n.depth >= 3 and n.label != "Year"],
            key=lambda n: (-n.depth, -len(n._norm)))
        self.departments = [n for n in self.all_nodes if n.label == "Department"]
        self.levels = [n for n in self.all_nodes if n.label == "Level"]
        logger.info(f"Tree: {len(self.all_nodes)} nodes | {len(self.specifics)} specs | {len(self.departments)} depts | {len(self.levels)} levels")

    def _load(self, path):
        with open(path, encoding="utf-8") as f: data = json.load(f)
        fac_data = data.get("faculty", data)
        self.faculty_name = fac_data.get("name","Faculty of Sciences")
        root = Node(self.faculty_name,"Faculty",0)
        self.all_nodes.append(root)
        for d in fac_data.get("departments",[]): self._dept(d, root)
        for spec in self.all_nodes:
            if spec.depth >= 3 and spec.label != "Year":
                spec_norm = spec._norm
                for alias_key, alias_value in _ALIASES.items():
                    if alias_value and norm(alias_value) == spec_norm:
                        alias_norm = norm(alias_key.replace("_", " "))
                        if alias_norm and alias_norm not in spec._aliases and alias_norm != spec_norm:
                            spec._aliases.append(alias_norm)

    def _dept(self, d, parent):
        n = Node(d["name"],"Department",1,parent)
        parent.children.append(n); self.all_nodes.append(n)
        for lv in d.get("levels",[]): self._level(lv, n)

    def _level(self, d, parent):
        n = Node(d["name"],"Level",2,parent)
        parent.children.append(n); self.all_nodes.append(n)
        for p in d.get("programs",[]): self._spec(p,n,"Program")
        for s in d.get("specializations",[]): self._spec(s,n,"Specialization")
        for c in d.get("categories",[]): self._cat(c,n)

    def _cat(self, d, parent):
        nm = d.get("type", d.get("name",""))
        n = Node(nm,"Category",3,parent)
        parent.children.append(n); self.all_nodes.append(n)
        for s in d.get("specializations",[]): self._spec(s,n,"Specialization")

    def _spec(self, d, parent, label="Specialization"):
        depth = 4 if label=="Specialization" else 3
        n = Node(d["name"],label,depth,parent)
        n.years = d.get("years",[])
        abbrev = d.get("abbrev", "")
        if abbrev:
            abbrev_norm = norm(abbrev)
            if abbrev_norm and abbrev_norm != n._norm:
                n._aliases.append(abbrev_norm)
        parent.children.append(n); self.all_nodes.append(n)
        for yr in n.years:
            yn = Node(yr,"Year",5,n)
            n.children.append(yn); self.all_nodes.append(yn)

    def find_specifics(self, text_norm: str) -> List[Tuple[Node, int]]:
        results, seen = [], set()
        for spec in self.specifics:
            for alias in spec._aliases:
                if fuzzy_match(alias, text_norm):
                    key = f"{spec.label}|{spec.name}"
                    if key not in seen: results.append((spec, len(alias.split()))); seen.add(key)
        for spec in self.specifics:
            nm = spec._norm; words = nm.split()
            if len(words) < 2: continue
            key = f"{spec.label}|{spec.name}"
            if key in seen: continue
            matched = False
            if nm and len(nm)>=5 and fuzzy_match(nm, text_norm):
                results.append((spec,len(words))); seen.add(key); continue
            for size in range(len(words),1,-1):
                prefix = " ".join(words[:size])
                if len(prefix)<8: continue
                if prefix in ("licence","master","licence en","master en","ingenieur","ingénieur"): continue
                if fuzzy_match(prefix, text_norm): results.append((spec,size)); seen.add(key); matched=True; break
            if matched: continue
            for n_gram in (3,2):
                for i in range(len(words)-n_gram+1):
                    gram = " ".join(words[i:i+n_gram])
                    if len(gram)<8: continue
                    if gram in ("licence en","master en","licence","master","ingenieur","ingénieur"): continue
                    if fuzzy_match(gram, text_norm): results.append((spec,n_gram+1)); seen.add(key); matched=True; break
                if matched: break
            if matched: continue
            stripped = re.sub(r"^(licence\s+(en\s+)?|master\s+(en\s+)?|option\s+|tronc\s+commun\s+|"
                r"premiere\s+annee\s+|ingenieur\s+informatique\s+|ingénieur\s+informatique\s+)","",nm).strip()
            if stripped and stripped!=nm:
                s_words = stripped.split()
                for size in range(len(s_words),1,-1):
                    prefix = " ".join(s_words[:size])
                    if len(prefix)<8: continue
                    if prefix in ("licence","master","ingenieur","ingénieur"): continue
                    if fuzzy_match(prefix, text_norm): results.append((spec,size)); seen.add(key); break
        results.sort(key=lambda x: (-x[1], len(x[0]._norm.split())))
        return results

    def find_departments(self, text_norm: str) -> List[Node]:
        return [d for d in self.departments if fuzzy_match(d._norm, text_norm)]

    def find_year_node(self, spec: Node, yr: str) -> Optional[Node]:
        for c in spec.children:
            if c.label=="Year" and c.name.upper()==yr.upper(): return c
        return None

# ═══════════════════════════ NEO4J CLIENT ═══════════════════════════

class Neo4jClient:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._id_cache: Dict[Tuple[str,str], str] = {}
        self._init()

    def close(self): self.driver.close()

    def _init(self):
        with self.driver.session() as s:
            for lbl in ("URL","Chunk"):
                try: s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.id IS UNIQUE")
                except Exception as e: logger.warning(f"Constraint {lbl}: {e}")
            
            try:
                logger.info(f"Creating vector index {NEO4J_VECTOR_INDEX}...")
                s.run(f"""
                    CREATE VECTOR INDEX {NEO4J_VECTOR_INDEX} IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}
                """)
                logger.info(f"✓ Vector index created/verified")
            except Exception as e:
                logger.error(f"Vector index creation failed: {e}")

    def get_node_id(self, label: str, name: str) -> Optional[str]:
        k = (label,name)
        if k in self._id_cache: return self._id_cache[k]
        actual = _LABEL_MAP.get(label,label)
        with self.driver.session() as s:
            r = s.run(f"MATCH (n:{actual}) WHERE n.name=$name RETURN n.id AS id LIMIT 1", name=name).single()
            if r: self._id_cache[k]=r["id"]; return r["id"]
        return None

    def get_node_id_by_parent(self, label: str, name: str, parent_id: str) -> Optional[str]:
        k = (label,name,parent_id)
        if k in self._id_cache: return self._id_cache[k]
        actual = _LABEL_MAP.get(label,label)
        with self.driver.session() as s:
            r = s.run(f"""MATCH (p {{id:$pid}})-[:HAS_SPECIALIZATION|HAS_PROGRAM|HAS_CATEGORY|HAS_YEAR]->(n:{actual})
                WHERE n.name=$name RETURN n.id AS id LIMIT 1""", pid=parent_id, name=name).single()
            if r: self._id_cache[k]=r["id"]; return r["id"]
        return self.get_node_id(label,name)

    def resolve_node_id(self, node: Node) -> Optional[str]:
        k = (node.label,node.name)
        if k in self._id_cache: return self._id_cache[k]
        if node.label=="Year" and node.parent:
            pid = self.resolve_node_id(node.parent)
            if pid:
                nid = self.get_node_id_by_parent("Year",node.name,pid)
                if nid: self._id_cache[k]=nid; return nid
        nid = self.get_node_id(node.label,node.name)
        if nid: self._id_cache[k]=nid; return nid
        if node.parent and node.label in ("Specialization","Program"):
            pid = self.resolve_node_id(node.parent)
            if pid:
                nid = self.get_node_id_by_parent(node.label,node.name,pid)
                if nid: self._id_cache[k]=nid; return nid
        return None

    def get_general_id(self) -> str:
        nid = self.get_node_id("General","general"); return nid or "general"

    def get_hierarchy_path(self, label: str, node_id: str) -> str:
        actual = _LABEL_MAP.get(label,label)
        with self.driver.session() as s:
            r = s.run(f"""
                MATCH path = (root)-[:{_HIERARCHY_RELS}*0..8]->(n {{id:$id}})
                WHERE $lbl IN labels(n)
                WITH path ORDER BY length(path) DESC LIMIT 1
                RETURN [node IN nodes(path) | node.name] AS names
            """, id=node_id, lbl=actual).single()
            if r and r["names"]: return " > ".join(n for n in r["names"] if n)
        return label

    def find_years_under_dept(self, dept_name: str, year_name: str) -> List[Dict]:
        with self.driver.session() as s:
            rows = s.run(f"""
                MATCH (d:Department {{name:$dept}})-[:{_HIERARCHY_RELS}*1..6]->(y:Year)
                WHERE y.name = $year
                OPTIONAL MATCH (spec)-[:HAS_YEAR]->(y)
                RETURN y.id AS id, y.name AS name, spec.name AS spec_name
            """, dept=dept_name, year=year_name.upper())
            results = []
            for r in rows:
                hp = self.get_hierarchy_path("Year", r["id"])
                results.append({"label":"Year","name":r["name"],"path":hp or "", "id":r["id"]})
            return results

    def find_all_years_by_name(self, year_name: str, detected_depts: List = None) -> List[Dict]:
        with self.driver.session() as s:
            if detected_depts:
                dept_names = [d.name for d in detected_depts]
                rows = s.run(f"""
                    MATCH (d:Department)-[:{_HIERARCHY_RELS}*1..6]->(y:Year)
                    WHERE y.name = $year AND d.name IN $depts
                    RETURN y.id AS id, y.name AS name
                """, year=year_name.upper(), depts=dept_names)
            else:
                rows = s.run(f"""
                    MATCH (d:Department)-[:{_HIERARCHY_RELS}*1..6]->(y:Year)
                    WHERE y.name = $year
                    RETURN y.id AS id, y.name AS name
                """, year=year_name.upper())
            results = []
            for r in rows:
                hp = self.get_hierarchy_path("Year", r["id"])
                results.append({"label": "Year", "name": r["name"], "path": hp or "", "id": r["id"]})
            return results

    def find_all_levels_by_name(self, level_name: str) -> List[Dict]:
        with self.driver.session() as s:
            rows = s.run("""
                MATCH (l:Level {name: $name})
                OPTIONAL MATCH (d:Department)-[:HAS_LEVEL]->(l)
                RETURN l.id AS id, l.name AS name, d.name AS dept_name
            """, name=level_name)
            results = []
            for r in rows:
                hp = self.get_hierarchy_path("Level", r["id"])
                results.append({"label": "Level", "name": r["name"], "path": hp or "",
                                 "id": r["id"], "dept_name": r["dept_name"]})
            return results

    def upsert_url(self, url_id, url, title, source_type, target_label, target_id,
                   hierarchy_path="", method="none", confidence=0.0, parent_url_id=None):
        """
        CRITICAL FIX: Uses CLASSIFIED_AS (not HAS_CONTENT) for RAG GraphTraversal compatibility.
        The RAG pipeline's GraphTraversal class queries:
          MATCH (u:URL)-[:CLASSIFIED_AS]->(n)
        """
        actual = _LABEL_MAP.get(target_label, target_label)
        with self.driver.session() as s:
            s.run("""MERGE (u:URL {id:$id}) SET u.url=$url, u.title=$title, u.source_type=$st,
                u.hierarchy_path=$hp, u.classification_method=$cm, u.confidence=$conf""",
                id=url_id, url=url, title=title, st=source_type, hp=hierarchy_path, cm=method, conf=confidence)
            
            if parent_url_id is None:
                # FIXED: CLASSIFIED_AS instead of HAS_CONTENT
                s.run(f"""
                    MATCH (n:{actual} {{id:$nid}})
                    MATCH (u:URL {{id:$uid}})
                    MERGE (u)-[:CLASSIFIED_AS]->(n)
                """, nid=target_id, uid=url_id)
            else:
                s.run("""MATCH (p:URL {id:$pid}) MATCH (f:URL {id:$fid}) MERGE (p)-[:HAS_FILE]->(f)""",
                    pid=parent_url_id, fid=url_id)

    def link_extra_targets(self, url_id, targets):
        """
        FIXED: CLASSIFIED_AS for extra targets too
        """
        with self.driver.session() as s:
            for t in targets:
                actual = _LABEL_MAP.get(t["label"], t["label"])
                s.run(f"""
                    MATCH (n:{actual} {{id:$nid}})
                    MATCH (u:URL {{id:$uid}})
                    MERGE (u)-[:CLASSIFIED_AS]->(n)
                """, nid=t["id"], uid=url_id)

    def create_chunks(self, url_id, chunks, classification):
        lbl = classification.get("label","General"); nid = classification.get("id","general")
        hp = classification.get("hierarchy_path",""); method = classification.get("method","none")
        conf = float(classification.get("confidence",0.0))
        with self.driver.session() as s:
            for i in range(0, len(chunks), NEO4J_BATCH):
                batch = chunks[i:i+NEO4J_BATCH]
                s.run("""UNWIND $batch AS ch MERGE (c:Chunk {id:ch.id})
                    SET c.text=ch.text, c.chunk_index=ch.chunk_index, c.token_count=ch.token_count,
                    c.language=ch.language, c.classification_id=ch.cid, c.classification_label=ch.clabel,
                    c.hierarchy_path=ch.hp, c.match_method=ch.method, c.confidence=ch.conf, c.embedding=ch.emb
                    WITH c,ch MATCH (u:URL {id:$uid}) MERGE (u)-[:HAS_CHUNK {order:ch.chunk_index}]->(c)""",
                    batch=[{"id":c["id"],"text":c["text"][:4000],"chunk_index":c["chunk_index"],
                        "token_count":c.get("token_count",0),"language":c.get("language",""),
                        "cid":nid,"clabel":lbl,"hp":hp,"method":method,"conf":conf,"emb":c.get("embedding",[])}
                        for c in batch], uid=url_id)
            if len(chunks)>1:
                s.run("""MATCH (u:URL {id:$uid})-[:HAS_CHUNK]->(c:Chunk) WITH c ORDER BY c.chunk_index
                    WITH collect(c) AS o UNWIND range(0,size(o)-2) AS i
                    WITH o[i] AS cur, o[i+1] AS nxt MERGE (cur)-[:NEXT_CHUNK]->(nxt)""", uid=url_id)

# ═══════════════════════════ SMART CLASSIFIER ═══════════════════════════

class SmartClassifier:
    def __init__(self, tree: AcademicTree, neo4j: Neo4jClient):
        self.tree = tree; self.neo4j = neo4j

    def classify(self, url="", title="", content="") -> Dict:
        url_norm     = self._norm_url(url)
        url_expanded = self._expand_abbrev(url_norm)
        title_norm   = norm(title)
        content_norm = norm(content[:300]) if content else ""

        url_raw_years = list(dict.fromkeys(extract_raw_years(url_expanded)))
        url_semesters = list(dict.fromkeys(extract_semesters(url_expanded)))

        combined_all  = " ".join(filter(None, [url, title, content[:300]]))
        all_raw_years = list(dict.fromkeys(extract_raw_years(combined_all)))
        all_semesters = list(dict.fromkeys(extract_semesters(combined_all)))

        gid = self.neo4j.get_general_id()

        context = detect_level_context(url_expanded + " " + title_norm)
        logger.info(f"   Context: {context} | Raw years: {all_raw_years} | Semesters: {all_semesters}")

        lvl_words = {"master","licence","license","doctorat","doctorate","ingénieur","ingenieur"}
        def rmlvl(t): return " ".join(w for w in (t or "").split() if w not in lvl_words)

        if url_expanded:
            r = self._try_match(url_expanded, url_raw_years, url_semesters, gid, context, "url")
            if r: return r

        if url_expanded and title_norm:
            combined_l2 = " ".join(filter(None, [url_expanded, rmlvl(title_norm)]))
            r = self._try_match(combined_l2, all_raw_years, all_semesters, gid, context, "keyword")
            if r: return r

        combined_l3 = " ".join(filter(None, [url_expanded, rmlvl(title_norm), rmlvl(title_norm), rmlvl(content_norm)]))
        if combined_l3:
            r = self._try_match(combined_l3, all_raw_years, all_semesters, gid, context, "keyword")
            if r: return r

        return self._make_general(gid)

    def _try_match(self, text, raw_years, semesters, gid, context, method):
        sp = self.tree.find_specifics(text)
        if sp:
            w = self._filter_all(sp)
            if w:
                spec_context = get_level_from_spec(w[0][0])
                t = self._build_targets(w, raw_years, semesters, gid, spec_context)
                if t: return self._make_result(t, method, gid)

        depts = self.tree.find_departments(text)
        if depts:
            t = self._build_targets_dept(depts, raw_years, semesters, gid, context)
            if t: return self._make_result(t, method, gid)

        level_keywords = {
            "doctorat": "Doctorat", "doctorate": "Doctorat",
            "master": "Master", "licence": "Licence", "license": "Licence",
            "ingenieur": "Ingénieur", "ingénieur": "Ingénieur"
        }
        found_levels = []
        for kw, ln in level_keywords.items():
            if kw in text.lower() and ln not in found_levels:
                found_levels.append(ln)

        if found_levels:
            detected_depts = self.tree.find_departments(text)
            targets = self._try_match_levels_with_dept_filter(found_levels, detected_depts)
            if targets:
                return self._make_result(targets, method, gid)

        y2l = {"L1":"Licence","L2":"Licence","L3":"Licence","M1":"Master","M2":"Master",
               "ING1":"Ingénieur","ING2":"Ingénieur","ING3":"Ingénieur",
               "ING4":"Ingénieur","ING5":"Ingénieur","D":"Doctorat"}
        all_years = list(raw_years)
        for s in semesters:
            yr = resolve_semester_to_year(s, context)
            if yr and yr not in all_years: all_years.append(yr)

        if all_years:
            detected_depts = self.tree.find_departments(text)
            targets, seen = [], set()
            for yr in all_years:
                neo4j_years = self.neo4j.find_all_years_by_name(yr, detected_depts)
                for ny in neo4j_years:
                    if ny["id"] not in seen:
                        seen.add(ny["id"]); targets.append(ny)
            if targets:
                return self._make_result(targets, method, gid)
            for yr in all_years:
                ln = y2l.get(yr.upper(), "")
                if ln:
                    neo4j_levels = self.neo4j.find_all_levels_by_name(ln)
                    filtered = self._filter_levels_by_dept(neo4j_levels, detected_depts)
                    if filtered:
                        seen2 = set(); deduped = []
                        for t in filtered:
                            if t["id"] not in seen2: seen2.add(t["id"]); deduped.append(t)
                        return self._make_result(deduped, method, gid)

        return None

    def _try_match_levels_with_dept_filter(self, found_levels, detected_depts):
        targets, seen = [], set()
        for ln in found_levels:
            neo4j_levels = self.neo4j.find_all_levels_by_name(ln)
            filtered = self._filter_levels_by_dept(neo4j_levels, detected_depts)
            for nl in filtered:
                if nl["id"] not in seen: seen.add(nl["id"]); targets.append(nl)
        return targets

    def _filter_levels_by_dept(self, neo4j_levels, detected_depts):
        if not detected_depts:
            return neo4j_levels
        dept_names_norm = {norm(d.name) for d in detected_depts}
        filtered = [nl for nl in neo4j_levels if norm(nl.get("dept_name") or "") in dept_names_norm]
        return filtered if filtered else neo4j_levels

    def _make_result(self, targets, method, gid):
        p = targets[0]; ps = list(dict.fromkeys(x.get("path","") for x in targets if x.get("path")))
        return {"targets":targets,"method":method,"confidence":1.0,"hierarchy_path":" | ".join(ps[:3]),
                "label":p.get("label","General"),"id":p.get("id",gid),"name":p.get("name","General")}

    def _make_general(self, gid):
        return {"targets":[{"label":"General","id":gid,"name":"General","path":"General"}],
            "method":"general","confidence":0.0,"hierarchy_path":"General","label":"General","id":gid,"name":"General"}

    def _norm_url(self, url):
        if not url: return ""
        path = re.sub(r"https?://[^/]+","",url); path = re.sub(r"\.\w{2,5}(\?.*)?$","",path)
        path = path.replace("_"," ").replace("-"," ").replace("/"," ")
        return re.sub(r"\s+"," ",norm(path)).strip()

    def _expand_abbrev(self, text: str) -> str:
        if not text or not _ALIASES: return text
        words = text.split(); result = []
        for w in words:
            exp = _ALIASES.get(w, w)
            if exp and exp != w:
                result.append(w)
                result.append(exp)
            else:
                result.append(w)
        return " ".join(result)

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
            dept_name = dept.name if dept else "__none__"
            by_dept[dept_name].append((nd, wc))
        result = []
        for dept_name, dept_matches in by_dept.items():
            if len(dept_matches) <= 1:
                result.extend(dept_matches)
            else:
                max_score = max(wc for _, wc in dept_matches)
                if max_score - min(wc for _, wc in dept_matches) >= 2:
                    result.extend([(nd, wc) for nd, wc in dept_matches if wc == max_score])
                else:
                    result.extend(dept_matches)
        return result

    def _filter_dominated(self, matches):
        if not matches or len(matches)<=1: return matches
        dom = set(); nodes = [nd for nd,_ in matches]
        for i,ni in enumerate(nodes):
            for j,nj in enumerate(nodes):
                if i==j: continue
                if ni.is_ancestor_of(nj): dom.add((ni.label,ni.name))
        return [(nd,wc) for nd,wc in matches if (nd.label,nd.name) not in dom]

    def _filter_prefix_overlap(self, matches):
        if len(matches)<=1: return matches
        keep = []
        for i,(nd_i,wc_i) in enumerate(matches):
            dominated = False
            for j,(nd_j,wc_j) in enumerate(matches):
                if i==j: continue
                di = nd_i.department_root(); dj = nd_j.department_root()
                if not di or not dj or di.name != dj.name: continue
                if wc_j > wc_i:
                    pi = " ".join(nd_i._norm.split()[:wc_i]); pj = " ".join(nd_j._norm.split()[:wc_j])
                    if pi and pj and pi in pj: dominated=True; break
            if not dominated: keep.append((nd_i,wc_i))
        return keep

    def _filter_subset_specs(self, matches):
        if len(matches)<=1: return matches
        to_remove = set()
        for i,(nd_i,wc_i) in enumerate(matches):
            for j,(nd_j,wc_j) in enumerate(matches):
                if i==j: continue
                di = nd_i.department_root(); dj = nd_j.department_root()
                if not di or not dj or di.name != dj.name: continue
                words_i = set(nd_i._norm.split()); words_j = set(nd_j._norm.split())
                if words_i and words_j and words_i.issubset(words_j) and wc_j >= wc_i:
                    to_remove.add(i)
        return [m for idx,m in enumerate(matches) if idx not in to_remove]

    def _rid(self, node): return self.neo4j.resolve_node_id(node)

    def _build_targets(self, winners, raw_years, semesters, gid, spec_context=None):
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
                        nid = self._rid(yn)
                        if nid and nid not in seen:
                            hp = yn.path_str
                            seen.add(nid); targets.append({"label":"Year","name":yn.name,"path":hp,"id":nid}); added=True
                if not added:
                    nid = self._rid(nd)
                    if nid and nid not in seen: seen.add(nid); targets.append({"label":nd.label,"name":nd.name,"path":nd.path_str,"id":nid})
            else:
                nid = self._rid(nd)
                if nid and nid not in seen: seen.add(nid); targets.append({"label":nd.label,"name":nd.name,"path":nd.path_str,"id":nid})
        return targets

    def _build_targets_dept(self, depts, raw_years, semesters, gid, context):
        targets, seen = [], set()
        y2l = {"L1":"Licence","L2":"Licence","L3":"Licence","M1":"Master","M2":"Master",
               "ING1":"Ingénieur","ING2":"Ingénieur","ING3":"Ingénieur","ING4":"Ingénieur","ING5":"Ingénieur","D":"Doctorat"}
        all_years = list(raw_years)
        for s in semesters:
            yr = resolve_semester_to_year(s, context)
            if yr and yr not in all_years: all_years.append(yr)
        for dept in depts:
            if not all_years:
                nid = self._rid(dept)
                if nid and nid not in seen: seen.add(nid); targets.append({"label":dept.label,"name":dept.name,"path":dept.path_str,"id":nid})
                continue
            added = False
            for yr in all_years:
                neo4j_years = self.neo4j.find_years_under_dept(dept.name, yr)
                for ny in neo4j_years:
                    if ny["id"] not in seen: seen.add(ny["id"]); targets.append(ny); added=True
            if not added:
                for yr in all_years:
                    ln = y2l.get(yr.upper(),"")
                    if ln:
                        for child in dept.children:
                            if child.label=="Level" and norm(child.name)==norm(ln):
                                nid = self._rid(child)
                                if nid and nid not in seen: seen.add(nid); targets.append({"label":"Level","name":child.name,"path":child.path_str,"id":nid}); added=True
                if not added:
                    nid = self._rid(dept)
                    if nid and nid not in seen: seen.add(nid); targets.append({"label":dept.label,"name":dept.name,"path":dept.path_str,"id":nid})
        if not targets:
            for dept in depts:
                nid = self._rid(dept)
                if nid and nid not in seen: seen.add(nid); targets.append({"label":dept.label,"name":dept.name,"path":dept.path_str,"id":nid})
        return targets

# ═══════════════════════════ GEMINI ═══════════════════════════

GEMINI_PROMPT = """You are a document classifier for Farhat Abbas University Sétif 1.
Output ONLY valid JSON:
{"target":{"label":"<Faculty|Department|Level|Category|Program|Specialization|Year>","id":"<id>","name":"<name>","reason":"<reason>"},"match_method":"llm","confidence":<0.0-1.0>}
If no match: {"target":null,"match_method":"general","confidence":0.0}"""

class GeminiClassifier:
    def __init__(self, api_key=GEMINI_API_KEY):
        self._model = None; self._last_call = 0.0
        if _GEMINI_OK and api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=GEMINI_PROMPT,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=256))
    @property
    def available(self): return self._model is not None
    def classify(self, url, title, content, candidates_text):
        if not self.available: return None
        elapsed = time.time()-self._last_call
        if elapsed < GEMINI_RATE_LIMIT: time.sleep(GEMINI_RATE_LIMIT-elapsed)
        self._last_call = time.time()
        try:
            resp = self._model.generate_content(f"{candidates_text}\n\n## DOCUMENT\nURL:{url}\nTitle:{title}\nContent:{content[:LLM_CONTENT_EXCERPT]}\n\nClassify to DEEPEST valid node.")
            raw = resp.text or ""; js = re.sub(r"^```(?:json)?\s*|\s*```$","",raw.strip())
            parsed = json.loads(js); tgt = parsed.get("target")
            if not tgt: return None
            return {"targets":[{"label":tgt.get("label",""),"id":tgt.get("id",""),"name":tgt.get("name",""),"path":""}],
                "method":"llm","confidence":float(parsed.get("confidence",0.55)),"hierarchy_path":"",
                "label":tgt.get("label",""),"id":tgt.get("id",""),"name":tgt.get("name","")}
        except: return None

# ═══════════════════════════ EMBEDDING ═══════════════════════════
# ═══════════════════════════ EMBEDDING ═══════════════════════════

class EmbeddingClassifier:
    def __init__(self, model_name=None):
        """Initialize Gemini embeddings client."""
        api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set!")
        
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_name = "models/gemini-embedding-001"
        self._index = None
        self._cache: Dict[str, np.ndarray] = {}
        logger.info("✓ Gemini embeddings client ready")
    
    def _encode_gemini(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Gemini API."""
        embeddings = []
        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
                continue
            
            try:
                response = self._client.models.embed_content(
                    model=self._model_name,
                    contents=text
                )
                vec = np.array(response.embeddings[0].values, dtype=np.float32)
                vec = vec / np.linalg.norm(vec)  # normalize
                
                # Cache (limit size)
                if len(self._cache) > 10000:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = vec
                embeddings.append(vec)
                
            except Exception as e:
                logger.error(f"Gemini encoding failed: {e}")
                # Fallback: zero vector
                embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
        
        return np.array(embeddings, dtype=np.float32)
    
    def build_index(self, tree):
        """Build embedding index for academic tree nodes."""
        nodes, sigs = [], []
        for n in tree.all_nodes:
            if n.label in ("Year","Specialization","Program","Category","Department"):
                sig = n.name
                if n.parent:
                    sig += f" | {n.path_str}"
                nodes.append(n)
                sigs.append(sig)
        
        if not nodes:
            self._index = ([], np.array([], dtype=np.float32))
            return
        
        # Encode with Gemini
        embeddings = self._encode_gemini(sigs)
        self._index = (nodes, embeddings)
        logger.info(f"Embedding index: {len(nodes)} nodes (Gemini 3072-dim)")
    
    def encode_chunks(self, texts):
        """Encode chunk texts for storage."""
        if not texts:
            return []
        embeddings = self._encode_gemini(texts)
        return embeddings.tolist()
    
    def classify(self, text):
        """Classify text using embedding similarity."""
        if not self._index:
            return None
        
        nodes, vecs = self._index
        if len(nodes) == 0:
            return None
        
        # Encode query
        dv = self._encode_gemini([text])[0]
        
        # Cosine similarity
        sims = np.dot(vecs, dv)
        bi = int(np.argmax(sims))
        bs = float(sims[bi])
        
        if bs < EMBED_MIN_CONFIDENCE:
            return None
        
        n = nodes[bi]
        return {
            "targets": [{"label": n.label, "name": n.name, "path": n.path_str, "id": ""}],
            "method": "embedding",
            "confidence": bs
        }
# ═══════════════════════════ CHUNKER ═══════════════════════════

class HierarchicalChunker:
    def __init__(self):
        self._enc = tiktoken.get_encoding("cl100k_base") if _TIKTOKEN_OK else None
    def split(self, text, title=""):
        text = normalize_text(text)
        if not text: return []
        segs = self._segment(text)
        if not segs: return []
        chunks, cur, cur_tok, cur_sec, cur_type, idx = [], [], 0, title or "", "paragraph", 0
        for seg in segs:
            st = self._tok(seg["content"])
            if seg["type"]=="heading":
                if cur:
                    c = self._make(cur, idx, title, cur_sec, cur_type)
                    if len(c["clean_body"])>=MIN_CHUNK_CHARS: chunks.append(c); idx+=1
                cur, cur_tok = [], 0; cur_sec=seg["content"]; cur_type="section"; continue
            if st > CHUNK_TOKENS:
                if cur:
                    c = self._make(cur, idx, title, cur_sec, cur_type)
                    if len(c["clean_body"])>=MIN_CHUNK_CHARS: chunks.append(c); idx+=1
                    cur, cur_tok = [], 0
                sub = self._split_long(seg["content"], title, cur_sec, idx); chunks.extend(sub); idx+=len(sub); continue
            if cur_tok+st > CHUNK_TOKENS and cur:
                c = self._make(cur, idx, title, cur_sec, cur_type)
                if len(c["clean_body"])>=MIN_CHUNK_CHARS: chunks.append(c); idx+=1
                over, ot = [], 0
                for prev in reversed(cur):
                    pt = self._tok(prev)
                    if ot+pt > OVERLAP_TOKENS: break
                    over.insert(0, prev); ot+=pt
                cur = over+[seg["content"]]; cur_tok=ot+st
            else: cur.append(seg["content"]); cur_tok+=st; cur_type=seg["type"]
        if cur:
            c = self._make(cur, idx, title, cur_sec, cur_type)
            if len(c["clean_body"])>=MIN_CHUNK_CHARS: chunks.append(c)
        if not chunks and text.strip():
            chunks = [self._make([text.strip()], 0, title, title, "paragraph")]
        return chunks
    def _segment(self, text):
        segs = []
        for block in _RE_PARA_BREAK.split(text):
            block = block.strip()
            if not block: continue
            if _RE_HEADING.match(block): segs.append({"type":"heading","content":block})
            elif _RE_TABLE_MARKER.search(block): segs.append({"type":"table","content":block})
            elif _RE_LIST_ITEM.search(block): segs.append({"type":"list","content":block})
            else: segs.append({"type":"paragraph","content":block})
        return segs
    def _split_long(self, para, title, section, start):
        sents = [s.strip() for s in _RE_SENT_BOUND.split(para) if s.strip()]
        if not sents: sents = [para]
        chunks, cur, tok, idx = [], [], 0, start
        for sent in sents:
            st = self._tok(sent)
            if cur and tok+st > CHUNK_TOKENS:
                c = self._make(cur, idx, title, section, "paragraph")
                if len(c["clean_body"])>=MIN_CHUNK_CHARS: chunks.append(c); idx+=1
                cur, tok = [sent], st
            else: cur.append(sent); tok+=st
        if cur:
            c = self._make(cur, idx, title, section, "paragraph")
            if len(c["clean_body"])>=MIN_CHUNK_CHARS: chunks.append(c)
        return chunks
    def _make(self, parts, idx, title, section, ctype):
        body = " ".join(parts)
        et = "\n".join(p for p in [title, section, body] if p)
        return {"embed_text":et,"text":et,"clean_body":body,"section":section,"chunk_type":ctype,"token_count":self._tok(body),"chunk_index":idx}
    def _tok(self, text):
        if self._enc: return len(self._enc.encode(text))
        return len(text)//4

# ═══════════════════════════ PARSER ═══════════════════════════

def parse_json_doc(data: dict) -> dict:
    meta = data.get("metadata",{}); content = data.get("content",{}); resources = data.get("resources",{})
    ext_docs = resources.get("documents",[]) if isinstance(resources,dict) else []
    if "page" in meta:
        page = meta["page"]; page_url = page.get("url","")
        parts = [content.get("text","")]
        for sec in content.get("sections",[]):
            if isinstance(sec,dict): parts.extend([sec.get("text",""),sec.get("title","")])
        return dict(text="\n\n".join(filter(None,parts)), title=page.get("title",""), url=page_url, ext_docs=ext_docs)
    return dict(text=content.get("text",""), title="", url="", ext_docs=ext_docs)

def collect_json_files(root: Path) -> List[Tuple[Path, str, str]]:
    results = []
    for faculty_dir in sorted(root.iterdir()):
        if not faculty_dir.is_dir(): continue
        fl = FACULTY_LABELS.get(faculty_dir.name.lower(), faculty_dir.name.upper())
        for sub in ("pages","extracted","tables"):
            sfolder = faculty_dir / sub
            if not sfolder.exists(): continue
            for dirpath, _, filenames in os.walk(str(sfolder)):
                for fname in filenames:
                    if not fname.endswith(".json"): continue
                    jf = Path(dirpath)/fname
                    rem = str(jf)[len(str(sfolder))+1:]
                    sp = rem.find(os.sep)
                    dept = (rem[:sp] if sp!=-1 else "General").replace("_"," ").replace("-"," ").title()
                    results.append((jf, fl, dept))
    return results

# ═══════════════════════════ PIPELINE ═══════════════════════════
class IngestionPipeline:
    def __init__(self):
        logger.info("Initializing Gemini Embeddings...")
        self.embed_model = None  # لم نعد نحتاج SentenceTransformer
        self.tree = AcademicTree(STRUCTURE_FILE)
        self.neo4j = Neo4jClient()
        self.classifier = SmartClassifier(self.tree, self.neo4j)
        self.gemini = GeminiClassifier()
        self.embed_clf = EmbeddingClassifier()  # يستخدم Gemini API
        self.embed_clf.build_index(self.tree)
        self.chunker = HierarchicalChunker()
        self._candidates_text = self._build_candidates_text()
        self._general_id = self.neo4j.get_general_id()
        logger.info("✅ Pipeline ready (Gemini embeddings)")
        
    def _build_candidates_text(self):
        lines = ["## Specializations"]
        for n in self.tree.all_nodes:
            if n.label=="Specialization": lines.append(f"  name={n.name}")
        lines.append("\n## Programs")
        for n in self.tree.all_nodes:
            if n.label=="Program": lines.append(f"  name={n.name}")
        lines.append("\n## Departments")
        for n in self.tree.departments: lines.append(f"  name={n.name}")
        return "\n".join(lines)

    def run(self):
        root = Path(ROOT_FOLDER)
        all_files = collect_json_files(root)
        page_files = [(jf,f,d) for jf,f,d in all_files if "/pages/" in str(jf)]
        logger.info(f"📂 {len(page_files)} page files to process")
        ok = skip = fail = 0
        for jf, faculty, department in page_files:
            try:
                doc = self._process_page(jf, faculty, department)
                if doc is None: skip += 1; continue
                self._store(doc); ok += 1
                n_tgt = len(doc["classification"].get("targets",[]))
                logger.info(f"✅ {jf.name} [{doc['classification'].get('method','?')}] → {n_tgt} target(s)")
            except Exception as exc:
                import traceback; logger.error(f"❌ {jf.name}: {exc}\n{traceback.format_exc()}"); fail += 1
        self.neo4j.close()
        logger.info(f"\nCOMPLETE: ✅ {ok} ⏭ {skip} ❌ {fail}")

    def _process_page(self, jf, faculty, department):
        with open(jf,"r",encoding="utf-8") as fh: raw = json.load(fh)
        parsed = parse_json_doc(raw)
        text = parsed.get("text","")
        if len(text.strip()) < MIN_DOC_CHARS: return None
        title = parsed.get("title","") or jf.stem
        url = parsed.get("url","")
        fp = hashlib.md5(text.encode()).hexdigest()[:16]
        url_id = hashlib.md5(url.encode()).hexdigest()[:16] if url else f"url_{fp}"

        logger.info(f"\n{'─'*60}\n📄 {title[:60]}\n   URL: {url[:80]}")

        classification = self.classifier.classify(url=url, title=title, content=text)
        logger.info(f"   Tree: {classification.get('method')} → {len(classification.get('targets',[]))} targets")

        if not classification.get("targets") or classification.get("method")=="general":
            if self.gemini.available:
                logger.info("   Trying Gemini...")
                gr = self.gemini.classify(url, title, text, self._candidates_text)
                if gr and gr.get("targets"): classification = gr; logger.info(f"   Gemini: {len(classification.get('targets',[]))} targets")

        if not classification.get("targets") or classification.get("method")=="general":
            logger.info("   Trying Embedding...")
            er = self.embed_clf.classify(text[:500])
            if er and er.get("targets"): classification = er; logger.info(f"   Embedding: {len(classification.get('targets',[]))} targets")

        if not classification.get("targets"):
            classification = self.classifier._make_general(self._general_id)

        for t in classification.get("targets",[])[:5]:
            logger.info(f"     🎯 [{t.get('label','?')}] {t.get('name','?')} | path: {t.get('path','?')}")

        raw_chunks = self.chunker.split(text, title=title)
        chunk_texts = [c["embed_text"] for c in raw_chunks]
        embeddings = self.embed_clf.encode_chunks(chunk_texts)
        chunks = []
        for i,(cd,emb) in enumerate(zip(raw_chunks, embeddings)):
            chunks.append({"id":f"{fp}_c{i}","text":cd["embed_text"],"chunk_index":i,
                "token_count":cd.get("token_count",0),"language":"fr",
                "section":cd.get("section",""),"chunk_type":cd.get("chunk_type","paragraph"),"embedding":emb})
        logger.info(f"   Chunks: {len(chunks)}")

        child_urls, pdf_docs = [], []
        for ext_doc in parsed.get("ext_docs",[]):
            pdf_url = ext_doc.get("url","")
            if not pdf_url: continue
            pdf_title = ext_doc.get("title","") or f"PDF from {title}"
            local_f = ext_doc.get("local_file","")
            child_url_id = hashlib.md5(pdf_url.encode()).hexdigest()[:16]
            child_urls.append({"url_id":child_url_id,"url":pdf_url,"title":pdf_title,"source_type":"pdf"})
            child_text = ""
            if local_f:
                local_name = Path(local_f).name
                extracted_path = jf.parent.parent / "extracted" / local_name.replace(".pdf",".json")
                if extracted_path.exists():
                    with open(extracted_path,"r",encoding="utf-8") as fh: ext_data = json.load(fh)
                    child_text = ext_data.get("content",{}).get("text","") or " ".join(
                        p.get("text","") for p in ext_data.get("content",{}).get("pages",[]))
                elif Path(local_f).exists() and _PYPDF_OK:
                    try:
                        reader = pypdf.PdfReader(str(Path(local_f)))
                        child_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                    except: pass
            if child_text and len(child_text.strip()) >= MIN_DOC_CHARS:
                child_chunks = self.chunker.split(child_text, title=pdf_title)
                child_embs = self.embed_clf.encode_chunks([c["embed_text"] for c in child_chunks])
                child_chunk_objs = []
                child_fp = hashlib.md5(child_text.encode()).hexdigest()[:16]
                for i,(cd,emb) in enumerate(zip(child_chunks, child_embs)):
                    child_chunk_objs.append({"id":f"{child_fp}_c{i}","text":cd["embed_text"],"chunk_index":i,
                        "token_count":cd.get("token_count",0),"language":"fr",
                        "section":cd.get("section",""),"chunk_type":cd.get("chunk_type","paragraph"),"embedding":emb})
                pdf_docs.append({"url_id":child_url_id,"chunks":child_chunk_objs,"classification":classification})
                logger.info(f"   PDF chunks: {len(child_chunk_objs)}")

        return {"url_id":url_id,"url":url,"title":title,"source_type":"page",
            "classification":classification,"chunks":chunks,"pdf_docs":pdf_docs,"child_urls":child_urls}

    def _store(self, doc):
        cls = doc["classification"]; targets = cls.get("targets",[])
        if not targets: targets = [{"label":"General","id":self._general_id,"name":"General","path":"General"}]
        primary = targets[0]
        self.neo4j.upsert_url(doc["url_id"],doc["url"],doc["title"],doc["source_type"],
            primary.get("label","General"),primary.get("id",self._general_id),
            cls.get("hierarchy_path",""),cls.get("method","none"),float(cls.get("confidence",0.0)))
        if len(targets) > 1: self.neo4j.link_extra_targets(doc["url_id"], targets[1:])
        self.neo4j.create_chunks(doc["url_id"], doc["chunks"], cls)
        for child in doc.get("child_urls",[]):
            self.neo4j.upsert_url(child["url_id"],child["url"],child["title"],child["source_type"],
                "General",self._general_id,"","inherited",0.0,doc["url_id"])
        for pdf in doc.get("pdf_docs",[]):
            self.neo4j.create_chunks(pdf["url_id"], pdf["chunks"], pdf["classification"])

# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    IngestionPipeline().run()

if __name__ == "__main__":
    main()
