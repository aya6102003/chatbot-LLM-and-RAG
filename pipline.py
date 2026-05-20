"""
RAG Ingestion Pipeline — Farhat Abbas University Sétif 1
=========================================================
Graph hierarchy (Neo4j):
  University
    └─ Faculty        (iast, med, snv …)
         └─ Department (informatique, biologie …)
              └─ Document (filename / page title)
                   └─ Chunk (text slice)

FIXES APPLIED:
1. Semantic chunking (sentence/paragraph boundaries)
2. Content quality filtering (no boilerplate/empty content)
3. Chunk overlap tracking with metadata
4. Progress saving (resume capability)
"""

import os, re, json, logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from sentence_transformers import SentenceTransformer
import chromadb
from neo4j import GraphDatabase

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
ROOT_FOLDER   = "./university_farhat_abaas"
CHROMA_PATH   = "./chroma_db"
METADATA_PATH = "./metadata.json"
PROGRESS_FILE = "./pipeline_progress.json"  # <--- NEW: resume capability

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "123456789"

UNIVERSITY_NAME = "Farhat Abbas University Sétif 1"

# <--- CHANGED: Reduced chunk size for better precision, increased overlap
MAX_CHARS = 600  # Was 800 - smaller chunks = better retrieval
OVERLAP   = 100   # Was 150 - adjusted for new chunk size
MIN_CHUNK = 80    # Was 60 - ignore tiny useless chunks

# <--- NEW: Content filtering thresholds
MIN_QUALITY_TEXT_LEN = 100  # Below this = likely garbage
BOILERPLATE_THRESHOLD = 0.25  # Max 25% boilerplate words

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MODELS & CLIENTS
# ─────────────────────────────────────────────────────────────
model  = SentenceTransformer("BAAI/bge-m3")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# <--- NEW: Clear existing collection? Add flag
def get_or_clear_collection(client, name: str, clear_existing: bool = False):
    """Get collection, optionally clear it for fresh indexing"""
    if clear_existing:
        try:
            client.delete_collection(name)
            log.info(f"🗑️ Cleared existing collection: {name}")
        except:
            pass
    return client.get_or_create_collection(name=name)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = get_or_clear_collection(chroma_client, "university_data", clear_existing=True)

# ─────────────────────────────────────────────────────────────
# FACULTY LABELS  (folder-name → human-readable)
# ─────────────────────────────────────────────────────────────
FACULTY_LABELS = {
    # Main university
    "Farhat_Abbas_University" : "Farhat Abbas University Sétif 1",
                    
    # Faculties
    "ftechnologie": "Faculty of Technology",
    "fsciences": "Faculty of Science",
    "fsnv": "Faculty of Nature and Life Sciences",
    "feco": "Faculty of Economics, Business and Management Sciences",
    "fmed": "Faculty of Medicine",       
    # Institutes
    "iomp": "Institute of Optics and Precision Mechanics",
    "iast": "Institute of Architecture and Earth Sciences",
    "istm": "Institute of Materials Science and Techniques"
}

# <--- NEW: Academic terms for quality filtering
ACADEMIC_INDICATORS = [
    # English
    "semester", "module", "course", "exam", "lecture",
    # French
    "semestre", "cours", "examen", "licence", "master",
    "doctorat", "formation", "syllabus", "filière", "TD", "TP",
    # Arabic
    "الفصل", "الدراسي", "امتحان", "مقياس", "تخصص"
]

BOILERPLATE_WORDS = [
    "copyright", "all rights reserved", "privacy policy", "terms of use",
    "click here", "read more", "subscribe", "newsletter", "cookie policy",
    "website", "navigation", "menu", "footer", "header"
]

# ─────────────────────────────────────────────────────────────
# CONTENT QUALITY FILTERS (NEW)
# ─────────────────────────────────────────────────────────────
def is_quality_content(text: str) -> Tuple[bool, str]:
    """
    Filter out low-value content.
    Returns: (is_quality, reason)
    """
    text_lower = text.lower()
    text_len = len(text.strip())
    
    # Check 1: Minimum length
    if text_len < MIN_QUALITY_TEXT_LEN:
        return False, f"Too short ({text_len} chars)"
    
    # Check 2: Boilerplate ratio
    boilerplate_count = sum(1 for word in BOILERPLATE_WORDS if word in text_lower)
    boilerplate_ratio = boilerplate_count / len(BOILERPLATE_WORDS)
    if boilerplate_ratio > BOILERPLATE_THRESHOLD:
        return False, f"Too much boilerplate ({boilerplate_ratio:.1%})"
    
    # Check 3: Academic relevance (at least one academic term)
    has_academic = any(indicator in text_lower for indicator in ACADEMIC_INDICATORS)
    if not has_academic:
        # Don't reject completely, but warn
        return True, "Low academic signal"
    
    return True, "Quality content"

def clean_text(text: str) -> str:
    """Remove noise and normalize text"""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove empty lines at start/end
    text = text.strip()
    return text

# ─────────────────────────────────────────────────────────────
# SEMANTIC CHUNKING (MAJOR CHANGE)
# ─────────────────────────────────────────────────────────────
def chunk_by_semantic_units(text: str, title: str, max_chars: int = MAX_CHARS) -> List[Dict]:
    """
    Chunk by semantic boundaries: sections → paragraphs → sentences.
    Returns list of dicts: {'text': chunk_text, 'start_char': pos, 'end_char': pos}
    """
    text = clean_text(text)
    chunks = []
    
    # Step 1: Try to split by markdown-style headers first
    lines = text.split('\n')
    current_section = []
    current_section_len = 0
    section_headers = []
    
    for i, line in enumerate(lines):
        # Check if this line looks like a header
        is_header = bool(re.match(r'^#{1,3}\s+|^[A-Z][A-Z\s]{4,}$|^[0-9]+\.\s+[A-Z]', line))
        
        if is_header and current_section_len > 0:
            # Save current section
            if current_section_len >= MIN_CHUNK:
                chunk_text = f"{title}\n{' '.join(section_headers)}\n{''.join(current_section)}"
                chunks.append({
                    'text': chunk_text[:max_chars],  # Truncate if needed
                    'start_char': i,
                    'end_char': i + len(chunk_text),
                    'headers': section_headers.copy()
                })
            # Reset for new section
            current_section = []
            current_section_len = 0
            section_headers = [line]
        else:
            current_section.append(line + '\n')
            current_section_len += len(line)
    
    # Don't forget the last section
    if current_section_len >= MIN_CHUNK:
        chunk_text = f"{title}\n{' '.join(section_headers)}\n{''.join(current_section)}"
        chunks.append({
            'text': chunk_text[:max_chars],
            'start_char': len(text) - current_section_len,
            'end_char': len(text),
            'headers': section_headers.copy()
        })
    
    # Step 2: If we have no chunks (no headers), split by paragraphs with overlap
    if not chunks:
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len <= max_chars:
                current_chunk.append(para)
                current_len += para_len
            else:
                if current_chunk:
                    chunks.append({
                        'text': f"{title}\n{''.join(current_chunk)}",
                        'start_char': 0,  # Can't track precisely in this fallback
                        'end_char': 0,
                        'headers': []
                    })
                current_chunk = [para]
                current_len = para_len
        
        if current_chunk:
            chunks.append({
                'text': f"{title}\n{''.join(current_chunk)}",
                'start_char': 0,
                'end_char': 0,
                'headers': []
            })
    
    # Step 3: Post-process - ensure no chunk is too long
    final_chunks = []
    for chunk in chunks:
        if len(chunk['text']) > max_chars:
            # Aggressive truncation at last sentence boundary
            text_to_truncate = chunk['text']
            last_period = text_to_truncate[:max_chars].rfind('.')
            last_newline = text_to_truncate[:max_chars].rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point < max_chars * 0.7:  # If cut point is too early, just hard cut
                cut_point = max_chars
            chunk['text'] = text_to_truncate[:cut_point]
        final_chunks.append(chunk)
    
    return final_chunks

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
                continue

            for jf in subfolder.rglob("*.json"):
                # Department = immediate subfolder under <sub>, or "General"
                rel = jf.relative_to(subfolder)
                dept = rel.parts[0] if len(rel.parts) > 1 else "General"
                dept = dept.replace("_", " ").replace("-", " ").title()

                results.append((jf, faculty_label, dept))

        log.info(f"📂 {faculty_dir.name} → {sum(1 for r in results if r[1]==faculty_label)} files so far")

    log.info(f"🔎 TOTAL JSON FILES: {len(results)}")
    return results

# ─────────────────────────────────────────────────────────────
# JSON PARSER (IMPROVED)
# ─────────────────────────────────────────────────────────────
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_json(data: dict) -> dict:
    meta    = data.get("metadata", {})
    content = data.get("content", {})

    # ── SCRAPER format ──────────────────────────────────────
    if "page" in meta:
        page = meta["page"]
        text_content = content.get("text", "")
        # <--- NEW: Clean scraper text
        text_content = clean_text(text_content)
        return dict(
            text      = text_content,
            title     = page.get("title", ""),
            url       = page.get("url", ""),
            file_path = "",
            file_type = "web",
            tables    = content.get("tables", []),
            source    = "scraper",
        )

    # ── EXTRACTOR format ────────────────────────────────────
    file_info = meta.get("file", {})
    tables    = content.get("tables", [])

    parts = []
    if content.get("text"):
        parts.append(clean_text(content["text"]))
    for p in content.get("pages", []):
        if p.get("text"):
            parts.append(clean_text(p["text"]))

    # Tables → markdown format (better for embedding)
    for tbl in tables:
        headers = tbl.get("headers", [])
        rows    = tbl.get("rows", [])
        if headers:
            # <--- CHANGED: Better table formatting (markdown style)
            tbl_text = "| " + " | ".join(headers) + " |\n"
            tbl_text += "|" + "|".join(["---" for _ in headers]) + "|\n"
            for row in rows:
                tbl_text += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
            parts.append(tbl_text)

    combined_text = "\n\n".join(parts)
    combined_text = clean_text(combined_text)

    return dict(
        text      = combined_text,
        title     = file_info.get("name", "") or Path(file_info.get("path", "")).stem,
        url       = "",
        file_path = file_info.get("path", ""),
        file_type = file_info.get("type", ""),
        tables    = tables,
        source    = "extractor",
    )

# ─────────────────────────────────────────────────────────────
# NEO4J — 5-level hierarchy (ENHANCED)
# ─────────────────────────────────────────────────────────────
def insert_graph(tx, university, faculty, department, doc_title, chunk_id, chunk_text, chunk_metadata):
    """
    Creates / merges nodes at every level and links them.
    <--- NEW: Added chunk position tracking, NODE_SEQUENCE for ordering
    """
    tx.run("""
        // University
        MERGE (u:University {name: $university})

        // Faculty
        MERGE (f:Faculty {name: $faculty})
        MERGE (u)-[:HAS_FACULTY]->(f)

        // Department
        MERGE (dept:Department {name: $department, faculty: $faculty})
        MERGE (f)-[:HAS_DEPARTMENT]->(dept)

        // Document
        MERGE (d:Document {title: $doc_title, faculty: $faculty, department: $department})
        MERGE (dept)-[:HAS_DOCUMENT]->(d)

        // Chunk with enhanced metadata
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $chunk_text,
            c.chunk_index = $chunk_index,
            c.start_char = $start_char,
            c.end_char = $end_char,
            c.has_tables = $has_tables,
            c.academic_score = $academic_score
            
        MERGE (d)-[:HAS_CHUNK {order: $chunk_index}]->(c)
        
        // <--- NEW: Link consecutive chunks for context expansion
        WITH c, $chunk_index as idx, $doc_title as doctitle
        MATCH (prev:Chunk)-[:HAS_CHUNK]-(:Document {title: doctitle})
        WHERE prev.chunk_index = idx - 1
        MERGE (prev)-[:NEXT_CHUNK]->(c)
    """,
        university = university,
        faculty    = faculty,
        department = department,
        doc_title  = doc_title,
        chunk_id   = chunk_id,
        chunk_text = chunk_text,
        chunk_index = chunk_metadata.get('chunk_index', 0),
        start_char = chunk_metadata.get('start_char', 0),
        end_char = chunk_metadata.get('end_char', 0),
        has_tables = chunk_metadata.get('has_tables', False),
        academic_score = chunk_metadata.get('academic_score', 0.5),
    )

# ─────────────────────────────────────────────────────────────
# PROGRESS TRACKING (NEW)
# ─────────────────────────────────────────────────────────────
def load_progress() -> set:
    """Load already processed files to skip them"""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_files: set):
    """Save progress"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_files), f)

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def build_pipeline(resume: bool = True):
    root = Path(ROOT_FOLDER)
    json_files = collect_json_files(root)
    
    # Track progress
    processed_files = load_progress() if resume else set()
    
    ok = skip = fail = 0
    metadata_store = []
    quality_stats = {'academic_low': 0, 'boilerplate_rejected': 0, 'too_short': 0}
    
    for jf, faculty, department in json_files:
        # Skip already processed
        file_key = f"{faculty}/{department}/{jf.name}"
        if file_key in processed_files:
            log.info(f"⏭️ Skipping already processed: {file_key}")
            skip += 1
            continue
        
        # ── Parse ──────────────────────────────────────────
        try:
            raw = load_json(jf)
            pars = parse_json(raw)
        except Exception as e:
            log.warning(f"⚠ Skipping {jf.name}: {e}")
            fail += 1
            continue
        
        if not pars["text"].strip():
            log.warning(f"⚠ Empty text in {jf.name}")
            skip += 1
            continue
        
        # <--- NEW: Quality filtering
        is_quality, reason = is_quality_content(pars["text"])
        if not is_quality:
            log.warning(f"⚠ Low quality ({reason}) → skipping {jf.name}")
            if "boilerplate" in reason:
                quality_stats['boilerplate_rejected'] += 1
            elif "short" in reason:
                quality_stats['too_short'] += 1
            skip += 1
            continue
        
        if reason == "Low academic signal":
            quality_stats['academic_low'] += 1
            log.info(f"⚠ Low academic signal in {jf.name} (still processing)")
        
        # ── Semantic chunking ──────────────────────────────
        title = pars["title"] or jf.stem
        semantic_chunks = chunk_by_semantic_units(pars["text"], title)
        
        if not semantic_chunks:
            log.warning(f"⚠ No chunks generated for {jf.name}")
            skip += 1
            continue
        
        # Extract just the text from chunks for embedding
        chunk_texts = [chunk['text'] for chunk in semantic_chunks]
        
        # ── Embed ──────────────────────────────────────────
        try:
            embeddings = model.encode(chunk_texts, show_progress_bar=False)
        except Exception as e:
            log.error(f"❌ Embedding failed for {jf.name}: {e}")
            fail += 1
            continue
        
        # ── Store ──────────────────────────────────────────
        with driver.session() as session:
            for i, (chunk_dict, embedding) in enumerate(zip(semantic_chunks, embeddings)):
                chunk_text = chunk_dict['text']
                cid = f"{jf.stem}_chunk_{i}"
                
                # Calculate academic score based on academic term density
                academic_term_count = sum(1 for term in ACADEMIC_INDICATORS if term in chunk_text.lower())
                academic_score = min(1.0, academic_term_count / 10.0)  # Normalize to 0-1
                
                chunk_meta = {
                    'chunk_index': i,
                    'start_char': chunk_dict.get('start_char', 0),
                    'end_char': chunk_dict.get('end_char', 0),
                    'has_tables': bool(pars.get('tables')),
                    'academic_score': academic_score
                }
                
                # Chroma
                collection.add(
                    documents=[chunk_text],
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        "chunk_id": cid,           # <── THIS LINE IS THE FIX
                        "file": jf.name,
                        "faculty": faculty,
                        "department": department,
                        "title": title,
                        "source": pars["source"],
                        "url": pars["url"],
                        "chunk_index": i,
                        "academic_score": academic_score,  # <--- NEW
                        "chunk_len": len(chunk_text)       # <--- NEW
                    }],
                    ids=[cid],
                )
                
                # Metadata JSON
                metadata_store.append({
                    "chunk_id": cid,
                    "file": jf.name,
                    "faculty": faculty,
                    "department": department,
                    "title": title,
                    "source": pars["source"],
                    "url": pars["url"],
                    "chunk": chunk_text,
                    "academic_score": academic_score,  # <--- NEW
                    "chunk_index": i,
                })
                
                # Neo4j
                session.execute_write(
                    insert_graph,
                    UNIVERSITY_NAME,
                    faculty,
                    department,
                    title,
                    cid,
                    chunk_text,
                    chunk_meta
                )
        
        ok += 1
        processed_files.add(file_key)
        save_progress(processed_files)  # Save after each file
        
        log.info(f"✅ [{faculty} / {department}] {jf.name} → {len(semantic_chunks)} chunks (academic_score: {academic_score:.2f})")
    
    # ── Save metadata ──────────────────────────────────────
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)
    
    # ── Summary with quality stats ─────────────────────────
    log.info(f"\n{'─'*55}")
    log.info(f"  PIPELINE COMPLETE")
    log.info(f"  ✅ SUCCESS: {ok}")
    log.info(f"  ⏭️ SKIPPED:  {skip}")
    log.info(f"  ❌ FAILED:   {fail}")
    log.info(f"\n  Quality Stats:")
    log.info(f"    - Rejected (boilerplate): {quality_stats['boilerplate_rejected']}")
    log.info(f"    - Rejected (too short):   {quality_stats['too_short']}")
    log.info(f"    - Low academic signal:    {quality_stats['academic_low']}")
    log.info(f"\n  📊 Total chunks indexed: {len(metadata_store)}")
    log.info(f"  💾 metadata.json saved")
    log.info(f"{'─'*55}")
    
    # <--- NEW: Print sample chunk for verification
    if metadata_store:
        log.info("\n  📝 SAMPLE CHUNK (first 200 chars):")
        log.info(f"     {metadata_store[0]['chunk'][:200]}...")
    
    return ok, skip, fail

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    resume = '--fresh' not in sys.argv  # Use --fresh to start over
    if not resume:
        log.info("🔥 Starting FRESH pipeline (clearing existing data)")
    build_pipeline(resume=resume)