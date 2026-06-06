#!/usr/bin/env python3
"""
Test Data Generator with Translation Nodes (French pivot)
===========================================================
- For French chunks: set c.is_french = True
- For non-French chunks: 
    * set c.has_translation = True
    * create a (:Translation {text: french_version, language:'fr'})
    * link c -[:HAS_TRANSLATION]-> translation
"""

from neo4j import GraphDatabase
import hashlib
import random
import logging
from sentence_transformers import SentenceTransformer
import argostranslate.package
import argostranslate.translate
from functools import lru_cache


def detect_language(text: str) -> str:
    """Simple language detection: Arabic, French, English."""
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "ar"
    # French common words
    if any(w in text.lower() for w in ["le", "la", "les", "de", "et", "est", "pour", "dans"]):
        return "fr"
    return "en"

# ─── Connection ───────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ─── Embedding model ─────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-m3"   # or intfloat/multilingual-e5-large
logging.info("Loading embedding model: %s", EMBED_MODEL_NAME)
_embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
PASSAGE_PREFIX = "passage: "
DIMENSION = 1024

def embed(text: str) -> list[float]:
    prefixed = PASSAGE_PREFIX + text
    vec = _embedding_model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if len(vec) != DIMENSION:
        raise RuntimeError(f"Expected {DIMENSION} dimensions, got {len(vec)}")
    return vec.tolist()

def build_embedded_text(chunk_text: str, page_title: str, source_type: str) -> str:
    prefix = f"[PDF] {page_title}" if source_type == "pdf" else page_title
    return f"{prefix} | {chunk_text}"

# ─── Test data (English, Arabic, French) ────────────────────────────────────
PAGES = [
    # English page
    {
        "url": "https://www.univ-setif.dz/fsciences/informatique/license/cs/en",
        "title": "Computer Science Department - Course Catalog",
        "source_type": "page",
        "link_to": {"label": "Department", "name": "Informatique", "parent": None},
        "chunks": [
            ("Data Structures and Algorithms", "en"),
            ("Operating Systems - Process Management", "en"),
            ("Database Design and SQL", "en"),
        ],
    },
    # Arabic page
    {
        "url": "https://www.univ-setif.dz/fsciences/mathematiques/ar",
        "title": "قسم الرياضيات - برنامج البكالوريوس",
        "source_type": "page",
        "link_to": {"label": "Department", "name": "Mathématiques", "parent": None},
        "chunks": [
            ("الجبر الخطي - المصفوفات والمحددات", "ar"),
            ("التحليل الرياضي - النهايات والاتصال", "ar"),
            ("المعادلات التفاضلية من الدرجة الأولى", "ar"),
        ],
    },
    # French page (already French)
    {
        "url": "https://www.univ-setif.dz/fsciences/actualites/inscriptions",
        "title": "Calendrier des Inscriptions 2024-2025",
        "source_type": "page",
        "link_to": {"label": "General", "name": "general", "parent": None},
        "chunks": [
            ("la responsabilité de Harag Fouaze", "fr"),
            ("Les inscriptions débuteront le 15 septembre 2024.", "fr"),
            ("Documents requis : baccalauréat, relevé de notes.", "fr"),
            ("Les étudiants internationaux doivent fournir une équivalence.", "fr"),
        ],
    },
    # Mixed: page title in French, but chunks in English/Arabic
    {
        "url": "https://www.univ-setif.dz/fsciences/informatique/actualites/workshop",
        "title": "Atelier International sur l'IA",
        "source_type": "page",
        "link_to": {"label": "General", "name": "general", "parent": None},
        "chunks": [
            ("This workshop covers deep learning and transformers.", "en"),
            ("المحاضرات ستكون باللغة العربية والفرنسية", "ar"),
            ("Des sessions pratiques sont prévues.", "fr"),
        ],
    },
]

def find_node(tx, link_to: dict) -> tuple[str | None, str | None]:
    label = link_to["label"]
    name = link_to["name"]
    parent = link_to.get("parent")
    if label == "General":
        rec = tx.run("MATCH (g:General) RETURN g.id AS id LIMIT 1").single()
        return ("General", rec["id"]) if rec else (None, None)
    if label == "Department":
        rec = tx.run("MATCH (d:Department {name: $name}) RETURN d.id AS id", name=name).single()
        return ("Department", rec["id"]) if rec else (None, None)
    if label == "Year" and parent:
        for query in [
            "MATCH (s:Specialization {name: $spec})-[:HAS_YEAR]->(y:Year {name: $year}) RETURN y.id",
            "MATCH (p:Program {name: $spec})-[:HAS_YEAR]->(y:Year {name: $year}) RETURN y.id"
        ]:
            rec = tx.run(query, spec=parent, year=name).single()
            if rec:
                return ("Year", rec["id"])
    return (None, None)

# ─── Chunk creation with translation nodes ───────────────────────────────────
def create_chunks(tx, chunks_with_lang, parent_url_id, page_title, source_type, rel="HAS_CHUNK"):
    chunk_ids = []
    for i, (chunk_text, lang) in enumerate(chunks_with_lang):
        cid = f"{parent_url_id}_c{i}"
        enriched_text = build_embedded_text(chunk_text, page_title, source_type)
        embedding_vec = embed(enriched_text)
        chunk_ids.append(cid)
        is_french = (lang == "fr")

        # ON CREATE SET ensures we don't overwrite existing chunks
        tx.run(f"""
            MERGE (c:Chunk {{id: $id}})
            ON CREATE SET
                c.text          = $text,
                c.embedded_text = $etext,
                c.embedding     = $embedding,
                c.chunk_index   = $idx,
                c.token_count   = $tokens,
                c.language      = $lang,
                c.is_french     = $is_french
            WITH c
            MATCH (u:URL {{id: $uid}})
            MERGE (u)-[:{rel}]->(c)
        """,
            id=cid, text=chunk_text, etext=enriched_text,
            embedding=embedding_vec, idx=i,
            tokens=len(chunk_text) // 4,
            lang=lang, is_french=is_french,
            uid=parent_url_id,
        )

    # Chain consecutive chunks — MERGE prevents duplicate NEXT_CHUNK relationships
    for j in range(len(chunk_ids) - 1):
        tx.run("""
            MATCH (a:Chunk {id: $aid}), (b:Chunk {id: $bid})
            MERGE (a)-[:NEXT_CHUNK]->(b)
        """, aid=chunk_ids[j], bid=chunk_ids[j + 1])

    return chunk_ids


def create_url_node(tx, url, title, source_type, target_label, target_id, chunks_with_lang):
    url_id = hashlib.md5(url.encode()).hexdigest()[:16]

    # ON CREATE SET avoids overwriting existing URL nodes
    tx.run("""
        MERGE (u:URL {id: $id})
        ON CREATE SET u.url = $url, u.title = $title, u.source_type = $stype
    """, id=url_id, url=url, title=title, stype=source_type)

    tx.run(f"""
        MATCH (n:{target_label} {{id: $nid}})
        MATCH (u:URL {{id: $uid}})
        MERGE (n)-[:HAS_CONTENT]->(u)
    """, nid=target_id, uid=url_id)

    create_chunks(tx, chunks_with_lang, url_id, title, source_type)
    return url_id


def ensure_hierarchy_nodes(tx):
    """
    Only create department nodes if they don't already exist by name.
    Avoids duplicating nodes that already exist with different IDs.
    """
    tx.run("MERGE (d:Department {name: 'Mathématiques'}) ON CREATE SET d.id = 'math'")
    tx.run("MERGE (d:Department {name: 'Informatique'})  ON CREATE SET d.id = 'info'")
    tx.run("MERGE (g:General {name: 'general'})          ON CREATE SET g.id = 'general'")

def create_file_node(tx, file: dict, parent_url_id: str) -> str:
    file_url_id = hashlib.md5(file["url"].encode()).hexdigest()[:16]
    tx.run("""
        MERGE (f:URL {id: $id})
        SET f.url = $url, f.title = $title, f.source_type = $stype
    """, id=file_url_id, url=file["url"], title=file["title"], stype=file["source_type"])
    tx.run("""
        MATCH (p:URL {id: $pid}), (f:URL {id: $fid})
        MERGE (p)-[:HAS_FILE]->(f)
    """, pid=parent_url_id, fid=file_url_id)
    create_chunks(tx, file["chunks"], file_url_id, file["title"], file["source_type"])
    return file_url_id

def create_test_data(tx):
    ensure_hierarchy_nodes(tx)
    for idx, p in enumerate(PAGES):
        link_to = p["link_to"]
        target_label, target_id = find_node(tx, link_to)
        if not target_id:
            print(f"  ⚠️  SKIPPED: hierarchy node not found for {link_to}")
            continue
        chunks_with_lang = p["chunks"]  # already list of (text, lang)
        url_id = create_url_node(
            tx, p["url"], p["title"], p["source_type"],
            target_label, target_id, chunks_with_lang
        )
        print(f"  📄 {p['title'][:50]:50s} → {len(chunks_with_lang)} chunks")
        # Files are not modified in this example; you can adapt similarly
        for f in p.get("files", []):
            create_file_node(tx, f, url_id)
            print(f"  📎 {f['title'][:50]:50s} → {len(f['chunks'])} chunks")
        print(f"✅ [{idx + 1}/{len(PAGES)}] {p['title'][:55]}\n")

def main():
    with driver.session() as session:
        session.execute_write(create_test_data)

    # Summary
    with driver.session() as session:
        print("=" * 55)
        print("📊  SUMMARY")
        for rec in session.run("""
            MATCH (c:Chunk) RETURN count(c) AS total_chunks
        """):
            print(f"  Total chunks         : {rec['total_chunks']}")    
        for rec in session.run("""
            MATCH (c:Chunk) WHERE c.is_french = true RETURN count(c) AS french_chunks
        """):
            print(f"  French chunks        : {rec['french_chunks']}")
        print("=" * 55)

    driver.close()
    print("✅ Done!")

if __name__ == "__main__":
    main()