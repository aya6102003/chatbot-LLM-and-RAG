#!/usr/bin/env python3
"""
RAG Retrieval Pipeline — Fixed to match ingestion pipeline
==========================================================
Fixes applied:
1. chunk_id added to Chroma metadata so lookup works
2. Academic score filter disabled by default (set to 0.0)
3. Hybrid search RRF score inversion fixed (no longer inverted)
4. Final sort changed to descending (best score first)
5. Neighbor expansion fetches real neighbor text from metadata store
6. Chunk text is used as-is (matching what pipeline stored)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (must match pipeline.py exactly)
# ─────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
METADATA_PATH = "./metadata.json"
OUTPUT_FILE   = "./rag_results.txt"

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "123456789"

EMBED_MODEL = "BAAI/bge-m3"          # must match pipeline
COLLECTION  = "university_data"       # must match pipeline

# ── Retrieval knobs ──────────────────────────────────────────
ENABLE_GRAPH_CONTEXT = True
ENABLE_HYBRID_SEARCH = True
EXPAND_NEIGHBORS     = 1
# FIX #2 — was 0.3, killed most valid chunks; set 0.0 to disable
MIN_ACADEMIC_SCORE   = 0.0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GRAPH RETRIEVER
# ─────────────────────────────────────────────────────────────
class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_chunk_neighbors(self, chunk_id: str) -> Dict:
        """Return prev/next chunk IDs using NEXT_CHUNK relationships."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
                OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
                RETURN
                    collect(DISTINCT prev.id) AS prev_chunks,
                    collect(DISTINCT next.id)  AS next_chunks
            """, chunk_id=chunk_id)
            record = result.single()
            if record:
                return {
                    "prev": record["prev_chunks"] or [],
                    "next": record["next_chunks"] or [],
                }
        return {"prev": [], "next": []}

    def get_chunks_for_scope(self,
                             faculty: Optional[str] = None,
                             department: Optional[str] = None) -> List[str]:
        """Return all chunk IDs that belong to the given faculty / department."""
        with self.driver.session() as session:
            query = """
                MATCH (f:Faculty)-[:HAS_DEPARTMENT]->(dept:Department)
                      -[:HAS_DOCUMENT]->(d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WHERE 1=1
            """
            params: Dict = {}
            if faculty:
                query += " AND f.name = $faculty"
                params["faculty"] = faculty
            if department:
                query += " AND dept.name = $department"
                params["department"] = department
            query += " RETURN collect(c.id) AS chunk_ids"

            result = session.run(query, **params)
            record = result.single()
            return record["chunk_ids"] if record else []


# ─────────────────────────────────────────────────────────────
# EMBEDDINGS  (same model as pipeline)
# ─────────────────────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────
# LOAD STORES
# ─────────────────────────────────────────────────────────────
def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )


def load_metadata_store() -> Dict[str, Dict]:
    """
    Returns a dict keyed by chunk_id.
    Pipeline stores each entry with key "chunk_id".
    """
    meta_path = Path(METADATA_PATH)
    if not meta_path.exists():
        log.warning(f"metadata.json not found at {METADATA_PATH}")
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["chunk_id"]: item for item in data}


# ─────────────────────────────────────────────────────────────
# HYBRID SEARCH  (FIX #3 + #4)
# ─────────────────────────────────────────────────────────────
def hybrid_search(
    vector_results: List[Tuple[Document, float]],
    query: str,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion over vector rank and BM25 rank.
    Returns list sorted best-first (highest RRF score first).
    FIX #3: score is now the RRF value itself (not 1/rrf).
    FIX #4: sorted descending so best results come first.
    """
    if not ENABLE_HYBRID_SEARCH or len(vector_results) < 2:
        return vector_results[:top_k]

    corpus_texts      = [doc.page_content for doc, _ in vector_results]
    tokenized_corpus  = [t.lower().split() for t in corpus_texts]
    bm25              = BM25Okapi(tokenized_corpus)
    bm25_scores       = bm25.get_scores(query.lower().split())

    K = 60  # standard RRF constant
    fused: Dict[str, Dict] = {}

    # Vector rank contribution
    for rank, (doc, vscore) in enumerate(vector_results):
        # FIX #1 (chunk_id in metadata) means this lookup now works
        cid = doc.metadata.get("chunk_id", f"vec_{rank}")
        fused.setdefault(cid, {"doc": doc, "rrf": 0.0, "vscore": vscore})
        fused[cid]["rrf"] += 1.0 / (K + rank + 1)

    # BM25 rank contribution
    bm25_order = sorted(range(len(bm25_scores)),
                        key=lambda i: bm25_scores[i], reverse=True)
    for rank, idx in enumerate(bm25_order):
        cid = vector_results[idx][0].metadata.get("chunk_id", f"bm25_{idx}")
        fused.setdefault(cid, {"doc": vector_results[idx][0], "rrf": 0.0, "vscore": 0.0})
        fused[cid]["rrf"] += 1.0 / (K + rank + 1)

    # FIX #4: sort descending by RRF score (higher = better)
    ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)

    # Return (Document, rrf_score) — callers that compare scores still work
    return [(item["doc"], item["rrf"]) for item in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────
# NEIGHBOR EXPANSION  (FIX #5)
# ─────────────────────────────────────────────────────────────
def expand_with_neighbors(
    results: List[Dict],
    graph_retriever: GraphRetriever,
    metadata_store: Dict[str, Dict],
) -> List[Dict]:
    """
    Append neighboring chunks (prev / next) for context.
    FIX #5: neighbor text is fetched from metadata_store, not copied from original.
    """
    if not ENABLE_GRAPH_CONTEXT:
        return results

    expanded  = []
    seen      = set()

    for res in results:
        if res["chunk_id"] not in seen:
            seen.add(res["chunk_id"])
            expanded.append(res)

        neighbors = graph_retriever.get_chunk_neighbors(res["chunk_id"])

        for direction, ids in [("prev", neighbors["prev"]),
                                ("next", neighbors["next"])]:
            for nbr_id in ids:
                if nbr_id in seen:
                    continue
                seen.add(nbr_id)

                # FIX #5 — fetch real text from metadata store
                nbr_meta = metadata_store.get(nbr_id, {})
                nbr_text = nbr_meta.get("chunk", "")   # pipeline stores key "chunk"

                if not nbr_text:
                    # chunk not in metadata (shouldn't happen, but be safe)
                    continue

                expanded.append({
                    **res,                          # inherit faculty/dept/doc info
                    "chunk_id":   nbr_id,
                    "chunk_text": nbr_text,
                    "score":      res["score"] * 0.85,   # small penalty for neighbor
                    "is_neighbor": True,
                    "neighbor_direction": direction,
                    # override metadata fields from the neighbor's own record
                    "document_name": nbr_meta.get("title",      res["document_name"]),
                    "file_name":     nbr_meta.get("file",       res["file_name"]),
                    "chunk_index":   nbr_meta.get("chunk_index", -1),
                    "academic_score": nbr_meta.get("academic_score", 0.0),
                })

    return expanded


# ─────────────────────────────────────────────────────────────
# FILE TYPE HELPER
# ─────────────────────────────────────────────────────────────
def _file_type(meta: Dict, doc: Document) -> str:
    source    = meta.get("source") or doc.metadata.get("source", "")
    file_type = meta.get("file_type") or doc.metadata.get("file_type", "")
    if source == "scraper":
        return "web"
    return file_type or "document"


# ─────────────────────────────────────────────────────────────
# MAIN RETRIEVAL
# ─────────────────────────────────────────────────────────────
def retrieve(
    query:             str,
    top_k:             int            = 5,
    faculty:           Optional[str]  = None,
    department:        Optional[str]  = None,
    min_academic_score: float         = MIN_ACADEMIC_SCORE,
) -> List[Dict]:
    """
    Full enhanced retrieval:
      1. Vector similarity search (with optional faculty/department filter)
      2. Hybrid re-rank via RRF
      3. Metadata enrichment  (uses chunk_id now correctly stored)
      4. Academic score filter (default 0.0 = off)
      5. Neighbor expansion   (with real neighbor text)
      6. Final sort descending by score
    """
    vector_store    = load_vector_store()
    metadata_store  = load_metadata_store()
    graph_retriever = GraphRetriever()

    # ── Step 1: build Chroma filter ──────────────────────────
    filter_dict: Dict = {}
    if faculty:
        filter_dict["faculty"] = faculty
    if department:
        filter_dict["department"] = department

    # ── Step 2: vector search ────────────────────────────────
    candidates = top_k * 3   # fetch extra so hybrid + filter still leave enough
    try:
        if filter_dict:
            raw = vector_store.similarity_search_with_score(
                query, k=candidates, filter=filter_dict
            )
        else:
            raw = vector_store.similarity_search_with_score(query, k=candidates)
    except Exception as e:
        log.warning(f"Filtered search failed, retrying without filter: {e}")
        raw = vector_store.similarity_search_with_score(query, k=candidates)

    if not raw:
        log.warning("Vector store returned no results.")
        graph_retriever.close()
        return []

    # ── Step 3: hybrid re-rank ───────────────────────────────
    raw = hybrid_search(raw, query, top_k * 2)

    # ── Step 4: enrich + filter ──────────────────────────────
    enriched: List[Dict] = []
    for doc, score in raw:
        # FIX #1: chunk_id is now in metadata (pipeline must also be fixed — see note below)
        chunk_id  = doc.metadata.get("chunk_id", "unknown")
        full_meta = metadata_store.get(chunk_id, {})

        # academic score: prefer metadata store value, fall back to Chroma metadata
        acad = float(
            full_meta.get("academic_score",
                          doc.metadata.get("academic_score", 0.5))
        )
        if acad < min_academic_score:
            log.debug(f"Skipping {chunk_id}: academic_score {acad:.2f} < {min_academic_score}")
            continue

        enriched.append({
            "chunk_id":      chunk_id,
            "document_name": full_meta.get("title")      or doc.metadata.get("title",      "Unknown"),
            "file_name":     full_meta.get("file")       or doc.metadata.get("file",       "Unknown"),
            "file_path":     full_meta.get("file_path",  ""),
            "file_type":     _file_type(full_meta, doc),
            "faculty":       full_meta.get("faculty")    or doc.metadata.get("faculty",    "Unknown"),
            "department":    full_meta.get("department") or doc.metadata.get("department", "Unknown"),
            "source":        full_meta.get("source")     or doc.metadata.get("source",     "unknown"),
            "url":           full_meta.get("url")        or doc.metadata.get("url",        ""),
            "score":         score,
            "chunk_text":    doc.page_content,
            "academic_score": acad,
            "chunk_index":   doc.metadata.get("chunk_index",
                             full_meta.get("chunk_index", -1)),
            "is_neighbor":   False,
        })

    if not enriched:
        log.warning("All candidates were filtered out (academic_score). "
                    "Try lowering MIN_ACADEMIC_SCORE.")
        graph_retriever.close()
        return []

    # ── Step 5: expand with neighbors ────────────────────────
    enriched = expand_with_neighbors(enriched[:top_k], graph_retriever, metadata_store)

    graph_retriever.close()

    # ── Step 6: sort descending (best score first) ───────────
    # For hybrid RRF scores: higher = better.
    # For raw Chroma distance scores: lower = better.
    # After hybrid_search the score IS rrf (higher=better), so sort descending.
    enriched.sort(key=lambda x: x["score"], reverse=True)

    return enriched[:top_k]


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────
def save_results(results: List[Dict], query: str, output_path: Path = Path(OUTPUT_FILE)):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("RAG RETRIEVAL RESULTS\n")
        f.write(f"Query         : {query}\n")
        f.write(f"Total results : {len(results)}\n")
        f.write(f"Graph context : {'ON' if ENABLE_GRAPH_CONTEXT else 'OFF'}\n")
        f.write(f"Hybrid search : {'ON' if ENABLE_HYBRID_SEARCH else 'OFF'}\n")
        f.write("=" * 75 + "\n\n")

        for idx, res in enumerate(results, 1):
            f.write("=" * 60 + "\n")
            f.write(f"RESULT #{idx}"
                    + (" [NEIGHBOR — " + res.get("neighbor_direction","?") + "]"
                       if res.get("is_neighbor") else "") + "\n")
            f.write("=" * 60 + "\n")
            f.write(f"Document      : {res['document_name']}\n")
            f.write(f"Chunk ID      : {res['chunk_id']}\n")
            f.write(f"File Name     : {res['file_name']}\n")
            f.write(f"Faculty       : {res['faculty']}\n")
            f.write(f"Department    : {res['department']}\n")
            f.write(f"Academic Score: {res['academic_score']:.3f}\n")
            f.write(f"Chunk Index   : {res['chunk_index']}\n")
            f.write(f"Score         : {res['score']:.6f}\n")
            f.write("\nContent:\n")
            f.write("-" * 60 + "\n")
            f.write(res["chunk_text"] + "\n")
            f.write("-" * 60 + "\n\n")

    log.info(f"Results saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def rag_retrieve(
    query:             str,
    top_k:             int           = 5,
    faculty:           Optional[str] = None,
    department:        Optional[str] = None,
    min_academic_score: float        = MIN_ACADEMIC_SCORE,
) -> List[Dict]:
    log.info(f"Query: {query}")
    if faculty:    log.info(f"Faculty filter   : {faculty}")
    if department: log.info(f"Department filter: {department}")

    results = retrieve(query, top_k, faculty, department, min_academic_score)

    if not results:
        log.warning("No results returned.")
        return []

    save_results(results, query)

    log.info("\n" + "=" * 60)
    log.info("RETRIEVAL SUMMARY")
    log.info("=" * 60)
    for i, r in enumerate(results, 1):
        tag = f" [NEIGHBOR-{r.get('neighbor_direction','')}]" if r.get("is_neighbor") else ""
        log.info(f"{i}. {r['document_name']}{tag}  score={r['score']:.4f}  acad={r['academic_score']:.2f}")

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag.py \"query\" [--faculty FAC] [--department DEPT] [--top-k N]")
        sys.exit(1)

    query      = sys.argv[1]
    faculty    = None
    department = None
    top_k      = 5

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--faculty"    and i + 1 < len(sys.argv): faculty    = sys.argv[i+1]; i += 2
        elif sys.argv[i] == "--department" and i + 1 < len(sys.argv): department = sys.argv[i+1]; i += 2
        elif sys.argv[i] == "--top-k"     and i + 1 < len(sys.argv): top_k      = int(sys.argv[i+1]); i += 2
        else: i += 1

    rag_retrieve(query, top_k=top_k, faculty=faculty, department=department)