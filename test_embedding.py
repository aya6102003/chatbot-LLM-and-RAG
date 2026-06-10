#!/usr/bin/env python3
"""
test_gemini_embedding.py
════════════════════════
Test Gemini embeddings (Google AI)
"""

import os
import numpy as np

# ── API KEY ─────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ضع المفتاح:")
    print("export GEMINI_API_KEY='YOUR_KEY'")
    exit()

# ── Import ──────────────────────────────
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

MODEL = "models/gemini-embedding-001"

queries = {
    "en": "what are computer science specializations?",
    "fr": "quelles sont les spécialités en informatique?",
    "ar": "ما هي تخصصات علوم الحاسوب؟"
}

vectors = {}

print("\n🔄 Testing Gemini Embeddings...\n")

for lang, text in queries.items():
    try:
        response = client.models.embed_content(
            model=MODEL,
            contents=text
        )

        vec = np.array(response.embeddings[0].values)
        vec = vec / np.linalg.norm(vec)

        vectors[lang] = vec

        print(f"[{lang}] OK  dim={len(vec)}")

    except Exception as e:
        print(f"❌ Error [{lang}]:", e)

# ── Similarity test ─────────────────────
print("\n📊 Similarity results:\n")

def cos(a, b):
    return float(np.dot(a, b))

print("en ↔ fr:", cos(vectors["en"], vectors["fr"]))
print("ar ↔ en:", cos(vectors["ar"], vectors["en"]))
print("ar ↔ fr:", cos(vectors["ar"], vectors["fr"]))

avg = (cos(vectors["ar"], vectors["en"]) + cos(vectors["ar"], vectors["fr"])) / 2

print("\n🎯 Arabic score avg:", avg)
