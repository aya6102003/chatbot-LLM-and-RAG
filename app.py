#!/usr/bin/env python3
"""
Web interface for the University RAG Assistant (Flask + Tailwind CSS)
Corrected version — JavaScript POST handling fixed.
"""

import logging
from flask import Flask, request, jsonify
from llm import ask

# Optional: pre-warm the RAG pipeline at startup
# from rag import warmup_retriever
# warmup_retriever()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ── Corrected HTML template with reliable form handling ─────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>University RAG Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .markdown a { color: #2563eb; text-decoration: underline; }
        .markdown p { margin-bottom: 0.75rem; }
        .loader {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2563eb;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen flex flex-col">
    <header class="bg-white shadow-sm border-b">
        <div class="max-w-4xl mx-auto py-4 px-4 sm:px-6 flex items-center justify-between">
            <h1 class="text-xl font-bold text-gray-800">🎓 University RAG Assistant</h1>
            <span class="text-sm text-gray-500">Farhat Abbas University Sétif 1</span>
        </div>
    </header>
    <main class="flex-grow max-w-4xl mx-auto w-full px-4 py-8">
        <!-- Search box -->
        <form id="search-form" class="flex gap-2 mb-8">
            <input type="text" id="query-input" name="query"
                   placeholder="Ask a question in French, Arabic, or English..."
                   class="flex-1 rounded-lg border border-gray-300 px-4 py-3 text-gray-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                   autofocus required>
            <button type="submit"
                    class="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-3 rounded-lg shadow-sm transition-colors">
                Ask
            </button>
        </form>

        <!-- Loading indicator -->
        <div id="loading" class="hidden flex items-center gap-3 mb-6 text-gray-600">
            <div class="loader"></div>
            <span>Searching and generating answer...</span>
        </div>

        <!-- Results area -->
        <div id="results" class="hidden">
            <!-- Answer card -->
            <div class="bg-white rounded-xl shadow-md p-6 mb-6">
                <h2 class="text-lg font-semibold text-gray-800 mb-3">Answer</h2>
                <div id="answer-content" class="prose max-w-none text-gray-700 markdown"></div>
            </div>

            <!-- Sources -->
            <div class="bg-white rounded-xl shadow-md p-6">
                <h2 class="text-lg font-semibold text-gray-800 mb-3">Sources & Documents</h2>
                <div id="sources-list" class="space-y-2"></div>
                <p id="no-sources" class="text-gray-500 italic hidden">No external sources were referenced.</p>
            </div>
        </div>

        <!-- Error message -->
        <div id="error" class="hidden bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 mt-6">
            <strong>Error:</strong> <span id="error-text"></span>
        </div>
    </main>

    <footer class="bg-white border-t py-4 text-center text-sm text-gray-500">
        This system strictly grounds answers in official university data. No hallucinations.
    </footer>

    <script>
        // Wait for the DOM to be fully loaded before attaching the event
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('search-form');
            const loading = document.getElementById('loading');
            const resultsDiv = document.getElementById('results');
            const errorDiv = document.getElementById('error');
            const answerContent = document.getElementById('answer-content');
            const sourcesList = document.getElementById('sources-list');
            const noSources = document.getElementById('no-sources');

            // Override form submission
            form.addEventListener('submit', async function(e) {
                e.preventDefault();                // Stop normal GET redirect
                const query = document.getElementById('query-input').value.trim();
                if (!query) return;

                // Reset UI
                resultsDiv.classList.add('hidden');
                errorDiv.classList.add('hidden');
                loading.classList.remove('hidden');

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await response.json();
                    loading.classList.add('hidden');

                    if (!response.ok) {
                        showError(data.error || 'Request failed');
                        return;
                    }

                    // Display answer
                    answerContent.innerHTML = convertToHtml(data.answer);
                    // Display sources
                    renderSources(data.sources);
                    resultsDiv.classList.remove('hidden');
                } catch (err) {
                    loading.classList.add('hidden');
                    showError('Unable to connect to server. Please try again.');
                }
            });

            function showError(msg) {
                document.getElementById('error-text').textContent = msg;
                errorDiv.classList.remove('hidden');
            }

            function renderSources(sources) {
                sourcesList.innerHTML = '';
                if (!sources || sources.length === 0) {
                    noSources.classList.remove('hidden');
                    return;
                }
                noSources.classList.add('hidden');
                sources.forEach(function(src) {
                    const div = document.createElement('div');
                    div.className = "flex items-center gap-2 p-2 rounded-lg hover:bg-gray-50";
                    div.innerHTML = `
                        <svg class="w-5 h-5 text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                        </svg>
                        <a href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer"
                           class="text-blue-600 hover:underline font-medium">${escapeHtml(src.title)}</a>
                        <span class="text-gray-400 text-sm truncate ml-2">${escapeHtml(src.url)}</span>
                    `;
                    sourcesList.appendChild(div);
                });
            }

            function escapeHtml(text) {
                const map = {
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#039;'
                };
                return text.replace(/[&<>"']/g, function(m) { return map[m]; });
            }

            function convertToHtml(text) {
                // Simple Markdown-like conversion: URLs become links, newlines become <p>
                const linked = text.replace(
                    /(https?:\/\/[^\s]+)/g,
                    '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-blue-600 underline">$1</a>'
                );
                const paragraphs = linked.split('\n').filter(function(p) { return p.trim() !== ''; });
                return paragraphs.map(function(p) { return '<p>' + p + '</p>'; }).join('');
            }
        });
    </script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML_TEMPLATE

@app.route("/ask", methods=["POST"])
def handle_ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    try:
        result = ask(query=query)
        return jsonify(result)
    except Exception as e:
        logging.error("Error processing query: %s", e)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5500)