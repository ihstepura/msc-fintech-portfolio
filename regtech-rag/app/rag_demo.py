"""
Multi-Regulation RAG — Web Interface
=====================================
Covers: DORA (EU) · NIS2 (EU) · PS21/3 (UK)

Run with:  python multi_reg_app.py
Opens at:  http://localhost:7860

Requirements:
    pip install gradio llama-index llama-index-llms-groq
                llama-index-llms-google-genai
                llama-index-embeddings-huggingface
                llama-index-vector-stores-chroma
                llama-index-retrievers-bm25
                chromadb sentence-transformers torch
                rank-bm25 PyStemmer google-generativeai
"""

import os
import re
import time
import json
import torch
import gradio as gr
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY",   "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY",  "")
GEMINI_MODEL   = "models/gemini-flash-latest"

BASE_DIR    = Path(__file__).parent        
CHUNKS_FILE = BASE_DIR / "dora_chunks" / "all_chunks.json"
CONFIG_FILE = BASE_DIR / "index_config.json"

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SETUP
# ─────────────────────────────────────────────────────────────────────────────

print("Initialising Multi-Regulation RAG pipeline...")

from llama_index.core import Settings, StorageContext, VectorStoreIndex, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.llms.groq import Groq
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb

device = "cuda" if torch.cuda.is_available() else "cpu"

# Router LLM: Groq 8b — prevents LlamaIndex OpenAI fallback, never used for generation
router_llm   = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
Settings.llm = router_llm

# Generation LLM: Gemini Flash — 1M context, no token limits
llm = GoogleGenAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
print(f"✓ Generation LLM: {GEMINI_MODEL}")

# Embedding model
config           = json.loads(CONFIG_FILE.read_text())
EMBED_MODEL_NAME = config["embed_model"]
CHROMA_DIR       = (BASE_DIR.parent / config["chroma_dir"]).resolve()
CHROMA_COLL      = config["chroma_collection"]

print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    query_instruction="Represent this sentence for searching relevant passages: ",
    embed_batch_size=16,
    device=device,
)
Settings.embed_model = embed_model
print("✓ Embedding model ready")

# Chroma index
chroma_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
chroma_collection = chroma_client.get_collection(CHROMA_COLL)
vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
index             = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)
print(f"✓ Index loaded — {chroma_collection.count()} vectors")

# BM25 nodes — load from all_chunks.json (all 3 regulations)
raw_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
all_nodes  = [
    TextNode(
        id_=c["chunk_id"], text=c["text_with_header"],
        metadata={
            "regulation":    c.get("regulation", "dora"),
            "jurisdiction":  c.get("jurisdiction", "eu"),
            "source_doc":    c["source_doc"],
            "doc_type":      c["doc_type"],
            "level":         c.get("level", 0),
            "article_num":   c["article_num"] if c["article_num"] is not None else -1,
            "article_title": c["article_title"],
            "section_ref":   c.get("section_ref") or "",
            "paragraph_idx": c["paragraph_idx"],
            "topic_tags":    "|".join(c["topic_tags"]),
            "entity_scope":  "|".join(c["entity_scope"]),
            "token_estimate":c["token_estimate"],
        }
    ) for c in raw_chunks
]

try:
    import Stemmer
    stemmer = Stemmer.Stemmer("english")
except ImportError:
    stemmer = None

by_reg = {}
for c in raw_chunks:
    r = c.get("regulation", "?")
    by_reg[r] = by_reg.get(r, 0) + 1
print(f"✓ BM25 nodes ready ({len(all_nodes)} chunks) — {by_reg}")

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def make_retriever(level=None, doc_type=None, regulation=None, jurisdiction=None, top_k=8):
    filters = []
    if level        is not None: filters.append(MetadataFilter(key="level",        value=level,        operator=FilterOperator.EQ))
    if doc_type     is not None: filters.append(MetadataFilter(key="doc_type",     value=doc_type,     operator=FilterOperator.EQ))
    if regulation   is not None: filters.append(MetadataFilter(key="regulation",   value=regulation,   operator=FilterOperator.EQ))
    if jurisdiction is not None: filters.append(MetadataFilter(key="jurisdiction", value=jurisdiction, operator=FilterOperator.EQ))
    if filters:
        return VectorIndexRetriever(index=index, similarity_top_k=top_k,
                                    filters=MetadataFilters(filters=filters))
    sem = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    bm  = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=top_k,
                                       stemmer=stemmer, language="english")
    return QueryFusionRetriever(
        retrievers=[sem, bm], llm=Settings.llm,
        similarity_top_k=top_k, num_queries=1,
        mode="reciprocal_rerank", use_async=False, verbose=False,
    )

def route_query(query: str, regulation: str = None):
    q = query.lower()
    if regulation:
        reg_scope = regulation
    else:
        detected = []
        if any(k in q for k in ["dora", "2022/2554", "financial entit", "eba", "esma", "eiopa"]): detected.append("dora")
        if any(k in q for k in ["nis2", "nis 2", "2022/2555", "essential entit", "important entit", "enisa"]): detected.append("nis2")
        if any(k in q for k in ["ps21", "ps21/3", "ss1/21", "fg22", "fca", "pra", "impact tolerance"]): detected.append("ps21_3")
        reg_scope = detected[0] if len(detected) == 1 else None

    if reg_scope == "nis2":
        if any(k in q for k in ["implementing regulation", "technical requirement", "2024/2690"]): return make_retriever(regulation="nis2", level=2, top_k=10), "NIS2 Level 2"
        if any(k in q for k in ["enisa", "guideline", "security measure"]): return make_retriever(regulation="nis2", level=3, top_k=8), "NIS2 Level 3"
        return make_retriever(regulation="nis2", top_k=10), "NIS2 full corpus"

    if reg_scope == "ps21_3":
        if any(k in q for k in ["ss1/21", "supervisory statement"]): return make_retriever(regulation="ps21_3", doc_type="supervisory_statement", top_k=8), "PS21/3 Supervisory Statement"
        if any(k in q for k in ["fg22", "finalised guidance"]): return make_retriever(regulation="ps21_3", doc_type="guidance", top_k=8), "PS21/3 FCA Guidance"
        return make_retriever(regulation="ps21_3", top_k=10), "PS21/3 full corpus"

    reg_filter = "dora" if reg_scope == "dora" else None
    if any(k in q for k in ["regulatory technical standard", " rts ", "rts on", "technical standard", "delegated regulation"]): return make_retriever(level=2, regulation=reg_filter, top_k=12), "Level 2 (RTS/ITS)"
    if any(k in q for k in ["tlpt", "penetration test", "threat-led", "tiber"]): return make_retriever(regulation=reg_filter, top_k=12), "Full corpus (TLPT)"
    if any(k in q for k in ["subcontract", "chain", "group entit", "third-party provider"]): return make_retriever(regulation=reg_filter, top_k=12), "Full corpus (adversarial)"
    if any(k in q for k in ["guideline", "aggregate", "oversight cooperation"]): return make_retriever(level=3, regulation=reg_filter, top_k=8), "Level 3 (guidelines)"
    if any(k in q for k in ["template", "reporting form"]): return make_retriever(doc_type="ITS", regulation=reg_filter, top_k=8), "ITS only"
    if any(k in q for k in ["article ", "regulation ", "directive ", " shall ", " must "]): return make_retriever(level=1, regulation=reg_filter, top_k=8), "Level 1 (main regulation)"
    if reg_filter: return make_retriever(regulation=reg_filter, top_k=8), f"{reg_filter.upper()} full corpus"
    return make_retriever(top_k=10), "All regulations"

def retrieve(query: str, regulation: str = None):
    retriever, scope = route_query(query, regulation=regulation)
    return retriever.retrieve(query), scope

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

DORA_QA_PROMPT = PromptTemplate("""\
You are a regulatory compliance assistant specialising in EU DORA \
(Digital Operational Resilience Act - Regulation (EU) 2022/2554) \
and its Level 2 regulatory technical standards and Level 3 guidelines.

CONTEXT (retrieved regulatory provisions):
---------------------------------------------
{context_str}
---------------------------------------------

QUESTION: {query_str}

INSTRUCTIONS:
1. Base your answer ONLY on the context above. Never use outside knowledge.
2. Cite every statement: [Source: <document_name>, Article <N>]
3. Distinguish: SHALL/MUST = mandatory, SHOULD = recommended, MAY = optional.
4. Flag entity-specific applicability where relevant.
5. STRICT REFUSAL: If the primary topic is NOT addressed in the context, respond ONLY with: \
'INSUFFICIENT CONTEXT: The DORA regulatory corpus does not directly address this question.'
6. Structure: direct answer first, then detail with citations.

ANSWER:\
""")

MULTI_REG_PROMPT = PromptTemplate("""\
You are a regulatory compliance expert specialising in {regulation_label}.

CONTEXT (retrieved regulatory provisions):
---------------------------------------------
{context_str}
---------------------------------------------

QUESTION: {query_str}

INSTRUCTIONS:
1. Answer ONLY from the context above. Never use outside knowledge.
2. Cite every statement: [Source: <document>, Article <N>] for EU docs, \
[Source: <document>, Section <ref>] for UK PS21/3 docs.
3. Distinguish: SHALL/MUST = mandatory, SHOULD = recommended, MAY = optional.
4. STRICT REFUSAL: If the topic falls outside {regulation_label} scope, respond ONLY with: \
'INSUFFICIENT CONTEXT: This topic falls outside the scope of {regulation_label}.'
5. Structure: direct answer first, then detail with citations.

ANSWER:\
""")

COMPARISON_PROMPT = PromptTemplate("""\
You are a multi-jurisdiction regulatory compliance expert covering DORA, NIS2, and PS21/3.
You have been given three regulatory answers to the same question.

DORA ANSWER:
{dora_answer}

NIS2 ANSWER:
{nis2_answer}

PS21/3 ANSWER:
{ps213_answer}

QUESTION: {query_str}

Write a CROSS-FRAMEWORK ANALYSIS structured exactly as follows:

**OVERLAP:** [Where all three frameworks align or require the same thing]

**CONFLICT:** [Where frameworks differ — be specific about what each requires]

**GAP:** [Where one framework covers something others do not address]

**JURISDICTION NOTE:** [Which regulation applies to which type of firm, \
and which firms face dual/triple compliance obligations]

Be specific. Cite articles and section references. If no meaningful conflict or gap exists, say 'None identified'.

ANALYSIS:\
""")

HYDE_PROMPT = """\
You are an expert in regulatory law covering DORA, NIS2, and PS21/3.

Write a short passage (4-6 sentences) that could plausibly appear in whichever \
of these regulations is most relevant to the question below. Write as actual \
regulatory text using SHALL/SHOULD/MAY. Do NOT answer conversationally.

Question: {question}

Hypothetical regulatory passage:"""

REG_LABELS = {
    "dora":   "EU DORA (Regulation 2022/2554)",
    "nis2":   "EU NIS2 (Directive 2022/2555)",
    "ps21_3": "UK PS21/3 (FCA/PRA Operational Resilience)",
}

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def llm_complete_with_retry(prompt: str, max_retries: int = 4) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return llm.complete(prompt).text.strip()
        except Exception as e:
            err = str(e)
            is_rate = any(k in err for k in [
                "RESOURCE_EXHAUSTED", "429", "quota", "rate_limit", "Too Many Requests"
            ])
            if not is_rate or attempt == max_retries:
                raise
            match = re.search(r"retry_delay.*?(\d+)", err)
            wait  = int(match.group(1)) + 2 if match else 30
            print(f"    Rate limit ({attempt}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)

def _build_context(nodes):
    parts = []
    for node in nodes:
        m   = node.metadata
        src = m.get("source_doc", "Unknown")
        art = m.get("article_num", -1)
        sec = m.get("section_ref", "")
        ttl = m.get("article_title", "")
        if art != -1:  hdr = f"[{src}, Article {art} — {ttl}]"
        elif sec:      hdr = f"[{src}, Section {sec} — {ttl}]"
        else:          hdr = f"[{src}]"
        parts.append(f"{hdr}\n{node.text}")
    return "\n\n".join(parts)

def _extract_sources(nodes):
    sources, seen = [], set()
    for node in nodes:
        m   = node.metadata
        art = m.get("article_num", -1)
        sec = m.get("section_ref", "")
        src = m.get("source_doc", "")
        key = f"{src}_{art}_{sec}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_doc":    src,
                "doc_type":      m.get("doc_type", ""),
                "regulation":    m.get("regulation", ""),
                "jurisdiction":  m.get("jurisdiction", ""),
                "level":         m.get("level", 0),
                "article_num":   art,
                "section_ref":   sec,
                "article_title": m.get("article_title", ""),
                "score":         round(getattr(node, "score", 0.0) or 0.0, 4),
            })
    return sources

def ask_single(question: str, regulation: str = "dora") -> dict:
    """Single-regulation ask with HyDE retrieval."""
    t0 = time.time()

    # HyDE: generate hypothesis, retrieve on it
    hypothesis = llm_complete_with_retry(HYDE_PROMPT.format(question=question))
    retriever, scope = route_query(question, regulation=regulation)
    nodes = retriever.retrieve(hypothesis)

    context_str = _build_context(nodes)
    reg_label   = REG_LABELS.get(regulation, regulation)

    if regulation == "dora":
        prompt = DORA_QA_PROMPT.format(context_str=context_str, query_str=question)
    else:
        prompt = MULTI_REG_PROMPT.format(
            regulation_label=reg_label,
            context_str=context_str,
            query_str=question
        )

    answer  = llm_complete_with_retry(prompt)
    latency = round(time.time() - t0, 2)

    return {
        "answer":     answer,
        "sources":    _extract_sources(nodes),
        "scope":      scope,
        "hypothesis": hypothesis,
        "latency_s":  latency,
        "regulation": regulation,
    }

def ask_comparison(question: str) -> dict:
    """
    Side-by-side comparison across DORA, NIS2, and PS21/3.
    4 LLM calls: 3 single-regulation answers + 1 comparison synthesis.
    """
    t0      = time.time()
    results = {}

    for reg in ["dora", "nis2", "ps21_3"]:
        results[reg] = ask_single(question, regulation=reg)
        time.sleep(4)   # stay within Gemini 15 RPM

    analysis = llm_complete_with_retry(COMPARISON_PROMPT.format(
        dora_answer  = results["dora"]["answer"],
        nis2_answer  = results["nis2"]["answer"],
        ps213_answer = results["ps21_3"]["answer"],
        query_str    = question,
    ))

    return {
        "dora":       results["dora"],
        "nis2":       results["nis2"],
        "ps21_3":     results["ps21_3"],
        "analysis":   analysis,
        "latency_s":  round(time.time() - t0, 2),
    }

print("✓ Pipeline ready\n")

# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

REG_COLOURS = {
    "dora":      "#3b82f6",   # blue
    "nis2":      "#10b981",   # green
    "ps21_3":    "#f59e0b",   # amber
    "cross_reg": "#8b5cf6",   # purple
}

REG_DISPLAY = {
    "dora":   "DORA (EU)",
    "nis2":   "NIS2 (EU)",
    "ps21_3": "PS21/3 (UK)",
}

LEVEL_LABELS = {
    1: "L1 — Regulation",
    2: "L2 — RTS/ITS",
    3: "L3 — Guidelines",
}

def format_sources_single(sources: list, regulation: str = None) -> str:
    if not sources:
        return "_No sources retrieved._"

    lines = ["**Retrieved Sources:**\n"]
    by_level = {}
    for s in sources:
        by_level.setdefault(s["level"], []).append(s)

    for lv in sorted(by_level.keys()):
        label = LEVEL_LABELS.get(lv, f"Level {lv}")
        lines.append(f"**{label}**")
        for s in by_level[lv]:
            art   = s["article_num"]
            sec   = s["section_ref"]
            title = s["article_title"]
            src   = s["source_doc"]
            dtype = s["doc_type"].upper()
            score = s["score"]
            if art != -1:
                lines.append(f"- `[{dtype}]` {src} — **Art {art}**: {title} _(score: {score})_")
            elif sec:
                lines.append(f"- `[{dtype}]` {src} — **Sec {sec}**: {title} _(score: {score})_")
            else:
                lines.append(f"- `[{dtype}]` {src} _(score: {score})_")
        lines.append("")

    return "\n".join(lines)

def format_sources_comparison(results: dict) -> str:
    """Format sources for all three regulations side by side."""
    lines = []
    for reg in ["dora", "nis2", "ps21_3"]:
        r = results.get(reg, {})
        sources = r.get("sources", [])
        label   = REG_DISPLAY.get(reg, reg)
        lines.append(f"### {label}\n")
        lines.append(format_sources_single(sources, regulation=reg))
        lines.append("")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO RESPONSE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def respond_single(question: str, regulation: str, history: list):
    if not question.strip():
        yield history, "_Ask a question to see retrieved sources._", "_HyDE hypothesis will appear here._"
        return

    # Show thinking state
    history_thinking = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": f"_Retrieving from **{REG_DISPLAY.get(regulation, regulation)}**..._"},
    ]
    yield history_thinking, "", ""

    try:
        result = ask_single(question, regulation=regulation)
    except Exception as e:
        history += [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": f"⚠️ **Error:** {str(e)[:300]}"},
        ]
        yield history, "", ""
        return

    answer     = result["answer"]
    sources_md = format_sources_single(result["sources"], regulation=regulation)
    scope      = result["scope"]
    latency    = result["latency_s"]
    hypothesis = result["hypothesis"]
    reg_label  = REG_DISPLAY.get(regulation, regulation)

    meta = (
        f"\n\n---\n"
        f"_Regulation: **{reg_label}** · "
        f"Scope: **{scope}** · "
        f"HyDE active · "
        f"Latency: **{latency}s**_"
    )
    history += [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer + meta},
    ]
    hyp_md = f"**HyDE Hypothesis** _(used as retrieval query)_:\n\n_{hypothesis[:500]}..._"

    yield history, sources_md, hyp_md

def respond_comparison(question: str, history: list):
    if not question.strip():
        yield history, "_Ask a question to see retrieved sources._", "_Analysis will appear here._"
        return

    # Show thinking
    history_thinking = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": "_Running comparison across DORA · NIS2 · PS21/3 (4 LLM calls — ~2 min)..._"},
    ]
    yield history_thinking, "", ""

    try:
        result = ask_comparison(question)
    except Exception as e:
        history += [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": f"⚠️ **Error:** {str(e)[:300]}"},
        ]
        yield history, "", ""
        return

    dora_ans  = result["dora"]["answer"]
    nis2_ans  = result["nis2"]["answer"]
    ps213_ans = result["ps21_3"]["answer"]
    analysis  = result["analysis"]
    latency   = result["latency_s"]

    # Format combined answer for chat
    combined = (
        f"### 🇪🇺 DORA\n{dora_ans}\n\n"
        f"---\n\n"
        f"### 🇪🇺 NIS2\n{nis2_ans}\n\n"
        f"---\n\n"
        f"### 🇬🇧 PS21/3\n{ps213_ans}\n\n"
        f"---\n\n"
        f"### Cross-Framework Analysis\n{analysis}"
        f"\n\n---\n_4 LLM calls · Total latency: **{latency}s**_"
    )

    history += [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": combined},
    ]

    sources_md  = format_sources_comparison(result)
    analysis_md = f"### Cross-Framework Analysis\n\n{analysis}"

    yield history, sources_md, analysis_md

def respond(question: str, scope_choice: str, history: list):
    """Route to single or comparison based on scope selector."""
    scope_map = {
        "dora":   ("single",     "dora"),
        "nis2":   ("single",     "nis2"),
        "ps21_3": ("single",     "ps21_3"),
        "all":    ("comparison", None),
    }
    mode, regulation = scope_map.get(scope_choice, ("single", "dora"))

    if mode == "comparison":
        yield from respond_comparison(question, history)
    else:
        yield from respond_single(question, regulation, history)

def clear_all():
    return [], "_Ask a question to see retrieved sources._", "_HyDE hypothesis / analysis will appear here._"

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = {
    "dora": [
        "What are the reporting timelines for a major ICT-related incident?",
        "Is TLPT mandatory for all financial entities?",
        "What contractual provisions must be included in ICT third-party contracts?",
        "Does DORA apply to small non-interconnected investment firms?",
        "What does DORA say about GDPR data subject rights?",
    ],
    "nis2": [
        "What are the NIS2 incident reporting timelines under Article 23?",
        "Which entities qualify as essential entities under NIS2?",
        "Is multi-factor authentication mandatory under NIS2?",
        "What supply chain security obligations does NIS2 impose?",
        "What does NIS2 say about impact tolerances for business services?",
    ],
    "ps21_3": [
        "What is an impact tolerance under PS21/3?",
        "What does SS1/21 require for a firm's self-assessment?",
        "Is scenario testing mandatory for all FCA/PRA firms?",
        "Which firms are subject to PS21/3 operational resilience requirements?",
        "What are the PS21/3 requirements for critical third parties?",
    ],
    "all": [
        "What are the incident reporting obligations and timelines?",
        "What third-party risk management is required?",
        "What governance and board responsibilities are required?",
        "What testing requirements apply and how do they compare?",
        "What obligations apply to a UK bank that also operates in the EU?",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark terminal aesthetic (extended from Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:          #0d0f12;
    --surface:     #161a1f;
    --surface2:    #1e2329;
    --border:      #2a3040;
    --accent:      #3b82f6;
    --accent-dim:  #1d4ed8;
    --green:       #10b981;
    --amber:       #f59e0b;
    --purple:      #8b5cf6;
    --red:         #ef4444;
    --text:        #e2e8f0;
    --text-muted:  #64748b;
    --text-dim:    #94a3b8;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'IBM Plex Sans', sans-serif;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Header */
.app-header {
    padding: 24px 32px 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.app-title {
    font-family: var(--mono) !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: var(--accent) !important;
    letter-spacing: 0.05em;
    margin: 0 !important;
}
.app-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: var(--mono);
}
.badge {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: var(--mono);
    color: var(--text-dim);
    margin-right: 6px;
    margin-top: 8px;
}
.badge.active { border-color: var(--green); color: var(--green); }
.badge.dora   { border-color: var(--accent); color: var(--accent); }
.badge.nis2   { border-color: var(--green);  color: var(--green); }
.badge.ps213  { border-color: var(--amber);  color: var(--amber); }

/* Scope selector */
.scope-selector .wrap { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 4px !important; }
.scope-selector label {
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.scope-selector label:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}
.scope-selector input[type=radio] { display: none !important; }
.scope-selector input[type=radio]:checked + span,
.scope-selector input[type=radio]:checked ~ span {
    color: var(--accent) !important;
}
.scope-selector input[type=radio]:checked + * {
    color: var(--accent) !important;
}
/* Active radio button label highlight */
.scope-selector .selected, 
.scope-selector [data-testid*="radio"] label.selected,
.scope-selector label:has(input:checked) {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(59,130,246,0.08) !important;
}
.scope-selector span { color: var(--text) !important; font-size: 0.92rem !important; font-family: var(--mono) !important; }

/* Panel labels */
.panel-label {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 10px 16px 8px !important;
    border-bottom: 1px solid var(--border) !important;
    background: var(--surface) !important;
}

/* Chatbot */
.chatbot-wrap, .chatbot-wrap > div, .chatbot-wrap .wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.chatbot-wrap *, .chatbot-wrap .message, .chatbot-wrap .message * {
    color: var(--text) !important;
}
.chatbot-wrap .message.user, .chatbot-wrap [data-testid="user"] {
    background: var(--accent-dim) !important;
    border-radius: 8px 8px 2px 8px !important;
    font-size: 0.9rem !important;
    color: var(--text) !important;
}
.chatbot-wrap .message.bot, .chatbot-wrap [data-testid="bot"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px 8px 8px 8px !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    color: var(--text) !important;
}

/* Input */
.input-row textarea,
.input-row textarea:focus,
.input-row .wrap,
.input-row > div,
.input-row .block {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.92rem !important;
    padding: 12px 16px !important;
    resize: none !important;
}
/* Force dark on all Gradio input wrappers */
.gradio-container input,
.gradio-container textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
}
.input-row textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* Buttons */
.btn-ask {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 12px 20px !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
.btn-ask:hover { background: var(--accent-dim) !important; }
.btn-clear {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
}
.btn-clear:hover { border-color: var(--red) !important; color: var(--red) !important; }

/* Example buttons */
.example-btn {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-dim) !important;
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
    padding: 8px 12px !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: border-color 0.15s, color 0.15s !important;
    margin-bottom: 4px !important;
    width: 100% !important;
}
.example-btn:hover { border-color: var(--accent) !important; color: var(--text) !important; }

/* Sources and info panels */
.sources-box {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    min-height: 80px !important;
    color: var(--text) !important;
}
.info-box {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    min-height: 120px !important;
    max-height: 340px !important;
    overflow-y: auto !important;
    color: var(--text) !important;
}
.sources-box *, .info-box * { color: var(--text) !important; }
.sources-box code, .info-box code {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 1px 5px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--amber) !important;
}

/* Session list */
.session-list {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-top: 6px;
}
.session-empty {
    padding: 16px;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: var(--mono);
    line-height: 1.6;
}
.session-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.12s;
}
.session-item:last-child { border-bottom: none; }
.session-item:hover { background: rgba(59,130,246,0.07); }
.session-num {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    min-width: 24px;
}
.session-q {
    font-size: 0.78rem;
    color: var(--text-dim);
    font-family: var(--sans);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Scope selector label */
.scope-selector > label > span { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

/* Force all Gradio blocks to dark */
.gradio-container .block, 
.gradio-container .form,
.gradio-container .wrap {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
"""

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

SCOPE_OPTIONS = [
    ("🇪🇺  DORA",   "dora"),
    ("🇪🇺  NIS2",   "nis2"),
    ("🇬🇧  PS21/3", "ps21_3"),
    ("🌐  ALL",     "all"),
]

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Multi-Reg RAG") as app:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">MULTI_REG_RAG //</div>
        <div class="app-subtitle">DORA · NIS2 · PS21/3 — Multi-Jurisdiction Regulatory Intelligence</div>
        <div style="margin-top:8px">
            <span class="badge active">● LIVE</span>
            <span class="badge dora">DORA (EU)</span>
            <span class="badge nis2">NIS2 (EU)</span>
            <span class="badge ps213">PS21/3 (UK)</span>
            <span class="badge">798 vectors</span>
            <span class="badge">HyDE enabled</span>
            <span class="badge">Gemini Flash</span>
        </div>
    </div>
    """)

    # ── Scope selector (full width, above columns) ────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="panel-label">Regulation Scope</div>')
            scope_selector = gr.Radio(
                choices=SCOPE_OPTIONS,
                value="dora",
                label="Select Regulation Scope",
                elem_classes="scope-selector",
            )

    with gr.Row(equal_height=False):

        # ── Left column: examples (dynamic per scope) + chat history ──────────
        with gr.Column(scale=1, min_width=240):
            gr.HTML('<div class="panel-label">Example Questions</div>')
            example_btns = []
            for i in range(5):
                btn = gr.Button(
                    EXAMPLES["dora"][i],
                    elem_classes="example-btn",
                )
                example_btns.append(btn)

            gr.HTML('<div class="panel-label" style="margin-top:16px">Chat Sessions</div>')
            new_session_btn = gr.Button("+ New Session", elem_classes="btn-clear")
            session_list    = gr.HTML(
                value='<div class="session-list"><div class="session-empty">No sessions yet.<br>Ask a question to begin.</div></div>'
            )

        # ── Centre column: chat ───────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<div class="panel-label">Conversation</div>')

            chatbot = gr.Chatbot(
                value=[],
                height=500,
                show_label=False,
                render_markdown=True,
                elem_classes="chatbot-wrap",
            )

            with gr.Row(elem_classes="input-row"):
                question_input = gr.Textbox(
                    placeholder="Ask a compliance question...",
                    show_label=False,
                    lines=2,
                    max_lines=4,
                    scale=5,
                )
                with gr.Column(scale=1, min_width=100):
                    ask_btn   = gr.Button("ASK →", elem_classes="btn-ask")
                    clear_btn = gr.Button("Clear",  elem_classes="btn-clear")

        # ── Right column: sources + info ──────────────────────────────────────
        with gr.Column(scale=2, min_width=280):
            gr.HTML('<div class="panel-label">Retrieved Sources</div>')
            sources_display = gr.Markdown(
                value="_Ask a question to see retrieved sources._",
                elem_classes="sources-box",
            )
            gr.HTML('<div class="panel-label" style="margin-top:12px">HyDE Hypothesis / Analysis</div>')
            info_display = gr.Markdown(
                value="_The HyDE hypothesis or cross-framework analysis will appear here._",
                elem_classes="info-box",
            )

    # ── Session history state ─────────────────────────────────────────────────
    # Stores list of {question, history, sources, info} dicts
    session_store       = gr.State([])
    active_session_idx  = gr.State(-1)     # index of current open session (-1 = new)
    selected_idx_input  = gr.Textbox(value="-1", visible=False, elem_id="sel-idx")

    # Inject JS so session-item onclick can trigger the hidden textbox change event
    gr.HTML("""
    <script>
    function restoreSession(idx) {
        // Find the hidden textbox by its elem_id and update its value
        const wrap = document.querySelector('#sel-idx textarea') ||
                     document.querySelector('#sel-idx input');
        if (!wrap) return;
        const nativeInputSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ) || Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
        nativeInputSetter.set.call(wrap, String(idx));
        wrap.dispatchEvent(new Event('input', { bubbles: true }));
    }
    </script>
    """)

    # ── Wire up scope selector → update examples ──────────────────────────────
    def update_examples(scope):
        examples = EXAMPLES.get(scope, EXAMPLES["dora"])
        return [gr.Button(value=examples[i]) for i in range(5)]

    scope_selector.input(
        fn=update_examples,
        inputs=scope_selector,
        outputs=example_btns,
    )

    # ── Wire up ask button and submit ─────────────────────────────────────────
    def on_ask(question, scope, history, sessions, active_idx):
        """
        Run pipeline and append to current conversation.
        Does NOT start a new session — that only happens via + New Session.
        If active_idx == -1 (no session yet), creates first session automatically.
        """
        if not question.strip():
            yield history, "", "", sessions, build_session_table(sessions), active_idx
            return

        final_history    = history
        final_sources_md = "_Ask a question to see retrieved sources._"
        final_info_md    = "_HyDE hypothesis / analysis will appear here._"

        for h, s, i in respond(question, scope, history):
            final_history    = h
            final_sources_md = s if s else final_sources_md
            final_info_md    = i if i else final_info_md
            yield h, s, i, sessions, build_session_table(sessions), active_idx

        # Update or create session
        if active_idx == -1 or active_idx >= len(sessions):
            # No active session — create first one
            sessions = sessions + [{
                "question": question[:60],
                "history":  final_history,
                "sources":  final_sources_md,
                "info":     final_info_md,
                "scope":    scope,
            }]
            active_idx = len(sessions) - 1
        else:
            # Update existing active session with latest state
            sessions = list(sessions)
            sessions[active_idx] = {
                **sessions[active_idx],
                "history": final_history,
                "sources": final_sources_md,
                "info":    final_info_md,
            }

        yield final_history, final_sources_md, final_info_md, sessions, build_session_table(sessions), active_idx

    def restore_session(sessions, evt: gr.SelectData):
        """Restore a previous session when its item is clicked in the list."""
        # Note: with HTML session list we use a hidden index input instead
        return sessions, build_session_table(sessions)

    def load_session(idx, sessions):
        """Load session at index idx into the chat view."""
        idx = int(idx)
        if idx < 0 or idx >= len(sessions):
            return [], "_No session selected._", "_No session selected._", idx
        s = sessions[idx]
        return s["history"], s["sources"], s["info"], idx

    def build_session_table(sessions):
        if not sessions:
            return '<div class="session-list"><div class="session-empty">No sessions yet.<br>Ask a question to begin.</div></div>'
        items = ""
        for i, s in enumerate(sessions):
            reg = s.get("scope", "dora")
            dot_col = {"dora": "var(--accent)", "nis2": "var(--green)",
                       "ps21_3": "var(--amber)", "all": "var(--purple)"}.get(reg, "var(--text-muted)")
            q = s["question"][:55] + ("..." if len(s["question"]) > 55 else "")
            items += (
                f'<div class="session-item" onclick="restoreSession({i})" title="{s['question']}">' +
                f'<span class="session-num" style="color:{dot_col}">#{i+1}</span>' +
                f'<span class="session-q">{q}</span>' +
                '</div>'
            )
        return f'<div class="session-list">{items}</div>'

    def new_session(sessions):
        """Start fresh chat — previous sessions saved in list."""
        return [], "_Ask a question to see retrieved sources._", "_HyDE hypothesis / analysis will appear here._", sessions, build_session_table(sessions), -1

    ask_btn.click(
        fn=on_ask,
        inputs=[question_input, scope_selector, chatbot, session_store, active_session_idx],
        outputs=[chatbot, sources_display, info_display, session_store, session_list, active_session_idx],
    ).then(fn=lambda: gr.Textbox(value=""), outputs=question_input)

    question_input.submit(
        fn=on_ask,
        inputs=[question_input, scope_selector, chatbot, session_store, active_session_idx],
        outputs=[chatbot, sources_display, info_display, session_store, session_list, active_session_idx],
    ).then(fn=lambda: gr.Textbox(value=""), outputs=question_input)

    clear_btn.click(fn=clear_all, outputs=[chatbot, sources_display, info_display])

    new_session_btn.click(
        fn=new_session,
        inputs=session_store,
        outputs=[chatbot, sources_display, info_display, session_store, session_list, active_session_idx],
    )

    # Restore session when user clicks a session item (updates hidden index)
    selected_idx_input.change(
        fn=load_session,
        inputs=[selected_idx_input, session_store],
        outputs=[chatbot, sources_display, info_display, active_session_idx],
    )

    # ── Wire example buttons → fill input ─────────────────────────────────────
    for i, btn in enumerate(example_btns):
        btn.click(
            fn=lambda scope, idx=i: EXAMPLES.get(scope, EXAMPLES["dora"])[idx],
            inputs=scope_selector,
            outputs=question_input,
        )

if __name__ == "__main__":
    print("Starting Multi-Regulation RAG web interface...")
    print("Open in browser: http://localhost:7860\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )