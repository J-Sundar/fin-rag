"""
evaluate.py

Evaluates the RAG pipeline against a golden question set.

Two kinds of golden-set items are supported, distinguished by "type":

  "positive" (default if "type" is omitted) — a question with a known
  expected_source. Measures retrieval and generation quality:
    · Hit Rate @k  : fraction of questions where expected source is in top-k
    · MRR          : Mean Reciprocal Rank
    · Faithfulness / Relevance (1-5, Groq-as-judge)

  "negative" — an out-of-corpus question with expected_source=null. There is
  no "right" document to retrieve, so retrieval isn't scored. What's scored
  is whether generation correctly refuses rather than hallucinating an answer
  from loosely-related retrieved chunks. This is the groundedness check.

Usage:
    python evaluate.py                   # full evaluation
    python evaluate.py --no-expand       # skip query expansion (faster, tests raw retrieval)
    python evaluate.py --retrieval-only  # skip generation and LLM judge (negative items are
                                          # skipped entirely in this mode — they need generation)
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.ingestion.chunker import process_markdown_chunks
from src.retrieval.vector_store import get_vector_store
from src.generation.llm_chain import get_llm_chain
from src.utils.config import (
    setup_logging,
    PROCESSED_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
    REFUSAL_MESSAGE,
)

setup_logging()
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set.json")
RESULTS_DIR     = Path("data/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Pipeline Initialisation ─────────────────────────────────────────────────

def load_pipeline():
    """Builds the same ensemble retriever and LLM chain used in app.py."""
    logger.info("Loading chunks for BM25...")
    chunks = process_markdown_chunks(str(PROCESSED_DIR))

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K

    logger.info("Connecting to Qdrant...")
    vector_store     = get_vector_store()
    qdrant_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.4, 0.6],
    )

    llm_chain = get_llm_chain()
    logger.info("Pipeline ready.")
    return ensemble_retriever, llm_chain


# ─── Query Expansion ─────────────────────────────────────────────────────────

def expand_query(user_query: str, llm: ChatGroq) -> str:
    prompt = PromptTemplate.from_template("""
You are an expert in Indian financial regulations (RBI/SEBI).
Rewrite the question below as a compact, space-separated string of legal 
keywords, synonyms, and formal legalese that would appear verbatim in an 
official RBI/SEBI document.

Rules:
1. Convert colloquial terms to formal equivalents.
2. Expand numerical concepts to words.
3. Output ONLY keywords separated by spaces. No sentences. No punctuation.

Question: {question}
Keywords:""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": user_query}).strip()


# ─── LLM Judge ───────────────────────────────────────────────────────────────

_JUDGE_PROMPT = PromptTemplate.from_template("""
You are a strict evaluator of a Retrieval-Augmented Generation (RAG) system 
built over Indian financial regulatory documents (RBI/SEBI).

Given a question, the retrieved context, and a generated answer, score two dimensions:

FAITHFULNESS (1-5): Does the answer make ONLY claims directly supported by the context?
  5 = every claim is verbatim or directly inferable from context
  3 = mostly grounded, minor extrapolation
  1 = significant hallucination or claims with no basis in context

RELEVANCE (1-5): Does the answer actually address what the question asked?
  5 = directly and completely answers the question
  3 = partially answers or addresses a related but different point
  1 = does not answer the question at all

---
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
---

Respond with ONLY a valid JSON object. No explanation, no markdown, no preamble.
{{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one concise sentence>"}}
""")


def judge_answer(question: str, context: str, answer: str, llm: ChatGroq) -> dict:
    """
    Sends question + context + answer to Groq and returns faithfulness/relevance scores.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    chain = _JUDGE_PROMPT | llm | StrOutputParser()
    raw   = chain.invoke({
        "question": question,
        "context":  context,
        "answer":   answer,
    }).strip()

    try:
        # Strip accidental markdown fences before parsing
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"Judge returned malformed JSON: {raw[:100]}")
        return {"faithfulness": None, "relevance": None, "reasoning": "parse error"}


# ─── Groundedness Check (negative items) ─────────────────────────────────────

def is_correct_refusal(answer: str) -> bool:
    """
    True if the model correctly declined an out-of-corpus question.
    Uses substring containment (not exact equality) so minor whitespace or
    trailing punctuation from the LLM doesn't cause a false negative.
    """
    return REFUSAL_MESSAGE.strip().lower() in answer.strip().lower()


# ─── Retrieval Metrics (positive items only) ─────────────────────────────────

def compute_hit(retrieved_docs, expected_source: str) -> bool:
    """True if the expected source filename appears in any retrieved chunk."""
    sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
    return expected_source in sources


def compute_rank(retrieved_docs, expected_source: str) -> int | None:
    """
    Returns the 1-based rank of the first chunk from the expected source.
    Returns None if the source was not retrieved at all.
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if doc.metadata.get("source", "") == expected_source:
            return i
    return None


def format_context(docs) -> str:
    """Stitches retrieved chunks into a single context string for the LLM,
    labeling each with source + page so the model can cite them."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page")
        label  = f"[{source}, p.{page}]" if page else f"[{source}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Display Helpers ─────────────────────────────────────────────────────────

def _col(text, width, align="left"):
    text = str(text) if text is not None else "—"
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


def print_results_table(results: list[dict], retrieval_only: bool):
    hit_w, rank_w, faith_w, rel_w = 6, 5, 13, 10

    if retrieval_only:
        header = (f"  {'#':>3}  {'Category':<22}  {'Question':<52}  "
                  f"{'Hit?':^{hit_w}}  {'Rank':^{rank_w}}")
        divider = "  " + "─" * (len(header) - 2)
        print("\n" + divider)
        print(header)
        print(divider)
        for r in results:
            hit_mark = "  ✓  " if r["hit"] else "  ✗  "
            rank_str = str(r["rank"]) if r["rank"] else "—"
            q_short  = r["question"][:50] + ".." if len(r["question"]) > 52 else r["question"]
            print(f"  {r['id']:>3}  {_col(r['category'], 22)}  {_col(q_short, 52)}"
                  f"  {_col(hit_mark, hit_w)}  {_col(rank_str, rank_w, 'right')}")
    else:
        header = (f"  {'#':>3}  {'Category':<22}  {'Question':<44}  "
                  f"{'Hit?':^{hit_w}}  {'Rank':^{rank_w}}  "
                  f"{'Faithfulness':^{faith_w}}  {'Relevance':^{rel_w}}")
        divider = "  " + "─" * (len(header) - 2)
        print("\n" + divider)
        print(header)
        print(divider)
        for r in results:
            hit_mark  = "  ✓  " if r["hit"] else "  ✗  "
            rank_str  = str(r["rank"]) if r["rank"] else "—"
            faith_str = f"{r['faithfulness']}/5" if r.get("faithfulness") else "—"
            rel_str   = f"{r['relevance']}/5"    if r.get("relevance")    else "—"
            q_short   = r["question"][:42] + ".." if len(r["question"]) > 44 else r["question"]
            print(f"  {r['id']:>3}  {_col(r['category'], 22)}  {_col(q_short, 44)}"
                  f"  {_col(hit_mark, hit_w)}  {_col(rank_str, rank_w, 'right')}"
                  f"  {_col(faith_str, faith_w, 'right')}  {_col(rel_str, rel_w, 'right')}")

    print(divider + "\n")


def print_negative_results(neg_results: list[dict]):
    """Prints the groundedness table for out-of-corpus questions."""
    if not neg_results:
        return

    refuse_w = 10
    header  = f"  {'#':>3}  {'Question':<62}  {'Refused?':^{refuse_w}}"
    divider = "  " + "─" * (len(header) - 2)
    print("\n" + divider)
    print("  GROUNDEDNESS CHECK — out-of-corpus questions")
    print(divider)
    print(header)
    print(divider)
    for r in neg_results:
        mark    = "   ✓   " if r["correctly_refused"] else "   ✗   "
        q_short = r["question"][:60] + ".." if len(r["question"]) > 62 else r["question"]
        print(f"  {r['id']:>3}  {_col(q_short, 62)}  {_col(mark, refuse_w)}")
    print(divider + "\n")


def print_summary(results: list[dict], neg_results: list[dict], retrieval_only: bool):
    total       = len(results)
    hits        = sum(1 for r in results if r["hit"])
    hit_rate    = hits / total if total else 0.0

    ranks       = [r["rank"] for r in results if r["rank"] is not None]
    mrr         = sum(1 / rank for rank in ranks) / total if total else 0.0

    # Per-category hit rate
    categories  = sorted(set(r["category"] for r in results))
    cat_hits    = {
        cat: (
            sum(1 for r in results if r["category"] == cat and r["hit"]),
            sum(1 for r in results if r["category"] == cat),
        )
        for cat in categories
    }

    print("=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions    : {total}")
    print(f"  Hit Rate @{TOP_K}       : {hit_rate:.1%}  ({hits}/{total})")
    print(f"  MRR                : {mrr:.3f}")

    if not retrieval_only:
        faith_scores = [r["faithfulness"] for r in results if r.get("faithfulness")]
        rel_scores   = [r["relevance"]    for r in results if r.get("relevance")]
        if faith_scores:
            print(f"  Avg Faithfulness   : {sum(faith_scores)/len(faith_scores):.2f} / 5")
        if rel_scores:
            print(f"  Avg Relevance      : {sum(rel_scores)/len(rel_scores):.2f} / 5")

    print()
    print("  Hit Rate by Category:")
    for cat, (cat_hit, cat_total) in cat_hits.items():
        bar_fill = int((cat_hit / cat_total) * 20)
        bar      = "█" * bar_fill + "░" * (20 - bar_fill)
        print(f"    {cat:<26} [{bar}]  {cat_hit}/{cat_total}")

    if neg_results:
        correct = sum(1 for r in neg_results if r["correctly_refused"])
        n       = len(neg_results)
        print()
        print(f"  Groundedness (refusal accuracy) : {correct}/{n}"
              f"  ({correct/n:.1%})")
        if correct < n:
            print("  ⚠ At least one out-of-corpus question got a hallucinated "
                  "answer instead of a refusal — check the flagged rows above.")
    elif not retrieval_only:
        print()
        print("  Groundedness (refusal accuracy) : no negative items in golden set")

    print("=" * 60)


# ─── Main Evaluation Loop ─────────────────────────────────────────────────────

def run_evaluation(expand: bool = True, retrieval_only: bool = False):
    logger.info("Loading golden set...")
    with open(GOLDEN_SET_PATH, "r") as f:
        golden_set = json.load(f)

    positive_items = [item for item in golden_set if item.get("type", "positive") == "positive"]
    negative_items = [item for item in golden_set if item.get("type") == "negative"]

    if negative_items and retrieval_only:
        logger.info(
            f"Skipping {len(negative_items)} negative/groundedness item(s) — "
            f"they require generation, which --retrieval-only disables."
        )
        negative_items = []

    ensemble_retriever, llm_chain = load_pipeline()

    # Shared Groq LLM instance for expansion + judging
    groq_llm = ChatGroq(
        temperature=LLM_TEMPERATURE,
        model_name=GROQ_MODEL,
        api_key=GROQ_API_KEY,
    )

    results     = []
    neg_results = []

    print(f"\nRunning evaluation — {len(positive_items)} positive"
          f" + {len(negative_items)} negative questions "
          f"| expand={'on' if expand else 'off'} "
          f"| judge={'off' if retrieval_only else 'on'}\n")

    # ── Positive items: retrieval + generation quality ──
    for item in positive_items:
        qid      = item["id"]
        question = item["question"]
        expected = item["expected_source"]
        category = item["category"]

        print(f"  [{qid:02d}/{len(positive_items)}] {question[:70]}...")

        search_query   = expand_query(question, groq_llm) if expand else question
        retrieved_docs = ensemble_retriever.invoke(search_query)[:TOP_K]

        hit  = compute_hit(retrieved_docs, expected)
        rank = compute_rank(retrieved_docs, expected)

        result = {
            "id":              qid,
            "question":        question,
            "category":        category,
            "expected_source": expected,
            "search_query":    search_query,
            "retrieved_sources": [d.metadata.get("source") for d in retrieved_docs],
            "hit":             hit,
            "rank":            rank,
        }

        if not retrieval_only:
            context_text = format_context(retrieved_docs)
            answer       = llm_chain.invoke({"context": context_text, "question": question})
            scores       = judge_answer(question, context_text, answer, groq_llm)

            result.update({
                "answer":       answer,
                "faithfulness": scores.get("faithfulness"),
                "relevance":    scores.get("relevance"),
                "reasoning":    scores.get("reasoning"),
            })

        results.append(result)
        time.sleep(1)  # stay within Groq free-tier rate limits

    # ── Negative items: groundedness / refusal check ──
    for item in negative_items:
        qid      = item["id"]
        question = item["question"]
        category = item.get("category", "Out-of-corpus")

        print(f"  [{qid:02d}/{len(negative_items)}] (negative) {question[:60]}...")

        search_query   = expand_query(question, groq_llm) if expand else question
        retrieved_docs = ensemble_retriever.invoke(search_query)
        context_text   = format_context(retrieved_docs)
        answer         = llm_chain.invoke({"context": context_text, "question": question})
        refused        = is_correct_refusal(answer)

        neg_results.append({
            "id":                qid,
            "question":          question,
            "category":          category,
            "search_query":      search_query,
            "retrieved_sources": [d.metadata.get("source") for d in retrieved_docs],
            "answer":            answer,
            "correctly_refused": refused,
        })
        time.sleep(1)

    # ── Output ──
    print_results_table(results, retrieval_only)
    print_negative_results(neg_results)
    print_summary(results, neg_results, retrieval_only)

    # Save full results with timestamp
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp":      timestamp,
                "config": {
                    "expand":         expand,
                    "retrieval_only": retrieval_only,
                    "top_k":          TOP_K,
                    "model":          GROQ_MODEL,
                },
                "results":          results,
                "negative_results": neg_results,
            },
            f,
            indent=2,
        )

    logger.info(f"Full results saved to {output_path}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the FinRAG pipeline.")
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Skip query expansion. Tests raw retrieval quality.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and LLM judge. Faster, measures retrieval only. "
             "Negative/groundedness items are skipped in this mode.",
    )
    args = parser.parse_args()

    run_evaluation(
        expand=not args.no_expand,
        retrieval_only=args.retrieval_only,
    )