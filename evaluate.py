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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.pipeline import load_pipeline
from src.retrieval.query_expansion import expand_query
from src.retrieval.reranker import rerank_documents
from src.generation.llm_chain import format_context
from src.utils.config import (
    setup_logging,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
    RETRIEVAL_CANDIDATES_K,
    RERANKER_MODEL_NAME,
    REFUSAL_MESSAGE,
)

setup_logging()
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set.json")
RESULTS_DIR     = Path("data/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── LLM Judge ───────────────────────────────────────────────────────────────

_JUDGE_PROMPT = PromptTemplate.from_template("""
You are a strict evaluator of a Retrieval-Augmented Generation (RAG) system 
built over Reserve Bank of India (RBI) regulatory documents.

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


def invoke_with_retry(chain, payload: dict, max_retries: int = 5):
    """Executes a LangChain runnable with exponential backoff on Groq 429 errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err:
                wait = 7 * (attempt + 1)
                logger.warning(f"Groq rate limit hit. Sleeping {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise e
    return chain.invoke(payload)


def judge_answer(question: str, context: str, answer: str, llm: ChatGroq) -> dict:
    """
    Sends question + context + answer to Groq and returns faithfulness/relevance scores.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    chain = _JUDGE_PROMPT | llm | StrOutputParser()
    try:
        raw = invoke_with_retry(chain, {
            "question": question,
            "context":  context,
            "answer":   answer,
        }).strip()
    except Exception as e:
        logger.warning(f"Judge invocation failed: {e}")
        return {"faithfulness": None, "relevance": None, "reasoning": "judge call failed"}

    try:
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
    print(f"  Doc-Hit Rate @{TOP_K}   : {hit_rate:.1%}  ({hits}/{total})")
    print(f"  MRR                : {mrr:.3f}")

    if not retrieval_only:
        faith_scores = [r["faithfulness"] for r in results if r.get("faithfulness")]
        rel_scores   = [r["relevance"]    for r in results if r.get("relevance")]
        if faith_scores:
            print(f"  Avg Faithfulness   : {sum(faith_scores)/len(faith_scores):.2f} / 5")
        if rel_scores:
            print(f"  Avg Relevance      : {sum(rel_scores)/len(rel_scores):.2f} / 5")

    print()
    print("  Doc-Hit Rate by Category:")
    for cat, (cat_hit, cat_total) in cat_hits.items():
        bar_fill = int((cat_hit / cat_total) * 20)
        bar      = "█" * bar_fill + "░" * (20 - bar_fill)
        print(f"    {cat:<26} [{bar}]  {cat_hit}/{cat_total}")

    if neg_results:
        correct = sum(1 for r in neg_results if r["correctly_refused"])
        n       = len(neg_results)
        print()
        print(f"  Guardrail Refusal Check (negative) : {correct}/{n}"
              f"  ({correct/n:.1%})")
        if correct < n:
            print("  ⚠ At least one out-of-corpus question got a hallucinated "
                  "answer instead of a refusal — check the flagged rows above.")
    elif not retrieval_only:
        print()
        print("  Groundedness (refusal accuracy) : no negative items in golden set")

    print("=" * 60)


# ─── Main Evaluation Loop ─────────────────────────────────────────────────────

def run_evaluation(expand: bool = True, retrieval_only: bool = False, rerank: bool = True):
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
          f"| rerank={'on' if rerank else 'off'} "
          f"| judge={'off' if retrieval_only else 'on'}\n")

    # ── Positive items: retrieval + generation quality ──
    for item in positive_items:
        qid      = item["id"]
        question = item["question"]
        expected = item["expected_source"]
        category = item["category"]

        print(f"  [{qid:02d}/{len(positive_items)}] {question[:70]}...")

        search_query = expand_query(question, groq_llm) if expand else question

        if rerank:
            candidates = ensemble_retriever.invoke(search_query)[:RETRIEVAL_CANDIDATES_K]
            retrieved_docs = rerank_documents(query=question, docs=candidates, top_k=TOP_K)
        else:
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
            answer       = invoke_with_retry(llm_chain, {"context": context_text, "question": question})
            scores       = judge_answer(question, context_text, answer, groq_llm)

            result.update({
                "answer":       answer,
                "faithfulness": scores.get("faithfulness"),
                "relevance":    scores.get("relevance"),
                "reasoning":    scores.get("reasoning"),
            })

        results.append(result)
        time.sleep(2)  # stay within Groq free-tier rate limits

    # ── Negative items: groundedness / refusal check ──
    for item in negative_items:
        qid      = item["id"]
        question = item["question"]
        category = item.get("category", "Out-of-corpus")

        print(f"  [{qid:02d}/{len(negative_items)}] (negative) {question[:60]}...")

        search_query = expand_query(question, groq_llm) if expand else question

        if rerank:
            candidates = ensemble_retriever.invoke(search_query)[:RETRIEVAL_CANDIDATES_K]
            retrieved_docs = rerank_documents(query=question, docs=candidates, top_k=TOP_K)
        else:
            retrieved_docs = ensemble_retriever.invoke(search_query)[:TOP_K]

        context_text = format_context(retrieved_docs)
        answer       = invoke_with_retry(llm_chain, {"context": context_text, "question": question})
        refused      = is_correct_refusal(answer)

        neg_results.append({
            "id":                qid,
            "question":          question,
            "category":          category,
            "search_query":      search_query,
            "retrieved_sources": [d.metadata.get("source") for d in retrieved_docs],
            "answer":            answer,
            "correctly_refused": refused,
        })
        time.sleep(2)

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
                    "expand":          expand,
                    "rerank":          rerank,
                    "reranker_model":  RERANKER_MODEL_NAME if rerank else None,
                    "retrieval_only":  retrieval_only,
                    "top_k":           TOP_K,
                    "candidates_k":    RETRIEVAL_CANDIDATES_K if rerank else TOP_K,
                    "model":           GROQ_MODEL,
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
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking.",
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
        rerank=not args.no_rerank,
    )