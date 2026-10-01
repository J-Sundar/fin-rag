from langchain_core.documents import Document
from src.ingestion.chunker import _attach_page_numbers
from src.generation.llm_chain import _get_top_header, format_context
from src.utils.config import PAGE_BREAK_MARKER, REFUSAL_MESSAGE


def test_get_top_header_hierarchy():
    meta_h3 = {"Header 1": "Chap 1", "Header 2": "Sec 2", "Header 3": "Part 3"}
    assert _get_top_header(meta_h3) == "Part 3"

    meta_h2 = {"Header 1": "Chap 1", "Header 2": "Sec 2"}
    assert _get_top_header(meta_h2) == "Sec 2"

    meta_h1 = {"Header 1": "Chap 1"}
    assert _get_top_header(meta_h1) == "Chap 1"

    assert _get_top_header({}) == ""


def test_format_context_with_page_and_header():
    doc1 = Document(
        page_content="Content clause 1",
        metadata={"source": "rbi_kyc.md", "page": 4, "Header 2": "CDD Guidelines"},
    )
    doc2 = Document(
        page_content="Content clause 2",
        metadata={"source": "rbi_lending.md"},
    )
    formatted = format_context([doc1, doc2])

    assert "[rbi_kyc.md, p.4] — CDD Guidelines" in formatted
    assert "Content clause 1" in formatted
    assert "[rbi_lending.md]" in formatted
    assert "Content clause 2" in formatted
    assert "---" in formatted


def test_attach_page_numbers():
    parent_text = (
        f"Page 1 start {PAGE_BREAK_MARKER} Page 2 start {PAGE_BREAK_MARKER} Page 3 start"
    )
    header_doc = Document(page_content=parent_text, metadata={})

    sub_doc1 = Document(page_content="Page 1 start", metadata={})
    sub_doc2 = Document(page_content="Page 2 start", metadata={})
    sub_doc3 = Document(page_content="Page 3 start", metadata={})

    sub_docs = [sub_doc1, sub_doc2, sub_doc3]
    _attach_page_numbers(header_doc, sub_docs, doc_start_page=1)

    assert sub_doc1.metadata["page"] == 1
    assert sub_doc2.metadata["page"] == 2
    assert sub_doc3.metadata["page"] == 3
    assert PAGE_BREAK_MARKER not in sub_doc1.page_content


def test_is_correct_refusal():
    from evaluate import is_correct_refusal

    exact_refusal = REFUSAL_MESSAGE
    assert is_correct_refusal(exact_refusal) is True

    case_variation = REFUSAL_MESSAGE.lower()
    assert is_correct_refusal(case_variation) is True

    with_period = f"{REFUSAL_MESSAGE}."
    assert is_correct_refusal(with_period) is True

    hallucination = "The current repo rate is 6.5% according to the latest announcement."
    assert is_correct_refusal(hallucination) is False


def test_retrieval_metrics():
    from evaluate import compute_hit, compute_rank

    docs = [
        Document(page_content="a", metadata={"source": "doc_a.md"}),
        Document(page_content="b", metadata={"source": "doc_b.md"}),
        Document(page_content="c", metadata={"source": "doc_c.md"}),
    ]

    assert compute_hit(docs, "doc_b.md") is True
    assert compute_hit(docs, "doc_z.md") is False

    assert compute_rank(docs, "doc_a.md") == 1
    assert compute_rank(docs, "doc_b.md") == 2
    assert compute_rank(docs, "doc_c.md") == 3
    assert compute_rank(docs, "doc_z.md") is None


def test_rerank_documents_mock():
    from src.retrieval.reranker import rerank_documents

    class MockCrossEncoder:
        def predict(self, pairs):
            return [10.0 if "relevant" in p[1] else -5.0 for p in pairs]

    docs = [
        Document(page_content="noise paragraph", metadata={"source": "doc_noise.md"}),
        Document(page_content="relevant clause", metadata={"source": "doc_target.md", "Header 1": "Section A"}),
    ]

    reranked = rerank_documents(
        query="What is the clause?",
        docs=docs,
        top_k=2,
        model=MockCrossEncoder(),
    )

    assert len(reranked) == 2
    assert reranked[0].metadata["source"] == "doc_target.md"
    assert reranked[0].metadata["rerank_score"] == 10.0
    assert reranked[1].metadata["source"] == "doc_noise.md"
    assert reranked[1].metadata["rerank_score"] == -5.0
    assert rerank_documents("query", [], top_k=2, model=MockCrossEncoder()) == []
