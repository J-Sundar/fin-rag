"""
src/generation/llm_chain.py

Manages prompt construction and LLM inference via Groq.
Enforces strict grounding to prevent hallucinations.
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, REFUSAL_MESSAGE

logger = logging.getLogger(__name__)

# REFUSAL_MESSAGE is interpolated once here (not a LangChain template
# variable) so this prompt and evaluate.py's groundedness check can never
# drift out of sync — both read the same constant from config.py.
_PROMPT_TEMPLATE = """
You are a strict, analytical regulatory assistant for Indian Finance (RBI/SEBI).
Use ONLY the retrieved context below to answer the question.

Each context chunk is labeled with its source document and page number,
like [rbi_master_direction_kyc.md, p.12]. Where your answer draws on a
specific chunk, cite it inline in parentheses right after the claim, e.g.
"...must be updated every two years (rbi_master_direction_kyc.md, p.12)."
Only cite a source/page pair that actually appears in the context labels —
never invent one.

If the answer is not contained within the context, respond with exactly:
"{refusal_message}"
Do not use external knowledge or make inferences beyond what is written.

Context:
{{context}}

Question: {{question}}

Answer:
""".format(refusal_message=REFUSAL_MESSAGE)


def get_llm_chain():
    """
    Builds and returns the Groq-backed generation chain:
        PromptTemplate → ChatGroq → StrOutputParser
    """
    llm = ChatGroq(
        temperature=LLM_TEMPERATURE,
        model_name=GROQ_MODEL,
        api_key=GROQ_API_KEY,
    )

    prompt = PromptTemplate(
        template=_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return prompt | llm | StrOutputParser()


def _get_top_header(metadata: dict) -> str:
    """Returns the most specific section header from chunk metadata."""
    for level in ("Header 3", "Header 2", "Header 1"):
        if metadata.get(level):
            return metadata[level]
    return ""


def format_context(docs: list) -> str:
    """Stitches retrieved chunks into a labeled context string for generation."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        header = _get_top_header(doc.metadata)
        label = f"[{source}, p.{page}]" if page else f"[{source}]"
        label += f" — {header}" if header else ""
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
