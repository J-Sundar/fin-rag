import logging
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

_EXPANSION_PROMPT = PromptTemplate.from_template("""
You are an expert search assistant for Indian financial regulations (RBI/SEBI).
Your goal is to enhance the user question into an effective search query for retrieving official circular clauses.

Rules:
1. Always preserve the core entities, actors, and subjects from the original question (e.g. "payment aggregators", "LSP", "small accounts").
2. Append 3 to 6 official regulatory terms or formal synonyms (e.g. "Customer Due Diligence CDD", "prudential norms").
3. Keep the query concise: under 20 words total. Do NOT list circular years or irrelevant laws.
4. Output ONLY the search query.

Question: {question}
Optimized Query:""")


def expand_query(user_query: str, llm: Optional[ChatGroq] = None) -> str:
    """Translates a user question into legal keywords for lexical retrieval."""
    if not user_query or not user_query.strip():
        return ""

    if llm is None:
        llm = ChatGroq(
            temperature=LLM_TEMPERATURE,
            model_name=GROQ_MODEL,
            api_key=GROQ_API_KEY,
        )

    chain = _EXPANSION_PROMPT | llm | StrOutputParser()
    expanded = chain.invoke({"question": user_query.strip()}).strip()

    if not expanded:
        logger.warning("Query expansion returned empty string; falling back to original query.")
        return user_query

    return expanded

