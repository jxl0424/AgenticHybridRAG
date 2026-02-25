"""
Prompt templates for RAG queries.
"""

# System prompt for the RAG assistant - STRICT about not hallucinating
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. If the provided context does NOT contain relevant information to answer the question, you MUST respond with "I don't have enough information to answer that question based on the provided documents."
2. Do NOT make up or hallucinate any information that is not explicitly in the context.
3. Do NOT refer to the context as "the document" or "the text" - just answer the question directly.
4. If asked about topics unrelated to the context (e.g., cooking, weather, general knowledge), say you don't have that information."""

# Template for user query with context
QUERY_TEMPLATE = """Use the following context to answer the question. If the context is not relevant to the question, say you don't know.

Context: 
{context}

Question: {question}

Instructions:
- Answer ONLY based on the context above
- If the context doesn't contain relevant information, say "I don't have enough information..."
- Keep your answer concise and accurate"""

# Template for building context from retrieved chunks
CONTEXT_TEMPLATE = "\n\n".join("- {chunk}" for chunk in "{chunks}")


def build_query_prompt(question: str, contexts: list[str]) -> str:
    """Build a query prompt with the given question and context chunks."""
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    return QUERY_TEMPLATE.format(context=context_block, question=question)


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    """Build the messages list for LLM inference."""
    user_content = build_query_prompt(question, contexts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
