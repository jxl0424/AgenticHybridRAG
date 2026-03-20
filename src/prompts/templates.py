"""
Prompt templates for RAG queries.
"""

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the provided context.

IMPORTANT RULES:
1. Base your answer primarily on the provided context.
2. If the context is topically related but does not fully answer the question, reason from what is available and note any gaps.
3. Only respond with "I don't have enough information to answer that question based on the provided documents." if the context is completely unrelated to the topic of the question.
4. Do NOT make up specific facts, numbers, or citations not present in the context.
5. Do NOT refer to the context as "the document" or "the text" - just answer the question directly."""

# Template for user query with context
QUERY_TEMPLATE = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the context above
- If the context is related but incomplete, provide a partial answer using what is available
- Only say you don't have enough information if the context is entirely off-topic
- Keep your answer concise and accurate"""

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
