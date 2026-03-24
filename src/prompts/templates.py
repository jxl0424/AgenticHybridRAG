"""
Prompt templates for RAG queries.
"""

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the provided context.

IMPORTANT RULES:
1. Base your answer on the provided context.
2. If the context contains NO information about the subject of the question, respond with: "I don't have enough information to answer that question based on the provided documents."
3. If the context discusses the subject of the question, you MUST use it to answer — including hypothetical or counterfactual questions ("what if X were NOT true?"). Reason from what the context says about X to answer the hypothetical. Do not refuse when relevant context is present.
4. Do NOT make up specific facts, numbers, or citations not present in the context.
5. Do NOT refer to the context as "the document" or "the text" - just answer the question directly."""

# Template for user query with context
QUERY_TEMPLATE = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer using the information in the context above
- Only say you don't have enough information if the context contains NO information about the subject
- For hypothetical or counterfactual questions, reason from what the context says about the subject — do not refuse
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
