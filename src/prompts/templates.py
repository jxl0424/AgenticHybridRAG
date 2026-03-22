"""
Prompt templates for RAG queries.
"""

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the provided context.

IMPORTANT RULES:
1. Base your answer strictly on the provided context. Do not reason beyond what the context explicitly states.
2. If the context does not mention the specific entity, system, or concept asked about at all, respond with: "I don't have enough information to answer that question based on the provided documents."
3. If the context directly discusses the subject of the question, you MAY reason from that information to answer hypothetical or counterfactual variants (e.g., "what if X were not true?"). Do NOT infer from merely topically adjacent content — the context must actually discuss the specific subject being asked about.
4. Do NOT make up specific facts, numbers, or citations not present in the context.
5. Do NOT refer to the context as "the document" or "the text" - just answer the question directly."""

# Template for user query with context
QUERY_TEMPLATE = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer only from facts explicitly stated in the context above
- If the context does not mention the question's specific subject at all, say you don't have enough information
- If the context does discuss the subject, you may reason from it to answer hypothetical or counterfactual questions
- Do not reason from tangentially related content where the specific subject is absent
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
