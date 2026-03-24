"""
Evaluation metrics for RAG systems.
"""
import json
import math
import re
from typing import List, Dict, Any

from tenacity import retry, wait_exponential, stop_after_attempt

class RAGMetrics:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_with_retry(self, prompt: str, max_tokens: int = 10) -> str:
        """Call LLM with exponential backoff retries."""
        return self.llm.generate([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0)

    def calculate_faithfulness(self, query: str, context: List[str], answer: str) -> float:
        """
        Measures how much of the answer is supported by the context.
        Uses LLM to verify each claim in the answer against the context.
        """
        refusal_phrases = [
            "i don't have enough information",
            "i do not have enough information",
            "i don't know",
            "cannot answer",
            "no information",
        ]
        if any(p in answer.lower() for p in refusal_phrases):
            return 1.0

        if not self.llm:
            return 0.0

        context_str = "\n".join(context)
        prompt = f"""
        Task: Verify if the following answer is faithful to the provided context.
        
        Context:
        {context_str}
        
        Answer:
        {answer}
        
        Step 1: Break the answer down into distinct claims/statements.
        Step 2: For each claim, check if it is explicitly stated in or directly inferred from the context.
        
        Output a score between 0 and 1, where 1 means all claims are supported and 0 means no claims are supported.
        Return ONLY the numerical score.
        """
        try:
            response = self._generate_with_retry(prompt, max_tokens=10)
            score_match = re.search(r"(\d*\.?\d+)", response)
            return float(score_match.group(1)) if score_match else 0.0
        except Exception as e:
            print(f"Faithfulness metric error: {e}")
            return 0.0

    def calculate_relevance(self, query: str, answer: str) -> float:
        """
        Measures how relevant the answer is to the query.
        """
        if not self.llm:
            return 0.0
            
        prompt = f"""
        Task: Rate the relevance of the answer to the research query.

        Query: {query}
        Answer: {answer}

        A relevant answer directly addresses the query using scientific and technical knowledge.
        Output a score between 0 and 1. Return ONLY the numerical score.
        """
        try:
            response = self._generate_with_retry(prompt, max_tokens=10)
            score_match = re.search(r"(\d*\.?\d+)", response)
            return float(score_match.group(1)) if score_match else 0.0
        except Exception as e:
            print(f"Relevance metric error: {e}")
            return 0.0

    def calculate_context_precision(self, query: str, contexts: List[str], ground_truth_context: str) -> float:
        """
        Measures how many of the retrieved contexts are actually relevant.
        Uses keyword overlap between ground truth and retrieved chunks.
        """
        if not contexts:
            return 0.0

        # Extract meaningful content words from ground truth (strip URLs, short tokens)
        import re
        stop_words = {"the", "a", "an", "is", "in", "of", "to", "and", "or",
                      "for", "was", "are", "be", "that", "this", "it", "at",
                      "by", "on", "as", "from", "with", "not", "but", "its"}
        # Strip URLs and special chars, then tokenize
        clean_gt = re.sub(r'https?://\S+', ' ', ground_truth_context)
        clean_gt = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_gt)
        gt_words = {w.lower() for w in clean_gt.split()
                    if len(w) > 3 and w.lower() not in stop_words}

        if not gt_words:
            return 0.0

        relevant_count = 0
        for ctx in contexts:
            clean_ctx = re.sub(r'[^a-zA-Z0-9\s]', ' ', ctx)
            ctx_words = {w.lower() for w in clean_ctx.split()}
            # A context is relevant if it shares >= 10% of GT content words
            overlap = len(gt_words & ctx_words) / len(gt_words)
            if overlap >= 0.10:
                relevant_count += 1

        return relevant_count / len(contexts)

    def calculate_context_recall(self, ground_truth_answer: str, retrieved_contexts: List[str]) -> float:
        """
        Measures if the retrieved contexts contain enough information to reconstruct the ground truth answer.
        """
        if not self.llm:
            return 0.0
            
        context_str = "\n".join(retrieved_contexts)
        prompt = f"""
        Task: Can the ground truth answer be fully answered using ONLY the retrieved context?
        
        Ground Truth Answer: {ground_truth_answer}
        Retrieved Context: {context_str}
        
        Output a score between 0 and 1. 1 means full coverage, 0 means no coverage.
        Return ONLY the numerical score.
        """
        try:
            response = self._generate_with_retry(prompt, max_tokens=10)
            score_match = re.search(r"(\d*\.?\d+)", response)
            return float(score_match.group(1)) if score_match else 0.0
        except Exception as e:
            print(f"Context recall metric error: {e}")
            return 0.0
    
    def calculate_hit_rate(self, retrieved_contexts: List[str], ground_truth_context: str) -> float:
        """
        Measures if at least one relevant context was retrieved.
        Uses LLM for semantic relevance判断 instead of string matching.
        
        Args:
            retrieved_contexts: List of contexts retrieved by the system
            ground_truth_context: The context that should have been retrieved
            
        Returns:
            1.0 if at least one relevant context found, 0.0 otherwise
        """
        if not retrieved_contexts or not ground_truth_context:
            return 0.0

        gt_lower = ground_truth_context.lower()
        gt_key_phrases = self._extract_key_phrases(gt_lower)

        for ctx in retrieved_contexts:
            ctx_lower = ctx.lower()
            for phrase in gt_key_phrases:
                if phrase in ctx_lower:
                    return 1.0

        return 0.0

    def _llm_relevance_check(self, retrieved_contexts: List[str], ground_truth_context: str) -> float:
        """
        Use LLM to check if any retrieved context is relevant to ground truth.
        This is more robust than string matching.
        """
        context_str = "\n\n---\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts[:3])])
        
        prompt = f"""
Task: Determine if any of the retrieved contexts is relevant to the ground truth context.

Ground Truth Context:
{ground_truth_context[:500]}

Retrieved Contexts:
{context_str}

Is any retrieved context semantically relevant to the ground truth? (They don't need to be identical, just cover similar topics)

Output a score between 0 and 1:
- 1.0 = At least one context is highly relevant
- 0.5 = Somewhat relevant
- 0.0 = Not relevant

Return ONLY the numerical score."""
        
        try:
            response = self._generate_with_retry(prompt, max_tokens=10)
            score_match = re.search(r"(\d*\.?\d+)", response)
            return float(score_match.group(1)) if score_match else 0.0
        except Exception as e:
            print(f"LLM relevance check error: {e}")
            return 0.0

    def calculate_mrr(self, retrieved_contexts: List[str], ground_truth_context: str) -> float:
        """
        Mean Reciprocal Rank - measures the position of the first relevant result.
        Uses LLM for semantic relevance judgment.
        
        Args:
            retrieved_contexts: Ordered list of contexts (most relevant first)
            ground_truth_context: The context that should have been retrieved
            
        Returns:
            Reciprocal rank (1/rank) or 0.0 if no relevant context found
        """
        if not retrieved_contexts or not ground_truth_context:
            return 0.0

        gt_lower = ground_truth_context.lower()
        gt_key_phrases = self._extract_key_phrases(gt_lower)

        for rank, ctx in enumerate(retrieved_contexts, start=1):
            ctx_lower = ctx.lower()
            for phrase in gt_key_phrases:
                if phrase in ctx_lower:
                    return 1.0 / rank

        return 0.0
    
    def _llm_mrr(self, retrieved_contexts: List[str], ground_truth_context: str) -> float:
        """
        Use LLM to find the first relevant context and calculate MRR.
        """
        prompt = f"""
Task: Find the position of the first relevant context in the retrieved list.

Ground Truth Context:
{ground_truth_context[:500]}

Retrieved Contexts (in order):
"""
        for i, ctx in enumerate(retrieved_contexts[:5], 1):
            prompt += f"\n{i}. {ctx[:300]}"
        
        prompt += """

Which position (1-5) contains the first context that is relevant to the ground truth?
If none are relevant, say "none".

Output just the number or "none":"""
        
        try:
            response = self._generate_with_retry(prompt, max_tokens=10)
            response = response.lower().strip()
            
            if "none" in response:
                return 0.0
            
            # Try to extract number
            num_match = re.search(r'(\d+)', response)
            if num_match:
                rank = int(num_match.group(1))
                return 1.0 / rank if rank > 0 else 0.0
        except Exception as e:
            print(f"LLM MRR calculation error: {e}")
            pass
        
        return 0.0
    
    def calculate_ndcg(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
        k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved_contexts: Ordered list of retrieved contexts
            ground_truth_contexts: List of relevant contexts
            k: Position cutoff
            
        Returns:
            NDCG score between 0 and 1
        """
        if not retrieved_contexts or not ground_truth_contexts:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, ctx in enumerate(retrieved_contexts[:k]):
            relevance = self._calculate_relevance(ctx, ground_truth_contexts)
            dcg += relevance / math.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG
        ideal_dcg = 0.0
        for i in range(min(k, len(ground_truth_contexts))):
            ideal_dcg += 1.0 / math.log2(i + 2)
        
        if ideal_dcg == 0.0:
            return 0.0
        
        return dcg / ideal_dcg
    
    def calculate_answer_correctness(
        self,
        query: str,
        ground_truth_answer: str,
        generated_answer: str
    ) -> float:
        """
        Measures how correct the generated answer is compared to ground truth.
        Uses LLM to evaluate factual accuracy, completeness, and semantic similarity.
        
        Args:
            query: The original question
            ground_truth_answer: The expected correct answer
            generated_answer: The answer generated by the RAG system
            
        Returns:
            Score between 0 and 1
        """
        if not self.llm:
            return 0.0
        
        refusal_phrases = [
            "i don't have enough information",
            "i do not have enough information",
            "i don't know",
            "cannot answer",
            "no information",
        ]
        if any(p in generated_answer.lower() for p in refusal_phrases):
            # If ground truth is short/vague (< 50 words), refusal is appropriate
            if len(ground_truth_answer.split()) < 50:
                return 0.5
            return 0.0

        prompt = f"""
        Task: Evaluate how correct the generated answer is compared to the ground truth.

        Query: {query}
        Ground Truth: {ground_truth_answer}
        Generated Answer: {generated_answer}

        Evaluate based on:
        1. Factual accuracy - Are the facts correct?
        2. Completeness - Does it cover all key points?
        3. Semantic similarity - Is the meaning preserved?

        Output a score between 0 and 1, where:
        - 1.0 = Perfect answer, fully correct
        - 0.5 = Partially correct, some information missing or incorrect
        - 0.0 = Completely incorrect or a refusal to answer

        Return ONLY the numerical score.
        """
        try:
            response = self._generate_with_retry(prompt, max_tokens=10)
            score_match = re.search(r"(\d*\.?\d+)", response)
            return float(score_match.group(1)) if score_match else 0.0
        except Exception as e:
            print(f"Answer correctness metric error: {e}")
            return 0.0
    
    def normalize(self, text: str) -> str:
        """NFKD -> lowercase -> strip punctuation -> remove articles -> collapse whitespace."""
        import unicodedata, re
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
        return " ".join(tokens)

    def calculate_exact_match(self, prediction: str, ground_truth: str) -> float:
        """1.0 if normalized prediction == normalized ground truth, else 0.0."""
        return 1.0 if self.normalize(prediction) == self.normalize(ground_truth) else 0.0

    def calculate_token_f1(self, prediction: str, ground_truth: str) -> float:
        """Counter-based (bag) token F1, SQuAD-style."""
        from collections import Counter
        pred_tokens = self.normalize(prediction).split()
        gt_tokens = self.normalize(ground_truth).split()
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _extract_key_phrases(self, text: str, min_length: int = 20) -> List[str]:
        """
        Extract key phrases from text for matching.
        Uses sentence-like chunks as key phrases.
        """
        import re
        
        # Split by sentences and get substantial chunks
        sentences = re.split(r'[.!?]+', text)
        phrases = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= min_length:
                phrases.append(sent)
            elif len(sent) >= 10:
                # For shorter sentences, use longer n-grams
                words = sent.split()
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    if len(phrase) >= min_length:
                        phrases.append(phrase)
        
        return phrases
    
    def _calculate_relevance(self, context: str, ground_truth_contexts: List[str]) -> float:
        """
        Calculate relevance score of a context against ground truth.
        """
        ctx_lower = context.lower()
        
        for gt in ground_truth_contexts:
            gt_lower = gt.lower()
            # Check for overlapping content
            if gt_lower[:50] in ctx_lower:
                return 1.0
            # Check for significant overlap
            common_words = set(gt_lower.split()) & set(ctx_lower.split())
            if len(common_words) >= 10:
                return 0.5
        
        return 0.0
