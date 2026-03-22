"""
LLM-as-Judge evaluation module for RAG systems.

This module provides pairwise comparison capabilities to evaluate
and compare different RAG configurations or responses.
"""
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.generation.llm_client import LLMClient
from tenacity import retry, wait_exponential, stop_after_attempt


class JudgeDecision(Enum):
    """Possible outcomes from LLM judge."""
    WIN_A = "WIN_A"
    WIN_B = "WIN_B"
    DRAW = "DRAW"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class JudgeResult:
    """Result from LLM judge comparison."""
    decision: JudgeDecision
    reasoning: str
    scores: Dict[str, float]  # scores for each response
    raw_output: str


@dataclass
class PairwiseComparisonResult:
    """Aggregated result from multiple pairwise comparisons."""
    winner: str  # "A", "B", or "DRAW"
    win_rate_a: float
    win_rate_b: float
    draw_rate: float
    avg_score_a: float
    avg_score_b: float
    individual_results: List[JudgeResult]


class LLMJudge:
    """
    LLM-based judge for pairwise comparison of RAG responses.
    
    Features:
    - Randomized order to reduce bias
    - Chain-of-thought reasoning
    - Multiple runs for consistency
    - Fair length normalization
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: float = 0,
        use_cot: bool = True,
        num_runs: int = 3
    ):
        """
        Initialize LLM judge.
        
        Args:
            llm_client: LLM client for evaluation
            temperature: Temperature for LLM (0 for deterministic)
            use_cot: Whether to use chain-of-thought reasoning
            num_runs: Number of comparison runs (for consistency)
        """
        self.llm = llm_client or LLMClient()
        self.temperature = temperature
        self.use_cot = use_cot
        self.num_runs = num_runs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_with_retry(self, prompt: str) -> str:
        """Call LLM with exponential backoff retries."""
        return self.llm.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=self.temperature
        )
    
    def compare(
        self,
        query: str,
        response_a: str,
        response_b: str,
        ground_truth: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> JudgeResult:
        """
        Compare two responses for a given query.
        
        Args:
            query: The original question
            response_a: First response to compare
            response_b: Second response to compare
            ground_truth: Optional ground truth answer
            context: Optional retrieved contexts
            
        Returns:
            JudgeResult with decision and reasoning
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            query, response_a, response_b, ground_truth, context
        )
        
        # Get LLM judgment
        try:
            raw_output = self._generate_with_retry(prompt)
        except Exception as e:
            return JudgeResult(
                decision=JudgeDecision.INCONCLUSIVE,
                reasoning=f"Error: {str(e)}",
                scores={"a": 0.0, "b": 0.0},
                raw_output=""
            )
        
        # Parse the result
        return self._parse_judge_output(raw_output, response_a, response_b)
    
    def _build_evaluation_prompt(
        self,
        query: str,
        response_a: str,
        response_b: str,
        ground_truth: Optional[str],
        context: Optional[List[str]]
    ) -> str:
        """Build the evaluation prompt with chain-of-thought."""
        
        # Normalize lengths for fair comparison
        len_a = len(response_a.split())
        len_b = len(response_b.split())
        
        prompt = f"""You are an expert evaluator comparing two AI responses to a medical question.

ORIGINAL QUESTION:
{query}

RESPONSE A ({len_a} words):
{response_a}

RESPONSE B ({len_b} words):
{response_b}
"""
        
        if ground_truth:
            prompt += f"""
GROUND TRUTH ANSWER:
{ground_truth}
"""
        
        if context:
            prompt += f"""
RETRIEVED CONTEXT:
{"=".join(context[:2])}  # Use first 2 contexts for reference
"""
        
        if self.use_cot:
            prompt += """
EVALUATION PROCESS (Chain-of-Thought):
1. Identify the key claims and facts in each response
2. Compare each claim against the question, ground truth, and context
3. Evaluate factual accuracy, completeness, and relevance
4. Consider if the response directly addresses the question
5. Identify any hallucinations or incorrect information

Then provide your final decision in this format:
DECISION: WIN_A | WIN_B | DRAW

EXPLANATION: [Brief reasoning]
"""
        else:
            prompt += """
Evaluate which response is better based on:
- Factual accuracy
- Completeness
- Relevance to the question
- Clarity

Output your decision as: WIN_A, WIN_B, or DRAW
"""
        
        return prompt
    
    def _parse_judge_output(
        self,
        raw_output: str,
        response_a: str,
        response_b: str
    ) -> JudgeResult:
        """Parse LLM output to extract decision."""
        
        output_lower = raw_output.lower()
        
        # Determine winner
        if "win_a" in output_lower and "win_b" not in output_lower:
            decision = JudgeDecision.WIN_A
            scores = {"a": 0.8, "b": 0.4}
        elif "win_b" in output_lower and "win_a" not in output_lower:
            decision = JudgeDecision.WIN_B
            scores = {"a": 0.4, "b": 0.8}
        elif "draw" in output_lower:
            decision = JudgeDecision.DRAW
            scores = {"a": 0.5, "b": 0.5}
        else:
            # Try to extract scores
            decision = JudgeDecision.INCONCLUSIVE
            scores = {"a": 0.5, "b": 0.5}
        
        import re
        # Extract reasoning (everything after EXPLANATION or ANALYSIS)
        reasoning = raw_output
        if "explanation:" in output_lower:
            parts = re.split(r'(?i)explanation:', raw_output, maxsplit=1)
            if len(parts) > 1:
                reasoning = parts[1].strip()
        elif "analysis:" in output_lower:
            parts = re.split(r'(?i)analysis:', raw_output, maxsplit=1)
            if len(parts) > 1:
                reasoning = parts[1].strip()
        
        return JudgeResult(
            decision=decision,
            reasoning=reasoning[:200],  # Limit length
            scores=scores,
            raw_output=raw_output
        )
    
    def pairwise_compare(
        self,
        query: str,
        response_a: str,
        response_b: str,
        ground_truth: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> PairwiseComparisonResult:
        """
        Run multiple comparisons with randomized order to reduce bias.
        
        Args:
            query: The original question
            response_a: First response
            response_b: Second response
            ground_truth: Optional ground truth
            context: Optional context
            
        Returns:
            Aggregated PairwiseComparisonResult
        """
        results = []
        wins_a = 0
        wins_b = 0
        draws = 0
        score_a_total = 0.0
        score_b_total = 0.0
        
        for run in range(self.num_runs):
            # Randomize order to reduce bias
            if random.random() > 0.5:
                # Use provided order
                result = self.compare(query, response_a, response_b, ground_truth, context)
            else:
                # Swap order
                result = self.compare(query, response_b, response_a, ground_truth, context)
                # Swap scores back
                result = JudgeResult(
                    decision=self._swap_decision(result.decision),
                    reasoning=result.reasoning,
                    scores={"a": result.scores["b"], "b": result.scores["a"]},
                    raw_output=result.raw_output
                )
            
            results.append(result)
            
            # Tally
            if result.decision == JudgeDecision.WIN_A:
                wins_a += 1
            elif result.decision == JudgeDecision.WIN_B:
                wins_b += 1
            else:
                draws += 1
            
            score_a_total += result.scores["a"]
            score_b_total += result.scores["b"]
        
        # Determine winner
        if wins_a > wins_b:
            winner = "A"
        elif wins_b > wins_a:
            winner = "B"
        else:
            winner = "DRAW"
        
        return PairwiseComparisonResult(
            winner=winner,
            win_rate_a=wins_a / self.num_runs,
            win_rate_b=wins_b / self.num_runs,
            draw_rate=draws / self.num_runs,
            avg_score_a=score_a_total / self.num_runs,
            avg_score_b=score_b_total / self.num_runs,
            individual_results=results
        )
    
    def _swap_decision(self, decision: JudgeDecision) -> JudgeDecision:
        """Swap A and B in decision."""
        if decision == JudgeDecision.WIN_A:
            return JudgeDecision.WIN_B
        elif decision == JudgeDecision.WIN_B:
            return JudgeDecision.WIN_A
        return decision


def compare_rag_configurations(
    query: str,
    pipeline,
    config_variations: List[Dict[str, Any]],
    ground_truth: Optional[str] = None
) -> Dict[str, PairwiseComparisonResult]:
    """
    Compare different RAG configurations on the same query.
    
    Args:
        query: Question to test
        pipeline: RAG pipeline instance
        config_variations: List of config dicts with 'name' and params
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary mapping config name to comparison results
    """
    # Get baseline response first
    baseline_response = pipeline.query(query)
    
    results = {}
    judge = LLMJudge()
    
    for config in config_variations:
        config_name = config.get("name", f"config_{len(results)}")
        
        # Run with this configuration
        response = pipeline.query(query, **config.get("params", {}))
        
        # Compare against baseline
        comparison = judge.pairwise_compare(
            query=query,
            response_a=baseline_response["answer"],
            response_b=response["answer"],
            ground_truth=ground_truth
        )
        
        results[config_name] = comparison
    
    return results


# Default judge instance
_default_judge: Optional[LLMJudge] = None


def get_llm_judge(
    llm_client: Optional[LLMClient] = None,
    num_runs: int = 3
) -> LLMJudge:
    """
    Get or create the default LLM judge.
    
    Args:
        llm_client: Optional LLM client
        num_runs: Number of comparison runs
        
    Returns:
        LLMJudge instance
    """
    global _default_judge
    if _default_judge is None:
        _default_judge = LLMJudge(llm_client=llm_client, num_runs=num_runs)
    return _default_judge
