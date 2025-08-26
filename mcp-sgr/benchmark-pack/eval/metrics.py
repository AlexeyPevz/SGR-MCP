#!/usr/bin/env python3
"""
Advanced metrics for benchmark evaluation
Includes RAGAS metrics, code quality metrics, and custom evaluators
"""

import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import difflib

@dataclass
class RAGASMetrics:
    """Metrics for evaluating RAG (Retrieval Augmented Generation) tasks."""
    
    @staticmethod
    def calculate_faithfulness(response: str, sources: List[Dict]) -> float:
        """
        Calculate how faithful the response is to the provided sources.
        Returns a score between 0 and 1.
        """
        if not response or not sources:
            return 0.0
        
        response_lower = response.lower()
        source_text = " ".join([s.get("content", "") for s in sources]).lower()
        
        # Extract key claims from response
        sentences = re.split(r'[.!?]+', response)
        faithful_sentences = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            # Check if key words from sentence appear in sources
            words = sentence.lower().split()
            key_words = [w for w in words if len(w) > 4]  # Focus on meaningful words
            
            if not key_words:
                continue
                
            matches = sum(1 for word in key_words if word in source_text)
            if matches / len(key_words) > 0.5:  # More than half the key words found
                faithful_sentences += 1
        
        return faithful_sentences / max(len(sentences), 1)
    
    @staticmethod
    def calculate_answer_relevancy(response: str, question: str) -> float:
        """
        Calculate how relevant the answer is to the question.
        Returns a score between 0 and 1.
        """
        if not response or not question:
            return 0.0
        
        # Extract key terms from question
        question_terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
        response_terms = set(re.findall(r'\b\w{4,}\b', response.lower()))
        
        # Calculate term overlap
        if not question_terms:
            return 0.5  # Default score if no meaningful terms
            
        overlap = len(question_terms.intersection(response_terms))
        relevancy = overlap / len(question_terms)
        
        # Boost score if response directly addresses the question type
        question_words = ["what", "how", "why", "when", "where", "who"]
        for qword in question_words:
            if qword in question.lower():
                if qword in ["what", "how"] and len(response) > 100:
                    relevancy = min(1.0, relevancy + 0.2)
                elif qword in ["why"] and any(word in response.lower() for word in ["because", "due to", "reason"]):
                    relevancy = min(1.0, relevancy + 0.2)
                break
        
        return relevancy
    
    @staticmethod
    def calculate_context_precision(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """
        Calculate precision of retrieved documents.
        Returns a score between 0 and 1.
        """
        if not retrieved_docs:
            return 0.0
        
        relevant_ids = set(relevant_docs)
        retrieved_ids = [doc.get("id", "") for doc in retrieved_docs]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
        
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    @staticmethod
    def calculate_context_recall(retrieved_docs: List[Dict], relevant_docs: List[str]) -> float:
        """
        Calculate recall of retrieved documents.
        Returns a score between 0 and 1.
        """
        if not relevant_docs:
            return 1.0  # If no relevant docs, any retrieval is perfect
        
        retrieved_ids = set(doc.get("id", "") for doc in retrieved_docs)
        relevant_ids = set(relevant_docs)
        
        found = len(retrieved_ids.intersection(relevant_ids))
        return found / len(relevant_ids)
    
    @staticmethod
    def calculate_answer_correctness(response: str, expected: str) -> float:
        """
        Calculate correctness of the answer compared to expected.
        Uses semantic similarity approximation.
        """
        if not response or not expected:
            return 0.0
        
        # Normalize texts
        response_norm = response.lower().strip()
        expected_norm = expected.lower().strip()
        
        # Direct match
        if response_norm == expected_norm:
            return 1.0
        
        # Calculate similarity using sequence matcher
        similarity = difflib.SequenceMatcher(None, response_norm, expected_norm).ratio()
        
        # Check for key facts
        expected_facts = re.findall(r'\b\d+\b|\b[A-Z]{2,}\b', expected)
        if expected_facts:
            facts_found = sum(1 for fact in expected_facts if fact in response)
            fact_score = facts_found / len(expected_facts)
            similarity = (similarity + fact_score) / 2
        
        return similarity


class CodeQualityMetrics:
    """Metrics for evaluating code generation quality."""
    
    @staticmethod
    def calculate_syntax_validity(code: str, language: str = "python") -> float:
        """
        Check if code has valid syntax.
        Returns 1.0 for valid, 0.0 for invalid.
        """
        if not code:
            return 0.0
        
        if language.lower() == "python":
            try:
                compile(code, "<string>", "exec")
                return 1.0
            except SyntaxError:
                return 0.0
        elif language.lower() in ["javascript", "js"]:
            # Basic JS syntax checks
            brackets = code.count("{") == code.count("}")
            parens = code.count("(") == code.count(")")
            quotes = code.count('"') % 2 == 0 and code.count("'") % 2 == 0
            
            return 1.0 if all([brackets, parens, quotes]) else 0.0
        else:
            # Basic syntax check for other languages
            return 0.5  # Neutral score
    
    @staticmethod
    def calculate_test_coverage(code: str, tests: str) -> float:
        """
        Estimate test coverage based on functions/methods tested.
        Returns a score between 0 and 1.
        """
        if not code or not tests:
            return 0.0
        
        # Extract function/method names from code
        code_functions = set(re.findall(r'def\s+(\w+)', code))
        code_functions.update(re.findall(r'function\s+(\w+)', code))
        code_functions.update(re.findall(r'(\w+)\s*:\s*function', code))
        
        if not code_functions:
            return 0.5  # No functions found, neutral score
        
        # Check how many functions are mentioned in tests
        tested_functions = sum(1 for func in code_functions if func in tests)
        
        return tested_functions / len(code_functions)
    
    @staticmethod
    def calculate_security_score(code: str) -> float:
        """
        Check for common security issues.
        Returns a score between 0 and 1 (1 being most secure).
        """
        security_issues = 0
        total_checks = 0
        
        # SQL injection patterns
        sql_patterns = [
            r'f".*SELECT.*{.*}.*"',  # f-string SQL
            r'".*SELECT.*"\s*\+',    # String concatenation SQL
            r'query\([^?]*%[^?]*\)',  # String formatting in query
        ]
        for pattern in sql_patterns:
            total_checks += 1
            if re.search(pattern, code, re.IGNORECASE):
                security_issues += 1
        
        # Command injection patterns
        cmd_patterns = [
            r'os\.system\s*\([^"\']*\+',  # Dynamic os.system
            r'subprocess\.\w+\s*\([^"\']*\+',  # Dynamic subprocess
            r'eval\s*\(',  # eval usage
            r'exec\s*\(',  # exec usage
        ]
        for pattern in cmd_patterns:
            total_checks += 1
            if re.search(pattern, code):
                security_issues += 1
        
        # Weak crypto patterns
        crypto_patterns = [
            r'md5',
            r'sha1',
            r'DES',
            r'Random\(\)',  # Non-crypto random
        ]
        for pattern in crypto_patterns:
            total_checks += 1
            if re.search(pattern, code, re.IGNORECASE):
                security_issues += 1
        
        # Good security practices
        good_patterns = [
            r'bcrypt',
            r'argon2',
            r'prepared\s+statement',
            r'parameterized',
            r'sanitize',
            r'validate',
            r'escape',
        ]
        good_practices = sum(1 for pattern in good_patterns if re.search(pattern, code, re.IGNORECASE))
        
        if total_checks == 0:
            return 0.5  # Neutral if no checks apply
        
        base_score = 1.0 - (security_issues / total_checks)
        bonus = min(0.2, good_practices * 0.05)  # Up to 0.2 bonus for good practices
        
        return min(1.0, base_score + bonus)
    
    @staticmethod
    def calculate_complexity_score(code: str) -> float:
        """
        Calculate code complexity (lower is better).
        Returns a score between 0 and 1 (1 being least complex).
        """
        if not code:
            return 0.0
        
        lines = code.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if not code_lines:
            return 1.0
        
        # Count complexity indicators
        complexity_indicators = {
            'nested_loops': len(re.findall(r'for.*:\s*\n\s+for', code)),
            'nested_ifs': len(re.findall(r'if.*:\s*\n\s+if', code)),
            'long_functions': len([1 for line in code_lines if len(line) > 100]),
            'deep_nesting': max([len(line) - len(line.lstrip()) for line in code_lines]) // 4,
        }
        
        # Calculate complexity score
        total_complexity = sum(complexity_indicators.values())
        max_acceptable_complexity = 10
        
        complexity_ratio = min(1.0, total_complexity / max_acceptable_complexity)
        return 1.0 - complexity_ratio  # Invert so higher score is better


class SummarizationMetrics:
    """Metrics for evaluating summarization tasks."""
    
    @staticmethod
    def calculate_rouge_score(summary: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (simplified version).
        Returns ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        if not summary or not reference:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        # Tokenize
        summary_tokens = summary.lower().split()
        reference_tokens = reference.lower().split()
        
        # ROUGE-1 (unigram overlap)
        summary_unigrams = set(summary_tokens)
        reference_unigrams = set(reference_tokens)
        
        if reference_unigrams:
            rouge1_precision = len(summary_unigrams & reference_unigrams) / len(summary_unigrams) if summary_unigrams else 0
            rouge1_recall = len(summary_unigrams & reference_unigrams) / len(reference_unigrams)
            rouge1_f1 = 2 * rouge1_precision * rouge1_recall / (rouge1_precision + rouge1_recall) if (rouge1_precision + rouge1_recall) > 0 else 0
        else:
            rouge1_f1 = 0.0
        
        # ROUGE-2 (bigram overlap)
        summary_bigrams = set(zip(summary_tokens[:-1], summary_tokens[1:]))
        reference_bigrams = set(zip(reference_tokens[:-1], reference_tokens[1:]))
        
        if reference_bigrams:
            rouge2_precision = len(summary_bigrams & reference_bigrams) / len(summary_bigrams) if summary_bigrams else 0
            rouge2_recall = len(summary_bigrams & reference_bigrams) / len(reference_bigrams)
            rouge2_f1 = 2 * rouge2_precision * rouge2_recall / (rouge2_precision + rouge2_recall) if (rouge2_precision + rouge2_recall) > 0 else 0
        else:
            rouge2_f1 = 0.0
        
        # ROUGE-L (longest common subsequence)
        lcs_length = SummarizationMetrics._lcs_length(summary_tokens, reference_tokens)
        rougeL_precision = lcs_length / len(summary_tokens) if summary_tokens else 0
        rougeL_recall = lcs_length / len(reference_tokens) if reference_tokens else 0
        rougeL_f1 = 2 * rougeL_precision * rougeL_recall / (rougeL_precision + rougeL_recall) if (rougeL_precision + rougeL_recall) > 0 else 0
        
        return {
            "rouge1": rouge1_f1,
            "rouge2": rouge2_f1,
            "rougeL": rougeL_f1
        }
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        if not seq1 or not seq2:
            return 0
            
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def calculate_compression_ratio(original: str, summary: str) -> float:
        """
        Calculate compression ratio.
        Returns ratio of summary length to original length.
        """
        if not original:
            return 0.0
        
        return len(summary) / len(original)
    
    @staticmethod
    def calculate_information_coverage(summary: str, key_points: List[str]) -> float:
        """
        Calculate how many key points are covered in the summary.
        Returns a score between 0 and 1.
        """
        if not key_points:
            return 1.0
        
        summary_lower = summary.lower()
        covered = sum(1 for point in key_points if point.lower() in summary_lower)
        
        return covered / len(key_points)


class WorkflowMetrics:
    """Metrics for evaluating multi-step workflow tasks."""
    
    @staticmethod
    def calculate_step_completion(completed_steps: List[str], required_steps: List[str]) -> float:
        """
        Calculate what percentage of required steps were completed.
        Returns a score between 0 and 1.
        """
        if not required_steps:
            return 1.0
        
        completed_set = set(completed_steps)
        required_set = set(required_steps)
        
        completed_required = len(completed_set & required_set)
        return completed_required / len(required_set)
    
    @staticmethod
    def calculate_step_order_score(completed_steps: List[str], expected_order: List[str]) -> float:
        """
        Calculate how well the step order matches expected order.
        Returns a score between 0 and 1.
        """
        if not expected_order or not completed_steps:
            return 0.0
        
        # Find positions of expected steps in completed steps
        positions = []
        for step in expected_order:
            try:
                pos = completed_steps.index(step)
                positions.append(pos)
            except ValueError:
                positions.append(float('inf'))  # Step not found
        
        # Check if positions are in ascending order
        in_order = 0
        for i in range(1, len(positions)):
            if positions[i] > positions[i-1] and positions[i] != float('inf'):
                in_order += 1
        
        return in_order / (len(expected_order) - 1) if len(expected_order) > 1 else 1.0
    
    @staticmethod
    def calculate_retry_efficiency(retries: int, max_retries: int = 3) -> float:
        """
        Calculate efficiency based on number of retries needed.
        Returns a score between 0 and 1 (1 being no retries needed).
        """
        if retries <= 0:
            return 1.0
        
        return max(0, 1.0 - (retries / max_retries))


def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Calculate a weighted composite score from multiple metrics.
    
    Args:
        metrics: Dictionary of metric names to scores
        weights: Optional dictionary of metric names to weights
    
    Returns:
        Composite score between 0 and 1
    """
    if not metrics:
        return 0.0
    
    if weights is None:
        # Equal weights by default
        weights = {k: 1.0 for k in metrics.keys()}
    
    # Normalize weights
    total_weight = sum(weights.get(k, 1.0) for k in metrics.keys())
    
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(
        metrics[k] * weights.get(k, 1.0) 
        for k in metrics.keys()
    )
    
    return weighted_sum / total_weight