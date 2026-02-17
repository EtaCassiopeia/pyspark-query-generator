"""Evaluation metrics for comparing generated PySpark code against references."""

import re
from collections.abc import Sequence


def bleu_score(reference: str, candidate: str) -> float:
    """Compute a simple BLEU-4 score between reference and candidate code strings.

    Uses nltk.translate.bleu_score if available, otherwise falls back to a
    simple unigram precision as an approximation.
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    except ImportError:
        # Fallback: unigram precision
        ref_tokens = set(reference.split())
        cand_tokens = candidate.split()
        if not cand_tokens:
            return 0.0
        matches = sum(1 for t in cand_tokens if t in ref_tokens)
        return matches / len(cand_tokens)


def exact_match(reference: str, candidate: str) -> bool:
    """Check if reference and candidate are identical after normalizing whitespace."""
    return _normalize(reference) == _normalize(candidate)


def _normalize(code: str) -> str:
    """Normalize whitespace in code for comparison."""
    return re.sub(r"\s+", " ", code.strip())


_PYSPARK_OPERATIONS = [
    "select",
    "filter",
    "where",
    "groupBy",
    "agg",
    "join",
    "orderBy",
    "sort",
    "withColumn",
    "drop",
    "distinct",
    "union",
    "limit",
    "alias",
    "Window",
    "partitionBy",
    "over",
    "when",
    "otherwise",
    "col",
    "lit",
    "count",
    "sum",
    "avg",
    "max",
    "min",
    "collect_list",
    "collect_set",
    "explode",
    "split",
    "coalesce",
    "broadcast",
]


def operation_coverage(reference: str, candidate: str) -> float:
    """Measure what fraction of PySpark operations in the reference appear in the candidate."""
    ref_ops = {op for op in _PYSPARK_OPERATIONS if op in reference}
    if not ref_ops:
        return 1.0  # No operations to match
    cand_ops = {op for op in _PYSPARK_OPERATIONS if op in candidate}
    return len(ref_ops & cand_ops) / len(ref_ops)


def structural_similarity(reference: str, candidate: str) -> float:
    """Measure structural similarity based on shared code patterns.

    Extracts structural tokens (method calls, column references) and computes Jaccard similarity.
    """
    ref_patterns = _extract_patterns(reference)
    cand_patterns = _extract_patterns(candidate)
    if not ref_patterns and not cand_patterns:
        return 1.0
    if not ref_patterns or not cand_patterns:
        return 0.0
    intersection = ref_patterns & cand_patterns
    union = ref_patterns | cand_patterns
    return len(intersection) / len(union)


def _extract_patterns(code: str) -> set[str]:
    """Extract structural patterns from PySpark code."""
    patterns: set[str] = set()
    # Method calls: .method(
    patterns.update(re.findall(r"\.(\w+)\s*\(", code))
    # String literals (table/column names): "name" or 'name'
    patterns.update(re.findall(r'["\']([^"\']+)["\']', code))
    # F.function calls
    patterns.update(re.findall(r"F\.(\w+)", code))
    return patterns


def compute_all_metrics(
    references: Sequence[str], candidates: Sequence[str]
) -> dict[str, float]:
    """Compute all metrics averaged over a set of reference/candidate pairs."""
    n = len(references)
    if n != len(candidates):
        raise ValueError("references and candidates must have the same length")
    if n == 0:
        return {"bleu": 0.0, "exact_match": 0.0, "operation_coverage": 0.0, "structural_similarity": 0.0}

    totals = {"bleu": 0.0, "exact_match": 0.0, "operation_coverage": 0.0, "structural_similarity": 0.0}
    for ref, cand in zip(references, candidates):
        totals["bleu"] += bleu_score(ref, cand)
        totals["exact_match"] += 1.0 if exact_match(ref, cand) else 0.0
        totals["operation_coverage"] += operation_coverage(ref, cand)
        totals["structural_similarity"] += structural_similarity(ref, cand)

    return {k: v / n for k, v in totals.items()}
