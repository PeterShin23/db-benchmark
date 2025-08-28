import math
from typing import Set, List, Union


def recall_at_k(truth_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """
    Calculate Recall@k metric.
    
    Args:
        truth_doc_ids: Set of ground truth document IDs
        retrieved_doc_ids: List of retrieved document IDs in order of relevance
        k: Number of top documents to consider
        
    Returns:
        Recall@k value
    """
    if not truth_doc_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_doc_ids[:k])
    relevant_retrieved = len(truth_doc_ids.intersection(retrieved_at_k))
    return relevant_retrieved / len(truth_doc_ids)


def mrr_at_k(truth_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """
    Calculate Mean Reciprocal Rank@k metric.
    
    Args:
        truth_doc_ids: Set of ground truth document IDs
        retrieved_doc_ids: List of retrieved document IDs in order of relevance
        k: Number of top documents to consider
        
    Returns:
        MRR@k value
    """
    if not truth_doc_ids:
        return 0.0
    
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in truth_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(truth_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k metric.
    
    Args:
        truth_doc_ids: Set of ground truth document IDs
        retrieved_doc_ids: List of retrieved document IDs in order of relevance
        k: Number of top documents to consider
        
    Returns:
        nDCG@k value
    """
    if not truth_doc_ids:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in truth_doc_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1+1) = log2(2) = 1
    
    # Calculate IDCG (ideal DCG)
    ideal_ranking_length = min(len(truth_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_ranking_length))
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(truth_doc_ids: Set[str], retrieved_doc_ids: List[str], k: int) -> float:
    """
    Calculate Precision@k metric.
    
    Args:
        truth_doc_ids: Set of ground truth document IDs
        retrieved_doc_ids: List of retrieved document IDs in order of relevance
        k: Number of top documents to consider
        
    Returns:
        Precision@k value
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved_doc_ids[:k])
    relevant_retrieved = len(truth_doc_ids.intersection(retrieved_at_k))
    return relevant_retrieved / k
