import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import subprocess


def _git_commit() -> Optional[str]:
    """Get the current git commit hash (best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def make_result(
    dataset: str,
    dataset_size: int,
    queries_count: int,
    model_name: str,
    vector_dim: int,
    dtype: str,
    normalized: bool,
    db_name: str,
    db_version: str = "",
    host: str = "",
    collection: str = "",
    params: Optional[Dict] = None,
    workload: Optional[Dict] = None,
    performance: Optional[Dict] = None,
    retrieval: Optional[Dict] = None,
    rag: Optional[Dict] = None,
    graphrag: Optional[Dict] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Create a standardized result dictionary for vector database evaluation.
    
    Args:
        dataset: Dataset name
        dataset_size: Number of documents in the dataset
        queries_count: Number of queries used
        model_name: Name of the embedding model
        vector_dim: Dimension of the embeddings
        dtype: Data type of embeddings (e.g., "float32")
        normalized: Whether embeddings are normalized
        db_name: Database name
        db_version: Database version (optional)
        host: Database host (optional)
        collection: Collection/table name (optional)
        params: Index/ANN parameters (optional)
        workload: Workload parameters (optional)
        performance: Performance metrics (optional)
        retrieval: IR metrics (optional)
        rag: RAG metrics (optional)
        graphrag: GraphRAG metrics (optional)
        notes: Additional notes (optional)
        
    Returns:
        Dictionary with the standardized result structure
    """
    return {
        "meta": {
            "schema_version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": _git_commit(),
            "runner": "vecgraphbench@local"
        },
        "context": {
            "dataset": dataset,
            "dataset_size": dataset_size,
            "queries_count": queries_count,
            "model_name": model_name,
            "vector_dim": vector_dim,
            "dtype": dtype,
            "normalized": normalized
        },
        "db": {
            "name": db_name,
            "version": db_version,
            "host": host,
            "collection": collection,
            "params": params or {}
        },
        "workload": workload or {},
        "metrics": {
            "performance": {
                "index_build_time_sec": None,
                "upsert_rate_vps": None,
                "mem_peak_gb": None,
                "disk_index_bytes": None,
                "latency_ms": {
                    "p50": None,
                    "p90": None,
                    "p95": None,
                    "p99": None
                },
                "embed_latency_ms": {
                    "p50": None,
                    "p90": None,
                    "p95": None,
                    "p99": None
                },
                "search_latency_ms": {
                    "p50": None,
                    "p90": None,
                    "p95": None,
                    "p99": None
                },
                "qps": None
            } | (performance or {}),
            "retrieval": {
                "recall@10": None,
                "mrr@10": None,
                "ndcg@10": None,
                "precision@10": None
            } | (retrieval or {}),
            "rag": rag,
            "graphrag": graphrag
        },
        "notes": notes
    }


def save_result(
    result: Dict[str, Any],
    dataset: str,
    db_name: str,
    model_name: str,
    results_dir: str = "results"
) -> str:
    """
    Save a result dictionary to a JSON file.
    
    Args:
        result: Result dictionary to save
        dataset: Dataset name
        db_name: Database name
        model_name: Model name
        results_dir: Directory to save results (default: "results")
        
    Returns:
        Path to the saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create safe model name (replace : with _)
    model_safe = model_name.split("/")[-1].replace(":", "_")
    
    # Create filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{dataset}__{db_name}__{model_safe}__{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Save result to JSON file
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    
    return filepath


def load_results(pattern: str = "results/*.json") -> Union["pandas.DataFrame", List[Dict]]:
    """
    Load all result JSON files and return as a DataFrame or list of dictionaries.
    
    Args:
        pattern: Glob pattern for result files (default: "results/*.json")
        
    Returns:
        DataFrame with results or list of dictionaries if pandas is not available
    """
    try:
        import pandas as pd
        import glob
        
        # Get all JSON files matching the pattern
        files = glob.glob(pattern)
        
        # Load all results
        results = []
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
                # Extract key metrics
                row = {
                    "file": os.path.basename(file),
                    "db": data["db"]["name"],
                    "dataset": data["context"]["dataset"],
                    "model": data["context"]["model_name"],
                    "recall@10": data["metrics"]["retrieval"]["recall@10"],
                    "ndcg@10": data["metrics"]["retrieval"]["ndcg@10"],
                    "mrr@10": data["metrics"]["retrieval"]["mrr@10"],
                    "p95_ms": data["metrics"]["performance"]["latency_ms"]["p95"],
                    "qps": data["metrics"]["performance"]["qps"]
                }
                results.append(row)
        
        # Return as DataFrame
        return pd.DataFrame(results)
    except ImportError:
        # Fallback if pandas is not available
        import glob
        
        # Get all JSON files matching the pattern
        files = glob.glob(pattern)
        
        # Load all results
        results = []
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
                # Extract key metrics
                row = {
                    "file": os.path.basename(file),
                    "db": data["db"]["name"],
                    "dataset": data["context"]["dataset"],
                    "model": data["context"]["model_name"],
                    "recall@10": data["metrics"]["retrieval"]["recall@10"],
                    "ndcg@10": data["metrics"]["retrieval"]["ndcg@10"],
                    "mrr@10": data["metrics"]["retrieval"]["mrr@10"],
                    "p95_ms": data["metrics"]["performance"]["latency_ms"]["p95"],
                    "qps": data["metrics"]["performance"]["qps"]
                }
                results.append(row)
        
        return results
