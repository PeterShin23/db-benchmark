import argparse
import json
import time
import numpy as np
import pandas as pd
import sys
import os
from sentence_transformers import SentenceTransformer

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database clients
from databases.qdrant_client import QdrantVectorDB
from databases.weaviate_client import WeaviateVectorDB
from databases.redis_client import RedisVectorDB
from databases.pgvector_client import PgVectorDB
from databases.neo4j_client import Neo4jVectorDB

# Import utils
from utils import make_result, save_result
from utils.metrics_ir import recall_at_k, mrr_at_k, ndcg_at_k


def get_db_client(db_name):
    """Get database client based on name."""
    if db_name == "qdrant":
        return QdrantVectorDB(
            host="localhost",
            port=6333
        )
    elif db_name == "weaviate":
        return WeaviateVectorDB(
            url="http://localhost:8080"
        )
    elif db_name == "redis":
        return RedisVectorDB(
            host="localhost",
            port=6379
        )
    elif db_name == "pgvector":
        return PgVectorDB(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="vectordb"
        )
    elif db_name == "neo4j":
        return Neo4jVectorDB(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
    else:
        raise ValueError(f"Unknown database: {db_name}")


def load_queries(queries_path):
    """Load queries from JSONL file."""
    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_qrels(qrels_path):
    """Load qrels from TSV file."""
    qrels = {}
    with open(qrels_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            query_id, corpus_id, score = parts[0], parts[1], int(parts[2])
            if query_id not in qrels:
                qrels[query_id] = []
            if score > 0:  # Only consider relevant documents
                qrels[query_id].append(corpus_id)
    return qrels


def main():
    parser = argparse.ArgumentParser(description='Evaluate vector database on FiQA dataset')
    parser.add_argument('--db', required=True, help='Database to evaluate (qdrant, weaviate, pgvector, redis, neo4j)')
    parser.add_argument('--parquet', required=True, help='Path to embeddings parquet file')
    
    args = parser.parse_args()
    
    # Initialize database client
    print(f"Initializing {args.db} client...")
    db = get_db_client(args.db)
    
    try:
        # Load data
        print("Loading queries and qrels...")
        queries = load_queries("data/fiqa/queries.jsonl")
        qrels = load_qrels("data/fiqa/qrels/test.tsv")
        
        # Load embeddings to get dimension
        print("Loading embeddings...")
        df = pd.read_parquet(args.parquet)
        dim = len(df.iloc[0]['emb'])
        
        # Setup database
        print("Setting up database...")
        db.setup(dim)
        
        # Index data
        print("Indexing data...")
        start_time = time.time()
        ids = df['id'].tolist()
        vectors = df['emb'].tolist()
        metas = df[['doc_id']].to_dict('records')
        
        # Add text to metadata for pgvector
        if args.db == "pgvector":
            for i, meta in enumerate(metas):
                meta['text'] = df.iloc[i]['text']
        
        # Batch upsert to avoid timeouts
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            db.upsert(batch_ids, batch_vectors, batch_metas)
            print(f"Indexed {min(i+batch_size, len(ids))}/{len(ids)} documents")
        
        index_build_time = time.time() - start_time
        print(f"Indexing completed in {index_build_time:.2f} seconds")
        
        # Initialize model
        print("Loading model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Evaluate
        print("Running evaluation...")
        latencies = []
        recalls = []
        mrrs = []
        ndcgs = []
        
        eval_start_time = time.time()
        
        # Filter queries to only those that have qrels
        eval_queries = [q for q in queries if q['_id'] in qrels]
        
        for query in eval_queries:
            query_id = query['_id']
            query_text = query['text']
            
            # Get ground truth
            truth_ids = set(qrels[query_id])
            if not truth_ids:
                continue
                
            # Embed query
            start_time = time.time()
            query_vec = model.encode(query_text, normalize_embeddings=True)
            embed_time = time.time() - start_time
            
            # Search
            start_time = time.time()
            results = db.search(query_vec, k=10)
            search_time = time.time() - start_time
            
            # Total latency
            latency_ms = (embed_time + search_time) * 1000
            latencies.append(latency_ms)
            
            # Extract retrieved IDs
            retrieved_ids = [str(result[0]) for result in results]
            
            # Compute metrics
            recalls.append(recall_at_k(truth_ids, retrieved_ids, 10))
            mrrs.append(mrr_at_k(truth_ids, retrieved_ids, 10))
            ndcgs.append(ndcg_at_k(truth_ids, retrieved_ids, 10))
        
        eval_time = time.time() - eval_start_time
        
        # Compute aggregate metrics
        mean_recall = np.mean(recalls) if recalls else 0.0
        mean_mrr = np.mean(mrrs) if mrrs else 0.0
        mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0
        qps = len(eval_queries) / eval_time if eval_time > 0 else 0.0
        
        print(f"Mean Recall@10: {mean_recall:.4f}")
        print(f"Mean MRR@10: {mean_mrr:.4f}")
        print(f"Mean nDCG@10: {mean_ndcg:.4f}")
        print(f"95th percentile latency: {p95_latency:.2f} ms")
        print(f"QPS: {qps:.2f}")
        
        # Create result
        result = make_result(
            dataset="fiqa",
            dataset_size=len(df),
            queries_count=len(eval_queries),
            model_name="all-MiniLM-L6-v2",
            vector_dim=dim,
            dtype="float32",
            normalized=True,
            db_name=args.db,
            db_version="",
            host="localhost",
            collection="vectors",
            params=None,
            workload={"top_k": 10, "concurrency": 1, "warmup_queries": 0},
            performance={
                "index_build_time_sec": index_build_time,
                "upsert_rate_vps": len(df) / index_build_time if index_build_time > 0 else 0,
                "mem_peak_gb": None,
                "disk_index_bytes": None,
                "latency_ms": {"p50": None, "p90": None, "p95": p95_latency, "p99": None},
                "qps": qps
            },
            retrieval={
                "recall@10": mean_recall,
                "mrr@10": mean_mrr,
                "ndcg@10": mean_ndcg,
                "precision@10": None
            },
            notes="FiQA evaluation"
        )
        
        # Save result
        filepath = save_result(result, "fiqa", args.db, "all-MiniLM-L6-v2")
        print(f"Results saved to {filepath}")
    finally:
        # Close database connection
        db.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
