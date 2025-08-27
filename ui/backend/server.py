from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database clients
from databases.qdrant_client import QdrantVectorDB
from databases.weaviate_client import WeaviateVectorDB
from databases.redis_client import RedisVectorDB
from databases.pgvector_client import PgVectorDB
from databases.neo4j_client import Neo4jVectorDB

app = FastAPI()

# Database factory
def get_db_client(db_name: str):
    if db_name == "qdrant":
        return QdrantVectorDB(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
    elif db_name == "weaviate":
        return WeaviateVectorDB(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080")
        )
    elif db_name == "redis":
        return RedisVectorDB(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379))
        )
    elif db_name == "pgvector":
        return PgVectorDB(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "vectordb")
        )
    elif db_name == "neo4j":
        return Neo4jVectorDB(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
    else:
        raise ValueError(f"Unknown database: {db_name}")

# Model cache
model_cache = {}

def get_model():
    global model_cache
    model_name = "all-MiniLM-L6-v2"
    if model_name not in model_cache:
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]

# Request models
class IndexRequest(BaseModel):
    parquet_path: str
    db: str

class SearchRequest(BaseModel):
    db: str
    text: str
    k: int = 5

# Routes
@app.post("/index")
async def index_data(request: IndexRequest):
    try:
        # Load parquet file
        df = pd.read_parquet(request.parquet_path)
        
        # Get first embedding to determine dimension
        first_emb = df.iloc[0]['emb']
        dim = len(first_emb)
        
        # Initialize database client
        db = get_db_client(request.db)
        
        # Setup database
        db.setup(dim)
        
        # Prepare data for upsert
        ids = df['id'].tolist()
        vectors = df['emb'].tolist()
        metas = df[['doc_id']].to_dict('records')
        
        # Add text to metadata for pgvector
        if request.db == "pgvector":
            for i, meta in enumerate(metas):
                meta['text'] = df.iloc[i]['text']
        
        # Upsert data
        db.upsert(ids, vectors, metas)
        
        return {"ok": True, "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_data(request: SearchRequest):
    try:
        # Get model and embed query
        model = get_model()
        query_vec = model.encode(request.text, normalize_embeddings=True).tolist()
        
        # Initialize database client
        db = get_db_client(request.db)
        
        # Search
        results = db.search(query_vec, request.k)
        
        # Format results
        formatted_results = [
            {
                "id": id,
                "score": score,
                "meta": meta
            }
            for id, score, meta in results
        ]
        
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_data(db: str):
    try:
        # Initialize database client
        db_client = get_db_client(db)
        
        # Clear data
        db_client.clear()
        
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="ui/frontend", html=True), name="frontend")
