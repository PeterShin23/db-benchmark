from typing import List, Dict, Tuple
import redis
import numpy as np
import json
import struct

from .base import VectorDB

class RedisVectorDB(VectorDB):
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self.index_name = 'vectors'
        self.prefix = 'vector:'
        
    def setup(self, dim: int):
        # Clear existing index and data
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        except:
            pass  # Index doesn't exist yet
            
        # Create index
        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        
        schema = (
            TextField("doc_id"),
            VectorField("vector",
                "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": dim,
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        )
        
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        self.client.ft(self.index_name).create_index(fields=schema, definition=definition)
        
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        pipe = self.client.pipeline()
        for idx, vector, meta in zip(ids, vectors, metas):
            key = f"{self.prefix}{idx}"
            # Convert vector to bytes
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            
            mapping = {
                'vector': vector_bytes,
                'doc_id': meta.get('doc_id', '')
            }
            pipe.hset(key, mapping=mapping)
        pipe.execute()
        
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        # Convert query vector to bytes
        query_bytes = np.array(query_vec, dtype=np.float32).tobytes()
        
        from redis.commands.search.query import Query
        q = Query(f'*=>[KNN {k} @vector $vec as score]').sort_by('score', asc=False).return_fields('id', 'doc_id', 'score').dialect(2)
        results = self.client.ft(self.index_name).search(q, query_params={'vec': query_bytes})
        
        return [
            (
                doc.id.replace(self.prefix, ''),  # Remove prefix from ID
                1.0 - float(doc.score),  # Convert distance to similarity
                {'doc_id': doc.doc_id}
            )
            for doc in results.docs
        ]
        
    def clear(self):
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        except:
            pass  # Index doesn't exist

    def close(self):
        """Close the database connection"""
        if self.client:
            self.client.close()
