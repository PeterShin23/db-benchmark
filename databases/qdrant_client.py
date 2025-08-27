from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchParams
from .base import VectorDB

class QdrantVectorDB(VectorDB):
    def __init__(self, host='localhost', port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = 'vectors'
        
    def setup(self, dim: int):
        # Delete collection if it exists
        collections = self.client.get_collections()
        if any(collection.name == self.collection_name for collection in collections.collections):
            self.client.delete_collection(self.collection_name)
            
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        points = [
            PointStruct(
                id=idx,
                vector=vector,
                payload=meta
            )
            for idx, vector, meta in zip(ids, vectors, metas)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=k,
            with_payload=True
        )
        
        return [
            (result.id, result.score, result.payload)
            for result in results
        ]
        
    def clear(self):
        collections = self.client.get_collections()
        if any(collection.name == self.collection_name for collection in collections.collections):
            self.client.delete_collection(self.collection_name)
