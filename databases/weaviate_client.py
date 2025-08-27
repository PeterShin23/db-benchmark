from typing import List, Dict, Tuple
import weaviate
from .base import VectorDB

class WeaviateVectorDB(VectorDB):
    def __init__(self, url='http://localhost:8080'):
        self.client = weaviate.Client(url)
        self.class_name = 'Vector'
        
    def setup(self, dim: int):
        # Delete class if it exists
        if self.client.schema.exists(self.class_name):
            self.client.schema.delete_class(self.class_name)
            
        # Create class
        class_obj = {
            'class': self.class_name,
            'properties': [
                {
                    'name': 'doc_id',
                    'dataType': ['string'],
                }
            ],
            'vectorIndexConfig': {
                'distance': 'cosine'
            },
            'vectorizer': 'none',
        }
        self.client.schema.create_class(class_obj)
        
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        with self.client.batch as batch:
            for idx, vector, meta in zip(ids, vectors, metas):
                data_object = {
                    'doc_id': meta.get('doc_id', ''),
                }
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self.class_name,
                    uuid=idx,
                    vector=vector
                )
        
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        results = (
            self.client.query
            .get(self.class_name, ['doc_id'])
            .with_near_vector({'vector': query_vec})
            .with_limit(k)
            .with_additional(['id', 'distance'])
            .do()
        )
        
        if 'data' not in results or 'Get' not in results['data'] or self.class_name not in results['data']['Get']:
            return []
            
        objects = results['data']['Get'][self.class_name]
        return [
            (
                obj['_additional']['id'],
                1.0 - float(obj['_additional']['distance']),  # Convert distance to similarity
                {'doc_id': obj['doc_id']}
            )
            for obj in objects
        ]
        
    def clear(self):
        if self.client.schema.exists(self.class_name):
            self.client.schema.delete_class(self.class_name)
