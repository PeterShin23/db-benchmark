from typing import List, Dict, Tuple
import weaviate
import uuid
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections.classes.data import DataObject
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.collections.classes.config_vectorizers import Vectorizers
from .base import VectorDB

class WeaviateVectorDB(VectorDB):
    def __init__(self, url='http://localhost:8080'):
        # Connect with increased timeout and skip init checks to avoid gRPC issues
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30)
            ),
            skip_init_checks=True
        )
        self.class_name = 'Vector'
        
    def setup(self, dim: int):
        # Delete class if it exists
        if self.client.collections.exists(self.class_name):
            self.client.collections.delete(self.class_name)
            
        # Create class
        from weaviate.collections.classes.config_vector_index import VectorDistances
        self.client.collections.create(
            name=self.class_name,
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            )
        )
        
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        collection = self.client.collections.get(self.class_name)
        
        # Prepare data objects
        data_objects = []
        for idx, vector, meta in zip(ids, vectors, metas):
            data_object = {
                'doc_id': str(meta.get('doc_id', '')),
                'text': str(meta.get('text', '')),
            }
            # Generate a valid UUID from the index
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idx)))
            data_objects.append(DataObject(
                properties=data_object,
                vector=vector,
                uuid=obj_uuid
            ))
        
        # Batch insert with explicit batch handling
        batch_result = collection.data.insert_many(data_objects)
        
        # Check for errors
        if batch_result.has_errors:
            errors = batch_result.errors
            raise Exception(f"Batch insert failed with errors: {errors}")
        
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        collection = self.client.collections.get(self.class_name)
        results = collection.query.near_vector(
            near_vector=query_vec,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["doc_id", "text"]
        )
        
        return [
            (
                str(obj.uuid),
                1.0 - obj.metadata.distance,  # Convert distance to similarity
                {'doc_id': obj.properties['doc_id'], 'text': obj.properties.get('text', '')}
            )
            for obj in results.objects
        ]
        
    def clear(self):
        if self.client.collections.exists(self.class_name):
            self.client.collections.delete(self.class_name)

    def close(self):
        """Close the database connection"""
        if self.client:
            self.client.close()
