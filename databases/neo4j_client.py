from typing import List, Dict, Tuple
from neo4j import GraphDatabase
from .base import VectorDB

class Neo4jVectorDB(VectorDB):
    def __init__(self, uri='bolt://localhost:7687', user='neo4j', password='password'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def setup(self, dim: int):
        # For Neo4j, we don't need to set up a specific schema for vectors
        # We'll handle this in the upsert method
        pass
        
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        # This is a simplified implementation that just clears and creates CO_OCCURS edges
        # between entities in the same document
        with self.driver.session() as session:
            # Clear existing CO_OCCURS relationships
            session.write_transaction(self._clear_cooccurs)
            
            # Group by doc_id to create CO_OCCURS relationships
            doc_entities = {}
            for idx, meta in zip(ids, metas):
                doc_id = meta.get('doc_id', '')
                # In a real implementation, we would extract entities from the text
                # For this demo, we'll just use the doc_id as the entity
                if doc_id not in doc_entities:
                    doc_entities[doc_id] = []
                doc_entities[doc_id].append(idx)
                
            # Create CO_OCCURS relationships
            for doc_id, entity_ids in doc_entities.items():
                if len(entity_ids) > 1:
                    session.write_transaction(self._create_cooccurs, entity_ids)
        
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        # For this demo, we'll return empty results as Neo4j is used for graph operations
        # not vector search
        return []
        
    def clear(self):
        with self.driver.session() as session:
            session.write_transaction(self._clear_database)
            
    @staticmethod
    def _clear_cooccurs(tx):
        tx.run("MATCH ()-[r:CO_OCCURS]->() DELETE r")
        
    @staticmethod
    def _clear_database(tx):
        tx.run("MATCH (n) DETACH DELETE n")
        
    @staticmethod
    def _create_cooccurs(tx, entity_ids):
        # Create CO_OCCURS relationships between all entities in the same document
        for i in range(len(entity_ids)):
            for j in range(i+1, len(entity_ids)):
                tx.run("""
                    MERGE (a:Entity {id: $id1})
                    MERGE (b:Entity {id: $id2})
                    MERGE (a)-[:CO_OCCURS]->(b)
                    """, 
                    id1=entity_ids[i], id2=entity_ids[j]
                )

    def close(self):
        self.driver.close()
