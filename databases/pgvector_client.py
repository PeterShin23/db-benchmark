from typing import List, Dict, Tuple
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from .base import VectorDB

class PgVectorDB(VectorDB):
    def __init__(self, host='localhost', port=5432, user='postgres', password='postgres', database='vectordb'):
        self.connection_params = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }
        self.table_name = 'vectors'
        
    def setup(self, dim: int):
        conn = psycopg2.connect(**self.connection_params)
        cur = conn.cursor()
        
        # Drop table if exists
        cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        
        # Create table with vector column
        cur.execute(f"""
            CREATE TABLE {self.table_name} (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                embedding VECTOR({dim})
            )
        """)
        
        # Create index
        cur.execute(f"CREATE INDEX ON {self.table_name} USING hnsw (embedding vector_cosine_ops)")
        
        conn.commit()
        cur.close()
        conn.close()

    def close(self):
        """Close the database connection"""
        # PgVector client creates new connections for each operation,
        # so there's no persistent connection to close.
        # This method is included for interface consistency.
        pass
        
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        conn = psycopg2.connect(**self.connection_params)
        cur = conn.cursor()
        
        # Prepare data for insertion
        data = [
            (idx, meta.get('doc_id', ''), meta.get('text', ''), np.array(vector, dtype=np.float32).tolist())
            for idx, vector, meta in zip(ids, vectors, metas)
        ]
        
        # Insert data
        insert_query = f"""
            INSERT INTO {self.table_name} (id, doc_id, text, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
        """
        execute_values(cur, insert_query, data, template=None, page_size=100)
        
        conn.commit()
        cur.close()
        conn.close()
        
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        conn = psycopg2.connect(**self.connection_params)
        cur = conn.cursor()
        
        # Convert query vector to PostgreSQL array format
        query_array = np.array(query_vec, dtype=np.float32).tolist()
        
        # Search for similar vectors
        cur.execute(f"""
            SELECT id, 1 - (embedding <=> %s::vector) AS similarity, doc_id
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_array, query_array, k))
        
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [
            (row[0], float(row[1]), {'doc_id': row[2]})
            for row in results
        ]
        
    def clear(self):
        conn = psycopg2.connect(**self.connection_params)
        cur = conn.cursor()
        
        cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        
        conn.commit()
        cur.close()
        conn.close()
