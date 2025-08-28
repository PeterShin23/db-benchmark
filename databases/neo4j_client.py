from typing import List, Dict, Tuple
from neo4j import GraphDatabase
from .base import VectorDB

LABEL = "Doc"
INDEX = "doc_emb_idx"

class Neo4jVectorDB(VectorDB):
    def __init__(self, uri='bolt://localhost:7687', user='neo4j', password='password'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.dim = None

    def setup(self, dim: int):
        self.dim = dim
        with self.driver.session() as s:
            # Wipe label to keep runs reproducible
            s.run(f"MATCH (n:{LABEL}) DETACH DELETE n")
            # Create vector index (Neo4j 5 native)
            s.run(f"""
            CREATE VECTOR INDEX {INDEX} IF NOT EXISTS
            FOR (d:{LABEL})
            ON d.emb
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """, dim=dim)

    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        rows = [{"id": ids[i], "doc_id": metas[i].get("doc_id",""), "emb": vectors[i]} for i in range(len(ids))]
        with self.driver.session() as s:
            s.run(f"""
            UNWIND $rows AS r
            MERGE (d:{LABEL} {{id: r.id}})
            SET d.doc_id = r.doc_id,
                d.text = r.text,
                d.emb = r.emb
            """, rows=[{"id": ids[i], "doc_id": metas[i].get("doc_id",""), "text": metas[i].get("text",""), "emb": vectors[i]} for i in range(len(ids))])

    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        with self.driver.session() as s:
            res = s.run(f"""
            CALL db.index.vector.queryNodes('{INDEX}', $k, $q) YIELD node, score
            RETURN node.id AS id, node.doc_id AS doc_id, node.text AS text, score
            """, k=k, q=query_vec)
            out = []
            for r in res:
                # Neo4j returns a *distance* for `score` with cosine; convert to similarity
                sim = 1.0 - float(r["score"])
                out.append((str(r["id"]), sim, {"doc_id": str(r["doc_id"]) if r["doc_id"] is not None else "", "text": str(r["text"]) if r["text"] is not None else ""}))
            return out

    def clear(self):
        with self.driver.session() as s:
            s.run(f"MATCH (n:{LABEL}) DETACH DELETE n")

    def close(self):
        self.driver.close()
