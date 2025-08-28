from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class VectorDB(ABC):
    @abstractmethod
    def setup(self, dim: int):
        """Set up the database with the specified dimension"""
        pass

    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]], metas: List[Dict]):
        """Upsert vectors with their IDs and metadata"""
        pass

    @abstractmethod
    def search(self, query_vec: List[float], k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all data from the database"""
        pass

    @abstractmethod
    def close(self):
        """Close the database connection"""
        pass
