from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional

class VectorStore:
    """Handles embedding storage and retrieval using Qdrant."""

    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "pdf_chunks", client: Optional[QdrantClient] = None):
        self.client = client if client else QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        collections = self.client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE), # Assuming OpenAI embedding size
            )

    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]] = None):
        """Stores chunks and their embeddings in Qdrant."""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = metadata[i] if metadata else {}
            payload["text"] = chunk # Store the original text chunk as well
            points.append(
                models.PointStruct(
                    id=i, # Simple ID for now, consider UUIDs for production
                    vector=embedding,
                    payload=payload
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def retrieve_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves similar chunks based on a query embedding."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        results = []
        for hit in search_result:
            results.append({"text": hit.payload["text"], "score": hit.score, "metadata": hit.payload})
        return results


