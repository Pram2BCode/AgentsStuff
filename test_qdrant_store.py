import unittest
from unittest.mock import MagicMock, patch
from qdrant_client import QdrantClient, models
from src.vector_store.qdrant_store import VectorStore

class TestVectorStore(unittest.TestCase):

    @patch("qdrant_client.QdrantClient")
    def setUp(self, MockQdrantClient):
        self.mock_client_instance = MagicMock()
        MockQdrantClient.return_value = self.mock_client_instance
        
        # Mock get_collections to simulate an empty collection initially
        self.mock_client_instance.get_collections.return_value.collections = []
        
        # Mock recreate_collection to do nothing
        self.mock_client_instance.recreate_collection.return_value = None

        # Pass the mocked client to the VectorStore constructor
        self.vector_store = VectorStore(client=self.mock_client_instance, collection_name="test_collection")

    def test_create_collection_if_not_exists(self):
        # The collection is created in setUp, so we just need to assert it was called
        self.mock_client_instance.recreate_collection.assert_called_once_with(
            collection_name="test_collection",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )

    def test_store_embeddings(self):
        chunks = ["chunk1", "chunk2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{"id": 1}, {"id": 2}]

        self.vector_store.store_embeddings(chunks, embeddings, metadata)

        self.mock_client_instance.upsert.assert_called_once()
        args, kwargs = self.mock_client_instance.upsert.call_args
        self.assertEqual(kwargs["collection_name"], "test_collection")
        self.assertEqual(len(kwargs["points"]), 2)
        self.assertEqual(kwargs["points"][0].payload["text"], "chunk1")
        self.assertEqual(kwargs["points"][1].payload["text"], "chunk2")

    def test_retrieve_similar(self):
        query_embedding = [0.5, 0.6]
        mock_search_result = [
            MagicMock(payload={"text": "result1", "doc_id": "docA"}, score=0.9),
            MagicMock(payload={"text": "result2", "doc_id": "docB"}, score=0.8)
        ]
        self.mock_client_instance.search.return_value = mock_search_result

        results = self.vector_store.retrieve_similar(query_embedding, top_k=2)

        self.mock_client_instance.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_embedding,
            limit=2
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "result1")
        self.assertEqual(results[1]["score"], 0.8)

if __name__ == "__main__":
    unittest.main()


