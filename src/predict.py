# src/predict.py

import numpy as np

class QPPPredictor:
    def predict_performance(self, query_embedding, doc_embeddings):
        """
        Predicts a simple QPP score by computing the mean cosine similarity
        between the query and each of the document embeddings.

        Parameters:
        - query_embedding (np.ndarray): Vector for the query
        - doc_embeddings (np.ndarray): List or array of vectors for the documents

        Returns:
        - float: A scalar score representing predicted query performance
        """
        # Normalize vectors
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)

        # Avoid division by zero
        if query_norm == 0 or np.any(doc_norms == 0):
            return 0.0

        similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)
        return float(np.mean(similarities))
