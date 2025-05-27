from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load model globally
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy database
database = ["machine learning", "deep learning", "information retrieval", "search engines"]
scores = [0.85, 0.9, 0.8, 0.75]

# Pre-compute embeddings
embeddings = model.encode(database)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def predict_query_performance(query, k=2):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, k)
    predicted_score = np.mean([scores[i] for i in I[0]])
    return predicted_score
