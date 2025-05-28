from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from src.predict import QPPPredictor

# Load the MS MARCO dataset from Hugging Face
print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v2.1", split="train[:1%]")  # small subset for testing

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Instantiate the QPP predictor
qpp_predictor = QPPPredictor()

# Loop over first N queries
N = 10
scores = []

for i in range(N):
    query = dataset[i]['query']
    documents = dataset[i]['passages']['passage_text'][:3]  # Top 3 passages

    print(f"\nQuery {i+1}: {query}")
    print(f"Documents: {documents}")

    # Compute embeddings
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(documents)

    # Predict performance
    score = qpp_predictor.predict_performance(query_embedding, doc_embeddings)
    print(f"Predicted QPP score: {score:.3f}")
    scores.append(score)

# Print average score
print(f"\nAverage predicted QPP score for {N} queries: {np.mean(scores):.3f}")
