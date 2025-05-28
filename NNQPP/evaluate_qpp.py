import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from src.predict import QPPPredictor
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr, kendalltau

print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v2.1", split="validation")
dataset = dataset.shuffle(seed=42).select(range(100))  # Use only 100 queries

# Load SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load your QPP model
predictor = QPPPredictor()

# Collect predicted and actual scores
predicted_scores = []
actual_ndcgs = []

print("Processing queries...")
for example in dataset:
    query = example["query"]
    passages = example["passages"]["passage_text"]
    labels = example["passages"]["is_selected"]

    if not any(labels):
        continue  # Skip if no relevant docs

    try:
        # Compute embeddings
        query_embedding = model.encode(query)
        doc_embeddings = model.encode(passages)

        # Predict performance score
        predicted = predictor.predict_performance(query_embedding, doc_embeddings)
        predicted_scores.append(predicted)

        # Create binary relevance labels (1 if selected, 0 otherwise)
        relevance = np.array(labels).reshape(1, -1)
        scores = np.ones(len(labels)).reshape(1, -1)  # All scores equal

        # Compute NDCG@10 as ground-truth performance
        ndcg = ndcg_score(relevance, scores, k=10)
        actual_ndcgs.append(ndcg)
    except Exception as e:
        print(f"Error: {e}")
        continue

# Compute correlations
print("\nEvaluation Results:")
print(f"Queries evaluated: {len(predicted_scores)}")

spearman_corr, _ = spearmanr(predicted_scores, actual_ndcgs)
kendall_corr, _ = kendalltau(predicted_scores, actual_ndcgs)

print(f"Spearman correlation: {spearman_corr:.3f}")
print(f"Kendall correlation: {kendall_corr:.3f}")
