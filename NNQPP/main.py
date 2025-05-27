import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# -----------------------------
# Helper functions
# -----------------------------

def load_queries(filepath):
    """Load queries from a TSV or TXT file."""
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(line.strip())
    return queries

def load_performance_scores(filepath):
    """Load ground-truth performance scores for queries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [float(line.strip()) for line in f]

def compute_embeddings(queries, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Compute dense embeddings for each query."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(queries, convert_to_tensor=True)
    return embeddings

def knn_predict(query_embed, all_embeds, perf_scores, k=10):
    """Predict performance score using k-NN."""
    sims = cosine_similarity(query_embed.reshape(1, -1), all_embeds)[0]
    top_k_idx = np.argsort(sims)[-k:]
    return np.mean([perf_scores[i] for i in top_k_idx])

# -----------------------------
# Main Pipeline
# -----------------------------

def main(args):
    print("Loading queries...")
    queries = load_queries(args.query_file)
    print(f"Loaded {len(queries)} queries.")

    print("Loading performance scores...")
    scores = load_performance_scores(args.score_file)

    print("Encoding queries...")
    embeddings = compute_embeddings(queries, model_name=args.encoder)

    print("Predicting performance...")
    predictions = []
    for i in range(len(queries)):
        query_embed = embeddings[i].cpu().numpy()
        other_embeds = np.delete(embeddings.cpu().numpy(), i, axis=0)
        other_scores = scores[:i] + scores[i+1:]
        pred = knn_predict(query_embed, other_embeds, other_scores, k=args.k)
        predictions.append(pred)

    print("Saving predictions...")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f)

    print("Done!")

# -----------------------------
# Command-line Arguments
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=str, default="data/queries.txt", help="Path to query file")
    parser.add_argument("--score_file", type=str, default="data/scores.txt", help="Path to ground-truth performance scores")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence encoder to use")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store predictions")

    args = parser.parse_args()
    main(args)
