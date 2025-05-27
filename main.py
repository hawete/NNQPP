# main.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util

def load_data(query_file, doc_file):
    queries = pd.read_csv(query_file, sep='\t')
    documents = pd.read_csv(doc_file, sep='\t')
    return queries, documents

def encode_texts(texts, model):
    return model.encode(texts, convert_to_tensor=True)

def main():
    # Load real data
    queries, documents = load_data("data/queries.tsv", "data/documents.tsv")

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode queries and documents
    query_embeddings = encode_texts(queries["query"].tolist(), model)
    doc_texts = (documents["title"] + " " + documents["body"]).tolist()
    doc_embeddings = encode_texts(doc_texts, model)

    # Compute similarity (e.g., for the first query)
    scores = util.cos_sim(query_embeddings[0], doc_embeddings)
    top_k = min(5, len(documents))
    top_results = scores[0].topk(top_k)

    print(f"\nQuery: {queries['query'][0]}")
    print("\nTop documents:")
    for score, idx in zip(*top_results):
        print(f"{documents.iloc[idx]['docid']}: {score:.4f}")

if __name__ == "__main__":
    main()
