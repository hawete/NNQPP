from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from src.predict import predict_performance

# Load the MS MARCO dataset from Hugging Face
print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v2.1", split="train[:1%]")  # Use 1% for speed

# Use first query and its top 3 passages as a test example
query = dataset[0]['query']
documents = dataset[0]['passages']['passage_text'][:3]

print(f"\nQuery: {query}")
print(f"Documents: {documents}")

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute embeddings
query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

# Predict QPP score
score = predict_performance(query_embedding, doc_embeddings)
print(f"\nPredicted performance score: {score:.3f}")
