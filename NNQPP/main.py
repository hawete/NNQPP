from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from src.predict import QPPPredictor

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
# Compute embeddings
query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

# Instantiate your QPP Predictor class
qpp_predictor = QPPPredictor()

# Predict QPP score using the instance
score = qpp_predictor.predict_performance(query_embedding, doc_embeddings)

print(f"\nPredicted performance rabouba score: {score:.3f}")
