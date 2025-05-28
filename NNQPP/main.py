from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from src.predict import QPPPredictor

# Load a subset of the MS MARCO dataset (adjust percentage or number as needed)
print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v2.1", split="train[:1%]")  # ~8,000 examples

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Instantiate the QPP predictor
qpp_predictor = QPPPredictor()

# Limit the number of queries to loop over for practicality
num_queries = 100

for i in range(num_queries):
    query = dataset[i]['query']
    documents = dataset[i]['passages']['passage_text'][:3]  # Use top 3 passages

    if not query or not documents:
        continue  # Skip if data is malformed

    # Compute embeddings
    query_embedding = model.encode(query)
    doc_embeddings = model.encode(documents)

    # Predict score
    score = qpp_predictor.predict_performance(query_embedding, doc_embeddings)

    # Print result
    print(f"\nQuery {i+1}: {query}")
    print(f"Documents: {documents}")
    print(f"Predicted performance score: {score:.3f}")
