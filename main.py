from src.predict import predict_query_performance

if __name__ == "__main__":
    # Example usage
    query = "information retrieval techniques"
    predicted_score = predict_query_performance(query)
    print(f"Predicted performance score: {predicted_score}")
