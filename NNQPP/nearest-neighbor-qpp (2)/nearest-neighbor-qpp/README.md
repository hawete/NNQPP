# Nearest Neighbor-based Query Performance Prediction (QPP)

This repository implements a simple yet effective QPP method using the nearest neighbor principle in semantic space.

## 📚 Dataset

We use public datasets:
- [MS MARCO Passage Ranking](https://microsoft.github.io/msmarco/)
- [TREC Deep Learning Tracks](https://trec.nist.gov/data/deep.html)

To download the datasets:
```bash
cd data
python download_data.py
```

## 🚀 Run Nearest Neighbor QPP

```bash
python src/knn_qpp.py --query_embedding_path ./embeddings/queries.pt
```

## 📦 Requirements

```bash
pip install -r requirements.txt
```

## 🔖 License

MIT License (see LICENSE file).
