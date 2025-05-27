import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(save_path, 'wb') as f, tqdm(
        desc=save_path,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in r.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

if __name__ == "__main__":
    os.makedirs('datasets', exist_ok=True)
    url = 'https://msmarco.blob.core.windows.net/msmarcoranking/trec-dl-2020/qrels.trec-dl-2020.txt'
    save_path = 'datasets/qrels_2020.txt'
    download_file(url, save_path)
