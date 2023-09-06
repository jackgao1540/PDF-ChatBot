import os
from sentence_transformers import SentenceTransformer
import numpy as np

data_files = [file for file in os.listdir() if file.startswith("data") and file.endswith(".txt")]

data_files.sort(key=lambda x: int(x.split("data")[1].split(".txt")[0]))

print("Loading SentenceTransformer model...")
embeddings_model = SentenceTransformer("sentence-transformers/gtr-t5-large")

for file_name in data_files:
    with open(file_name, 'r', encoding='utf-8') as f:
        paragraphs = f.read().split("\n\n")
        print(f"Loaded {file_name}")
    embeddings = embeddings_model.encode(paragraphs, show_progress_bar=True)
    np_embeddings = np.array(embeddings)
    
    embeddings_file_name = f"embeddings_{file_name.split('data')[1].split('.txt')[0]}.npy"
    np.save(embeddings_file_name, np_embeddings)
    print(f"Saved embeddings for {file_name}")
