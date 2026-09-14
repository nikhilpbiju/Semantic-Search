import faiss
import numpy as np

embeddings = np.load("data/embeddings.npy")

dim = embeddings.shape[1]

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "data/faiss.index")

print("Index built with", index.ntotal, "vectors")