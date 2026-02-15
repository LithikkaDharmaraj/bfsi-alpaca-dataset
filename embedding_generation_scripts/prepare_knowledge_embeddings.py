import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"

def flatten_knowledge(data):
    chunks = []

    def extract(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                extract(value)
        elif isinstance(obj, str):
            chunks.append(obj)

    extract(data)
    return chunks


# Load JSON
with open("../data/knowledge_base.json", "r") as f:
    kb_data = json.load(f)

# Flatten
knowledge_chunks = flatten_knowledge(kb_data)

print(f"Total knowledge chunks: {len(knowledge_chunks)}")

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Generate embeddings
embeddings = model.encode(knowledge_chunks)

# Save
np.save("../embeddings/knowledge_embeddings.npy", embeddings)

# Save chunks separately
with open("knowledge_chunks.json", "w") as f:
    json.dump(knowledge_chunks, f, indent=2)

print("Knowledge embeddings saved successfully.")
