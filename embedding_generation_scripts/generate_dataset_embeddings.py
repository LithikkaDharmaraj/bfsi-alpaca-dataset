import json
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
JSON_FILE = "../data/alpaca_bfsi_sample.json"
EMBEDDINGS_FILE = "../embeddings/alpaca_embeddings.npy"
METADATA_FILE = "alpaca_metadata.json"
# MODEL_NAME = "all-MiniLM-L6-v2"   # Small + fast (384 dim)
MODEL_NAME = "all-mpnet-base-v2"

model = SentenceTransformer(MODEL_NAME)

print("Loading Alpaca dataset...")
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

match_texts = []
metadata = []

for entry in data:
    instruction = entry["instruction"]
    input_text = entry["input"]
    output_text = entry["output"]

    # Combine instruction + input
    match_text = instruction + " " + input_text
    match_texts.append(match_text)

    # Store metadata for later retrieval
    metadata.append({
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    })

print(f"Total records loaded: {len(match_texts)}")

print("Generating embeddings...")
embeddings = model.encode(match_texts, convert_to_numpy=True)

print("Embedding shape:", embeddings.shape)

# SAVE EMBEDDINGS
np.save(EMBEDDINGS_FILE, embeddings)

# Save metadata
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Embeddings saved to:", EMBEDDINGS_FILE)
print("Metadata saved to:", METADATA_FILE)
print("Done.")
