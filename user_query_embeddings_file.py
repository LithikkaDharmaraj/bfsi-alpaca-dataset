from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# GET USER QUERY
# -----------------------------
user_query = input("\nEnter customer query: ")

# -----------------------------
# GENERATE EMBEDDING
# -----------------------------
query_embedding = model.encode(user_query, convert_to_numpy=True)

# -----------------------------
# PRINT INFO
# -----------------------------
print("\nEmbedding generated successfully.")
print("Embedding shape:", query_embedding.shape)
print("First 5 values:", query_embedding[:5])
