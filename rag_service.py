import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-mpnet-base-v2"
RAG_THRESHOLD = 0.65

# Load embedding model once
model = SentenceTransformer(MODEL_NAME)

# Load embeddings once
knowledge_embeddings = np.load("embeddings/knowledge_embeddings.npy")

# Load chunks once
with open("knowledge_chunks.json", "r") as f:
    knowledge_chunks = json.load(f)


def retrieve_context(user_query: str):
    """
    Performs semantic retrieval using embeddings.
    Returns (context, score)
    """

    query_embedding = model.encode([user_query])

    similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    best_chunk = knowledge_chunks[best_index]

    return best_chunk, float(best_score)