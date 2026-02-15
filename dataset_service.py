import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# config
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDINGS_FILE = "embeddings/alpaca_embeddings.npy"
METADATA_FILE = "alpaca_metadata.json"

model = SentenceTransformer(MODEL_NAME)
dataset_embeddings = np.load(EMBEDDINGS_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def match_dataset(user_query: str):
    """
    Returns best matched dataset answer and similarity score.
    """

    # Generate query embedding
    query_embedding = model.encode(user_query, convert_to_numpy=True)

    # Compute cosine similarity
    scores = util.cos_sim(query_embedding, dataset_embeddings)[0]
    scores = scores.cpu().numpy()

    # Best match
    best_index = np.argmax(scores)
    best_score = scores[best_index]

    best_answer = metadata[best_index]["output"]
    print(f"Best Match Answer: {best_answer}")
    print(f"Best Match Score: {best_score}")

    return best_answer, float(best_score)