import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from constants import IMDB_INDEX_FIELD, IMDB_REVIEW_TEXT_FIELD, logger
from load_imdb_data import get_wordcount, load_imdb

EMBEDDING_DIR = os.path.abspath("embedding")
EMBEDDING_INDEX_FILE = os.path.join(EMBEDDING_DIR, "imdb_embedding_index.pkl")
EMBEDDING_ID_MAP_FILE = os.path.join(EMBEDDING_DIR, "imdb_id_map.pkl")


def get_embedding_model(name: str) -> SentenceTransformer:
    """Return a sentence embedding transformer model."""
    return SentenceTransformer(name)


def embed_text(model: SentenceTransformer, texts: list) -> np.ndarray:
    """Embed text(s) with a given embedding transformer model."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def create_faiss_index(embeddings: np.ndarray, ids: list):
    """Create a FAISS vector embedding index.

    Args:
        embeddings: A 2D numpy array of embeddings.
        ids: A list of identifiers corresponding to the embeddings.

    Returns:
        index: A FAISS index object.
        id_map: A dictionary mapping FAISS indices to original identifiers.
    """
    # Create a FAISS index
    d = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    # Map FAISS internal indices to original IDs
    id_map = {i: ids[i] for i in range(len(ids))}
    
    return index, id_map


def save_faiss_index(index: faiss.IndexFlatL2, id_map: dict, index_path: str, id_map_path: str):
    """Save FAISS index and ID map."""
    faiss.write_index(index, index_path)
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map, f)


def load_faiss_index(index_path: str, id_map_path: str):
    """Load FAISS index and ID map from disk."""
    index = faiss.read_index(index_path)
    with open(id_map_path, "rb") as f:
        id_map = pickle.load(f)
    return index, id_map


def retrieve_most_similar_texts(query_text: str,
                                embedding_model: SentenceTransformer,
                                index: faiss.IndexFlatL2,
                                id_map: dict,
                                top_n: int,
                                valid_ids: set = None):
    """Find the N most similar texts to a query using FAISS.

    Args:
        query_text: The input text to find similar texts for.
        embedding_model: The SentenceTransformer model used for embedding.
        index: The FAISS index for similarity search.
        id_map: A dictionary mapping FAISS indices to original identifiers.
        top_n: The number of most similar texts to retrieve.
        valid_ids: An optional set of valid IDs to narrow the search scope.

    Returns:
        list: A list of tuples (identifier, similarity_score), sorted by similarity.
    """
    # Embed the query text
    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)

    # If valid_ids is specified, create a temporary subset index
    valid_indices_map = {}
    if valid_ids is not None:
        idx = 0
        valid_indices = []
        for faiss_index, original_id in id_map.items():
            if original_id in valid_ids:
                valid_indices_map[idx] = original_id
                valid_indices.append(faiss_index)
                idx += 1
        
        # Convert to numpy array for FAISS indexing
        valid_embeddings = index.reconstruct_n(0, index.ntotal)[valid_indices]

        # Create a subset index
        subset_index = faiss.IndexFlatL2(valid_embeddings.shape[1])
        subset_index.add(valid_embeddings)
    else:
        subset_index = index

    # Search the FAISS index
    distances, indices = subset_index.search(query_embedding, top_n)

    # Map FAISS indices to original IDs
    if valid_ids is not None:
        index_map = valid_indices_map
    else:
        index_map = id_map
    results = [
        (index_map[idx], 1 / (1 + dist))
        for idx, dist in zip(indices[0], distances[0])
        if idx != -1 #and (valid_ids is None or id_map[idx] in valid_ids)
    ]
    return results


# Load default sentence embedding model
EMBEDDING_MODEL = get_embedding_model("all-MiniLM-L6-v2")


if __name__ == "__main__":
    # Load IMDB train set
    imdb_train_data = load_imdb("train")

    # Add wordcount field
    imdb_train_data["wordcount"] = imdb_train_data[IMDB_REVIEW_TEXT_FIELD].apply(get_wordcount)

    # Extract tuples (index, review text) from train set
    texts_with_ids = [
        (row[IMDB_INDEX_FIELD], row[IMDB_REVIEW_TEXT_FIELD])
        for _, row in imdb_train_data.iterrows()
        # Limit to relatively short reviews between 75-150 words
        if row["wordcount"] >= 75 and row["wordcount"] <= 150
    ]

    # Create IMDB embedding index
    ids, texts = zip(*texts_with_ids)
    embeddings = embed_text(EMBEDDING_MODEL, texts)
    imdb_embedding_index, id_map = create_faiss_index(embeddings, ids)

    # Write embedding index to pickle file
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    save_faiss_index(
        imdb_embedding_index, id_map,
        EMBEDDING_INDEX_FILE, EMBEDDING_ID_MAP_FILE
    )
    logger.info(f"Wrote FAISS vector embedding index to {EMBEDDING_INDEX_FILE}")
    logger.info(f"Wrote FAISS vector embedding ID map to {EMBEDDING_ID_MAP_FILE}")
