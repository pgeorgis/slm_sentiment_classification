import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import IMDB_INDEX_FIELD, IMDB_REVIEW_TEXT_FIELD
from load_imdb_data import load_imdb


def get_embedding_model(name: str) -> SentenceTransformer:
    """Return a sentence embedding transformer model."""
    return SentenceTransformer(name)


def embed_text(model: SentenceTransformer, texts: list) -> np.ndarray:
    """Embed text(s) with a given embedding transformer model."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings


def create_embedding_index(embedding_model: SentenceTransformer, texts_with_ids: list):
    """
    Create a dictionary mapping text identifiers to embeddings.
    
    Args:
        embedding_model: A pretrained SentenceTransformer model.
        texts_with_ids: A list of tuples where each tuple is (identifier, text).
    
    Returns:
        dict: A dictionary mapping identifiers to their embeddings.
    """
    ids, texts = zip(*texts_with_ids)
    embeddings = embed_text(embedding_model, texts)
    embedding_index = {id_: emb for id_, emb in zip(ids, embeddings)}
    return embedding_index


def save_embedding_index(index: dict, file_path: str):
    """
    Save the embedding index to a binary file using pickle.
    
    Args:
        index: The embedding index dictionary to save.
        file_path: Path to the binary file.
    """
    with open(file_path, "wb") as file:
        pickle.dump(index, file)


def load_embedding_index(file_path: str) -> dict:
    """
    Load the embedding index from a binary file using pickle.
    
    Args:
        file_path: Path to the binary file.
        
    Returns:
        dict: The loaded embedding index.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)

# Load default sentence embedding model
EMBEDDING_MODEL = get_embedding_model("all-MiniLM-L6-v2")

if __name__ == "__main__":
    # Load IMDB train set
    imdb_train_data = load_imdb("train")

    # Extract tuples (index, review text) from train set
    texts_with_ids = [
        (row[IMDB_INDEX_FIELD], row[IMDB_REVIEW_TEXT_FIELD])
        for _, row in imdb_train_data.iterrows()
    ]

    # Create IMDB embedding index
    imdb_embedding_index = create_embedding_index(EMBEDDING_MODEL, texts_with_ids)

    # Write embedding index to pickle file
    embedding_dir = "embedding"
    os.makedirs(embedding_dir, exist_ok=True)
    embedding_index_file = os.path.join(embedding_dir, "imdb_embedding_index.pkl")
    save_embedding_index(imdb_embedding_index, embedding_index_file)
