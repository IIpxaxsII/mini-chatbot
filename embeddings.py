from sentence_transformers import SentenceTransformer

# Load the embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text_chunks: list[str]):
    """
    Convert a list of text chunks into embedding vectors.
    Each chunk becomes one semantic vector.
    """
    embeddings = embed_model.encode(text_chunks)
    return embeddings