from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"

)

embeddings = embed_model.encode(
    "hello i am paras, dealing with embeddings"
)

print(embeddings)