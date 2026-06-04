from pathlib import Path
from embeddings import embed_model, get_embeddings
from sentence_transformers import util
import faiss
import numpy as np

def load_text_file(file_path: str) -> str:
    """
    Load a plain text file and return its contents as a string.
    This is the first ingestion step of the project.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open("r", encoding="utf-8") as file:
        return file.read() 




def split_into_chunks(text: str) -> list:
    """
    Split the document into smaller semantic chunks.

    Currently we use paragraph-based chunking
    by splitting on double line breaks.
    """

    chunks = text.split("\n\n")

    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return cleaned_chunks





def retrieve_relevant_chunks(query, chunks, embeddings, embed_model, top_k=2):
    """
    Convert the user query into an embedding
    and return the top_k most relevant chunks.
    """

    query_embedding = embed_model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]

    top_indices = similarities.argsort(descending=True)[:top_k]

    top_chunks = []
    for idx in top_indices:
        top_chunks.append((chunks[idx], similarities[idx].item()))

    return top_chunks



def build_faiss_index(embeddings):
    """
    Build a FAISS index from chunk embeddings.
    This allows fast vector similarity search.
    """
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index



def search_faiss(query, embed_model, index, chunks, top_k=3):
    """
    Search the FAISS index using a query.
    Returns the most relevant chunks.
    """

    query_embedding = embed_model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results



def build_context(retrieved_chunks):
    """
    Turn retrieved chunks into a clean context block
    that can later be sent to the language model.
    """
    context = "\n\n"

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context += f"Context {i}:\n{chunk}\n\n"

    return context


def build_prompt(context, question):
    """
    Build the final prompt that will be sent to the language model.
    The prompt tells the model to answer only using the provided context.
    """
    prompt = f"""
You are a helpful AI assistant.

Use only the context below to answer the question.
If the answer is not in the context, say that you do not know.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


def main():
    file_path = "documents/knowledge.txt"

    text = load_text_file(file_path)
    chunks = split_into_chunks(text)

    # print(f"\nTotal chunks created: {len(chunks)}\n")

    # for index, chunk in enumerate(chunks):
    #     print(f"Chunk {index + 1}:\n")
    #     print(chunk)
    #     print("\n" + "-" * 50 + "\n")

    embeddings = get_embeddings(chunks)

    # print("Embeddings created successfully.")
    # print("Embedding shape:", embeddings.shape)
    index = build_faiss_index(embeddings)
    print("FAISS index built successfully.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break

        top_chunks = retrieve_relevant_chunks(
            query,
            chunks,
            embeddings,
            embed_model,
            top_k=2,
        )

        # print("\nMost Relevant Chunks:\n")

        # for i, (chunk, score) in enumerate(top_chunks, start=1):
        #     print(f"Chunk {i}:")
        #     print(chunk)
        #     print(f"Similarity Score: {score:.4f}")
        #     print("\n" + "-" * 50 + "\n")

        faiss_results = search_faiss(
            query,
            embed_model,
            index,
            chunks,
            top_k=3
        )

        # print("\nFAISS Results:\n")

        # for i, chunk in enumerate(faiss_results, start=1):
        #     print(f"Result {i}:")
        #     print(chunk)


        #     print("\n" + "=" * 50 + "\n")
            

        context = build_context(faiss_results)

        # print("\nBuilt Context:\n")
        # print(context)
        prompt = build_prompt(context, query)
        print("\nBuilt Prompt:\n")
        print(prompt)

if __name__ == "__main__":
    main()