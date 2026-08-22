import numpy as np

from embedding import (
    load_embedding_model,
    create_query_embedding
)

from database import get_all_documents


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(
        vector_a,
        dtype=np.float32
    )

    vector_b = np.array(
        vector_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:

        return 0.0

    similarity = (
        np.dot(vector_a, vector_b)
        / denominator
    )

    return float(similarity)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    embedding_client,
    top_k=3
):

    print(
        "\nSemantic search yapılıyor..."
    )

    # ========================================================
    # 1. QUERY EMBEDDING
    # ========================================================

    query_embedding = create_query_embedding(
        embedding_client,
        query
    )

    print(
        f"Query vector boyutu: "
        f"{len(query_embedding)}"
    )

    # ========================================================
    # 2. SQLITE DOCUMENTS
    # ========================================================

    documents = get_all_documents()

    print(
        f"SQLite documents: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "Database içerisinde doküman bulunamadı."
        )

        return []

    # ========================================================
    # 3. COSINE SIMILARITY
    # ========================================================

    results = []

    for document in documents:

        similarity = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append(
            {
                "id": document["id"],
                "document": document["document"],
                "metadata": document["metadata"],
                "similarity": similarity
            }
        )

    # ========================================================
    # 4. SORT
    # ========================================================

    results.sort(
        key=lambda result: result["similarity"],
        reverse=True
    )

    # ========================================================
    # 5. TOP-K
    # ========================================================

    return results[:top_k]


# ============================================================
# CONVENIENCE WRAPPER (rag.py bunu bekliyor)
# ============================================================

def retrieve(question, top_k=3):
    """
    rag.py orchestrator'ının kullandığı basit arayüz: embedding
    modelini kendi içinde yükleyip semantic_search'ü çalıştırır,
    böylece rag.py bir embedding client yönetmek zorunda kalmaz.
    """

    embedding_client = load_embedding_model()

    return semantic_search(question, embedding_client, top_k=top_k)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOCAL RAG RETRIEVER")
    print("=" * 60)

    # ========================================================
    # QUESTION
    # ========================================================

    question = input(
        "\nTarih hakkında bir soru sor: "
    )

    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    embedding_client = load_embedding_model()

    # ========================================================
    # SEARCH
    # ========================================================

    results = semantic_search(
        question,
        embedding_client,
        top_k=20
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("TOP RELEVANT DOCUMENTS")
    print("=" * 60)

    if not results:

        print(
            "\nSonuç bulunamadı."
        )

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print()
        print(
            f"RESULT {index}"
        )

        print("-" * 60)

        print(
            f"Source: "
            f"{metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section')}"
        )

        print(
            f"Cosine Similarity: "
            f"{result['similarity']:.4f}"
        )

        print()

        print(
            result["document"]
        )