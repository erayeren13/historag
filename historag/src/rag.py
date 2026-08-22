from retriever import retrieve
from generator import load_llm, generate_answer


TOP_K = 3


def build_context(results):
    """
    Retrieved chunks'ları LLM'in kullanacağı
    tek bir context metnine dönüştürür.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown")

        text = result["document"]

        context_parts.append(
            f"""
SOURCE {i}
Source: {source}
Page: {page}
Section: {section}

{text}
"""
        )

    return "\n".join(context_parts)


def run_rag(question):
    """
    Complete RAG pipeline:

    Question
        ↓
    Embedding
        ↓
    Cosine Similarity
        ↓
    Top-K chunks
        ↓
    Context
        ↓
    Phi-3.5 Mini
        ↓
    Answer
    """

    print()
    print("=" * 60)
    print("LOCAL RAG ASSISTANT")
    print("=" * 60)

    # ============================================================
    # 1. RETRIEVAL
    # ============================================================

    print("\n[1/3] Dokümanlarda arama yapılıyor...")

    results = retrieve(
        question,
        top_k=TOP_K
    )

    if not results:

        return "Bu bilgi sağlanan dokümanlarda bulunmuyor."

    print(f"Toplam {len(results)} ilgili chunk bulundu.")

    # ============================================================
    # 2. CONTEXT OLUŞTURMA
    # ============================================================

    print("\n[2/3] Context oluşturuluyor...")

    context = build_context(results)

    # ============================================================
    # 3. LLM
    # ============================================================

    print("\n[3/3] Phi-3.5 Mini cevap oluşturuyor...")

    client = load_llm()

    answer = generate_answer(
        client,
        question,
        context
    )

    return answer


if __name__ == "__main__":

    question = input(
        "\nSorunuzu yazın: "
    )

    answer = run_rag(
        question
    )

    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)