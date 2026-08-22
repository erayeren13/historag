import streamlit as st

from embedding import load_embedding_model
from retriever import semantic_search
from generator import load_llm, generate_answer
from rag import build_context, TOP_K


st.set_page_config(
    page_title="Yerel Tarih RAG Asistanı",
    page_icon="📜"
)

st.title("📜 Yerel Tarih RAG Asistanı")
st.caption(
    "Tamamen çevrimdışı çalışır — Microsoft Foundry Local üzerinde "
    "yerel embedding + LLM ile, internet bağlantısı gerekmez."
)


# ============================================================
# MODELLERİ BİR KEZ YÜKLE (session boyunca cache'lenir)
# ============================================================

@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def get_embedding_client():
    return load_embedding_model()


@st.cache_resource(show_spinner="Foundry Local LLM yükleniyor (ilk seferde birkaç dakika sürebilir)...")
def get_llm_client():
    return load_llm()


embedding_client = get_embedding_client()
llm_client = get_llm_client()


# ============================================================
# SOHBET GEÇMİŞİ
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['source']}**, Sayfa "
                        f"{source['page']} "
                        f"(benzerlik: {source['similarity']:.3f})"
                    )
                    st.caption(source["text"])


# ============================================================
# SORU-CEVAP
# ============================================================

question = st.chat_input("Tarih hakkında bir soru sor...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Dokümanlarda aranıyor..."):
            results = semantic_search(
                question, embedding_client, top_k=TOP_K
            )

        if not results:
            answer = "Bu bilgi sağlanan dokümanlarda bulunmuyor."
            sources = []
        else:
            context = build_context(results)

            with st.spinner("Cevap üretiliyor..."):
                answer = generate_answer(llm_client, question, context)

            if not answer:
                answer = "Cevap üretilirken bir hata oluştu, lütfen tekrar deneyin."

            sources = [
                {
                    "source": r["metadata"].get("source", "Unknown"),
                    "page": r["metadata"].get("page", "Unknown"),
                    "similarity": r["similarity"],
                    "text": r["document"],
                }
                for r in results
            ]

        st.markdown(answer)

        if sources:
            with st.expander("Kaynaklar"):
                for source in sources:
                    st.markdown(
                        f"- **{source['source']}**, Sayfa "
                        f"{source['page']} "
                        f"(benzerlik: {source['similarity']:.3f})"
                    )
                    st.caption(source["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )