import time
import sys

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

from embedding import load_embedding_model
from retriever import semantic_search


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ALIAS = "phi-3.5-mini"

TOP_K = 3

MAX_GENERATION_ATTEMPTS = 2

# Bilinen CPU varyant ID'leri (bazı GPU/WebGPU varyantları bu
# makinede f16 desteği olmadığı için "Operation was cancelled"
# hatası veriyordu — CPU varyantını açıkça seçiyoruz).
KNOWN_CPU_VARIANT_IDS = [
    f"{MODEL_ALIAS}-instruct-generic-cpu:4",
    f"{MODEL_ALIAS}-instruct-generic-cpu:3",
    f"{MODEL_ALIAS}-instruct-generic-cpu:2",
    f"{MODEL_ALIAS}-instruct-generic-cpu:1",
]


# ============================================================
# SYSTEM PROMPT
# ============================================================
# Spesifikasyon gereği: halüsinasyon yasağı, kaynak atfı,
# "bilmeme durumu" fallback'i.

SYSTEM_PROMPT = """
Sen bir tarih dokümanı soru-cevap asistanısın.

Kurallar:

1. SADECE aşağıda sana verilen bağlamdaki (context) bilgileri
   kullanarak cevap ver. Kendi genel bilginden veya tahminden
   kesinlikle yararlanma.
2. Bağlamda yer almayan hiçbir bilgiyi uydurma.
3. Eğer sorulan bilgi verilen bağlamda yoksa, kesinlikle şu şekilde
   belirt: "Bu bilgi sağlanan dokümanlarda bulunmuyor."
4. Cevabını verirken, bilgiyi hangi kaynaktan/sayfadan aldığını
   belirt (örn. "(Kaynak: ata.pdf, Sayfa 2)").
5. Kısa ve net cevap ver, gereksiz uzatma.
6. Türkçe cevap ver.
"""


# ============================================================
# 1. EMBEDDING MODEL + RETRIEVAL
# ============================================================

def retrieve_context(question: str):

    print("\n1. Embedding modeli hazırlanıyor...")

    embedding_client = load_embedding_model()

    print(f"\n2. İlgili dokümanlar aranıyor (top_k={TOP_K})...")

    results = semantic_search(question, embedding_client, top_k=TOP_K)

    if not results:
        print("Hiç sonuç bulunamadı. Veritabanı boş olabilir.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        print(f"\n[{i}] Source: {source} | Page: {page} | "
              f"Similarity: {result['similarity']:.4f}")
        print(result["document"])

        context_parts.append(
            f"[Kaynak: {source}, Sayfa {page}]\n{result['document']}"
        )

    context = "\n\n".join(context_parts)

    return context


# ============================================================
# 2. FOUNDRY LOCAL SETUP
# ============================================================

def setup_foundry_local():

    print("\n3. Foundry Local başlatılıyor...")

    try:
        config = Configuration(app_name="historag")

        try:
            FoundryLocalManager.initialize(config)
        except FoundryLocalException as e:
            # embedding.py (load_embedding_model) already initialized
            # the singleton for the embedding model — that's fine,
            # we just reuse the existing instance.
            if "already been initialized" not in str(e):
                raise

        manager = FoundryLocalManager.instance

        print("   Foundry Local manager hazır.")

    except Exception as e:
        print("\nFoundry Local başlatılamadı.")
        print(type(e).__name__)
        print(str(e))
        sys.exit(1)

    print("\n4. Execution provider'lar kontrol ediliyor...")

    try:
        eps = manager.discover_eps()

        for ep in eps:
            status = "kayıtlı" if ep.is_registered else "kayıtlı değil"
            print(f"   - {ep.name}: {status}")

        if not any(ep.is_registered for ep in eps):
            print("   Hiçbir EP kayıtlı değil, indirilip kaydediliyor...")
            result = manager.download_and_register_eps()
            print(f"   Success: {result.success}, Status: {result.status}")

    except Exception as e:
        print("\nEP kontrolü sırasında hata oluştu (devam ediliyor).")
        print(type(e).__name__)
        print(str(e))

    return manager


def get_cpu_model(manager):

    print("\n5. Model katalogdan alınıyor (CPU varyantı)...")

    for variant_id in KNOWN_CPU_VARIANT_IDS:
        try:
            model = manager.catalog.get_model_variant(variant_id)
            if model is not None:
                print(f"   CPU varyantı bulundu: {variant_id}")
                return model
        except Exception:
            continue

    print("   Bilinen CPU varyant ID'leri bulunamadı, katalogda aranıyor...")

    try:
        base_model = manager.catalog.get_model(MODEL_ALIAS)

        if base_model is None:
            print(f"\nModel katalogda bulunamadı: {MODEL_ALIAS}")
            sys.exit(1)

        base_model.download()

        cached = manager.catalog.get_cached_models()

        cpu_candidates = [
            v for v in cached
            if getattr(v, "alias", "") == MODEL_ALIAS
            and "cpu" in getattr(v, "id", "").lower()
        ]

        if cpu_candidates:
            print(f"   CPU varyantı bulundu: {cpu_candidates[0].id}")
            return cpu_candidates[0]

        print(
            "   Uyarı: CPU varyantı otomatik bulunamadı, "
            "GPU varyantıyla devam edilecek (hataya sebep olabilir)."
        )
        return base_model

    except Exception as e:
        print("\nModel katalogdan alınamadı.")
        print(type(e).__name__)
        print(str(e))
        sys.exit(1)


def load_chat_model(manager):

    model = get_cpu_model(manager)

    print("\n6. Model download kontrolü yapılıyor...")

    try:
        model.download()
        print("   Model download kontrolü tamamlandı.")
    except Exception as e:
        print("\nModel download sırasında hata oluştu.")
        print(type(e).__name__)
        print(str(e))
        sys.exit(1)

    print("\n7. Model yükleniyor...")

    try:
        model.load()
        print("   Model başarıyla yüklendi.")
    except Exception as e:
        print("\nModel yüklenemedi.")
        print(type(e).__name__)
        print(str(e))
        sys.exit(1)

    print("\n8. Model hazırlanıyor...")
    print("   Modelin tamamen hazır olması için 10 saniye bekleniyor...")

    time.sleep(10)

    print("\n9. Chat client oluşturuluyor...")

    try:
        client = model.get_chat_client()
        print("   Chat client hazır.")
        return model, client

    except Exception as e:
        print("\nChat client oluşturulamadı.")
        print(type(e).__name__)
        print(str(e))

        try:
            model.unload()
        except Exception:
            pass

        sys.exit(1)


# ============================================================
# 3. GENERATE ANSWER
# ============================================================

def _generate_answer_impl(client, context: str, question: str):

    user_prompt = f"""
Aşağıda tarih dokümanından alınmış bağlam (context) bulunuyor:

============================================================

{context}

============================================================

Soru:

{question}

============================================================

Yukarıdaki bağlamı kullanarak soruyu cevapla. Bağlamda bilgi yoksa
bunu açıkça belirt.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    print("\n10. Model cevap üretiyor...")

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):

        try:
            response = client.complete_chat(messages)
            return response.choices[0].message.content

        except FoundryLocalException as e:
            print(f"\nDeneme {attempt}/{MAX_GENERATION_ATTEMPTS} "
                  f"başarısız: {e}")

            if attempt < MAX_GENERATION_ATTEMPTS:
                print("Kısa bir bekleme sonrası tekrar deneniyor...")
                time.sleep(5)
            else:
                print("\nModel cevap üretemedi.")
                return None

        except Exception as e:
            print("\nModel cevap üretirken beklenmeyen hata oluştu.")
            print(type(e).__name__)
            print(str(e))
            return None


# ============================================================
# CONVENIENCE WRAPPERS (rag.py bunları bekliyor)
# ============================================================

def load_llm():
    """
    rag.py orchestrator'ının kullandığı basit arayüz: Foundry Local'i
    kurar, CPU varyantını yükler ve sadece chat client'ı döndürür.
    """

    manager = setup_foundry_local()
    _model, client = load_chat_model(manager)

    return client


def generate_answer(client, question: str, context: str):
    """
    rag.py'nin çağırdığı sırayla (client, question, context) — asıl
    üretim mantığı yukarıdaki generate_answer(client, context,
    question) fonksiyonunda, burada sadece parametre sırasını
    uyarlıyoruz.
    """

    return _generate_answer_impl(client, context, question)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOCAL RAG GENERATOR")
    print("=" * 70)

    question = input("\nTarih hakkında bir soru sor: ").strip()

    if not question:
        print("Soru boş bırakılamaz.")
        sys.exit(1)

    context = retrieve_context(question)

    manager = setup_foundry_local()

    model, client = load_chat_model(manager)

    answer = _generate_answer_impl(client, context, question)

    if answer:
        print("\n")
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(answer)

        print("\n")
        print("=" * 70)
        print("RAG PIPELINE TAMAMLANDI")
        print("=" * 70)

    print("\nModel bellekten boşaltılıyor...")

    try:
        model.unload()
        print("Model unloaded.")
    except Exception as e:
        print(f"Model unload sırasında hata: {e}")


if __name__ == "__main__":
    main()