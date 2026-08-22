import re
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from foundry_local_sdk import Configuration, FoundryLocalManager


# ==================================================
# 1. Belgeyi oku
# ==================================================
text = Path("docs/rag_nedir.txt").read_text(encoding="utf-8")
# Başlık + devam eden satırları birlikte tut
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

chunks = []
current = ""

for p in paragraphs:
    current += p + "\n\n"

    # Eğer paragraf numaralı liste içeriyorsa devam et
    if re.search(r"\d+\.", p):
        continue

    if len(current) > 300:
        chunks.append(current.strip())
        current = ""

if current:
    chunks.append(current.strip())

print(f"Toplam chunk: {len(chunks)}")

# ==================================================
# 2. Embedding modeli
# ==================================================
print("Embedding modeli yükleniyor...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = embedder.encode(chunks)

# ==================================================
# 3. Kullanıcı sorusu
# ==================================================
question = "RAG sisteminin temel adımları nelerdir?"
print(f"\nSORU: {question}")

question_embedding = embedder.encode([question])

scores = cosine_similarity(question_embedding, chunk_embeddings)[0]
top_indices = np.argsort(scores)[::-1][:2]

# En ilgili chunk'ları seç
selected_chunks = [chunks[i] for i in top_indices]
context = "\n\n".join(selected_chunks)

print("\nBulunan ilgili bilgiler:")
print(context)

# ==================================================
# 4. Foundry Local başlat
# ==================================================
print("\nFoundry başlatılıyor...")

config = Configuration(app_name="rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

alias = "phi-3.5-mini"
model = manager.catalog.get_model(alias)

model.download()
model.load()

print("Model hazır olması için bekleniyor...")
time.sleep(20)

client = model.get_chat_client()

# ==================================================
# 5. Gerçek RAG promptu
# ==================================================
prompt = f"""
Sen bir RAG asistanısın.

Sadece aşağıdaki BELGE parçalarını kullanarak cevap ver.
Belgede olmayan bilgi ekleme.

BELGE PARÇALARI:
{context}

SORU:
{question}

Kısa, maddeli ve anlaşılır cevap ver.
"""

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

print("\nModel cevap üretiyor...")

response = client.complete_chat(messages)

# ==================================================
# 6. Sonuç
# ==================================================
print("\n" + "=" * 60)
print("GERÇEK RAG CEVABI")
print("=" * 60)
print(response.choices[0].message.content)

model.unload()

