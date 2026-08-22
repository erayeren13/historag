
import time
from pathlib import Path
from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Foundry başlatılıyor...")
config = Configuration(app_name="rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("2. Belge okunuyor...")
doc_path = Path("docs/rag_nedir.txt")
text = doc_path.read_text(encoding="utf-8")

chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
print(f"Chunk sayısı: {len(chunks)}")

print("3. Model hazırlanıyor...")
alias = "phi-3.5-mini"
model = manager.catalog.get_model(alias)

print("4. Download kontrolü...")
model.download()

print("5. Model yükleniyor...")
model.load()

print("6. 20 saniye bekleniyor...")
time.sleep(20)

client = model.get_chat_client()

question = "RAG sisteminin temel adımları nelerdir?"
context = "\n\n".join(chunks)

prompt = f"""
Sadece aşağıdaki belgeye dayanarak cevap ver.

BELGE:
{context}

SORU: {question}
"""

messages = [
    {"role": "user", "content": prompt}
]

print("7. Modele soru gönderiliyor...")

response = client.complete_chat(messages)

print("\n=== SONUÇ ===")
print(response.choices[0].message.content)

print("\n8. Model kapatılıyor...")
model.unload()

