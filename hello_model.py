
import time
from foundry_local_sdk import Configuration, FoundryLocalManager

# Foundry Local başlat
config = Configuration(app_name="rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

alias = "phi-3.5-mini"
print(f"Model hazırlanıyor: {alias} ...")

model = manager.catalog.get_model(alias)

# İndirilmişse tekrar indirmez
model.download()

print("Model yükleniyor...")
model.load()

# ÖNEMLİ: model warm-up süresi
print("Modelin tamamen hazır olması için 30 saniye bekleniyor...")
time.sleep(30)

client = model.get_chat_client()

messages = [
    {
        "role": "user",
        "content": "Merhaba! Bana tek cümleyle RAG nedir açıkla."
    }
]

print("İstek gönderiliyor...")

try:
    response = client.complete_chat(messages)

    print("\nMODEL CEVABI:")
    print(response.choices[0].message.content)

except Exception as e:
    print("\nHATA:")
    print(type(e).__name__)
    print(str(e))

finally:
    print("\nModel bellekten boşaltılıyor...")
    model.unload()

