from transformers import pipeline

# Load translation pipeline for English → Spanish
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

# Input text in English
text = "Machine learning enables computers to learn from data and make decisions without being explicitly programmed."

# Translate
result = translator(text, max_length=100)[0]['translation_text']

print("📘 English:\n", text)
print("\n📙 Spanish Translation:\n", result)
