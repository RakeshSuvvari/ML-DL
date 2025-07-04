# 🛠️ Transformer for Text Classification (with Hugging Face)


from transformers import pipeline

# Load a sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test sentences
sentences = [
    "I love studying machine learning!",
    "This is the worst movie I have ever seen.",
    "What is sentiment-analysis?"
]

# Run predictions
for sentence in sentences:
    result = classifier(sentence)[0]
    print(f"Text: {sentence}\n→ Label: {result['label']}, Score: {result['score']:.2f}\n")


