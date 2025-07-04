

from transformers import pipeline


# Load summarization pipeline with pretrained model (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Long text input
text = """
Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed to do so.
In data science, an algorithm is a sequence of statistical processing steps. In machine learning, algorithms are 'trained' to find patterns and features in massive amounts of data.
The better the algorithm, the more accurate the predictions and insights.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)[0]["summary_text"]

print("📄 Original Text:\n", text)
print("\n📝 Summary:\n", summary)

