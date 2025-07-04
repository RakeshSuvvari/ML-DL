# 💬 Question Answering (QA) with Transformers
# Ask a question based on a paragraph — the model finds and returns the most relevant answer.


from transformers import pipeline

# Load a pretrained QA pipeline
qa_pipeline = pipeline("question-answering")

# Provide context and question
context = """
Machine learning is a field of artificial intelligence that gives computers the ability to learn without being explicitly programmed.
It is used in various applications such as recommendation systems, autonomous vehicles, and voice assistants.
"""

question = "What is machine learning?"

# Get answer
result = qa_pipeline(question=question, context=context)

# Show result
print("❓ Question:", question)
print("📄 Answer:", result['answer'])
print("📊 Score:", round(result['score'], 3))
