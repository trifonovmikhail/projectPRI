from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st
import numpy as np

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('s-nlp/russian_toxicity_classifier')
    model = AutoModelForSequenceClassification.from_pretrained('s-nlp/russian_toxicity_classifier')
    return tokenizer, model

def classify_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return scores

st.title("Классификатор токсичности текста")

user_input = st.text_area("Введите текст:", height=150)

tokenizer, model = load_model()

if st.button("Анализировать"):
    if user_input.strip():
        scores = classify_text(user_input, tokenizer, model)
        non_toxic, toxic = scores
        st.write(f"**Нейтральный:** {non_toxic:.2f}")
        st.write(f"**Токсичный:** {toxic:.2f}")
    else:
        st.warning("Пожалуйста, введите текст для анализа.")
