import streamlit as st
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Membaca data CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Membersihkan data dari karakter-karakter yang tidak diinginkan
    data['Tweet'] = data['Tweet'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

    # Memuat model yang telah disimpan
    model = load_model("model_sentimen.h5")

    # Menginisialisasi tokenizer yang sama dengan tokenizer pada saat pelatihan
    tokenizer = Tokenizer(num_words=5000, split=" ")
    tokenizer.fit_on_texts(data['Tweet'].values)

    # Membangun antarmuka web
    st.title("Analisis Sentimen Komentar")
    st.write("Upload file CSV dengan kolom 'komentar'")
    st.write("Kolom 'sentiment' akan diisi dengan nilai 'positif', 'negatif', atau 'netral'")

    # Input dari file CSV
    if st.button("Proses dari file CSV"):
        X = tokenizer.texts_to_sequences(data['Tweet'].values)
        X = pad_sequences(X, maxlen=124)
        y_pred = model.predict(X)
        y_pred = y_pred.argmax(axis=1)
        sentiment_map = {0: "negatif", 1: "netral", 2: "positif"}
        data['sentiment'] = [sentiment_map[s] for s in y_pred]
        st.write(data)

    # Input dari teks
    st.write("Input komentar:")
    text_input = st.text_input("")
    if st.button("Proses dari input kata"):
        X = tokenizer.texts_to_sequences([text_input])
        X = pad_sequences(X, maxlen=124)
        y_pred = model.predict(X)
        y_pred = y_pred.argmax(axis=1)
        sentiment_map = {0: "negatif", 1: "netral", 2: "positif"}
        st.write(f"Sentiment: {sentiment_map[y_pred[0]]}")
