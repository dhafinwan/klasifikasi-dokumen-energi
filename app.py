

import re
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Dokumen",
    layout="centered",
    initial_sidebar_state="expanded"
)

# GANTI DENGAN REPO HUGGING FACE-MU
MODEL_NAME = "Dhafinwan/model-klasifikasi-model"
MIN_CHAR  = 20

CLASS_DESC = {
    "Energi": "Konversi energi, termodinamika, sistem pembangkit, dan efisiensi termal.",
    "Manufaktur": "Proses permesinan, kontrol kualitas, optimasi proses, dan CNC.",
    "Material": "Karakterisasi material, uji mekanik, komposit, dan metalurgi."
}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL & CACHE
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# LOAD MODEL & CACHE
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
   
    MODEL_NAME = "Dhafinwan/model-klasifikasi-model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    
    id2label = {0: "Energi", 1: "Manufaktur", 2: "Material"}
    label2id = {"Energi": 0, "Manufaktur": 1, "Material": 2}

    return {
        "tokenizer"  : tokenizer,
        "model"      : model,
        "device"     : device,
        "id2label"   : id2label,
        "label2id"   : label2id,
        "max_length" : 256,
        "num_labels" : 3,
    }

# ─────────────────────────────────────────────────────────────
# FUNGSI PEMROSESAN TEKS
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    # Disamakan dengan proses training agar hasil inference tidak berbeda jauh.
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text: str, resources: dict) -> dict:
    t0 = time.time()
    text_clean = clean_text(text)
    
    inputs = resources["tokenizer"](
        text_clean,
        padding="max_length",
        truncation=True,
        max_length=resources["max_length"],
        return_tensors="pt"
    ).to(resources["device"])

    with torch.no_grad():
        outputs = resources["model"](**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(logits, dim=1).item()

    pred_label = resources["id2label"][pred_idx]
    confidence = probs[pred_idx].item()
    
    elapsed = (time.time() - t0) * 1000

    return {
        "text_raw": text,
        "text_clean": text_clean,
        "label": pred_label,
        "confidence": confidence,
        "elapsed_ms": elapsed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ─────────────────────────────────────────────────────────────
# KOMPONEN SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Informasi Sistem")
st.sidebar.write("Aplikasi ini menggunakan arsitektur IndoBERT yang telah dilatih untuk mengidentifikasi konteks teknis dari dokumen abstrak jurnal Teknik Mesin. Proses pembersihan teks di aplikasi dibuat sama dengan proses saat pelatihan model.")

st.sidebar.subheader("Kategori Label")
for label, desc in CLASS_DESC.items():
    st.sidebar.write(f"**{label}**")
    st.sidebar.caption(desc)

# ─────────────────────────────────────────────────────────────
# APLIKASI UTAMA (MAIN)
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

with st.spinner("Memuat model pembelajaran mendalam..."):
    resources = load_model()

# Header
st.title("Klasifikasi Dokumen Akademik")
st.caption("Analisis Semantik Abstrak Teknik Mesin berbasis IndoBERT")

# Input Area
user_input = st.text_area(
    label="Masukkan Teks Abstrak",
    label_visibility="collapsed",
    placeholder="Ketik atau tempel teks abstrak jurnal Teknik Mesin di sini...",
    height=200
)

# Tombol Prediksi menggunakan parameter native Streamlit (type="primary")
if st.button("PROSES KLASIFIKASI", type="primary", use_container_width=True):
    if len(user_input.strip()) < MIN_CHAR:
        st.warning("Teks terlalu pendek. Mohon masukkan abstrak yang lebih komprehensif.")
    else:
        with st.spinner("Menganalisis pola teks..."):
            result = predict_text(user_input, resources)
            st.session_state.history.insert(0, result)

        st.divider()
        st.subheader("Hasil Analisis")
        
        lbl = result['label']
        conf = result['confidence']

        # Menampilkan Label Hasil
        st.header(lbl.upper())
        st.write(CLASS_DESC[lbl])
        
        st.write("") # Spasi kosong

        # Menggunakan native metrics Streamlit
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Keyakinan Model", value=f"{conf * 100:.2f}%")
        col2.metric(label="Waktu Proses", value=f"{result['elapsed_ms']:.2f} ms")
        col3.metric(label="Panjang Teks", value=f"{len(result['text_clean'].split())} Kata")

        # Menggunakan native progress bar untuk keyakinan
        st.progress(float(conf), text="Visualisasi Tingkat Keyakinan")

# Riwayat
if st.session_state.history:
    st.divider()
    st.subheader("Riwayat Klasifikasi")
    
    history_data = []
    for item in st.session_state.history:
        history_data.append({
            "Waktu": item["timestamp"].split(" ")[1],
            "Prediksi": item["label"],
            "Keyakinan": f"{item['confidence']*100:.1f}%",
            "Cuplikan Teks": item["text_raw"][:80] + "..."
        })
    
    # Native Streamlit DataFrame
    st.dataframe(pd.DataFrame(history_data), hide_index=True, use_container_width=True)
