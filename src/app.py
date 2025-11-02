import streamlit as st
import torch
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
from io import BytesIO

# path 'src' untuk impor modul lain
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dari file .py lain di folder src
from model_recognition import PointNet2Recognition 
try:
    from extract_morphometry_utils import run_morphometry_pipeline 
except ImportError:
    st.error("ERROR: Tidak dapat menemukan 'extract_morphometry_utils.py'.")
    # fungsi dummy agar aplikasi tidak crash total
    def run_morphometry_pipeline(points, labels):
        return pd.DataFrame({"Error": ["Fungsi morfometrik tidak ditemukan."]})

# --- KONFIGURASI APLIKASI ---
MODEL_PATH = "../models/best_recognition_triplet_model.pth"
GALLERY_DIR = "../data/test_set_npz" 
EMBEDDING_DIM = 256
THRESHOLD_JARAK = 1.0141 # Ganti dengan threshold dari evaluate_final.py

# --- CACHING FUNGSI BERAT ---
@st.cache_resource
def load_model(device):
    st.spinner("Memuat model AI...")
    model = PointNet2Recognition(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_gallery_embeddings(_model, device):
    st.spinner("Memuat galeri embedding pasien...")
    gallery_embeddings = []
    gallery_labels = []
    
    patient_folders = sorted([d for d in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, d))])

    with torch.no_grad():
        for patient_id in patient_folders:
            patient_path = os.path.join(GALLERY_DIR, patient_id)
            # Ambil 1 file per pasien (file asli, bukan augmentasi) untuk galeri
            file_path = glob.glob(os.path.join(patient_path, "*_final.npz"))
            if file_path:
                file_path = file_path[0]
                data = np.load(file_path)
                points_tensor = torch.from_numpy(data['points']).float().unsqueeze(0).to(device)
                embedding = _model(points_tensor).cpu().numpy()
                gallery_embeddings.append(embedding)
                gallery_labels.append(patient_id)
                
    return np.vstack(gallery_embeddings), gallery_labels

def create_2d_plot(points, labels):
    fig, ax = plt.subplots()
    colors = {0: 'gray', 1: 'blue', 2: 'red'}
    labels_map = {0: 'Background', 1: 'Palatum', 2: 'Rugae'}
    
    for label_idx, color in colors.items():
        subset_points = points[labels == label_idx]
        if subset_points.shape[0] > 0:
            ax.scatter(subset_points[:, 0], subset_points[:, 1], c=color, s=0.1, label=labels_map[label_idx])
    
    ax.set_title("Visualisasi 2D Data Anotasi (Ground Truth)")
    ax.set_xlabel("Sumbu X")
    ax.set_ylabel("Sumbu Y")
    ax.legend(markerscale=10)
    ax.axis('equal')
    return fig

# --- APLIKASI UTAMA ---
st.set_page_config(layout="wide")
st.title("🚀 MVP Demo Identifikasi Forensik Palatal Rugae")
st.info("Aplikasi ini mendemonstrasikan dua pipeline: (1) Identifikasi Pasien dan (2) Ekstraksi Morfometrik.")

# Setup device dan muat model & galeri
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)
gallery_embeddings, gallery_labels = load_gallery_embeddings(model, device)

# --- SIDEBAR UNTUK UPLOAD ---
st.sidebar.title("Panel Kontrol")
st.sidebar.info(f"Menggunakan device: **{str(device).upper()}**")
st.sidebar.write(f"Model: `best_recognition_triplet_model.pth`")
st.sidebar.write(f"Galeri: `{len(gallery_labels)}` pasien.")
st.sidebar.write(f"Threshold Jarak: `{THRESHOLD_JARAK}`")

# Untuk demo, minta user untuk upload file .npz yang sudah diproses
st.sidebar.subheader("Unggah Sampel Uji")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file .npz yang sudah diproses:", 
    type=["npz"]
)
st.sidebar.caption("Catatan: Demo ini menerima file .npz yang sudah melalui tahap normalisasi.")

# --- AREA UTAMA ---
if uploaded_file is not None:
    # Muat data dari file .npz yang di-upload
    file_bytes = BytesIO(uploaded_file.getvalue())
    data = np.load(file_bytes)
    points = data['points']
    labels = data['labels'] 

    st.header(f"Analisis untuk: `{uploaded_file.name}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Tampilkan Visualisasi 2D
        st.subheader("Visualisasi 2D Anotasi")
        fig = create_2d_plot(points, labels)
        st.pyplot(fig)

    with col2:
        # 2. Pipeline Identifikasi (Recognition)
        st.subheader("Pipeline 1: Identifikasi Pasien (Live)")
        if st.button("Jalankan Identifikasi"):
            with st.spinner(f"Menjalankan model AI pada {len(points)} titik..."):
                points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    query_embedding = model(points_tensor)
                
                distances = torch.cdist(query_embedding, torch.from_numpy(gallery_embeddings).to(device), p=2.0).cpu().numpy()[0]
                
                results_df = pd.DataFrame({
                    "ID Pasien Kandidat": gallery_labels,
                    "Jarak Embedding": distances
                })
                results_df = results_df.sort_values(by="Jarak Embedding")
                
                results_df["Keputusan"] = np.where(results_df["Jarak Embedding"] <= THRESHOLD_JARAK, "✅ Match", "❌ Berbeda")
                
                st.dataframe(results_df.head(10)) 

    st.markdown("---")

    # 3. Pipeline Morfometri (Prototype)
    st.subheader("Pipeline 2: Ekstraksi Morfometrik (Prototype)")
    st.write("Menjalankan pipeline morfometrik pada **data anotasi** (ground truth) dari file ini.")
    if st.button("Jalankan Ekstraksi Morfometri"):
        with st.spinner("Menghitung panjang rugae..."):
            morpho_results_df = run_morphometry_pipeline(points, labels)
            st.dataframe(morpho_results_df)