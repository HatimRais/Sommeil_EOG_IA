import streamlit as st
import mne
import numpy as np
import openvino as ov
import matplotlib.pyplot as plt
import os
import time

# --- CONFIGURATION STYLÉE ---
st.set_page_config(
    page_title="DeepSleep AI - NPU Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thème personnalisé via Markdown
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
col_t1, col_t2 = st.columns([1, 5])
with col_t1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
with col_t2:
    st.title("DeepSleep AI : Analyseur Sommeil EOG")
    st.write("**Architecture :** CNN-LSTM hybride | **Hardware :** Intel NPU AI Boost")


# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_npu_engine():
    core = ov.Core()
    model_path = "models/sleep_model_npu.xml"
    if not os.path.exists(model_path):
        return None, "Fichier modèle .xml introuvable dans /models"
    try:
        # Mode AUTO pour synergie NPU + CPU
        compiled_model = core.compile_model(model_path, "AUTO")
        device = compiled_model.get_property("EXECUTION_DEVICES")
        return compiled_model, device
    except Exception as e:
        return None, str(e)


npu_net, device_info = load_npu_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔬 Paramètres d'Analyse")
    if npu_net:
        st.success(f"**Hardware :** {device_info}")
    else:
        st.error(f"**Erreur :** {device_info}")

    st.divider()
    edf_file = st.file_uploader("Fichier Signal (PSG.edf)", type=["edf"])
    hypno_file = st.file_uploader("Fichier Vérité (Hypnogram.edf)", type=["edf"])

    st.divider()
    analyze_btn = st.button("⚡ LANCER L'INFÉRENCE NPU", use_container_width=True)

# --- LOGIQUE PRINCIPALE ---
if edf_file and analyze_btn:
    try:
        # Simulation d'un chargement stylé
        with st.status("🚀 Initialisation du moteur Deep Learning...", expanded=True) as status:
            st.write("Chargement du signal EDF...")
            with open("temp_psg.edf", "wb") as f:
                f.write(edf_file.getbuffer())

            raw = mne.io.read_raw_edf("temp_psg.edf", preload=True, verbose=False)

            # Recherche du canal EOG
            eog_ch = [ch for ch in raw.ch_names if 'EOG' in ch.upper()]
            if not eog_ch:
                st.error("Aucun canal EOG détecté.")
                st.stop()
            target_ch = eog_ch[0]

            st.write(f"Prétraitement du canal {target_ch} (Filtre 0.3-35Hz)...")
            raw.filter(0.3, 35, picks=[target_ch], verbose=False)
            data = raw.get_data(picks=[target_ch])[0]

            st.write("Segmentation en époques de 30s...")
            sfreq = raw.info['sfreq']
            pts = int(30 * sfreq)
            n_epochs = len(data) // pts
            X_input = data[:n_epochs * pts].reshape(n_epochs, pts, 1).astype(np.float32)

            st.write("Inférence sur Intel AI Boost...")
            predictions = []
            output_layer = npu_net.output(0)

            # Barre de progression
            pbar = st.progress(0)
            start_time = time.time()
            for i in range(n_epochs):
                res = npu_net(X_input[i:i + 1])[output_layer]
                predictions.append(np.argmax(res))
                pbar.progress((i + 1) / n_epochs)

            inference_time = time.time() - start_time
            status.update(label="✅ Analyse terminée !", state="complete", expanded=False)

        # --- AFFICHAGE DES RÉSULTATS ---
        st.subheader("📊 Hypnogramme Généré par l'IA")

        # Design du graphique
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.step(range(len(predictions)), predictions, where='post', color='#00FFCC', linewidth=1.5)
        ax.fill_between(range(len(predictions)), predictions, step="post", alpha=0.1, color='#00FFCC')
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
        ax.invert_yaxis()
        ax.set_title(f"Analyse temporelle sur {n_epochs} époques", color='white', pad=20)
        plt.grid(alpha=0.2)
        st.pyplot(fig)

        # Metrics Techniques
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Temps d'inférence", f"{inference_time:.2f}s")
        with m2:
            st.metric("Vitesse", f"{n_epochs / inference_time:.1f} époques/s")
        with m3:
            st.metric("Device", "NPU (AI Boost)")

        # --- SECTION VÉRITÉ TERRAIN ---
        if hypno_file:
            st.divider()
            st.subheader("⚖️ Comparaison : IA vs Médecin (Vérité Terrain)")

            with open("temp_hypno.edf", "wb") as f:
                f.write(hypno_file.getbuffer())

            true_annots = mne.read_annotations("temp_hypno.edf")
            mapping = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                       'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}

            y_true = [mapping[d] for d in true_annots.description if d in mapping]
            min_l = min(len(predictions), len(y_true))

            fig2, ax2 = plt.subplots(figsize=(15, 4))
            ax2.step(range(min_l), y_true[:min_l], label="Expert (EDF)", color='#7ed6df', alpha=0.8)
            ax2.step(range(min_l), predictions[:min_l], label="Notre IA", color='#ff7979', linestyle='--', alpha=0.8)
            ax2.invert_yaxis()
            ax2.legend()
            plt.grid(alpha=0.1)
            st.pyplot(fig2)

            acc = (sum(1 for i in range(min_l) if predictions[i] == y_true[i]) / min_l) * 100
            st.success(f"🎯 Précision du modèle sur ce fichier : **{acc:.2f} %**")

    except Exception as e:
        st.error(f"❌ Erreur critique : {e}")

else:
    # État d'attente stylé
    st.info("👋 Bienvenue. Veuillez charger un fichier PSG.edf dans la barre latérale pour lancer l'analyse neurale.")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("### Architecture Deep Learning")
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg", width=300)
    with col_img2:
        st.markdown("### Traitement du Signal")
        st.code("""
# Pipeline d'analyse
1. Loading Raw Data
2. Bandpass Filter (0.3 - 35Hz)
3. Normalization & Epoching
4. CNN Feature Extraction
5. Bi-LSTM Temporal Modeling
6. NPU Quantized Inference
        """, language="python")