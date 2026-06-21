import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import PyPDF2
from annotated_text import annotated_text

# 1. Konfigurasi Halaman (Lebih Lebar Biar Kelihatan Profesional)
st.set_page_config(page_title="Entify: NER System", page_icon="🔍", layout="wide")

# --- SIDEBAR: Model Info Card ---
with st.sidebar:
    st.header("Tentang Model")
    st.write("**Arsitektur:** RoBERTa-base")
    st.write("**Dataset:** WikiNER (English)")
    st.write("**F1-Score:** 0.95 (Weighted Average)")
    st.write("**Target Label:** 5 Kelas (PER, LOC, ORG, MISC, O)")
    st.divider()
    st.caption("Proyek UAS NLP - COMP6885001")

st.title("🔍 Entify")
st.markdown("**RoBERTa-Based Named Entity Recognition for Historical Texts**")

#fungsi load model
@st.cache_resource
def load_roberta_model_final():
    model_path = "./RoBERTa_NER_Final_ASLI" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    model.config.id2label = {
        0: "O",
        1: "LOC",
        2: "PER",
        3: "MISC",
        4: "ORG"
    }
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    
    ner_pipe = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )
    return ner_pipe

with st.spinner("Memuat model RoBERTa..."):
    ner_pipeline = load_roberta_model_final()

#inputan
st.subheader("Coba Ekstraksi Entitas")
input_method = st.radio("Pilih metode input teks:", ("Ketik Manual", "Upload File (.txt / .pdf)"))

teks_siap_proses = ""

if input_method == "Ketik Manual":
    default_text = "Queen Victoria ruled the United Kingdom from London, while the East India Company expanded its global trade."
    teks_siap_proses = st.text_area("Masukkan teks sejarah di sini:", default_text, height=120)

elif input_method == "Upload File (.txt / .pdf)":
    uploaded_file = st.file_uploader("Upload artikel sejarah (.txt atau .pdf)", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            teks_siap_proses = uploaded_file.getvalue().decode("utf-8")
            st.success(f"✅ File {uploaded_file.name} berhasil dibaca!")
        
        elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            teks_ekstrak = ""
            for page in pdf_reader.pages:
                teks_ekstrak += page.extract_text() + "\n"
            teks_siap_proses = teks_ekstrak
            st.success(f"✅ File {uploaded_file.name} berhasil dibaca!")
            
        st.info("Cuplikan teks: " + teks_siap_proses[:300] + " ...")

#tombol eskekusi
if st.button("Jalankan Entify", type="primary"):
    if teks_siap_proses.strip() == "":
        st.warning("Teks belum ada, silakan ketik atau upload file terlebih dahulu.")
    else:
        with st.spinner("Model sedang menganalisis teks..."):
            results = ner_pipeline(teks_siap_proses)
        
        if len(results) == 0:
            st.info("Tidak ada entitas yang ditemukan pada teks tersebut.")
        else:
            st.success("Analisis Selesai!")
            
            # --- FITUR BARU 1: STATISTIK HASIL ---
            st.subheader("Statistik Penemuan")
            counts = {"PER": 0, "LOC": 0, "ORG": 0, "MISC": 0}
            for ent in results:
                label = ent['entity_group']
                if label in counts:
                    counts[label] += 1
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("👤 Person (PER)", counts["PER"])
            col2.metric("🌍 Location (LOC)", counts["LOC"])
            col3.metric("🏢 Organization (ORG)", counts["ORG"])
            col4.metric("📌 Misc (MISC)", counts["MISC"])
            
            st.divider()
            
            st.subheader("Visualisasi Teks")
            annotated_data = []
            last_idx = 0
            
            warna_label = {
                "PER": "#ffcccc",   # Merah Muda
                "ORG": "#cce5ff",   # Biru Muda
                "LOC": "#d9f2d9",   # Hijau Muda
                "MISC": "#fff2cc"   # Kuning Muda
            }

            for ent in results:
                start = ent['start']
                end = ent['end']
                label = ent['entity_group']
                
                if start > last_idx:
                    annotated_data.append(teks_siap_proses[last_idx:start])
                
                kata_asli = teks_siap_proses[start:end]
                warna = warna_label.get(label, "#eee")
                
                annotated_data.append((kata_asli, label, warna, "#000000"))
                last_idx = end
                
            if last_idx < len(teks_siap_proses):
                annotated_data.append(teks_siap_proses[last_idx:])
                
            #nampilin si highlight
            annotated_text(*annotated_data)
            
            st.divider()
            
            #CONFIDENCE
            st.subheader("📋 Detail Prediksi")
            with st.expander("Klik di sini untuk melihat detail Confidence Score tiap entitas"):
                for entity in results:
                    # Tambahin .strip() buat ngehapus spasi gaib di awal/akhir kata
                    kata_bersih = entity['word'].replace("#", "").strip()
                    label = entity['entity_group']
                    
                    # Nggak perlu di-round di sini, kita format langsung di teksnya
                    skor = entity['score'] * 100
                    
                    # Pakai {:.3f} biar mutlak cuma nampilin 3 angka di belakang koma
                    st.markdown(f"- **{kata_bersih}** [{label}] — Confidence: {skor:.3f}%")