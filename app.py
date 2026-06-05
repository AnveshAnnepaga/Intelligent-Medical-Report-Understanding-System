import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from fpdf import FPDF
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
from train import PositionalEncoding, clean_text, build_model

# --- UI Configuration & Styling ---
st.set_page_config(page_title="IntelliMed | Healthcare NLP", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# Injecting Premium CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        /* Global Font & Background */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Headers */
        h1, h2, h3 {
            color: #1A365D;
            font-weight: 800;
        }
        
        h1 {
            font-size: 2.8rem;
            background: -webkit-linear-gradient(45deg, #2B6CB0, #4299E1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(49, 130, 206, 0.2);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(49, 130, 206, 0.3);
            background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%);
        }
        
        /* Metric Cards */
        .metric-container {
            display: flex;
            gap: 20px;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            flex: 1;
            display: flex;
            flex-direction: column;
            border-top: 4px solid #4299E1;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #2D3748;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #718096;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 8px;
        }
        
        /* Result Panels */
        .result-panel {
            background: #ffffff;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border: 1px solid #edf2f7;
            height: 100%;
        }
        
        .result-title {
            font-size: 1.2rem;
            color: #4A5568;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .result-value {
            font-size: 2rem;
            color: #38A169;
            font-weight: 800;
            margin-bottom: 20px;
        }
        
        /* Badges */
        .badge {
            background-color: #EBF8FF;
            color: #2B6CB0;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 8px;
            margin-bottom: 8px;
            border: 1px solid #BEE3F8;
        }
        
        /* Text areas and inputs */
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #CBD5E0;
            padding: 15px;
            font-size: 1rem;
            line-height: 1.6;
            transition: border-color 0.2s;
        }
        
        .stTextArea textarea:focus {
            border-color: #4299E1;
            box-shadow: 0 0 0 1px #4299E1;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Artifacts ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists('medical_attention_model.keras'):
        with st.spinner("Model not found. Auto-training a new model (this will take a minute)..."):
            # Import training module dynamically
            import train
            import generate_data
            
            # Forcibly overwrite any corrupted lingering datasets on the server
            st.toast("Generating synthetic medical dataset...")
            df = generate_data.generate_dataset(1000)
            df.to_csv('mtsamples.csv', index=False)
                
            st.toast("Training Self-Attention Model...")
            train.train()
            st.toast("Model trained successfully!", icon="✅")
            
        if not os.path.exists('medical_attention_model.keras'):
             return None, None, None, None
        
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('model_config.pkl', 'rb') as f:
        config = pickle.load(f)
        
    num_classes = len(label_encoder.classes_)
    _, attention_model = build_model(config['max_len'], config['max_vocab_size'], num_classes)
    trained_model = load_model('medical_attention_model.keras', custom_objects={'PositionalEncoding': PositionalEncoding})
    attention_model.set_weights(trained_model.get_weights())
    
    return attention_model, tokenizer, label_encoder, config

# --- PDF Generation ---
def create_pdf_report(report_text, prediction, confidence, top_words):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=18, style='B')
    pdf.set_text_color(26, 54, 93) # Dark blue
    pdf.cell(200, 15, txt="IntelliMed Diagnostic Report", ln=True, align='C')
    
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 30, 200, 30)
    pdf.ln(15)
    
    pdf.set_font("Arial", size=12, style='B')
    pdf.set_text_color(74, 85, 104)
    pdf.cell(50, 10, txt="Predicted Specialty: ", border=0)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(43, 108, 176)
    pdf.cell(100, 10, txt=prediction, ln=True)
    
    pdf.set_font("Arial", size=12, style='B')
    pdf.set_text_color(74, 85, 104)
    pdf.cell(50, 10, txt="Confidence Score: ", border=0)
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(56, 161, 105)
    pdf.cell(100, 10, txt=f"{confidence:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=14, style='B')
    pdf.set_text_color(26, 54, 93)
    pdf.cell(200, 10, txt="Key Diagnostic Indicators", ln=True)
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    words_str = ", ".join(top_words)
    pdf.multi_cell(0, 8, txt=words_str)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=14, style='B')
    pdf.set_text_color(26, 54, 93)
    pdf.cell(200, 10, txt="Original Transcription", ln=True)
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, txt=report_text)
    
    return pdf.output(dest='S').encode('latin1')

# --- Main App ---
def main():
    attention_model, tokenizer, label_encoder, config = load_artifacts()

    # Sidebar
    with st.sidebar:
        st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1 style='font-size: 2rem;'>IntelliMed</h1><p style='color: #718096;'>AI Intelligence Platform</p></div>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("Navigation", [
            "📝 AI Report Analyzer", 
            "📊 Dataset Analytics", 
            "🧠 Attention & Embeddings"
        ], label_visibility="collapsed")
        st.markdown("---")
        st.caption("Powered by Custom Self-Attention Architecture")

    if attention_model is None:
        st.error("Model artifacts not found! Please ensure you have run `python train.py` with the dataset present.")
        return

    # Routing
    if menu == "📊 Dataset Analytics":
        st.title("Dataset Analytics")
        st.markdown("Explore the underlying Medical Transcriptions dataset and vocabulary metrics.")
        
        if os.path.exists('mtsamples.csv'):
            df = pd.read_csv('mtsamples.csv').dropna(subset=['transcription', 'medical_specialty'])
            
            # Premium Metric Cards
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-value">{len(df):,}</div>
                        <div class="metric-label">Total Reports Analyzed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{df['medical_specialty'].nunique()}</div>
                        <div class="metric-label">Medical Specialties</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(tokenizer.word_index):,}</div>
                        <div class="metric-label">Unique Medical Terms</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Specialty Distribution")
                st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(y=df['medical_specialty'], order=df['medical_specialty'].value_counts().iloc[:10].index, palette='crest', ax=ax)
                ax.set_ylabel("")
                ax.set_xlabel("Count")
                sns.despine()
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.subheader("Medical Vocabulary Builder")
                st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
                word_counts = pd.DataFrame.from_dict(tokenizer.word_counts, orient='index', columns=['Frequency']).sort_values('Frequency', ascending=False)
                st.dataframe(word_counts.head(50), use_container_width=True, height=400)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("mtsamples.csv not found.")
            
    elif menu == "🧠 Attention & Embeddings":
        st.title("Network Introspection")
        st.markdown("Visualizing the internal representations of the custom Self-Attention model.")
        
        st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
        st.subheader("Positional Encoding Map")
        st.markdown("The mathematical representation of sentence and token positions using sinusoidal waves, ensuring the model understands the sequence order of medical symptoms.")
        
        pe_layer = PositionalEncoding(sequence_length=config['max_len'], vocab_size=config['max_vocab_size'], embed_dim=128)
        _ = pe_layer(tf.zeros((1, config['max_len'])))
        pe_weights = pe_layer.pe.numpy()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pe_weights, cmap="mako", ax=ax, cbar_kws={'label': 'Encoding Amplitude'})
        ax.set_title("")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Token Position")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif menu == "📝 AI Report Analyzer":
        st.title("AI Medical Report Analyzer")
        st.markdown("Instantly classify medical transcriptions and understand the AI's diagnostic reasoning.")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
            input_method = st.radio("Input Source", ["Text Entry", "File Upload"], horizontal=True, label_visibility="collapsed")
            
            report_text = ""
            if input_method == "Text Entry":
                report_text = st.text_area("Patient Transcription", placeholder="Type or paste the clinical notes here...", height=250, label_visibility="collapsed")
            else:
                uploaded_file = st.file_uploader("Upload Medical Note (.txt)", type=["txt"])
                if uploaded_file is not None:
                    report_text = uploaded_file.read().decode("utf-8")
                    st.text_area("File Contents", value=report_text, height=250, disabled=True)
            
            analyze_btn = st.button("Analyze Transcription")
            st.markdown("</div>", unsafe_allow_html=True)

        if analyze_btn and report_text:
            with st.spinner("Neural networks processing text..."):
                cleaned_text = clean_text(report_text)
                sequence = tokenizer.texts_to_sequences([cleaned_text])
                padded_sequence = pad_sequences(sequence, maxlen=config['max_len'], padding='post', truncating='post')
                
                pred_probs, attention_scores = attention_model.predict(padded_sequence)
                pred_class_idx = np.argmax(pred_probs[0])
                confidence = pred_probs[0][pred_class_idx] * 100
                prediction_label = label_encoder.inverse_transform([pred_class_idx])[0]
                
                avg_attention = np.mean(attention_scores[0], axis=0)
                word_importance = np.sum(avg_attention, axis=0)
                
                words = cleaned_text.split()[:config['max_len']]
                importance_dict = {}
                for i, word in enumerate(words):
                    if i < config['max_len']:
                        importance_dict[word] = word_importance[i]
                        
                sorted_words = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
                top_n_words = [w[0] for w in sorted_words[:8]]
                
            with col2:
                st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
                st.markdown("<div class='result-title'>Predicted Specialty</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-value'>{prediction_label}</div>", unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.markdown("<div class='result-title'>Confidence Score</div>", unsafe_allow_html=True)
                st.progress(int(confidence))
                st.caption(f"{confidence:.1f}% Match")
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("<div class='result-title'>Key Identifiers</div>", unsafe_allow_html=True)
                badges_html = "".join([f"<span class='badge'>{w}</span>" for w in top_n_words])
                st.markdown(badges_html, unsafe_allow_html=True)
                
                st.markdown("<br><hr>", unsafe_allow_html=True)
                
                pdf_bytes = create_pdf_report(report_text, prediction_label, confidence, top_n_words)
                b64 = base64.b64encode(pdf_bytes).decode('latin1')
                href = f'<a href="data:application/pdf;base64,{b64}" download="IntelliMed_Report.pdf" style="text-decoration: none;"><button style="width: 100%; border-radius: 12px; font-weight: 600; background: #E2E8F0; color: #2D3748; border: none; padding: 0.75rem 1.5rem; cursor: pointer; transition: 0.3s; margin-top: 10px;">📄 Download PDF Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Diagnostic Explainability: Self-Attention Map")
            st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
            st.markdown("The heatmap visualizes where the AI focused its attention. High-intensity areas indicate word relationships that strongly drove the classification decision.")
            
            actual_len = len(words)
            trim_len = min(actual_len, 25) 
            
            if trim_len > 0:
                trim_attention = avg_attention[:trim_len, :trim_len]
                trim_words = words[:trim_len]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(trim_attention, xticklabels=trim_words, yticklabels=trim_words, cmap="flare", ax=ax, square=True)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                sns.despine()
                st.pyplot(fig)
            else:
                st.warning("Not enough text to generate an attention map.")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
