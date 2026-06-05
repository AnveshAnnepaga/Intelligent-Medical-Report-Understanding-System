# Intelligent Medical Report Understanding System

An end-to-end Healthcare NLP Dashboard that processes medical transcriptions, categorizes them by medical specialty, and provides an explainable AI interface using custom Self-Attention mechanisms.

## Project Structure
- `train.py`: Data loading, preprocessing, Custom Positional Encoding, and Self-Attention model training.
- `app.py`: Streamlit dashboard for medical text analysis, prediction, attention visualization, and PDF report generation.
- `generate_data.py`: (Optional) Script to generate synthetic data if you want to test without the Kaggle dataset.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have the required Python packages installed.
   ```bash
   pip install pandas numpy tensorflow scikit-learn streamlit matplotlib seaborn fpdf
   ```

2. **Download the Dataset:**
   - Go to the Kaggle link: [Medical Transcriptions Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
   - Download the data and place the `mtsamples.csv` file directly into this project folder.

3. **Train the Model:**
   Before running the dashboard, you must train the Multi-Head Attention model on the dataset.
   ```bash
   python train.py
   ```
   *This will preprocess the text, train the custom Positional Encoding + Self Attention model, and save `medical_attention_model.keras` along with tokenizers.*

4. **Run the Healthcare Dashboard:**
   Start the Streamlit UI.
   ```bash
   streamlit run app.py
   ```

## Features Complete
- **Task 1 & 2:** Dataset Analytics and Medical Vocabulary Builder (Available in the Streamlit Sidebar).
- **Task 3, 4, 5:** Baseline and Self-Attention Models with Custom Positional Encoding (Implemented in `train.py`).
- **Task 6 & 7:** Diagnostic Importance Analysis and Positional Encoding Heatmaps (Visualized in `app.py`).
- **Bonus:** One-click PDF generation of the medical analysis report.