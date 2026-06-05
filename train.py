import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import re

print("TensorFlow Version:", tf.__version__)

# 1. Data Loading and Preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(filepath='mtsamples.csv', top_n_classes=5):
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please download it from Kaggle.")
        return None, None, None, None, None, None, None
        
    df = df.dropna(subset=['transcription', 'medical_specialty'])
    
    # Filter top N specialties for better classification
    top_specialties = df['medical_specialty'].value_counts().nlargest(top_n_classes).index
    df = df[df['medical_specialty'].isin(top_specialties)]
    
    df['clean_transcription'] = df['transcription'].apply(clean_text)
    
    print(f"Dataset shape after filtering: {df.shape}")
    print("Class distribution:")
    print(df['medical_specialty'].value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['medical_specialty'])
    
    # Tokenize text
    max_vocab_size = 10000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_transcription'])
    
    X = tokenizer.texts_to_sequences(df['clean_transcription'])
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, tokenizer, label_encoder, max_len, max_vocab_size

# 2. Custom Positional Encoding (Task 5)
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        
        # Pre-compute positional encodings
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe = np.zeros((sequence_length, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.cast(pe, dtype=tf.float32)

    def call(self, inputs):
        x = self.token_emb(inputs)
        return x + self.pe

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

# 3. Model Architecture
def build_model(max_len, max_vocab_size, num_classes, embed_dim=128, num_heads=4):
    inputs = Input(shape=(max_len,))
    
    # Embedding + Positional Encoding
    x = PositionalEncoding(sequence_length=max_len, vocab_size=max_vocab_size, embed_dim=embed_dim, name="pos_encoding")(inputs)
    
    # Self-Attention (Task 4)
    attention_output, attention_scores = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, name="multi_head_attention"
    )(x, x, return_attention_scores=True)
    
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    # We also create a sub-model to output attention scores for Task 6
    attention_model = Model(inputs=inputs, outputs=[outputs, attention_scores])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, attention_model

# 4. Training
def train():
    X_train, X_test, y_train, y_test, tokenizer, label_encoder, max_len, max_vocab_size = load_and_preprocess_data()
    
    if X_train is None:
        return
        
    num_classes = len(label_encoder.classes_)
    model, _ = build_model(max_len, max_vocab_size, num_classes)
    
    print("Model Summary:")
    model.summary()
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    print("Saving artifacts...")
    model.save('medical_attention_model.keras')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('model_config.pkl', 'wb') as f:
        pickle.dump({'max_len': max_len, 'max_vocab_size': max_vocab_size}, f)
        
    print("Training complete. Models and tokenizers saved.")

if __name__ == '__main__':
    train()
