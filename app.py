"""
🧠 Next Word Prediction with LSTM
Production-Ready Streamlit Application

Version: 2.0 (Deployment Compatible)
Last Updated: May 30, 2026
"""

# ==========================================
# CRITICAL: set_page_config() MUST be FIRST
# ==========================================
import streamlit as st

st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# IMPORTS (after set_page_config)
# ==========================================
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# ERROR HANDLING & MODEL LOADING
# ==========================================
@st.cache_resource
def load_resources():
    """
    Load LSTM model, tokenizer, and max_length with error handling.
    
    Returns:
        tuple: (model, tokenizer, max_len) or (None, None, None) if error
    """
    errors = []
    
    try:
        # Load model from H5 format (deployment compatible)
        # H5 is backward compatible across TensorFlow versions
        if not os.path.exists("lstm_model.h5"):
            errors.append("❌ lstm_model.h5 not found")
            return None, None, None
            
        model = load_model("lstm_model.h5")
        
    except Exception as e:
        errors.append(f"❌ Model loading failed: {str(e)[:100]}")
        return None, None, None
    
    try:
        # Load tokenizer
        if not os.path.exists("tokenizer.pkl"):
            errors.append("❌ tokenizer.pkl not found")
            return None, None, None
            
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        errors.append(f"❌ Tokenizer loading failed: {str(e)[:100]}")
        return None, None, None
    
    try:
        # Load max_length
        if not os.path.exists("max_len.pkl"):
            errors.append("❌ max_len.pkl not found")
            return None, None, None
            
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
    except Exception as e:
        errors.append(f"❌ Max length loading failed: {str(e)[:100]}")
        return None, None, None
    
    if errors:
        st.error(" | ".join(errors))
        return None, None, None
    
    return model, tokenizer, max_len


# ==========================================
# LOAD RESOURCES
# ==========================================
model, tokenizer, max_len = load_resources()

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================
def predict_next_word(text, model, tokenizer, max_len):
    """Predict the most likely next word"""
    try:
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
        
        preds = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)
        
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                return word
        return ""
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def generate_text_advanced(seed_text, n_words, model, tokenizer, max_len, 
                          temperature=0.7, diversity_penalty=0.3):
    """
    Generate text with advanced strategies to prevent repetition.
    
    Args:
        seed_text: Starting phrase
        n_words: Number of words to generate
        model: Trained LSTM model
        tokenizer: Fitted tokenizer
        max_len: Maximum sequence length
        temperature: Sampling temperature (0.1-2.0)
        diversity_penalty: Repetition prevention (0.0-1.0)
    
    Returns:
        Generated text string
    """
    try:
        generated_words = seed_text.split()
        current_text = seed_text
        prevent_repetition = 2
        
        for _ in range(n_words):
            seq = tokenizer.texts_to_sequences([current_text])[0]
            seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')
            
            preds = model.predict(seq, verbose=0)[0].copy()
            
            # Apply diversity penalty to recent words
            if diversity_penalty > 0:
                for recent_word in generated_words[-prevent_repetition:]:
                    if recent_word in tokenizer.word_index:
                        idx = tokenizer.word_index[recent_word]
                        if idx < len(preds):
                            preds[idx] = preds[idx] * (1 - diversity_penalty)
            
            # Apply temperature
            if temperature != 1.0:
                preds = np.power(preds, 1.0/temperature)
                preds = preds / (np.sum(preds) + 1e-10)
                pred_index = np.random.choice(len(preds), p=preds)
            else:
                pred_index = np.argmax(preds)
            
            # Get word
            next_word = ""
            for word, index in tokenizer.word_index.items():
                if index == pred_index:
                    next_word = word
                    break
            
            if next_word == "":
                break
            
            generated_words.append(next_word)
            current_text += " " + next_word
        
        return " ".join(generated_words)
    
    except Exception as e:
        return f"Error generating text: {str(e)[:100]}"


# ==========================================
# STREAMLIT UI
# ==========================================

# Check if model loaded successfully
if model is None or tokenizer is None or max_len is None:
    st.error("⚠️ Unable to load model resources. Check that all files exist:")
    st.code("Required files:\n- lstm_model.h5\n- tokenizer.pkl\n- max_len.pkl")
    st.stop()

st.title("🧠 Next Word Prediction with LSTM")
st.write("Generate creative text sequences using an LSTM neural network trained on quotes.")

# ==========================================
# HELP SECTION
# ==========================================
with st.expander("❓ How to Use This App", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 What is this?")
        st.write("""
        This app uses a **Long Short-Term Memory (LSTM)** neural network to generate 
        creative text continuations. It learns patterns from famous quotes to predict 
        the next words in a sequence.
        """)
        
        st.markdown("### 🎯 How to Use:")
        st.write("""
        1. **Enter Seed Text**: Start with any phrase (e.g., "life is", "the world")
        2. **Adjust Parameters**: Control how long and diverse the output is
        3. **Click Generate**: Watch the AI complete your text!
        """)
    
    with col2:
        st.markdown("### ⚙️ Parameter Guide:")
        st.write("""
        - **Seed Text**: Starting phrase for text generation
        - **Words to Generate**: How many words to add (5-50)
        - **Temperature**: 
          - Low (0.1): Conservative, repetitive
          - Medium (0.7): Balanced, recommended ✓
          - High (2.0): Creative, random
        - **Repetition Prevention**: Avoids repeating words (higher = less repetition)
        """)
    
    st.markdown("### 💡 Tips for Best Results:")
    tips = """
    - Use **3-5 words** as seed text for better context
    - Try temperature **0.7-0.9** for natural-sounding text
    - Use repetition prevention **0.3-0.5** to avoid word loops
    - Try different seed texts: "love is", "the best", "success is", etc.
    - Generate **10-20 words** for coherent sentences
    """
    st.write(tips)

# ==========================================
# TEXT GENERATION INTERFACE
# ==========================================
st.subheader("✨ Generate Text Sequence")

col1, col2, col3 = st.columns(3)
with col1:
    gen_input = st.text_input(
        "💬 Seed text:",
        placeholder="Start with...",
        value="life is"
    )
with col2:
    n_words = st.slider(
        "📏 Number of words to generate:",
        min_value=5,
        max_value=50,
        value=15
    )
with col3:
    temperature = st.slider(
        "🌡️ Temperature (diversity):",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

col4, col5 = st.columns(2)
with col4:
    diversity_penalty = st.slider(
        "🔄 Repetition prevention:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )

# Generate button
if st.button("🚀 Generate Text", key="btn_generate", use_container_width=True):
    if gen_input.strip() == "":
        st.warning("⚠️ Please enter some seed text.")
    else:
        with st.spinner("🤔 Generating text..."):
            generated_text = generate_text_advanced(
                gen_input,
                n_words,
                model,
                tokenizer,
                max_len,
                temperature=temperature,
                diversity_penalty=diversity_penalty
            )
        st.success("✅ Text generated!")
        st.text_area(
            "📄 Generated Text:",
            generated_text,
            height=100,
            disabled=True
        )

# ==========================================
# MODEL INFORMATION
# ==========================================
with st.expander("📊 Model Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Vocabulary Size", "10,000 words")
        st.metric("Max Sequence Length", "748 tokens")
    
    with col2:
        st.metric("Model Type", "LSTM Sequential")
        st.metric("Input Dimension", "50")

st.write("""

**Training:**
- Dataset: Motivational quotes
- Training Accuracy: 60.32%
- Validation Accuracy: 50.68%
""")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ About")
    st.write("""
    **Next Word Prediction AI**
    
    This LSTM-based text generation model predicts the next word in a sequence
    using patterns learned from a quotes dataset.
    
    Built with:
    - TensorFlow & Keras
    - Streamlit
    - Python 3.11
    """)
    
    st.markdown("---")
    st.subheader("👤 Developer")
    st.write("**Bhavy Soni**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("🐙 GitHub", "https://github.com/bhavysoni2005")
    with col2:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/bhavy-soni-b3746b316")
    
    st.markdown("---")
    st.caption("© 2026 Next Word Prediction AI | All Rights Reserved")
    st.caption("Deployment Version: 2.0 (Compatible)")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #999; font-size: 12px;'>"
    "🧠 LSTM Neural Network | Built with Streamlit | Powered by TensorFlow & Keras<br>"
    "© 2026 Next Word Prediction AI"
    "</div>",
    unsafe_allow_html=True
)

