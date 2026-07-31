import streamlit as st
import numpy as np
import pickle
import re
import time
from datetime import datetime
from scipy.sparse import csr_matrix, hstack

# ── NLTK setup ──
import nltk
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('omw-1.4',    quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StyleSense — Fashion Review Sentiment Analyzer",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLE — Bulletproof Professional Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* 1. FORCE MAIN APP BACKGROUND */
    .stApp {
        background-color: #f8fafc !important; /* Soft light gray/blue */
    }

    /* 2. BULLETPROOF LIGHT MODE FIX FOR MAIN AREA */
    /* This stops Streamlit's Dark Mode from turning text white */
    section[data-testid="stMain"] .stMarkdown h1,
    section[data-testid="stMain"] .stMarkdown h2,
    section[data-testid="stMain"] .stMarkdown h3,
    section[data-testid="stMain"] .stMarkdown h4,
    section[data-testid="stMain"] .stMarkdown h5,
    section[data-testid="stMain"] .stMarkdown h6,
    section[data-testid="stMain"] .stMarkdown p,
    section[data-testid="stMain"] .stMarkdown span,
    section[data-testid="stMain"] label {
        color: #1e293b !important; 
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* 3. HERO BANNER */
    .hero-wrap {
        background: linear-gradient(135deg, #831843 0%, #be185d 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(131, 24, 67, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 42px;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
    }
    .hero-sub {
        color: #fce7f3 !important;
        font-size: 16px;
        margin-top: 8px;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 16px;
        backdrop-filter: blur(4px);
    }

    /* 4. SECTION CARDS */
    .section-card {
        background: #ffffff !important;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid #f1f5f9;
        margin-bottom: 24px;
    }

    /* 5. INPUT CONTROLS */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 16px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    .stTextArea textarea:focus {
        border-color: #be185d !important;
        box-shadow: 0 0 0 2px rgba(190, 24, 93, 0.2) !important;
    }

    .stButton>button {
        background-color: #be185d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(190, 24, 93, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #9d174d !important;
        transform: translateY(-2px);
    }

    .chip-row .stButton>button {
        background-color: #fdf2f8 !important;
        color: #9d174d !important;
        border: 1px solid #fbcfe8 !important;
        box-shadow: none !important;
        font-size: 13px !important;
        padding: 6px 12px !important;
    }
    .chip-row .stButton>button:hover {
        background-color: #fce7f3 !important;
        transform: none;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }

    /* 6. RESULT CARDS */
    .result-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        transition: transform 0.15s ease;
        height: 100%;
    }
    .result-card:hover { transform: translateY(-3px); }
    .result-model { font-size: 12px; color: #64748b !important; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 1px;}
    .result-emoji { font-size: 36px; margin: 12px 0; }
    .result-label { font-size: 18px; font-weight: 700; margin: 0; }

    .verdict-card {
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        background: #ffffff;
        margin-top: 16px;
    }

    /* 7. SIDEBAR STYLING (Forced Dark Mode) */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important; 
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] table {
        color: #f1f5f9 !important;
        background: #1e293b !important;
        border-radius: 8px;
        overflow: hidden;
        width: 100%;
    }
    section[data-testid="stSidebar"] th, section[data-testid="stSidebar"] td {
        border-bottom: 1px solid #334155 !important;
        padding: 10px 12px !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* 8. HISTORY PILLS */
    .history-pill {
        background: #fdf2f8;
        border-left: 4px solid #be185d;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        font-size: 14px;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .history-pill .meta-text {
        color: #64748b;
        font-size: 12px;
    }
    .history-pill .quote-text {
        color: #334155;
        font-style: italic;
        margin-top: 4px;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open("naive_bayes_model.pkl", "rb") as f:
            nb = pickle.load(f)
        with open("svm_model.pkl", "rb") as f:
            svm = pickle.load(f)
        with open("logistic_regression_model.pkl", "rb") as f:
            lr = pickle.load(f)
    except FileNotFoundError:
        st.error("Model files missing. Ensure your .pkl files are in the same directory.")
        return None, None, None
    return nb, svm, lr

nb_model, svm_bundle, lr_model = load_models()

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# ─────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────
def predict_nb(text, model):
    processed = preprocess(text)
    X = model['tfidf'].transform([processed])
    review = X.toarray().flatten()
    scores = {}
    for s in model['classes']:
        scores[s] = model['log_prior'][s] + np.sum(review * model['log_likelihood'][s])
    min_s = min(scores.values())
    max_s = max(scores.values())
    rng = max_s - min_s
    if rng > 0:
        normalized = {s: float((v - min_s) / rng) for s, v in scores.items()}
    else:
        normalized = {s: float(1/len(scores)) for s in scores}
    return max(scores, key=scores.get), normalized


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_lr(text, model):
    processed = preprocess(text)
    X = model['tfidf'].transform([processed])
    ones = csr_matrix(np.ones((1, 1)))
    X_b = hstack([ones, X]).tocsr()
    thetas = model['thetas']
    classes = model['classes']
    inv_label = model['inv_label']
    probs = np.zeros(len(classes))
    for c in classes:
        probs[c] = float(sigmoid(X_b.dot(thetas[c]))[0])
    pred_idx = int(np.argmax(probs))
    scores = {inv_label[c]: float(np.clip(probs[c], 0.0, 1.0)) for c in classes}
    return inv_label[pred_idx], scores


def predict_svm(text, bundle):
    processed = preprocess(text)
    X = bundle['tfidf'].transform([processed])
    try:
        pred = bundle['model'].predict(X)[0]
    except Exception:
        try:
            pred = bundle['model'].predict(X.toarray())[0]
        except Exception as e:
            st.error(f"SVM prediction error: {e}")
            pred = "Unknown"
    return pred, {}

def safe_progress(val):
    return float(max(0.0, min(1.0, val)))

def sentiment_style(sentiment):
    # Slightly darker hex codes so text is highly readable on white backgrounds
    if sentiment == 'Positive':
        return "🟢", "#059669", "Positive" # Dark Emerald
    elif sentiment == 'Negative':
        return "🔴", "#dc2626", "Negative" # Dark Red
    else:
        return "🟡", "#d97706", "Neutral" # Dark Amber

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    "I absolutely love this dress! It fits perfectly and the fabric is amazing.",
    "The material felt cheap and it ran two sizes too small, very disappointed.",
    "It's okay — not bad, not great. Fits fine but the color is a bit different from the photo.",
]

def set_example(txt):
    st.session_state.review_text = txt

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">👗 StyleSense</div>
    <div class="hero-sub">AI-powered sentiment analysis for women's clothing reviews. Instantly know how a customer feels.</div>
    <span class="hero-badge">✨ Group 096 · NLP Applied Project</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<h2 style="font-family:\'Playfair Display\', serif; color:#0f172a !important; font-size: 24px; font-weight:700; margin-bottom: 4px;">✍️ Write or Paste a Review</h2>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b !important; font-size:14.5px; margin-bottom:20px;">Try a real customer review, or tap an example below to see it in action.</p>', unsafe_allow_html=True)

st.markdown('<div class="chip-row">', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLES))
chip_labels = ["😍 Positive example", "😠 Negative example", "😐 Neutral example"]
for i, col in enumerate(chip_cols):
    with col:
        if st.button(chip_labels[i], key=f"chip_{i}", use_container_width=True):
            set_example(EXAMPLES[i])
st.markdown('</div><br>', unsafe_allow_html=True)

review_text = st.text_area(
    "Review text",
    value=st.session_state.review_text,
    placeholder="e.g. I absolutely love this dress! It fits perfectly and the fabric is amazing.",
    height=140,
    label_visibility="collapsed",
    key="review_input"
)
st.session_state.review_text = review_text

st.markdown("<br>", unsafe_allow_html=True)
col_a, col_b = st.columns([2.5, 1])
with col_a:
    model_choice = st.selectbox(
        "🤖 Choose Analytical Model",
        ["All Models (Compare)", "Naive Bayes", "Logistic Regression", "SVM"]
    )
with col_b:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if analyze_clicked:
    if not review_text.strip():
        st.warning("Please enter a review first!")
    elif not nb_model:
        st.error("Cannot analyze: Models failed to load. Check .pkl files.")
    else:
        with st.spinner("Analyzing text architecture..."):
            time.sleep(0.4)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="font-family:\'Playfair Display\', serif; color:#0f172a !important; font-size: 26px; font-weight:700; margin-bottom: 16px;">📊 Analysis Results</h2>', unsafe_allow_html=True)

        if model_choice == "All Models (Compare)":
            nb_pred,  nb_scores  = predict_nb(review_text,  nb_model)
            lr_pred,  lr_scores  = predict_lr(review_text,  lr_model)
            svm_pred, _          = predict_svm(review_text, svm_bundle)

            col1, col2, col3 = st.columns(3)
            for col, name, pred in [
                (col1, "Naive Bayes",         nb_pred),
                (col2, "Logistic Regression", lr_pred),
                (col3, "Support Vector",      svm_pred)
            ]:
                emoji, color, label = sentiment_style(pred)
                with col:
                    st.markdown(f"""
                    <div class="result-card" style="border-top:4px solid {color}; background:{color}05;">
                        <p class="result-model">{name}</p>
                        <p class="result-emoji">{emoji}</p>
                        <p class="result-label" style="color:{color} !important;">{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

            votes = [nb_pred, lr_pred, svm_pred]
            majority = max(set(votes), key=votes.count)
            emoji, color, label = sentiment_style(majority)
            
            st.markdown(f"""
            <div class="verdict-card" style="border:1px solid {color}40; background:{color}08;">
                <p style="font-family:'Inter',sans-serif; font-weight:600; font-size:16px; margin:0; color:#475569 !important;">🗳️ ENSEMBLE VERDICT</p>
                <p class="result-emoji" style="font-size:48px;">{emoji}</p>
                <p class="result-label" style="font-size:26px; color:{color} !important;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

            if lr_scores:
                st.markdown("<hr style='border-color:#e2e8f0; margin: 30px 0;'/>", unsafe_allow_html=True)
                # Using inline HTML header to guarantee it isn't turned white by Streamlit
                st.markdown("<h4 style='color: #0f172a !important; font-weight: 600; font-size:18px;'>📈 Logistic Regression — Confidence Scores</h4>", unsafe_allow_html=True)
                for s, score in sorted(lr_scores.items(), key=lambda x: -x[1]):
                    e2, c2, _ = sentiment_style(s)
                    st.markdown(f"<span style='color:#0f172a !important; font-weight:500;'>{e2} {s}</span>", unsafe_allow_html=True)
                    st.progress(safe_progress(score))
                    st.caption(f"Score: {score:.4f}")

            if nb_scores:
                st.markdown("<hr style='border-color:#e2e8f0; margin: 30px 0;'/>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #0f172a !important; font-weight: 600; font-size:18px;'>📈 Naive Bayes — Relative Scores</h4>", unsafe_allow_html=True)
                for s, score in sorted(nb_scores.items(), key=lambda x: -x[1]):
                    e2, c2, _ = sentiment_style(s)
                    st.markdown(f"<span style='color:#0f172a !important; font-weight:500;'>{e2} {s}</span>", unsafe_allow_html=True)
                    st.progress(safe_progress(score))
                    st.caption(f"Relative Score: {score:.4f}")

            final_pred = majority

        else:
            if model_choice == "Naive Bayes":
                pred, scores = predict_nb(review_text, nb_model)
            elif model_choice == "Logistic Regression":
                pred, scores = predict_lr(review_text, lr_model)
            else:
                pred, scores = predict_svm(review_text, svm_bundle)

            emoji, color, label = sentiment_style(pred)
            st.markdown(f"""
            <div class="verdict-card" style="border:1px solid {color}40; background:{color}08; border-top: 5px solid {color};">
                <p style="font-family:'Inter',sans-serif; font-weight:600; font-size:16px; margin:0; color:#475569 !important;">{model_choice.upper()}</p>
                <p class="result-emoji" style="font-size:56px;">{emoji}</p>
                <p class="result-label" style="font-size:32px; color:{color} !important;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

            if scores:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #0f172a !important; font-weight: 600; font-size:18px;'>📈 Confidence Scores</h4>", unsafe_allow_html=True)
                for s, score in sorted(scores.items(), key=lambda x: -x[1]):
                    e2, c2, _ = sentiment_style(s)
                    st.markdown(f"<span style='color:#0f172a !important; font-weight:500;'>{e2} {s}</span>", unsafe_allow_html=True)
                    st.progress(safe_progress(score))
                    st.caption(f"Score: {score:.4f}")

            final_pred = pred

        st.markdown('</div>', unsafe_allow_html=True)

        # Save to history
        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": review_text[:80] + ("..." if len(review_text) > 80 else ""),
            "model": model_choice,
            "pred": final_pred
        })
        st.session_state.history = st.session_state.history[:8]

# ─────────────────────────────────────────────
# RECENT HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:\'Playfair Display\', serif; color:#0f172a !important; font-size: 22px; font-weight:700; margin-bottom: 16px;">🕓 History</h2>', unsafe_allow_html=True)
    for h in st.session_state.history:
        emoji, color, _ = sentiment_style(h["pred"])
        st.markdown(f"""
        <div class="history-pill">
            {emoji} <b style="color:{color};">{h['pred']}</b> <span class="meta-text">· {h['model']} · {h['time']}</span><br>
            <span class="quote-text">"{h['text']}"</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👗 StyleSense")
    st.markdown("<p style='color:#94a3b8 !important;'>Fashion Review Sentiment Analyzer</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Model Accuracy")
    st.markdown("""
    | Model | Accuracy |
    |---|---|
    | Naive Bayes | 79.91% |
    | Logistic Regression | 79.60% |
    | SVM | 83.07% |
    """)
    st.markdown("---")
    st.markdown("### 🏷️ Labels")
    st.markdown("🟢 **Positive** — Rating 4–5")
    st.markdown("🟡 **Neutral** — Rating 3")
    st.markdown("🔴 **Negative** — Rating 1–2")
    st.markdown("---")
    st.markdown("### 📚 Processing Pipeline")
    st.markdown("""
    1. Lowercase Extraction
    2. Regex Cleaning
    3. Special Char Removal
    4. Tokenization
    5. Stopword Removal
    6. Lemmatization
    7. TF-IDF (5000 features, bigrams)
    """)
    st.markdown("---")
    st.caption("Group 096 | NLP Applied Project")