"""
================================================================
TRAIN ALL 3 MODELS FROM ORIGINAL DATA (Cleaned_Reviews_File.csv)
================================================================
Group 096 — Sentiment Analysis Dashboard
Yeh script proof hai ke .pkl files humne khud original data se
train ki hain (Naive Bayes, Logistic Regression, SVM) — exact wahi
logic jo teenon original notebooks (Phase_06_Hadia, Phase_06_SVM,
LogisticRegressionWithSMOTE) mein use hui thi, sirf ek file mein
combine ki gayi hai taake .pkl files seedhi generate ho jayen.

RUN COMMAND:
    pip install streamlit pandas numpy scikit-learn scipy imbalanced-learn nltk
    python train_all_models.py

INPUT REQUIRED (same folder mein rakho):
    Cleaned_Reviews_File.csv

OUTPUT (yeh 3 files banengi):
    naive_bayes_model.pkl
    logistic_regression_model.pkl
    svm_model.pkl

NOTE: Logistic Regression wala hissa apna poora hyperparameter
tuning (alpha -> lambda -> iterations) khud dobara chalata hai,
bilkul LogisticRegressionWithSMOTE.ipynb ki tarah — isliye yeh
step thoda time le sakta hai (kuch minutes), lekin isi wajah se
result 100% original notebook jaisa guaranteed hai.
================================================================
"""

import pandas as pd
import numpy as np
import pickle
import re
import nltk

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# ================================================================
# STEP 1: LOAD ORIGINAL CLEANED DATA
# ================================================================
print("STEP 1: Loading Cleaned_Reviews_File.csv ...")
df = pd.read_csv("Cleaned_Reviews_File.csv")
df["Review Text"] = df["Review Text"].fillna("no review")
print("Shape:", df.shape)

# ================================================================
# STEP 2: NLP PREPROCESSING (same as Phase_04 notebook)
# lowercase -> URL/space clean -> special char remove ->
# tokenize -> stopword remove -> lemmatize -> join
# ================================================================
print("STEP 2: Preprocessing text (lowercase, clean, tokenize, lemmatize)...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

df["Processed Text"] = df["Review Text"].apply(preprocess)
print("Preprocessing done!")

# ================================================================
# STEP 3: TF-IDF FEATURE GENERATION (same params as notebooks)
# ================================================================
print("STEP 3: TF-IDF feature extraction...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["Processed Text"])
y_raw = df["Sentiment"]
print("TF-IDF matrix shape:", X.shape)

# ================================================================
# STEP 4: TRAIN / VALIDATION / TEST SPLIT (70/15/15, same seed)
# ================================================================
print("STEP 4: Splitting data (70% train / 15% val / 15% test)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_raw, test_size=0.30, random_state=42, stratify=y_raw
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
y_train_arr = np.array(y_train)
classes_str = np.unique(y_train_arr)   # ['Negative','Neutral','Positive']
print("Train:", X_train.shape[0], " Val:", X_val.shape[0], " Test:", X_test.shape[0])

# ================================================================
# MODEL 1: NAIVE BAYES (from scratch — Multinomial NB)
# Exact same as Phase_06_Model_Development_Hadia.ipynb
# ================================================================
print("\n" + "="*60)
print("TRAINING MODEL 1: NAIVE BAYES (from scratch)")
print("="*60)

log_prior = {}
log_likelihood = {}
total_documents = len(y_train_arr)
vocabulary_size = X_train.shape[1]

for sentiment in classes_str:
    class_documents = np.sum(y_train_arr == sentiment)
    prior = class_documents / total_documents
    log_prior[sentiment] = np.log(prior)

    class_matrix = X_train[y_train_arr == sentiment]
    word_counts = np.asarray(class_matrix.sum(axis=0)).flatten()
    total_word_counts = word_counts.sum()
    likelihood = (word_counts + 1) / (total_word_counts + vocabulary_size)
    log_likelihood[sentiment] = np.log(likelihood)

def predict_nb(X_data):
    predictions = []
    for i in range(X_data.shape[0]):
        review = X_data[i].toarray().flatten()
        scores = {}
        for sentiment in classes_str:
            scores[sentiment] = log_prior[sentiment] + np.sum(review * log_likelihood[sentiment])
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

nb_test_pred = predict_nb(X_test)
nb_acc = accuracy_score(y_test, nb_test_pred) * 100
print(f"Naive Bayes Test Accuracy: {nb_acc:.2f}%")

nb_bundle = {
    "tfidf": tfidf,
    "classes": list(classes_str),
    "log_prior": log_prior,
    "log_likelihood": log_likelihood,
}
with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb_bundle, f)
print("Saved: naive_bayes_model.pkl")

# ================================================================
# MODEL 2: LOGISTIC REGRESSION (from scratch, One-vs-Rest,
# L2 regularization + SMOTE)
# Exact same as LogisticRegressionWithSMOTE.ipynb, including the
# full sequential hyperparameter tuning (alpha -> lambda -> iters)
# ================================================================
print("\n" + "="*60)
print("TRAINING MODEL 2: LOGISTIC REGRESSION (from scratch + SMOTE)")
print("="*60)

label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
inv_label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
class_names = ['Negative', 'Neutral', 'Positive']
y_train_enc = np.array([label_map[v] for v in y_train_arr])
y_val_enc   = np.array([label_map[v] for v in y_val])
y_test_enc  = np.array([label_map[v] for v in y_test])
classes = [0, 1, 2]

# --- SMOTE on RAW features (before bias column) — sirf training data pe ---
print("Applying SMOTE on training data only (raw features, before bias)...")
smote = SMOTE(random_state=42)
X_train_sm_raw, y_train_sm = smote.fit_resample(X_train.toarray(), y_train_enc)
X_train_sm_raw = csr_matrix(X_train_sm_raw)

# --- Bias column add karo SMOTE ke baad, teeno sets mein ---
ones_train = csr_matrix(np.ones((X_train_sm_raw.shape[0], 1)))
ones_val   = csr_matrix(np.ones((X_val.shape[0],           1)))
ones_test  = csr_matrix(np.ones((X_test.shape[0],          1)))
X_train_sm = hstack([ones_train, X_train_sm_raw]).tocsr()
X_val_b    = hstack([ones_val,   X_val          ]).tocsr()
X_test_b   = hstack([ones_test,  X_test         ]).tocsr()
print("Bias added after SMOTE. Final training shape:", X_train_sm.shape)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta, lam=0):
    m = len(y)
    z = X.dot(theta)
    h = sigmoid(z)
    h = np.clip(h, 1e-7, 1 - 1e-7)
    cross_entropy = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    l2_penalty = (lam / (2 * m)) * np.sum(theta[1:] ** 2)
    return cross_entropy + l2_penalty

def gradient_descent(X, y, theta, alpha, iterations, lam=0, track_cost=False):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        z = X.dot(theta)
        h = sigmoid(z)
        error = h - y
        gradient = (1/m) * X.T.dot(error)
        reg_term = (lam/m) * theta.copy()
        reg_term[0] = 0
        theta = theta - alpha * (gradient + reg_term)
        if track_cost:
            cost_history.append(compute_cost(X, y, theta, lam))
    return theta, cost_history

def train_one_vs_rest(X, y, classes, alpha, iterations, lam, track_cost=False):
    thetas_local = {}
    for c in classes:
        y_bin = (y == c).astype(int)
        theta0 = np.zeros(X.shape[1])
        theta_c, _ = gradient_descent(X, y_bin, theta0, alpha, iterations, lam, track_cost)
        thetas_local[c] = theta_c
    return thetas_local

def predict_multiclass(X, thetas, classes):
    probs = np.zeros((X.shape[0], len(classes)))
    for c in classes:
        probs[:, c] = sigmoid(X.dot(thetas[c]))
    return np.argmax(probs, axis=1)

# ----------------------------------------------------------------
# STEP 7A: ALPHA TUNING — SEQUENTIAL, STOP-ON-INCREASE METHOD
# ----------------------------------------------------------------
print("\n--- Alpha tuning (sequential method) ---")
alphas_to_try = [0.001, 0.01, 0.1, 1, 3, 5, 7, 10]
tuning_iterations = 500
tolerance = 1e-4

prev_cost = None
best_alpha = alphas_to_try[0]

for a in alphas_to_try:
    theta = np.zeros(X_train_sm.shape[1])
    y_bin = (y_train_sm == 2).astype(int)   # representative classifier
    cost_hist = []

    for i in range(tuning_iterations):
        z = X_train_sm.dot(theta)
        h = sigmoid(z)
        error = h - y_bin
        gradient = (1/len(y_bin)) * X_train_sm.T.dot(error)
        theta = theta - a * gradient
        cost_hist.append(compute_cost(X_train_sm, y_bin, theta, 0))
        if i > 0 and abs(cost_hist[-2] - cost_hist[-1]) < tolerance:
            break

    final_cost = cost_hist[-1]
    print(f"Alpha={a:<8} Final Cost={final_cost:.6f}")

    if prev_cost is not None and final_cost > prev_cost:
        print(f"  -> Cost barh gaya, ruk rahe hain. Best alpha = {best_alpha}")
        break

    prev_cost = final_cost
    best_alpha = a

print(f"Best Alpha: {best_alpha}")

# ----------------------------------------------------------------
# STEP 7B: LAMBDA TUNING — U-CURVE, STOP-ON-INCREASE METHOD
# ----------------------------------------------------------------
print("\n--- Lambda tuning (validation error U-curve) ---")
lambdas_to_try = [0, 0.001, 0.01, 0.1, 1, 10]

prev_val_error = None
best_lambda = lambdas_to_try[0]

for lam in lambdas_to_try:
    thetas_cand = train_one_vs_rest(X_train_sm, y_train_sm, classes, best_alpha, tuning_iterations, lam)
    y_val_pred_cand = predict_multiclass(X_val_b, thetas_cand, classes)
    val_acc_cand = accuracy_score(y_val_enc, y_val_pred_cand)
    val_error_cand = 1 - val_acc_cand

    print(f"Lambda={lam:<8} Val Error={val_error_cand:.4f} (Val Acc={val_acc_cand*100:.2f}%)")

    if prev_val_error is not None and val_error_cand > prev_val_error:
        print(f"  -> Val error barh gaya, ruk rahe hain. Best lambda = {best_lambda}")
        break

    prev_val_error = val_error_cand
    best_lambda = lam

print(f"Best Lambda: {best_lambda}")

# ----------------------------------------------------------------
# STEP 8: BEST ITERATIONS — TOLERANCE METHOD (avg cost across 3 classifiers)
# ----------------------------------------------------------------
print("\n--- Iteration tuning (tolerance / elbow method) ---")
lam = best_lambda
max_iter_long = 1500
tolerance_iter = 1e-5

thetas_long = {c: np.zeros(X_train_sm.shape[1]) for c in classes}
avg_cost_history = []
best_iter_found = max_iter_long

for i in range(max_iter_long):
    step_costs = []
    for c in classes:
        y_bin = (y_train_sm == c).astype(int)
        z = X_train_sm.dot(thetas_long[c])
        h = sigmoid(z)
        error = h - y_bin
        gradient = (1/len(y_bin)) * X_train_sm.T.dot(error)
        reg_term = (lam/len(y_bin)) * thetas_long[c].copy()
        reg_term[0] = 0
        thetas_long[c] = thetas_long[c] - best_alpha * (gradient + reg_term)
        step_costs.append(compute_cost(X_train_sm, y_bin, thetas_long[c], lam))

    avg_cost_history.append(np.mean(step_costs))

    if i > 0:
        improvement = abs(avg_cost_history[-2] - avg_cost_history[-1])
        if improvement < tolerance_iter:
            best_iter_found = i + 1
            print(f"  Converged at iteration: {best_iter_found} (improvement={improvement:.8f})")
            break

if best_iter_found == max_iter_long:
    print(f"  Max iterations ({max_iter_long}) reached")

best_iterations = best_iter_found
print(f"\nFinal Selected Hyperparameters -> alpha={best_alpha}, lambda={best_lambda}, iterations={best_iterations}")

# ----------------------------------------------------------------
# STEP 9: TRAIN FINAL MODEL — ONE-VS-REST WITH TUNED HYPERPARAMETERS
# ----------------------------------------------------------------
print("\n--- Training final One-vs-Rest model ---")
thetas = {}
for c in classes:
    print(f"  Training classifier: {class_names[c]} vs Rest")
    y_binary = (y_train_sm == c).astype(int)
    theta_init = np.zeros(X_train_sm.shape[1])
    theta_c, _ = gradient_descent(X_train_sm, y_binary, theta_init, best_alpha, best_iterations, lam, track_cost=False)
    thetas[c] = theta_c

lr_val_pred = predict_multiclass(X_val_b, thetas, classes)
lr_val_acc = accuracy_score(y_val_enc, lr_val_pred) * 100
lr_test_pred = predict_multiclass(X_test_b, thetas, classes)
lr_acc = accuracy_score(y_test_enc, lr_test_pred) * 100
print(f"Logistic Regression Validation Accuracy: {lr_val_acc:.2f}%")
print(f"Logistic Regression Test Accuracy: {lr_acc:.2f}%")

lr_bundle = {
    "tfidf": tfidf,
    "thetas": thetas,
    "classes": classes,
    "label_map": label_map,
    "inv_label": inv_label,
    "best_alpha": best_alpha,
    "best_lambda": best_lambda,
    "best_iters": best_iterations,
}
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lr_bundle, f)
print("Saved: logistic_regression_model.pkl")

# ================================================================
# MODEL 3: SVM (sklearn, linear kernel)
# Exact same as Phase_06_Model_Development_SVM.ipynb
# ================================================================
print("\n" + "="*60)
print("TRAINING MODEL 3: SVM (linear kernel)")
print("="*60)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train_arr)
svm_test_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_test_pred) * 100
print(f"SVM Test Accuracy: {svm_acc:.2f}%")

svm_bundle = {"tfidf": tfidf, "model": svm_model}
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_bundle, f)
print("Saved: svm_model.pkl")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*60)
print("ALL 3 MODELS TRAINED & SAVED SUCCESSFULLY")
print("="*60)
print(f"Naive Bayes          : {nb_acc:.2f}%")
print(f"Logistic Regression  : {lr_acc:.2f}%")
print(f"SVM (Linear Kernel)  : {svm_acc:.2f}%")
print("\nAb 'streamlit run app.py' chalao — dashboard inhi 3 files ko use karega.")
