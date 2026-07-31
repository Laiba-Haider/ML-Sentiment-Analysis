# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn scipy imbalanced-learn nltk

# 2. Train all 3 models from scratch (Generates all .pkl files)
python train_all_models.py

# 3. Launch the Web Application
streamlit run app.py