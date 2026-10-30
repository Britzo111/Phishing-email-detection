# AI-Based Phishing Email Detection

A machine learning tool that classifies emails as **phishing/spam** or **legitimate** using Natural Language Processing, with an interactive Streamlit web app and SHAP explainability.

Built by **Britto Benny** as part of an MSc in Software Design with Cyber Security at the Technological University of the Shannon (TUS), Athlone.

## Results

Three models were trained on about 6,000 emails (SpamAssassin public corpus) and tested on a held-out set of 1,200 emails:

| Model | Accuracy |
|---|---|
| Logistic Regression | 97.83% |
| XGBoost | 99.75% |
| **Random Forest (tuned)** | **99.83%** |

The tuned Random Forest caught **all 199 spam emails** in the test set, with only **2 false positives** out of 1,001 legitimate emails.

![Model comparison](results/compariaon%20of%20allthe%20models.png)

## How it works

1. **Parsing** – reads raw `.eml` files, including HTML emails (BeautifulSoup), and extracts the subject, sender, body and URLs.
2. **Preprocessing** – lowercasing, URL and punctuation removal, stop-word removal and lemmatisation (NLTK).
3. **Feature extraction** – TF-IDF with 5,000 features.
4. **Modelling** – Logistic Regression, XGBoost and Random Forest, with the Random Forest tuned using GridSearchCV and 5-fold cross-validation.
5. **Explainability** – SHAP values show which words drove each prediction.
6. **Web app** – a Streamlit app that rates each email as high, medium or low risk and flags suspicious URLs and keywords.

![Streamlit app detecting a phishing email](results/phishing%20mail%20streamlit.png)

## Project structure

```
app/          Streamlit web app
notebooks/    EDA, preprocessing, model training, SHAP explainability
src/          Dataset builder, email parser, feature extraction, training script
models/       Trained models and TF-IDF vectoriser
results/      Charts, confusion matrices and screenshots
```

## Run it locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Data

The dataset is not included in this repository. Download the public [SpamAssassin corpus](https://spamassassin.apache.org/old/publiccorpus/) into `data/raw/spamassassin/`, then run `python src/build_dataset.py`.

## Limitations and next steps

- The model was trained on the SpamAssassin spam/ham corpus. Next step: evaluate it on a dedicated phishing dataset (Nazario phishing corpus).
- Add sender-domain and URL-reputation features alongside the text features.
