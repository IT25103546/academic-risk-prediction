# Academic Risk Prediction Streamlit Deployment

## Run locally

Use Python 3.11 or 3.12 (recommended) and install:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy online

1. Upload `app.py`, `requirements.txt`, and `academic_risk_model.pkl` to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Create a new app from the repository, choose branch `main`, main file `app.py`, then deploy.
4. Test a sample input and check the deployed app logs if loading fails.

The serialized model was saved with scikit-learn 1.6.1; the requirements pin that version. Only load trusted pickle/joblib files. Check that the app's input feature names, category encodings and training preprocessing match the notebook before relying on predictions. This is a risk estimate, not a definitive student assessment.
