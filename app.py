import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Academic Risk Prediction', page_icon='🎓')
st.title('Academic Risk Prediction')
st.caption('Student academic risk estimate using a trained Random Forest model')

@st.cache_resource
def load_model():
    return joblib.load('academic_risk_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error('Missing academic_risk_model.pkl. Export best_model from the Random Forest notebook and place it next to app.py.')
    st.stop()

with st.form('student'):
    age = st.number_input('Age', min_value=1, max_value=120, value=17)
    study = st.number_input('Study time per week (hours)', min_value=0.0, max_value=168.0, value=5.0, step=0.5)
    absences = st.number_input('Absences', min_value=0, max_value=365, value=20)
    tutoring = st.selectbox('Tutoring', ['No', 'Yes'])
    support = st.selectbox('Parental support', ['None (0)', 'Low (1)', 'Moderate (2)', 'High (3)', 'Very high (4)'])
    education = st.selectbox('Parental education', ['None (0)', 'High School (1)', 'Some College (2)', 'Bachelor (3)', 'Higher (4)'])
    extra = st.selectbox('Extracurricular activities', ['No', 'Yes'])
    sports = st.selectbox('Sports', ['No', 'Yes'])
    music = st.selectbox('Music', ['No', 'Yes'])
    volunteering = st.selectbox('Volunteering', ['No', 'Yes'])
    submitted = st.form_submit_button('Predict Academic Risk', type='primary')

if submitted:
    row = pd.DataFrame([{
        'Age': age, 'StudyTimeWeekly': study, 'Absences': absences,
        'Tutoring': int(tutoring == 'Yes'), 'ParentalSupport': int(support[-2]),
        'ParentalEducation': int(education[-2]),
        'Extracurricular': int(extra == 'Yes'), 'Sports': int(sports == 'Yes'),
        'Music': int(music == 'Yes'), 'Volunteering': int(volunteering == 'Yes')
    }])
    result = int(model.predict(row)[0])
    if result == 1:
        st.error('Prediction: AT RISK (1)')
    else:
        st.success('Prediction: NOT AT RISK (0)')
    if hasattr(model, 'predict_proba'):
        classes = list(model.classes_)
        probability = float(model.predict_proba(row)[0, classes.index(1)])
        st.metric('Estimated probability of academic risk', f'{probability:.1%}')
    st.caption('This prediction is a statistical estimate, not a definitive assessment of a student.')
