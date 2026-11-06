
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from catboost import CatBoostClassifier

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="💙",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# PROFESSIONAL WHITE, BLACK AND BLUE DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
}

.main .block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stWidgetLabel"] p {
    color: #222222 !important;
}

h1, h2, h3, h4 {
    color: #142D4E !important;
}

.welcome-card {
    background-color: #174F91;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 25px;
}

.welcome-card h1,
.welcome-card p {
    color: #FFFFFF !important;
}

.section-box {
    background-color: #174F91;
    padding: 13px 18px;
    border-radius: 9px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.section-box h3 {
    color: #FFFFFF !important;
    margin: 0;
}

.stButton button,
.stFormSubmitButton button {
    background-color: #174F91 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px;
    font-weight: bold;
    padding: 10px 20px;
}

.stButton button p,
.stFormSubmitButton button p {
    color: #FFFFFF !important;
}

[data-testid="stForm"] {
    background-color: #F5F8FC;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #DCE5EF;
}

.stApp input, .stApp textarea {
    background-color: #FFFFFF !important;
    color: #222222 !important;
}


.notice-box {
    background-color: #174F91;
    color: #FFFFFF !important;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.notice-box p {
    color: #FFFFFF !important;
}
.result-box {

    background-color: #174F91;
    padding: 22px;
    border-radius: 12px;
}

.result-box h2, .result-box p {
    color: #FFFFFF !important;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND METADATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(BASE_DIR / "diabetes_catboost_model.cbm"))

    with open(BASE_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    return model, metadata

model, metadata = load_model()

FEATURES = metadata["features"]
THRESHOLD = metadata["threshold"]

# --------------------------------------------------
# WELCOME SECTION
# --------------------------------------------------

st.markdown("""
<div class="welcome-card">
    <h1>Diabetes Prediction System</h1>
    <p>Welcome to your health assessment interface.</p>
    <p>Enter your health information to receive a
    machine-learning-based prediction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="notice-box">
        <p>This system is designed for academic research and
        educational purposes. It does not provide a medical diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# PATIENT ASSESSMENT FORM
# --------------------------------------------------

with st.form("diabetes_assessment"):

    st.markdown("""
    <div class="section-box">
        <h3>1. General Health Information</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=12,
            max_value=98,
            value=28,
            step=1
        )

        gen_health = st.selectbox(
            "General health",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Excellent",
                2: "Very good",
                3: "Good",
                4: "Fair",
                5: "Poor"
            }[x]
        )

        age = st.selectbox(
            "Age group",
            list(range(1, 14)),
            format_func=lambda x: {
                1: "18–24 years",
                2: "25–29 years",
                3: "30–34 years",
                4: "35–39 years",
                5: "40–44 years",
                6: "45–49 years",
                7: "50–54 years",
                8: "55–59 years",
                9: "60–64 years",
                10: "65–69 years",
                11: "70–74 years",
                12: "75–79 years",
                13: "80 years or older"
            }[x]
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: {
                0: "Female",
                1: "Male"
            }[x]
        )

        education = st.selectbox(
            "Education level",
            list(range(1, 7)),
            format_func=lambda x: {
                1: "Never attended / Kindergarten",
                2: "Primary (Grades 1–8)",
                3: "Secondary (Grades 9–11)",
                4: "Senior secondary / GED",
                5: "Some college",
                6: "College graduate"
            }[x]
        )

        income = st.selectbox(
            "Annual household income (approximate Naira equivalents)",
            list(range(1, 9)),
            format_func=lambda x: {
                1: "Below ₦13.4 million",
                2: "₦13.4m – below ₦20.2m",
                3: "₦20.2m – below ₦26.9m",
                4: "₦26.9m – below ₦33.6m",
                5: "₦33.6m – below ₦47.1m",
                6: "₦47.1m – below ₦67.2m",
                7: "₦67.2m – below ₦100.8m",
                8: "₦100.8 million or more"
            }[x]
        )

    st.caption(
        "Income amounts are approximate conversions of the original "
        "U.S. dataset categories, not Nigerian-specific income bands. "
        "The model retains the original category codes."
    )

    # MEDICAL HISTORY

    st.markdown("""
    <div class="section-box">
        <h3>2. Medical History</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        high_bp = st.selectbox("High blood pressure?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        high_chol = st.selectbox("High cholesterol?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        chol_check = st.selectbox("Cholesterol check in the last 5 years?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        stroke = st.selectbox("Ever had a stroke?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

    with col2:
        heart_disease = st.selectbox("Heart disease or heart attack?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        smoker = st.selectbox("Smoked at least 100 cigarettes in your lifetime?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        healthcare = st.selectbox("Have any kind of healthcare coverage?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        no_doc_cost = st.selectbox("Could not see a doctor because of cost in the past 12 months?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

    # LIFESTYLE

    st.markdown("""
    <div class="section-box">
        <h3>3. Lifestyle and Physical Health</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        physical_activity = st.selectbox("Any physical activity in the past 30 days, excluding activity at work?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        fruits = st.selectbox("Consume fruit daily?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        veggies = st.selectbox("Consume vegetables daily?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        heavy_alcohol = st.selectbox("Heavy alcohol consumption (men: more than 14 drinks/week; women: more than 7 drinks/week)?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

    with col2:
        diff_walk = st.selectbox("Serious difficulty walking or climbing stairs?", [0, 1],
            format_func=lambda x: "Yes" if x else "No")

        ment_health = st.number_input(
            "Days of poor mental health (last 30 days)",
            min_value=0, max_value=30, value=0
        )

        phys_health = st.number_input(
            "Days of poor physical health (last 30 days)",
            min_value=0, max_value=30, value=0
        )

    st.divider()

    consent = st.checkbox(
        "I understand that this is an educational prediction tool "
        "and not a medical diagnosis."
    )

    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if not consent:
        st.warning("Please accept the disclaimer before proceeding.")

    else:

        input_values = {
            "HighBP": high_bp,
            "HighChol": high_chol,
            "CholCheck": chol_check,
            "BMI": bmi,
            "Smoker": smoker,
            "Stroke": stroke,
            "HeartDiseaseorAttack": heart_disease,
            "PhysActivity": physical_activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "HvyAlcoholConsump": heavy_alcohol,
            "AnyHealthcare": healthcare,
            "NoDocbcCost": no_doc_cost,
            "GenHlth": gen_health,
            "MentHlth": ment_health,
            "PhysHlth": phys_health,
            "DiffWalk": diff_walk,
            "Sex": sex,
            "Age": age,
            "Education": education,
            "Income": income
        }

        input_data = pd.DataFrame(
            [[input_values[f] for f in FEATURES]],
            columns=FEATURES
        )

        probability = float(model.predict_proba(input_data)[0][1])

        st.markdown("---")
        st.header("Prediction Result")

        st.metric(
            "Model-predicted likelihood",
            f"{probability * 100:.2f}%"
        )



        if probability >= THRESHOLD:
            result_title = "Higher predicted likelihood of diabetes"
            result_text = "The model identified a higher likelihood of the combined positive category."
        else:
            result_title = "Lower predicted likelihood of diabetes"
            result_text = "The model identified a lower likelihood of the combined positive category."

        st.markdown(f"""
        <div style="background-color:#174F91; padding:20px;
                    border-radius:12px; margin-top:15px;">
            <h3 style="color:#FFFFFF !important;
                       -webkit-text-fill-color:#FFFFFF !important;">
                {result_title}
            </h3>
            <p style="color:#FFFFFF !important;
                      -webkit-text-fill-color:#FFFFFF !important;">
                {result_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            f"Classification threshold: {THRESHOLD:.2f}"
        )

        st.markdown("""
        <div style="background-color:#174F91; padding:20px;
                    border-radius:12px; margin-top:15px;">
            <h4 style="color:#FFFFFF !important;
                       -webkit-text-fill-color:#FFFFFF !important;">
                Important Medical Notice
            </h4>
            <p style="color:#FFFFFF !important;
                      -webkit-text-fill-color:#FFFFFF !important;">
                This model cannot distinguish diabetes from prediabetes.
                A qualified healthcare professional and appropriate blood
                glucose testing are needed for diagnosis. A lower prediction
                does not rule out either condition.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("---")
st.header("About This Project")

st.write("**Project:** Diabetes Prediction Using Machine Learning")
st.write("**Student:** Abduljabar Aminu Yau")
st.write("**Registration Number:** UG22CSC1006")
st.write("**Department:** Computer Science")
st.write("**Institution:** Northwest University, Kano")
st.write("**Supervisor:** Abba Mukhtar Bala")
st.write("**Machine Learning Algorithm:** CatBoost Classifier")
st.write("**Dataset:** CDC Diabetes Health Indicators (UCI Repository)")

st.markdown("""
<div class="footer">
    Diabetes Prediction System | Academic Research Project | 2026
</div>
""", unsafe_allow_html=True)
