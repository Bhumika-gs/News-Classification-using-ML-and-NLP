import streamlit as st
import joblib
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="News Classification",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f6f8;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* ================= HEADER ================= */

.header-container {
    background: linear-gradient(135deg, #102d4f, #2169a3);
    padding: 32px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 6px 18px rgba(20, 55, 90, 0.18);
}

.header-title {
    color: white;
    font-size: 42px;
    font-weight: 750;
    margin: 0;
    letter-spacing: -1px;
}

.header-subtitle {
    color: #dceaf7;
    font-size: 17px;
    margin-top: 8px;
}

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #eef6ff;
    padding: 7px 15px;
    border-radius: 20px;
    font-size: 13px;
    margin-top: 18px;
}

/* ================= SECTION HEADERS ================= */

.section-title {
    color: #173b60;
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 7px;
}

.section-description {
    color: #68798a;
    font-size: 15px;
    margin-bottom: 18px;
}

/* ================= INFO CARDS ================= */

.info-card {
    background: white;
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid #d9e1e8;
    box-shadow: 0 2px 8px rgba(31,45,61,0.05);
    min-height: 90px;
}

.card-label {
    color: #718096;
    font-size: 12px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.card-value {
    color: #173b60;
    font-size: 20px;
    font-weight: 700;
    margin-top: 6px;
}

.card-description {
    color: #718096;
    font-size: 13px;
    line-height: 1.4;
    margin-top: 7px;
}

/* ================= RESULT ================= */

.prediction-box {
    background: white;
    border-left: 6px solid #2474b5;
    padding: 24px 28px;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0 3px 12px rgba(31,45,61,0.08);
}

.prediction-label {
    color: #718096;
    font-size: 13px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.prediction-category {
    color: #173b60;
    font-size: 32px;
    font-weight: 750;
    margin-top: 5px;
}

.prediction-description {
    color: #68798a;
    font-size: 14px;
    margin-top: 5px;
}

/* ================= CATEGORY CARDS ================= */

.category-card {
    background: white;
    border: 1px solid #d9e1e8;
    border-radius: 12px;
    padding: 18px;
    min-height: 125px;
    box-shadow: 0 2px 7px rgba(31,45,61,0.05);
}

.category-name {
    color: #173b60;
    font-size: 19px;
    font-weight: 700;
}

.category-description {
    color: #718096;
    font-size: 13px;
    line-height: 1.45;
    margin-top: 8px;
}

/* ================= BUTTON ================= */

.stButton > button {
    background: linear-gradient(135deg, #174a7c, #2474b5);
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-size: 16px;
    font-weight: 650;
    box-shadow: 0 4px 10px rgba(23,74,124,0.20);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #123d67, #1e629a);
    color: white;
}

/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background-color: #172b40;
}

[data-testid="stSidebar"] * {
    color: #e8f0f7;
}

.sidebar-title {
    text-align: center;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 8px;
}

.sidebar-subtitle {
    text-align: center;
    color: #b9cad9;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 22px;
}

/* Pipeline */

.pipeline-title {
    text-align: center;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 18px;
}

.pipeline-step {
    text-align: center;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 10px 6px;
    margin: 0 auto;
    font-size: 14px;
    width: 90%;
}

.pipeline-arrow {
    text-align: center;
    font-size: 20px;
    color: #8fc3ec;
    line-height: 1;
    padding: 7px 0;
}

/* Footer */

.footer {
    text-align: center;
    color: #7a8794;
    font-size: 13px;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("model/logistic_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# CATEGORY MAPPING
# ============================================================

categories = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

category_descriptions = {
    "World": "International news and global events",
    "Sports": "Sports, teams, matches and competitions",
    "Business": "Markets, companies and financial news",
    "Sci/Tech": "Science, technology and innovation"
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📰 News Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'An NLP-powered news categorization system.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="pipeline-title">'
        'Machine Learning Pipeline'
        '</div>',
        unsafe_allow_html=True
    )

    # Pipeline steps

    pipeline = [
        "📝 Text Input",
        "🧹 Text Preprocessing",
        "🔢 TF-IDF Vectorization",
        "🤖 Logistic Regression",
        "📰 News Category"
    ]

    for i, step in enumerate(pipeline):

        st.markdown(
            f'<div class="pipeline-step">{step}</div>',
            unsafe_allow_html=True
        )

        if i < len(pipeline) - 1:

            st.markdown(
                '<div class="pipeline-arrow">↓</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown(
        '<div class="pipeline-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.metric("Test Accuracy", "92%")
    st.metric("Macro F1-Score", "0.92")

    st.markdown("---")

    st.caption(
        "Python • Scikit-learn • Streamlit"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="header-container">'
    '<div class="header-title">📰 News Classification</div>'
    '<div class="header-subtitle">'
    'Intelligent News Categorization using Natural Language Processing '
    'and Machine Learning'
    '</div>'
    '<div class="header-badge">'
    'NLP &nbsp;•&nbsp; TF-IDF &nbsp;•&nbsp; Logistic Regression'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Model Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The trained machine learning model analyzes news text and '
    'predicts its most likely category.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-label">Algorithm</div>'
        '<div class="card-value">Logistic Regression</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-label">Feature Extraction</div>'
        '<div class="card-value">TF-IDF</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-label">Training Data</div>'
        '<div class="card-value">120K Articles</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-label">Test Accuracy</div>'
        '<div class="card-value">92%</div>'
        '</div>',
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# NEWS INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Analyze News Article</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Paste a headline, article description, or complete news article below.'
    '</div>',
    unsafe_allow_html=True
)

news_text = st.text_area(
    "News Article",
    height=220,
    placeholder=(
        "Paste or type your news article here..."
    ),
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# CLASSIFICATION
# ============================================================

if st.button(
    "🔍  Classify News Article",
    use_container_width=True
):

    if not news_text.strip():

        st.warning(
            "Please enter a news article before classification."
        )

    else:

        text_tfidf = vectorizer.transform(
            [news_text]
        )

        prediction = model.predict(
            text_tfidf
        )[0]

        probabilities = model.predict_proba(
            text_tfidf
        )[0]

        predicted_category = categories[prediction]

        confidence = probabilities[
            list(model.classes_).index(prediction)
        ]

        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Classification Result'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-box">'
            '<div class="prediction-label">'
            'Predicted Category'
            '</div>'
            f'<div class="prediction-category">'
            f'{predicted_category}'
            '</div>'
            f'<div class="prediction-description">'
            f'{category_descriptions[predicted_category]}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence

        confidence_col1, confidence_col2 = st.columns(
            [1, 3]
        )

        with confidence_col1:

            st.metric(
                "Model Confidence",
                f"{confidence:.2%}"
            )

        with confidence_col2:

            st.progress(
                float(confidence)
            )

        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Category Probabilities'
            '</div>',
            unsafe_allow_html=True
        )

        probability_data = pd.DataFrame(
            {
                "Category": [
                    categories[c]
                    for c in model.classes_
                ],
                "Probability": probabilities
            }
        )

        probability_cols = st.columns(4)

        for col, (_, row) in zip(
            probability_cols,
            probability_data.iterrows()
        ):

            with col:

                st.markdown(
                    '<div class="category-card">'
                    f'<div class="category-name">'
                    f'{row["Category"]}'
                    '</div>'
                    f'<div class="category-description">'
                    f'{row["Probability"]:.2%}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart

        chart_data = probability_data.set_index(
            "Category"
        )

        st.bar_chart(
            chart_data,
            y="Probability",
            height=350
        )


# ============================================================
# SUPPORTED CATEGORIES
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    'Supported News Categories'
    '</div>',
    unsafe_allow_html=True
)

category_cols = st.columns(4)

for col, (category, description) in zip(
    category_cols,
    category_descriptions.items()
):

    with col:

        st.markdown(
            '<div class="category-card">'
            f'<div class="category-name">'
            f'{category}'
            '</div>'
            f'<div class="category-description">'
            f'{description}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'News Classification System'
    '<br>'
    'NLP + TF-IDF + Logistic Regression'
    '<br>'
    'Built with Python, Scikit-learn and Streamlit'
    '</div>',
    unsafe_allow_html=True
)