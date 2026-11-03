
import streamlit as st
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="SMS Shield | Spam Detector",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------
# LOAD SAVED MODEL
# --------------------------------

model = joblib.load("model/spam_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f1024, #191633, #101827);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg, #5426a8, #7c3aed, #2563eb);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 35px rgba(91, 45, 180, 0.3);
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    color: #e9ddff;
    font-size: 17px;
}

.section-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
}

textarea {
    background-color: #20213b !important;
    color: white !important;
    border-radius: 12px !important;
}

.stTextArea label {
    color: #e0d7ff !important;
    font-weight: 600;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    height: 52px;
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    border: 1px solid #c4b5fd;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(124,58,237,0.4);
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.spam {
    background: rgba(239, 68, 68, 0.13);
    border: 1px solid #ef4444;
}

.ham {
    background: rgba(34, 197, 94, 0.13);
    border: 1px solid #22c55e;
}

.result-title {
    font-size: 30px;
    font-weight: bold;
    margin: 8px 0;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

.footer {
    text-align: center;
    color: #aaa6c8;
    margin-top: 45px;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.markdown("""
<div class="hero">
    <h1>🛡️ SMS SHIELD</h1>
    <p>Intelligent SMS Spam Detection System</p>
    <p>Protect yourself from unwanted and suspicious messages.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------
# PROJECT INFORMATION
# --------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>🤖</h2>
        <h3>AI Powered</h3>
        <p>Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>🎯</h2>
        <h3>98.26%</h3>
        <p>Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>🔍</h2>
        <h3>Smart Scan</h3>
        <p>Spam Classification</p>
    </div>
    """, unsafe_allow_html=True)


st.write("")

# --------------------------------
# MESSAGE INPUT SECTION
# --------------------------------

st.markdown("""
<div class="section-card">
    <h2>📩 Analyze Your Message</h2>
    <p style="color:#b9b5d0;">
        Enter an SMS message below to check whether it is Spam or Ham.
    </p>
</div>
""", unsafe_allow_html=True)

message = st.text_area(
    "Enter your SMS message",
    placeholder="Example: Congratulations! You have won a free prize. Claim now!",
    height=150
)

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("🔍  ANALYZE MESSAGE"):

    if message.strip() == "":
        st.warning("Please enter a message before analyzing.")

    else:

        # Convert message into TF-IDF features
        message_tfidf = tfidf.transform([message])

        # Predict category
        prediction = model.predict(message_tfidf)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(message_tfidf)[0]

        # Match probability to its class
        probability_dict = dict(zip(model.classes_, probabilities))

        ham_probability = probability_dict["ham"] * 100
        spam_probability = probability_dict["spam"] * 100

        confidence = max(ham_probability, spam_probability)

        st.markdown("---")

        st.subheader("📊 Analysis Result")

        if prediction == "spam":

            st.markdown(f"""
            <div class="result-card spam">
                <h1>🚨</h1>
                <div class="result-title" style="color:#ff6b6b;">
                    SPAM MESSAGE DETECTED
                </div>
                <p style="color:#ffd1d1;">
                    This message has been classified as suspicious.
                </p>
                <h3 style="color:white;">Confidence: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card ham">
                <h1>🛡️</h1>
                <div class="result-title" style="color:#4ade80;">
                    HAM MESSAGE
                </div>
                <p style="color:#c7f9d4;">
                    This message has been classified as a normal message.
                </p>
                <h3 style="color:white;">Confidence: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        # Probability details
        st.write("")
        st.subheader("Prediction Probability")

        p1, p2 = st.columns(2)

        with p1:
            st.write("🟢 Ham")
            st.progress(float(ham_probability / 100))
            st.write(f"{ham_probability:.2f}%")

        with p2:
            st.write("🔴 Spam")
            st.progress(float(spam_probability / 100))
            st.write(f"{spam_probability:.2f}%")


# --------------------------------
# FOOTER
# --------------------------------

st.markdown("""
<div class="footer">
    <p>SMS SHIELD | Python Machine Learning Project</p>
    <p>Built with Python • Scikit-learn • TF-IDF • Streamlit</p>
</div>
""", unsafe_allow_html=True)
