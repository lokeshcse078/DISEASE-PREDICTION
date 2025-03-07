import streamlit as st
import smtplib
import random
import bcrypt
import os
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import numpy as np
import joblib
import pandas as pd
import webbrowser
import base64
import os

os.environ["EMAIL_USER"] = "your-email@gmail.com"
os.environ["EMAIL_PASS"] = "app-pass"


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Replace 'your_image.jpg' with your actual image path
image_path = "2.jpg"
base64_img = get_base64_of_image(image_path)

# CSS for background image with a dark overlay
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{base64_img}");
    background-size: cover;
    background-attachment: fixed; /* Keeps background fixed */
    background-position: center;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed; /* Use fixed instead of absolute */
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);  /* Adjust opacity */
    z-index: -1;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar Header
st.sidebar.header("Need Help?")

# Display Contact Email in Sidebar
st.sidebar.write("📩 **Contact Email:**")
st.sidebar.markdown("[your-email@gamil.com.com](mailto:your-email-@gamil.com)")

# Support Email Link
support_email = "mailto: your-email-@gamil.com?subject=Help%20Needed&body=Hello,%20I%20need%20assistance%20regarding%20the%20disease%20prediction%20app."

# Help Button in Sidebar
if st.sidebar.button("📧 Contact Support"):
    webbrowser.open(support_email)
    st.sidebar.success("Opening your email client... 📩")
if st.sidebar.button("About Us"):
    st.sidebar.success("This is an model web browser that predicts the disease based on the provided patients test result details and its 80% true based on the model we provided")
if st.sidebar.button("FAQ'S"):
    webbrowser.Chrome()
    st.sidebar.success("opening")
# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            username TEXT,
            password TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT,
            otp INTEGER,
            expiry DATETIME,
            PRIMARY KEY (email)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to verify passwords
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# Function to send OTP
def send_otp(email):
    otp = random.randint(100000, 999999)
    expiry_time = datetime.now() + timedelta(minutes=5)
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO otps (email, otp, expiry) VALUES (?, ?, ?)", (email, otp, expiry_time))
    conn.commit()
    conn.close()
    
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")
    receiver_email = email
    
    if not sender_email or not sender_password:
        st.error("Email credentials are not set. Configure environment variables EMAIL_USER and EMAIL_PASS.")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "Your OTP Code"
        msg.attach(MIMEText(f"Your OTP code is {otp}. It expires in 5 minutes.", 'plain'))
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")
        return False

def register():
    st.title("Register New User")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    email = st.text_input("Email")
    
    if st.button("Send OTP"):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            st.error("Email already registered.")
        elif send_otp(email):
            st.session_state["otp_sent"] = True
            st.success("OTP sent to your email!")
        conn.close()
    
    if "otp_sent" in st.session_state and st.session_state["otp_sent"]:
        otp_input = st.text_input("Enter OTP")
        if st.button("Verify OTP"):
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT otp, expiry FROM otps WHERE email = ?", (email,))
            result = cursor.fetchone()
            if result and datetime.strptime(result[1], "%Y-%m-%d %H:%M:%S.%f") > datetime.now() and int(otp_input) == result[0]:
                cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, new_username, hash_password(new_password)))
                conn.commit()
                st.success("Registration successful! Please login.")
                cursor.execute("DELETE FROM otps WHERE email = ?", (email,))
                conn.commit()
            else:
                st.error("Invalid or expired OTP")
            conn.close()

def login():
    st.title("Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT username, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        if user and verify_password(password, user[1]):
            st.session_state["authenticated"] = True
            st.session_state["username"] = user[0]
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid email or password")

def main_app():
    st.title("DISEASE PREDICTION")
    st.write(f"Welcome, {st.session_state['username']}")
    model_diabetes = joblib.load("model_diabetes.pkl")
    scaler_diabetes = joblib.load("scaler_diabetes.pkl")

    model_heart = joblib.load("model_heart.pkl")
    scaler_heart = joblib.load("scaler_heart.pkl")

    parkinsons_model = joblib.load("parkinsons_model.pkl")
    parkinsons_scaler = joblib.load("parkinsons_scaler.pkl")

    alzheimers_model = joblib.load("alzheimers_model.pkl")
    scaler_alzheimers = joblib.load("scaler_alzheimers.pkl")

    model_skin = joblib.load("model_skin.pkl")
    scaler_skin = joblib.load("scaler_skin.pkl")

    pancreas_model = joblib.load("pancreas_model.pkl")
    pancreas_scaler = joblib.load("pancreas_scaler.pkl")

    breast_cancer_model = joblib.load("breast_cancer_model.pkl")
    breast_cancer_scaler = joblib.load("breast_cancer_scaler.pkl")

    # Function to process diabetes input data
    def process_diabetes_results(input_data, model, scaler):
        columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        df = pd.DataFrame([input_data], columns=columns)
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        return prediction[0]

    # Function to process heart disease input data
    def process_heart_results(input_data, model, scaler):
        columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        df = pd.DataFrame([input_data], columns=columns)
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        return prediction[0]


    def process_parkinsons_results(input_data, model, scaler):
        # Define feature columns based on the dataset
        columns = ["Age", "MMSE", "CDR", "Education_Years", "Memory_Score"]
        # Convert input data into a DataFrame
        df = pd.DataFrame([input_data], columns=columns)
        
        # Scale the input data
        df_scaled = scaler.transform(df)
        
        # Make a prediction
        prediction = model.predict(df_scaled)
        
        return prediction[0]  # Return the predicted class (0 = Healthy, 1 = Parkinson’s)

    def process_alzheimers_results(input_data, model, scaler):
        # Define feature columns based on the dataset
        columns = ["MDVP_Fo(Hz)", "MDVP_Fhi(Hz)", "MDVP_Flo(Hz)", "Jitter(%)", "Shimmer(dB)", "HNR"]
        
        # Convert input data into a DataFrame
        df = pd.DataFrame([input_data], columns=columns)
        
        # Scale the input data
        df_scaled = scaler.transform(df)
        
        # Make a prediction
        prediction = model.predict(df_scaled)
        
        return prediction[0]  # Return the predicted class (0 = Healthy, 1 = alzheimers)

    # Function for cancer prediction
    def process_cancer_results(input_data, model, scaler, feature_columns):
        df = pd.DataFrame([input_data], columns=feature_columns)
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        return prediction[0]

    # Function for Skin Cancer Prediction
    def predict_skin_cancer(texture_mean, area_mean, smoothness_mean):
        input_data = [texture_mean, area_mean, smoothness_mean]
        return process_cancer_results(input_data, model_skin, scaler_skin, ["Texture Mean", "Area Mean", "Smoothness Mean"])

    # Function for Pancreatic Cancer Prediction
    def predict_pancreatic_cancer(age, tumor_size, ca_19_9):
        input_data = [age, tumor_size, ca_19_9]
        return process_cancer_results(input_data, pancreas_model, pancreas_scaler, ["Age", "Tumor Size", "CA 19-9 Level"])

    # Function for Breast Cancer Prediction
    def predict_breast_cancer(age, tumor_size, family_history):
        input_data = [age, tumor_size, family_history]
        return process_cancer_results(input_data, breast_cancer_model, breast_cancer_scaler, ["Age", "Tumor Size", "Family History"])


    # Streamlit UI
    st.title("Disease Prediction App")
    option = st.selectbox("Select the disease to predict:", [
        "Diabetes", "Heart Disease", "Parkinson's Disease", "Alzheimer's disease", "Skin Cancer", "Pancreatic Cancer", "Breast Cancer"
    ])

    if option == "Diabetes":
        st.header("Diabetes Prediction")
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        
        if st.button("Predict Diabetes"):
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
            result = process_diabetes_results(input_data, model_diabetes, scaler_diabetes)
            st.success("✅ No Diabetes" if result == 0 else "⚠️ Has Diabetes")

    elif option == "Heart Disease":
        st.header("Heart Disease Prediction")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, step=1)
        chol = st.number_input("Cholesterol Level", min_value=0, max_value=600, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
        restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, step=1)
        exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1)
        slope = st.number_input("Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", min_value=0, max_value=4, step=1)
        thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, step=1)
        
        if st.button("Predict Heart Disease"):
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            result = process_heart_results(input_data, model_heart, scaler_heart)
            st.success("✅ No Heart Disease" if result == 0 else "⚠️ Has Heart Disease")

    elif option == "Parkinson's Disease":
        st.header("Parkinson's Disease Prediction")

        # Input fields for Parkinson’s dataset features
        mdvp_fo = st.number_input("MDVP_Fo (Fundamental Frequency in Hz)", min_value=50.0, max_value=300.0, step=0.1)
        mdvp_fhi = st.number_input("MDVP_Fhi (Highest Frequency in Hz)", min_value=50.0, max_value=300.0, step=0.1)
        mdvp_flo = st.number_input("MDVP_Flo (Lowest Frequency in Hz)", min_value=50.0, max_value=300.0, step=0.1)
        jitter = st.number_input("Jitter (%)", min_value=0.0, max_value=0.1, step=0.0001, format="%.5f")
        shimmer = st.number_input("Shimmer (dB)", min_value=0.0, max_value=1.0, step=0.01)
        hnr = st.number_input("HNR (Harmonics-to-Noise Ratio)", min_value=0.0, max_value=40.0, step=0.1)

        if st.button("Predict Parkinson's Disease"):
            input_data = [mdvp_fo, mdvp_fhi, mdvp_flo, jitter, shimmer, hnr]
            result = process_parkinsons_results(input_data, parkinsons_model, parkinsons_scaler)
            st.success("✅ No Parkinson's Disease" if result == 0 else "⚠️ Parkinson's Disease Detected")

    elif option == "Alzheimer's disease":
        st.header("Alzheimer's Disease Prediction")

        # Input fields for Alzheimer's dataset features
        age = st.number_input("Age", min_value=40, max_value=100, step=1)
        mmse = st.number_input("MMSE (Mini-Mental State Examination Score)", min_value=0, max_value=30, step=1)
        cdr = st.selectbox("CDR (Clinical Dementia Rating)", [0.0, 0.5, 1.0, 2.0])
        education_years = st.number_input("Education Years", min_value=0, max_value=25, step=1)
        memory_score = st.number_input("Memory Score (0-1)", min_value=0.0, max_value=1.0, step=0.01)

        if st.button("Predict Alzheimer's Disease"):
            input_data = [age, mmse, cdr, education_years, memory_score]
            result = process_alzheimers_results(input_data, alzheimers_model, scaler_alzheimers)
            st.success("✅ No Alzheimer's Disease" if result == 0 else "⚠️ Alzheimer's Disease Detected")

    elif option == "Skin Cancer":
        st.header("Skin Cancer Prediction")
        texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, step=0.1)
        area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2000.0, step=1.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, step=0.01)
        if st.button("Predict Skin Cancer"):
            result = predict_skin_cancer(texture_mean, area_mean, smoothness_mean)
            st.success("✅ No Skin Cancer" if result == 0 else "⚠️ Skin Cancer Detected")

    elif option == "Pancreatic Cancer":
        st.header("Pancreatic Cancer Prediction")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=20.0, step=0.1)
        ca_19_9 = st.number_input("CA 19-9 Level", min_value=0.0, max_value=5000.0, step=1.0)
        if st.button("Predict Pancreatic Cancer"):
            result = predict_pancreatic_cancer(age, tumor_size, ca_19_9)
            st.success("✅ No Pancreatic Cancer" if result == 0 else "⚠️ Pancreatic Cancer Detected")

    elif option == "Breast Cancer":
        st.header("Breast Cancer Prediction")
        age = st.number_input("Age", min_value=20, max_value=100, step=1)
        tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=20.0, step=0.1)
        family_history = st.selectbox("Family History (Yes=1, No=0)", [0, 1])
        if st.button("Predict Breast Cancer"):
            result = predict_breast_cancer(age, tumor_size, family_history)
            st.success("✅ No Breast Cancer" if result == 0 else "⚠️ Breast Cancer Detected")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if st.session_state["authenticated"]:
        main_app()
    else:
        option = st.radio("Choose an option", ["Login", "Register"])
        if option == "Login":
            login()
        else:
            register()

if __name__ == "__main__":
    main()






