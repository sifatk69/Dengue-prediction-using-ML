import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the model and label encoders
def load_model():
    with open('voting_classifier_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Load label encoders for categorical variables
label_encoders = {}
categorical_columns = ['severe_headache', 'pain_behind_the_eyes', 'joint_muscle_aches', 'metallic_taste_of_mouth',
                       'appetite_loss', 'abdominal_pain', 'nausea_vomiting']

for col in categorical_columns:
    with open(f'{col}_label_encoder.pkl', 'rb') as file:
        label_encoders[col] = pickle.load(file)

# Load the machine learning model
loaded_model = load_model()


# Preprocess user inputs
def preprocess_input(user_inputs):
    for col, encoder in label_encoders.items():
        user_inputs[col] = encoder.transform([user_inputs[col]])[0]



    input_features = [user_inputs['temp'], user_inputs['wbc'], user_inputs['severe_headache'], user_inputs['pain_behind_the_eyes'],
                      user_inputs['joint_muscle_aches'], user_inputs['metallic_taste_of_mouth'], user_inputs['appetite_loss'],
                      user_inputs['abdominal_pain'], user_inputs['nausea_vomiting'], user_inputs['hemo'],user_inputs['hematocrite'], user_inputs['plttelte']]

    return input_features


# Streamlit UI for user inputs
def show_predict_page():
    st.title("Dengue Fever Prediction")
    st.write("""### Please provide the following information:""")
    temp = st.slider("Body Temperature (Farenheit)", 0.00, 105.00, 00.1)
    wbc = st.slider("WBC White Blood Cell (K)", 0.00, 13.00, 00.1)
    headache = st.selectbox("Severe Headache", ["yes", "no"])
    pain_eyes = st.selectbox("Pain Behind the Eyes", ["yes", "no"])
    joint_muscle = st.selectbox("Joint Muscle Aches", ["yes", "no"])
    metal_taste = st.selectbox("Metallic Taste of Mouth", ["yes", "no"])
    appetit_loss = st.selectbox("Appetite Loss", ["yes", "no"])
    abdominal_pain = st.selectbox("Abdominal Pain", ["yes", "no"])
    nau_vomiting = st.selectbox("Nausea-Vomiting", ["yes", "no"])
    hemo = st.slider("Hemoglobin (gram per DL)", 0.00, 15.00, 0.10)
    hematocrite = st.slider("Hematrocrite (%)", 0, 100, 1)
    plttelte = st.slider("Platelet", 0, 400, 1)

    ok = st.button("Predict Dengue")

    if ok:
        # Preprocess user inputs
        user_inputs = {
            "temp": temp,
            "wbc": wbc,
            "severe_headache": headache,
            "pain_behind_the_eyes": pain_eyes,
            "joint_muscle_aches": joint_muscle,
            "metallic_taste_of_mouth": metal_taste,
            "appetite_loss": appetit_loss,
            "abdominal_pain": abdominal_pain,
            "nausea_vomiting": nau_vomiting,
            "hemo": hemo,
            "hematocrite": hematocrite,
            "plttelte": plttelte
        }

        input_features = preprocess_input(user_inputs)

        # Predict using the loaded model
        prediction = loaded_model.predict([input_features])[0]

        # Display the prediction
        if prediction == 1:
            st.write("Prediction: Dengue Fever")
        else:
            st.write("Prediction: No Dengue Fever")
