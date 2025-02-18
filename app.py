import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64

# Set page configuration for a wider layout and custom icon
st.set_page_config(
    page_title="Medical Disease Prediction & Recommendation",
    page_icon="🩺",
    layout="wide"
)

# Function to convert an image file to a base64 string


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Encode the background image (make sure "medicineimg.jpeg" is in your working directory)
img_base64 = get_base64_of_bin_file("medicineimg.jpeg")

# Inject custom CSS with the base64 encoded image for the form container background
st.markdown(
    f"""
    <style>
    /* Set a light background for the app */
    .reportview-container {{
        background-color: #f7f7f7;
    }}
    /* Centering elements */
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    /* Custom title styling */
    .custom-title {{
        font-size: 3em;
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
    }}
    .custom-subtitle {{
        font-size: 1.5em;
        color: #34495e;
        text-align: center;
        margin-bottom: 1rem;
    }}
    /* Styling the form container with a background image */
    .form-container {{
        background: url("data:image/jpeg;base64,{img_base64}") no-repeat center center;
        background-size: cover;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #fff; /* Adjust text color for contrast */
    }}
    /* Styling the result box */
    .result-box {{
        background-color: #dff0d8;
        border: 1px solid #d6e9c6;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Load datasets and the trained model
symptom_disease = pd.read_csv("data/Symptom-severity.csv")
precautions = pd.read_csv("data/precautions_df.csv")
workout = pd.read_csv("data/workout_df.csv")
description = pd.read_csv("data/description.csv")
medication = pd.read_csv("data/medications.csv")
diet = pd.read_csv("data/diets.csv")
model = pickle.load(open("model/svc_rbf_2.pkl", "rb"))

symptoms_dictionary = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
                       'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
Disease_dictionary = {0: '(vertigo) Paroymsal Positional Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis', 4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis', 8: 'Chicken pox', 9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue', 12: 'Diabetes', 13: 'Dimorphic hemmorhoids(piles)', 14: 'Drug Reaction', 15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack', 19: 'Hepatitis B',
                      20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthristis', 32: 'Paralysis (brain hemorrhage)', 33: 'Peptic ulcer diseae', 34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection', 39: 'Varicose veins', 40: 'hepatitis A'}
diaseas = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
           'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dictionary))
    for item in patient_symptoms:
        input_vector[symptoms_dictionary[item]] = 1
    return Disease_dictionary[model.predict([input_vector])[0]]


def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([dc for dc in desc])

    pre = precautions[precautions['Disease'] == dis][[
        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [pc for pc in pre.values]

    med = medication[medication['Disease'] == dis]['Medication']
    med = [md for md in med.values]

    diett = diet[diet['Disease'] == dis]['Diet']
    diett = [dt for dt in diett.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [wo for wo in wrkout.values]

    return desc, pre, med, diett, wrkout


# Main Title and Subtitle
st.markdown("<div class='custom-title'>Medical Disease Prediction & Recommendation</div>",
            unsafe_allow_html=True)
st.markdown("<div class='custom-subtitle'>Know your symptoms and get personalized recommendations</div>",
            unsafe_allow_html=True)

# Centered banner image
st.markdown("<div class='centered'>", unsafe_allow_html=True)
st.image('medicineimg.jpeg', use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# Form for input using a styled container with the background image
with st.container():
    with st.form("Disease", clear_on_submit=True):
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)

        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            first_name = st.text_input("First Name")
        with col2:
            last_name = st.text_input("Last Name")
        with col3:
            contact_no = st.text_input("Contact Number")

        st.markdown("---")
        st.subheader("Select Your Symptoms")
        symp = st.multiselect("Choose symptoms from the list", options=diaseas)

        st.markdown("</div>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Predict and Recommend")

        if submit_button:
            predicted_disease = get_predicted_value(symp)
            desc, pre, med, diett, wrkout = helper(predicted_disease)

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.write("**Your Report:**")
            st.write(f"**Symptoms Provided:** {symp}")
            st.success(f"**Predicted Disease:** {predicted_disease}")
            st.write("")
            st.write(f"**Description:** {desc}")
            st.write("")
            st.write(f"**Precautions:** {pre[0] if pre else 'N/A'}")
            st.write(
                f"**Medication Recommendation:** {med[0] if med else 'N/A'}")
            st.write(
                f"**Diet Recommendation:** {diett[0] if diett else 'N/A'}")
            st.write(f"**Workout Suggestion:** {wrkout if wrkout else 'N/A'}")
            st.write("")
            st.info("Model used: Support Vector Machine (kernel='rbf')")
            st.markdown("</div>", unsafe_allow_html=True)
