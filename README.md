# Medical Disease Prediction & Recommendation 🩺

Welcome to the **Medical Disease Prediction & Recommendation System**! This project leverages machine learning to predict diseases based on user-input symptoms and then provides recommendations for precautions, medications, diet, and workouts. The interactive web app is built using Streamlit with an attractive, custom UI.

![](https://raw.github.com/alphatechlogics/MedicalDiseasePredictionRecommendation/b837dd9ac6606005a4b15aa5463c836dbf124652/Screenshot%202025-06-17%20232440.png)

---

## Table of Contents

- [Overview](#overview-)
- [Features](#features-)
- [Dataset](#dataset-)
- [Installation](#installation-)

---

## Overview 🚀

The **Medical Disease Prediction & Recommendation System** aims to help users quickly identify potential health issues based on their symptoms. The app utilizes a Support Vector Machine (SVM) with an RBF kernel to generate predictions. In addition, it provides personalized recommendations for:

- **Precautions**
- **Medications**
- **Diet Plans**
- **Workout Suggestions**

All this is delivered through a user-friendly, interactive interface built with Streamlit.

---

## Features 💡

- **Interactive User Interface:** Designed with Streamlit and enhanced with custom CSS and a background image.
- **Disease Prediction:** Uses a pre-trained SVM model to predict diseases based on selected symptoms.
- **Personalized Recommendations:** Offers detailed advice on precautions, medications, diet, and workouts.
- **Dataset-Driven Insights:** Leverages a comprehensive dataset from Kaggle.
- **Aesthetic Design:** The UI features a visually appealing background image (embedded via base64 encoding) and modern styling.

---

## Dataset 📊

The dataset used for this project is sourced from Kaggle:

- **[Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)**

This dataset includes detailed information about various diseases, their symptoms, descriptions, and recommendations, making it a robust resource for building a healthcare prediction system.

---

## Installation ⚙️

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/alphatechlogics/MedicalDiseasePredictionRecommendation.git
   cd MedicalDiseasePredictionRecommendation
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**

- Visit the [Kaggle Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) and download the files.

- Place the CSV files in the data/ folder.
- Place the Background Image
- Ensure that medicineimg.jpeg is located in the root directory of the project (this image is used for the app’s background).

5. **Run the App**
   ```bash
   streamlit run app.py
   ```
