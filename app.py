from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from googlesearch import search  

app = Flask(__name__)

# 94 symptoms
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
      'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

# List of diseases
nmap = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}
disease = sorted(nmap, key=lambda x: nmap[x])

# Placeholder for trained models
dt_model = rf_model = nb_model = None

def train_models():
    global dt_model, rf_model, nb_model
    # Load and preprocess data
    df = pd.read_csv('dataset/Training.csv')
    tr = pd.read_csv('dataset/Testing.csv')
    df.replace({'prognosis': nmap}, inplace=True)
    tr.replace({'prognosis': nmap}, inplace=True)
    X, y = df[l1], df['prognosis']
    X_test, y_test = tr[l1], tr['prognosis']

    # Train classifiers
    dt_model = DecisionTreeClassifier().fit(X, y)
    rf_model = RandomForestClassifier().fit(X, y)
    nb_model = GaussianNB().fit(X, y)

    # Log accuracies
    print(f"Decision Tree accuracy: {accuracy_score(y_test, dt_model.predict(X_test))*100:.2f}%")
    print(f"Random Forest accuracy: {accuracy_score(y_test, rf_model.predict(X_test))*100:.2f}%")
    print(f"Naive Bayes accuracy: {accuracy_score(y_test, nb_model.predict(X_test))*100:.2f}%")

train_models()

def get_doctors_scrape(disease_name, num_results=5):
    """
    Scrape Google for top 'best <disease_name> specialist near me' URLs.
    Returns list of dicts: [{'title': domain, 'url': full_url}, ...]
    """
    query = f"best {disease_name} specialist near me"
    doctors = []
    for url in search(query, num_results=num_results, lang="en"):
        # Use domain as title placeholder; for richer titles you'd fetch page <title> via requests+BS4
        title = url.split("//")[-1].split("/")[0]
        doctors.append({"title": title, "url": url})
    return doctors

@app.route('/', methods=['GET', 'POST'])
def index():
    original = {}
    enhanced = {}
    if request.method == 'POST':
        # 1. Collect selected symptoms
        selected = {request.form.get(f'symptom{i}') for i in range(1, 6) if request.form.get(f'symptom{i}')}
        # 2. Build full feature vector (length = len(l1))
        l2 = [1 if s in selected else 0 for s in l1]

        # 3. Predict indices
        dt_idx = dt_model.predict([l2])[0]
        rf_idx = rf_model.predict([l2])[0]
        nb_idx = nb_model.predict([l2])[0]

        # 4. Get confidences
        dt_conf = round(dt_model.predict_proba([l2])[0][dt_idx]*100, 1)
        rf_conf = round(rf_model.predict_proba([l2])[0][rf_idx]*100, 1)
        nb_conf = round(nb_model.predict_proba([l2])[0][nb_idx]*100, 1)

        # 5. Original simple names
        original = {
            'Decision Tree':  disease[dt_idx],
            'Random Forest':  disease[rf_idx],
            'Naive Bayes':    disease[nb_idx]
        }

        # 6. Enhanced info with doctors
        for model_name, idx, conf in [
            ('Decision Tree', dt_idx, dt_conf),
            ('Random Forest', rf_idx, rf_conf),
            ('Naive Bayes',   nb_idx, nb_conf)
        ]:
            dis = disease[idx]
            enhanced[model_name] = {
                'disease':   dis,
                'confidence': conf,
                'doctors':   get_doctors_scrape(dis)  # scrape live results :contentReference[oaicite:1]{index=1}
            }

    return render_template(
    'index.html',
    symptoms=l1,
    original=original,
    enhanced=enhanced
)

if __name__ == '__main__':
    app.run(debug=True)
   