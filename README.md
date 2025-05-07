# AIES-CCP


---

## 🩺 Project Report: **College Doctor – Symptom-Based Disease Predictor**

### **Overview**

**College Doctor** is a web-based application designed to assist users in identifying possible diseases based on selected symptoms. It uses manually implemented machine learning algorithms to predict the most likely disease and suggests relevant specialists for consultation.

---

### **Objectives**

* Help users (e.g., students or general public) quickly assess potential illnesses based on symptoms.
* Provide recommendations for medical specialists.
* Demonstrate the use of custom-built machine learning models without relying on external libraries like `scikit-learn`.

---

### **Key Features**

* **User-Friendly Interface:**
  A clean, responsive frontend built with Bootstrap 5, allowing users to select up to 5 symptoms via dropdown menus.

* **Manual Machine Learning Models:**
  The backend uses:

  * **Decision Tree** (built from scratch using ID3 algorithm)
  * **Random Forest** (ensemble of Decision Trees)
  * **Gaussian Naive Bayes** (probabilistic model for classification)

* **Prediction Output:**
  After submitting symptoms, the app displays:

  * Predicted disease from each model
  * Confidence score
  * A list of recommended medical specialists (doctors) with links

* **Robust Validation:**
  Includes both client-side (JavaScript) and server-side validation to ensure complete input.

---

### **Tech Stack**

* **Frontend:** HTML5, CSS, Bootstrap 5, Jinja2 templating
* **Backend:** Python, Flask
* **ML Algorithms:** Manually implemented in Python (`DecisionTree`, `RandomForest`, `GaussianNB`)
* **Data Source:** CSV datasets (`Training.csv`, `Testing.csv`) with symptom-disease mappings

---

### **How It Works**

1. User selects symptoms from dropdown menus.
2. The Flask backend receives the selected symptoms and converts them into feature vectors.
3. Each custom model makes a prediction based on trained data.
4. The application displays predictions along with suggested doctors.

---

### **Future Enhancements**

* Add real-time symptom search with auto-suggestion.
* Integrate with online health APIs for live doctor data.
* Extend the model to handle more symptoms and diseases dynamically.
* Add multilingual support for wider accessibility.

---

