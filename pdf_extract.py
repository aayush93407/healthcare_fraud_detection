import pdfplumber
import re
import joblib  
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using pdfplumber (text-based only)."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_entities(text):
    """Extract hospital name, doctor name, and billing amount using keyword search and regex."""
    hospital_name = "Unknown"
    doctor_name = "Unknown"
    billing_amount = None
    
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "Name of Insurer" in line:
            hospital_name = lines[i + 1].strip() if i + 1 < len(lines) else "Unknown"
        elif "Doctor Name" in line:
            doctor_name = lines[i + 1].strip() if i + 1 < len(lines) else "Unknown"
    
    # Match billing amount like "Total Rs. 12 345"
    pattern = r"Total\s*Rs\.?\s*((?:\d\s*)+)"
    match = re.search(pattern, text)
    if match:
        billing_amount = int(match.group(1).replace(" ", ""))
    
    return hospital_name, doctor_name, billing_amount

def train_model(csv_path, model_path):
    """Train an ML model to predict billing amounts with additional features."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    label_encoders = {}
    for col in ['Doctor', 'Hospital', 'Medical Condition', 'Admission Type', 'Insurance Provider']:
        df[col] = df[col].str.lower().str.strip()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df['Billing Amount'] = df['Billing Amount'].replace('[^0-9]', '', regex=True).astype(int)

    X = df[['Doctor', 'Hospital', 'Medical Condition', 'Admission Type', 'Insurance Provider']]
    y = df['Billing Amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump((model, label_encoders), model_path)

def predict_billing(hospital, doctor, medical_condition, admission_type, insurance_provider, model_path):
    """Predict billing amount using a trained ML model with additional features."""
    model, label_encoders = joblib.load(model_path)

    def encode_label(value, encoder):
        value = value.lower().strip()
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1  # unknown value

    features = np.array([
        encode_label(doctor, label_encoders['Doctor']),
        encode_label(hospital, label_encoders['Hospital']),
        encode_label(medical_condition, label_encoders['Medical Condition']),
        encode_label(admission_type, label_encoders['Admission Type']),
        encode_label(insurance_provider, label_encoders['Insurance Provider'])
    ]).reshape(1, -1)

    predicted_amount = model.predict(features)
    return predicted_amount[0]

def main(pdf_path, model_path):
    text = extract_text_from_pdf(pdf_path)
    hospital, doctor, actual_amount = extract_entities(text)

    medical_condition = "Diabetes"  
    admission_type = "Emergency"  
    insurance_provider = "Medicare"  

    predicted_amount = predict_billing(hospital, doctor, medical_condition, admission_type, insurance_provider, model_path)

    print(f"Extracted Hospital: {hospital}")
    print(f"Extracted Doctor: {doctor}")
    print(f"Actual Billing Amount: {actual_amount}")
    print(f"Predicted Billing Amount: {predicted_amount}")

    if actual_amount:
        difference = abs(actual_amount - predicted_amount)
        print(f"Billing Difference: {difference}")
    else:
        print("Could not extract actual billing amount from PDF.")

# Uncomment these lines only for local training/testing
# train_model("healthcare_dataset.csv", "billing_model.pkl")
# main("rewq.pdf", "billing_model.pkl")
