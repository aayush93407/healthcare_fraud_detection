from flask import Flask, render_template, request
import os
from pdf_extract import extract_text_from_pdf, extract_entities, predict_billing
import tempfile

app = Flask(__name__)
MODEL_PATH = "billing_model.pkl"

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['pdf']
    if not file:
        return "No file uploaded.", 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        file.save(temp_pdf.name)
        pdf_path = temp_pdf.name

    try:
        text = extract_text_from_pdf(pdf_path)
        hospital, doctor, actual_amount = extract_entities(text)

        medical_condition = "Diabetes"
        admission_type = "Emergency"
        insurance_provider = "Medicare"

        predicted_amount = predict_billing(
            hospital, doctor, medical_condition,
            admission_type, insurance_provider,
            MODEL_PATH
        )

        difference = abs(actual_amount - predicted_amount) if actual_amount else None

        return render_template(
            'result.html',
            hospital=hospital,
            doctor=doctor,
            actual_amount=actual_amount,
            predicted_amount=int(predicted_amount),
            difference=int(difference) if difference else None
        )

    except Exception as e:
        return f"Error processing PDF: {e}", 500
    finally:
        os.remove(pdf_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

