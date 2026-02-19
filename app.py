from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ---------- LOAD MODEL SAFELY ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "payments.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ---------- HOME PAGE ----------
@app.route('/')
def home():
    return render_template('home.html')

# ---------- PREDICTION PAGE ----------
@app.route('/predict')
def predict_page():
    return render_template('predict.html')

# ---------- SUBMIT FORM ----------
@app.route('/submit', methods=['POST'])
def submit():
    try:
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        

        features = np.array([[amount,
                              oldbalanceOrg,
                              newbalanceOrig]])

        # SCALE FEATURES
        features = scaler.transform(features)

        prediction = int(model.predict(features)[0])

        return render_template(
            'submit.html',
            prediction=prediction
        )

    except Exception as e:
        return f"Error occurred: {e}"

# ---------- RUN APP ----------
if __name__ == '__main__':
    app.run(debug=True)
