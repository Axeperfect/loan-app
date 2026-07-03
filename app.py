from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn.ensemble
import sys

# FIX: Map the pickle import path directly to scikit-learn's class
sys.modules['GradientBoostingClassifier'] = sklearn.ensemble.GradientBoostingClassifier

app = Flask(__name__)

try:
    with open('loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Success: 'loan_model.pkl' loaded perfectly!")
except FileNotFoundError:
    print("Warning: 'loan_model.pkl' not found. Running in test mode.")
    model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    decision = None
    status_color = ""
    error = None
    
    if request.method == 'POST':
        try:
            monthly_salary = float(request.form.get('monthly_salary', 0))
            employment_status = int(request.form.get('employment_status', 0))
            years_employed = float(request.form.get('years_employed', 0))
            requested_amount = float(request.form.get('requested_amount', 0))
            
            features = np.array([[monthly_salary, employment_status, years_employed, requested_amount]])
            
            if model is not None:
                # Use your loaded pkl model to predict
                score = model.predict(features)[0]
                if isinstance(score, np.ndarray):
                    score = score[0]
                score = max(0, min(1000, score))
            else:
                # Fallback test formula if model fails to load
                # Adjusted multiplier for monthly salary to reflect Tugriks scaling
                score = (monthly_salary * 0.00015 + years_employed * 40) - (requested_amount * 0.00001)
                score = max(0, min(1000, score))
            
            prediction = round(score, 1)
            
            if prediction >= 700:
                decision = "Approved"
                status_color = "bg-green-50 text-green-700 border-green-200"
            elif prediction >= 450:
                decision = "Review Required"
                status_color = "bg-amber-50 text-amber-700 border-amber-200"
            else:
                decision = "Denied"
                status_color = "bg-red-50 text-red-700 border-red-200"
                
        except Exception as e:
            error = f"Error: {str(e)}"
            
    return render_template('index.html', prediction=prediction, decision=decision, status_color=status_color, error=error)

if __name__ == '__main__':
    app.run(debug=True)