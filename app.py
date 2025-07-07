from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your heart disease model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract 13 input features from form
        inputs = [float(request.form[key]) for key in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]]
        
        prediction = model.predict([np.array(inputs)])
        result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk / No Heart Disease"

        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
