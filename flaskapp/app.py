import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

loaded = joblib.load("math_model.pkl")

if isinstance(loaded, tuple):
    scaler, model = loaded
else:
    scaler = None
    model = loaded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')

    if features is None or len(features) != 9:
        return jsonify({'error': 'Exactly 9 features are required.'}), 400

    try:
        feature_names = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'goout', 'G1', 'G2']
        features_array = np.array(features, dtype=float).reshape(1, -1)
        if scaler:
            features_array = pd.DataFrame([features], columns=feature_names)
            features_array = scaler.transform(features_array)
    except Exception as e:
        return jsonify({'error': 'Invalid input format. Ensure all values are numbers.'}), 400

    try:
        prediction = model.predict(features_array)[0]
        return jsonify({
            'prediction': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({'error': 'Prediction failed. Model error.'}), 500

if __name__ == '__main__':
    app.run(debug=True)