
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the preprocessor and model
try:
    model = joblib.load('best_decision_tree_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    print("Model and Preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500

    try:
        data = request.get_json(force=True)

        if not isinstance(data, list):
            data = [data]

        input_df = pd.DataFrame(data)
        expected_features = ['Age', 'Gender', 'CityTier']
        input_df = input_df[expected_features]

        X_processed = preprocessor.transform(input_df)

        categorical_features = ['Gender', 'CityTier']
        numerical_features = ['Age']
        categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = list(categorical_feature_names) + numerical_features
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

        predictions = model.predict(X_processed_df)
        prediction_proba = model.predict_proba(X_processed_df)[:, 1]

        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': int(predictions[i]),
                'probability': float(prediction_proba[i])
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
