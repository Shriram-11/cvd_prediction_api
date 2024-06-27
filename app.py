from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model and scaler
model = tf.keras.models.load_model('heart_disease_model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Extract features from the JSON
    features = np.array(data['features']).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = (model.predict(features_scaled) > 0.5).astype("int32")

    # Return prediction as JSON
    return jsonify({'prediction': int(prediction[0][0])})


if __name__ == '__main__':
    app.run(debug=True)
