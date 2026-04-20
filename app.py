from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model sekali saat startup
model = joblib.load('xgboost_mbti_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # {"features": [val1, val2, ...]}
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"mbti": str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)