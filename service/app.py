from flask import Flask, render_template, request, jsonify
import logging
import joblib
import numpy as np
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='flask.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# ABSOLUTE PATHS - CONFIRMED BY YOUR ls COMMAND
MODEL_PATH = "/Users/mac/PABD25/models/linear_regression_model.pkl"
SCALER_PATH = "/Users/mac/PABD25/models/scaler.pkl"

# Debug output
print(f"=== PATH VERIFICATION ===")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Scaler exists: {os.path.exists(SCALER_PATH)}")

# Load models with verification
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("=== MODELS LOADED SUCCESSFULLY ===")
except Exception as e:
    print("=== LOADING FAILED ===")
    print(f"Error: {e}")
    print("Please verify:")
    print(f"1. {MODEL_PATH} exists")
    print(f"2. {SCALER_PATH} exists")
    print("3. Both files are readable")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    data = request.get_json()
    try:
        area = float(data['number1'])
        scaled_area = scaler.transform([[area]])
        predicted_price = model.predict(scaled_area)[0]
        return jsonify({
            'status': 'success',
            'price': round(predicted_price, 2)
        })
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)