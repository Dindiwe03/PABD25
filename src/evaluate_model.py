import os
import logging
import joblib
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(file_dir, "logs", "model_evaluation.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define required features
REQUIRED_FEATURES = ['total_meters', 'floor_ratio', 'rooms_count']
TARGET_COLUMN = 'price'

def load_model_artifacts(model_path, preprocessor_path):
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading preprocessor from: {preprocessor_path}")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def load_test_data(input_path):
    logger.info(f"Loading test data from: {input_path}")
    data = pd.read_csv(input_path, index_col='url_id')
    return data

def evaluate_model(model, preprocessor, data):
    logger.info("Evaluating model...")
    
    # Data validation
    required_columns = REQUIRED_FEATURES + [TARGET_COLUMN]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Preprocessing - fixed syntax
    data['rooms_count'] = data['rooms_count'].astype(float).fillna(1).astype(int).clip(upper=3)
    
    # Feature/target selection
    X = data[REQUIRED_FEATURES]
    y = data[TARGET_COLUMN]

    # Model evaluation
    X_processed = preprocessor.transform(X)
    predictions = model.predict(X_processed)

    # Calculate metrics
    metrics = {
        "mae": mean_absolute_error(y, predictions),
        "rmse": np.sqrt(mean_squared_error(y, predictions)),
        "r2": r2_score(y, predictions),
        "mean_actual_price": y.mean(),
        "mean_predicted_price": predictions.mean()
    }

    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.2f}")

    return metrics

def predict_price(model, preprocessor, features):
    logger.info("Making prediction...")
    
    # Check if we need to calculate floor_ratio
    if 'floor_ratio' not in features and all(k in features for k in ['floor', 'floors_count']):
        features['floor_ratio'] = features['floor'] / features['floors_count']
    
    # Validate input features
    missing_features = [feat for feat in REQUIRED_FEATURES if feat not in features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Create input DataFrame
    input_df = pd.DataFrame([features])
    
    # Preprocess rooms_count - simplified syntax
    input_df['rooms_count'] = input_df['rooms_count'].astype(float).fillna(1).astype(int).clip(upper=3)
    
    # Transform and predict
    input_processed = preprocessor.transform(input_df)
    predicted_price = model.predict(input_processed)[0]
    
    logger.info(f"Predicted price: {predicted_price:,.0f} руб")
    return predicted_price

def save_metrics(metrics, reports_dir):
    try:
        os.makedirs(reports_dir, exist_ok=True)
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        metrics_json_path = os.path.join(reports_dir, "metrics.json")
        with open(metrics_json_path, "w", encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        metrics_txt_path = os.path.join(reports_dir, "metrics.txt")
        with open(metrics_txt_path, "w", encoding='utf-8') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        metrics_csv_path = os.path.join(reports_dir, "metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)

        logger.info("Metrics saved in JSON, TXT, and CSV formats.")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Evaluate apartment price prediction model.')
    parser.add_argument('--model_path', type=str,
                      default=os.path.join(file_dir, "..", "models", "apartment_price_model.pkl"),
                      help='Path to trained model')
    parser.add_argument('--preprocessor_path', type=str,
                      default=os.path.join(file_dir, "..", "models", "preprocessor.pkl"),
                      help='Path to preprocessor')
    parser.add_argument('--test_data_path', type=str,
                      default=os.path.join(file_dir, "..", "data", "processed", "train.csv"),
                      help='Path to test data')

    args = parser.parse_args()

    try:
        logger.info("Model evaluation started.")
        model, preprocessor = load_model_artifacts(args.model_path, args.preprocessor_path)
        test_data = load_test_data(args.test_data_path)

        metrics = evaluate_model(model, preprocessor, test_data)

        # Sample prediction with raw features
        sample_features = {
            'total_meters': 60,
            'floor': 10,
            'floors_count': 25,
            'rooms_count': 2
        }
        predicted_price = predict_price(model, preprocessor, sample_features)
        metrics['sample_prediction'] = predicted_price

        reports_dir = os.path.join(file_dir, "..", "reports")
        save_metrics(metrics, reports_dir)

        logger.info("Model evaluation completed successfully.")
        return 0
    except Exception as e:
        logger.critical(f"Critical error during model evaluation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())