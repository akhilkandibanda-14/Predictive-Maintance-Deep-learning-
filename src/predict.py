import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

class PredictiveMaintenanceModel:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self._load_assets()

    def _load_assets(self):
        try:
            model_path = os.path.join(self.models_dir, 'ann_model.keras')
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
            feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
                
            self.model = tf.keras.models.load_model(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_names = joblib.load(feature_names_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model assets: {str(e)}")

    def validate_input(self, raw_input):
        required_keys = [
            'Type', 
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        
        for key in required_keys:
            if key not in raw_input:
                raise ValueError(f"Missing required input: {key}")
                
        # Validate Type
        if raw_input['Type'] not in ['L', 'M', 'H']:
            raise ValueError("Invalid machine Type. Must be 'L', 'M', or 'H'")
            
        # Validate numerics
        numeric_keys = [k for k in required_keys if k != 'Type']
        for key in numeric_keys:
            val = raw_input[key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"Input '{key}' must be a number.")
            if val < 0:
                raise ValueError(f"Input '{key}' cannot be negative.")
        
        # Domain-specific validation to catch extremely out-of-range sensor values
        if raw_input['Air temperature [K]'] < 200 or raw_input['Air temperature [K]'] > 400:
            raise ValueError("Air temperature out of expected realistic range (200-400 K).")
        if raw_input['Process temperature [K]'] < 200 or raw_input['Process temperature [K]'] > 400:
            raise ValueError("Process temperature out of expected realistic range (200-400 K).")
            
        return True

    def predict(self, raw_input, threshold=0.5):
        """
        Predict failure risk for a single machine instance.
        Uses the default classification threshold of 0.5, but can be configured.
        """
        self.validate_input(raw_input)
        
        # Convert to DataFrame to match training format, preserving exact columns
        df = pd.DataFrame([raw_input])
        
        # Apply preprocessor (transform only, DO NOT fit, no leakage)
        try:
            X_processed = self.preprocessor.transform(df)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {str(e)}")
            
        # Make prediction without retraining or updating weights
        prob = self.model.predict(X_processed, verbose=0)[0][0]
        
        # Determine classification and risk level based on the configurable threshold
        is_failure = bool(prob >= threshold)
        
        if prob < threshold * 0.6:
            risk_level = "Low"
        elif prob < threshold:
            risk_level = "Medium"
        elif prob < threshold + ((1.0 - threshold) * 0.5):
            risk_level = "High"
        else:
            risk_level = "Critical"
            
        status = "Failure Predicted" if is_failure else "Normal Operation"
        
        return {
            "failure_probability": float(prob),
            "prediction": is_failure,
            "status": status,
            "risk_level": risk_level,
            "threshold_used": float(threshold)
        }

if __name__ == '__main__':
    # Simple command-line test for verification
    try:
        print("Initializing Predictive Maintenance Model Inference Pipeline...")
        # Paths are relative to the root when executing `python src/predict.py`
        predictor = PredictiveMaintenanceModel(models_dir='models')
        
        print("Model assets loaded successfully. Running a test prediction...")
        
        # Sample Machine 1 (Likely Normal)
        sample_normal = {
            'Type': 'L',
            'Air temperature [K]': 298.1,
            'Process temperature [K]': 308.6,
            'Rotational speed [rpm]': 1551,
            'Torque [Nm]': 42.8,
            'Tool wear [min]': 10
        }
        
        # Sample Machine 2 (Likely Failure - high tool wear, high torque, low speed)
        sample_failure = {
            'Type': 'L',
            'Air temperature [K]': 302.3,
            'Process temperature [K]': 311.2,
            'Rotational speed [rpm]': 1300,
            'Torque [Nm]': 65.0,
            'Tool wear [min]': 230
        }
        
        print("\n--- Test Case 1: Normal Machine ---")
        print(f"Input: {sample_normal}")
        result_1 = predictor.predict(sample_normal)
        print(f"Result: {result_1}")
        
        print("\n--- Test Case 2: High Risk Machine ---")
        print(f"Input: {sample_failure}")
        result_2 = predictor.predict(sample_failure, threshold=0.5)
        print(f"Result: {result_2}")
        
        print("\n--- Test Case 3: Invalid Input (Expected to fail) ---")
        bad_sample = sample_normal.copy()
        bad_sample['Type'] = 'Z'
        print(f"Input: {bad_sample}")
        try:
            predictor.predict(bad_sample)
        except ValueError as e:
            print(f"Caught expected Error: {e}")
            
    except Exception as e:
        print(f"Pipeline verification failed: {e}")
