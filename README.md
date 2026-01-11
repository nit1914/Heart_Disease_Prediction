# Heart_Disease_Prediction
The project develops an intelligent ML system analyzing health parameters             like blood pressure, cholesterol, and heart rate for early risk assessment and real-time predictions. 


# 🫀 Heart Disease Prediction - Complete Implementation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**A Complete Machine Learning Project with Source Code**

[📥 Installation](#-installation) • [💻 Code Structure](#-code-structure) • [🚀 Quick Start](#-quick-start) • [📊 Examples](#-examples)

</div>

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Core Code Files](#-core-code-files)
- [Data Processing](#1-data-processing)
- [Model Training](#2-model-training)
- [Prediction System](#3-prediction-system)
- [Web Application](#4-web-application)
- [API Endpoints](#5-api-endpoints)
- [Testing](#6-testing)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements File

Create `requirements.txt`:

```txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
flask==2.3.3
flask-cors==4.0.0
xgboost==1.7.6
lightgbm==4.0.0
plotly==5.16.1
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── 📂 data/
│   ├── heart.csv                    # Raw dataset
│   └── processed_data.csv           # Preprocessed data
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data cleaning and preprocessing
│   ├── feature_engineering.py       # Feature creation
│   ├── model_training.py            # Model training pipeline
│   ├── model_evaluation.py          # Evaluation metrics
│   └── predictor.py                 # Prediction interface
│
├── 📂 models/
│   └── heart_disease_model.pkl      # Saved model
│
├── 📂 notebooks/
│   └── heart_disease_analysis.ipynb # Jupyter notebook
│
├── 📂 api/
│   ├── app.py                       # Flask API
│   └── routes.py                    # API routes
│
├── 📂 templates/
│   └── index.html                   # Web interface
│
├── 📂 static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── 📂 tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── config.py                         # Configuration settings
├── main.py                          # Main execution script
└── requirements.txt                 # Dependencies
```

---

## 💻 Core Code Files

### 1. Data Preprocessing

**File: `src/data_preprocessing.py`**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    Handles all data preprocessing tasks for heart disease prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load the heart disease dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing = df.isnull().sum()
        
        if missing.sum() > 0:
            print(f"Missing values found: {missing.sum()}")
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    def remove_outliers(self, df, columns, threshold=3):
        """Remove outliers using Z-score method"""
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        # Already encoded in this dataset, but keeping for reference
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def split_features_target(self, df, target_column='target'):
        """Split data into features and target"""
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        self.feature_names = X.columns.tolist()
        return X, y
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, filepath, test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        # Load data
        df = self.load_data(filepath)
        
        if df is None:
            return None, None, None, None
        
        # Preprocessing steps
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        
        # Remove outliers from specific columns
        outlier_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
        df = self.remove_outliers(df, outlier_cols)
        
        # Split features and target
        X, y = self.split_features_target(df)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"✓ Training set: {X_train_scaled.shape}")
        print(f"✓ Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data('data/heart.csv')
```

---

### 2. Model Training

**File: `src/model_training.py`**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Train and evaluate multiple machine learning models
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        print("✓ Models initialized")
        return self.models
    
    def train_model(self, model, X_train, y_train):
        """Train a single model"""
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def cross_validate(self, model, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores.mean(), scores.std()
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        self.initialize_models()
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60 + "\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            trained_model = self.train_model(model, X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test)
            
            # Cross-validation
            cv_mean, cv_std = self.cross_validate(
                trained_model, X_train, y_train
            )
            
            # Store results
            self.results[name] = {
                'model': trained_model,
                'metrics': metrics,
                'cv_score': cv_mean,
                'cv_std': cv_std
            }
            
            # Print results
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print("-" * 60)
        
        # Find best model
        self.find_best_model()
        
        return self.results
    
    def find_best_model(self):
        """Identify the best performing model"""
        best_score = 0
        
        for name, result in self.results.items():
            score = result['metrics']['accuracy']
            if score > best_score:
                best_score = score
                self.best_model = result['model']
                self.best_model_name = name
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Accuracy: {best_score:.4f}\n")
        
        return self.best_model, self.best_model_name
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on Random Forest"""
        print("\nPerforming hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model, filepath='models/heart_disease_model.pkl'):
        """Save trained model to disk"""
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/heart_disease_model.pkl'):
        """Load trained model from disk"""
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data('data/heart.csv')
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    trainer.save_model(trainer.best_model)
```

---

### 3. Prediction System

**File: `src/predictor.py`**

```python
import numpy as np
import joblib

class HeartDiseasePredictor:
    """
    Make predictions using trained model
    """
    
    def __init__(self, model_path='models/heart_disease_model.pkl'):
        self.model = self.load_model(model_path)
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            model = joblib.load(model_path)
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Convert dict to array if needed
        if isinstance(input_data, dict):
            input_array = np.array([input_data[feat] for feat in self.feature_names])
            input_array = input_array.reshape(1, -1)
        else:
            input_array = np.array(input_data).reshape(1, -1)
        
        return input_array
    
    def predict(self, input_data):
        """Make prediction on input data"""
        if self.model is None:
            return None, None
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_input)[0]
        probability = self.model.predict_proba(processed_input)[0]
        
        # Interpret results
        result = {
            'prediction': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            },
            'confidence': float(max(probability)) * 100
        }
        
        return result
    
    def predict_batch(self, input_data_list):
        """Make predictions on multiple inputs"""
        results = []
        
        for input_data in input_data_list:
            result = self.predict(input_data)
            results.append(result)
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return sorted_importance
        else:
            return None


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Sample patient data
    patient_data = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    # Make prediction
    result = predictor.predict(patient_data)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Low Risk Probability: {result['probability']['low_risk']:.4f}")
    print(f"High Risk Probability: {result['probability']['high_risk']:.4f}")
```

---

### 4. Web Application

**File: `api/app.py`**

```python
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
sys.path.append('..')
from src.predictor import HeartDiseasePredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = HeartDiseasePredictor()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Make prediction
        result = predictor.predict(data)
        
        if result:
            return jsonify({
                'success': True,
                'result': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Get feature importance"""
    importance = predictor.get_feature_importance()
    
    if importance:
        return jsonify({
            'success': True,
            'importance': importance
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': 'Feature importance not available'
        }), 404

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    }), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🫀 HEART DISEASE PREDICTION API")
    print("="*60)
    print("Server starting on http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

### 5. Main Execution Script

**File: `main.py`**

```python
#!/usr/bin/env python3
"""
Heart Disease Prediction - Main Execution Script
"""

import argparse
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.predictor import HeartDiseasePredictor

def train_pipeline(data_path):
    """Complete training pipeline"""
    print("\n🫀 HEART DISEASE PREDICTION - TRAINING PIPELINE\n")
    
    # Step 1: Data Preprocessing
    print("STEP 1: Data Preprocessing")
    print("-" * 60)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_path)
    
    # Step 2: Model Training
    print("\nSTEP 2: Model Training")
    print("-" * 60)
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 3: Hyperparameter Tuning (optional)
    print("\nSTEP 3: Hyperparameter Tuning")
    print("-" * 60)
    best_model = trainer.hyperparameter_tuning(X_train, y_train)
    
    # Step 4: Save Model
    print("\nSTEP 4: Saving Model")
    print("-" * 60)
    trainer.save_model(best_model)
    
    print("\n✓ Training pipeline completed successfully!\n")

def predict_sample():
    """Make a sample prediction"""
    print("\n🫀 HEART DISEASE PREDICTION - SAMPLE PREDICTION\n")
    
    predictor = HeartDiseasePredictor()
    
    # Sample patient data
    patient_data = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    result = predictor.predict(patient_data)
    
    print("PATIENT DATA:")
    print("-" * 60)
    for key, value in patient_data.items():
        print(f"  {key}: {value}")
    
    print("\nPREDICTION RESULT:")
    print("-" * 60)
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  Low Risk Probability: {result['probability']['low_risk']:.4f}")
    print(f"  High Risk Probability: {result['probability']['high_risk']:.4f}")
    print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Heart Disease Prediction System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        default='predict',
        help='Mode: train or predict'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/heart.csv',
        help='Path to dataset'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pipeline(args.data)
    else:
        predict_sample()

if __name__ == "__main__":
    main()
```

---

## 📊 Usage Examples

### Example 1: Train the Model

```bash
python main.py --mode train --data data/heart.csv
```

### Example 2: Make Predictions

```bash
python main.py --mode predict
```

### Example 3: Start Web Server

```bash
python api/app.py
```

### Example 4: Use in Python Script

```python
from src.predictor import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor()

# Patient data
patient = {
    'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 140,
    'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 155,
    'exang': 0, 'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
}

# Get prediction
result = predictor.predict(patient)
print(f"Risk: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

---

## ⚙️ Configuration

**File: `config.py`**

```python
# Model Configuration
MODEL_CONFIG = {
    'model_path': 'models/heart_disease_model.pkl',
    'scaler_path': 'models/scaler.pkl',
    'random_state': 42,
    'test_size': 0.2
}

# Feature Names
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Feature Descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina',
    'oldpeak': 'ST depression',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Number of major vessels (0-3)',
    'thal': 'Thalassemia (0-3)'
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}
```

---

## 🧪 Testing

**File: `tests/test_model.py`**

```python
import unittest
import sys
sys.path.append('..')
from src.predictor import HeartDiseasePredictor

class TestHeartDiseasePredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = HeartDiseasePredictor()
        self.sample_data = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
            'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
            'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }
    
    def test_model_loaded(self):
        """Test if model is loaded"""
        self.assertIsNotNone(self.predictor.model)
    
    def test_prediction(self):
        """Test prediction functionality"""
        result = self.predictor.predict(self.sample_data)
        self.assertIsNotNone(result)
        self.assertIn('prediction', result)
        self.assertIn('risk_level', result)
    
    def test_prediction_probability(self):
        """Test prediction probabilities"""
        result = self.predictor.predict(self.sample_data)
        prob_sum = sum(result['probability'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
```

### Run Tests

```bash
python -m pytest tests/
```

---

## 🚀 Quick Start Guide

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python main.py --mode train --data data/heart.csv
   ```

3. **Start the API**
   ```bash
   python api/app.py
   ```

4. **Make predictions**
   ```bash
   curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 63, "sex": 1, "cp": 3, ...}'
   ```

---

## 📝 License

MIT License - See LICENSE file for details

---

<div align="center">

**Made with ❤️ for better healthcare**

[⭐ Star this repo](https://github.com/yourusername/heart-disease-prediction) • [🐛 Report Bug](https://github.com/yourusername/heart-disease-prediction/issues) • [✨ Request Feature](https://github.com/yourusername/heart-disease-prediction/issues)

</div>
