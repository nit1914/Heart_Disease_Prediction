import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

model = None
le_dict = {}
scaler = None
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Medical normal ranges for abnormality detection
ABNORMALITY_THRESHOLDS = {
    'age': {'min': 1, 'max': 120},
    'trestbps': {'min': 80, 'max': 140, 'critical_high': 180},  # Resting Blood Pressure
    'chol': {'min': 125, 'max': 200, 'critical_high': 240},  # Cholesterol
    'thalach': {'min': 60, 'max': 100, 'critical_low': 40},  # Max Heart Rate
    'oldpeak': {'min': 0, 'max': 6.5},  # ST Depression
}

def detect_abnormalities(data):
    """Detect abnormal medical values"""
    abnormalities = []
    
    # Check age
    if data['age'] < 1 or data['age'] > 120:
        abnormalities.append(f"Age {data['age']} is out of normal range")
    
    # Check Resting Blood Pressure
    if data['trestbps'] < 80 or data['trestbps'] > 140:
        if data['trestbps'] >= 180:
            abnormalities.append(f"Critical High Blood Pressure: {data['trestbps']} mmHg")
        else:
            abnormalities.append(f"Abnormal Resting BP: {data['trestbps']} mmHg (Normal: 80-140)")
    
    # Check Cholesterol
    if data['chol'] < 125 or data['chol'] > 200:
        if data['chol'] >= 240:
            abnormalities.append(f"Critical High Cholesterol: {data['chol']} mg/dl")
        else:
            abnormalities.append(f"Abnormal Cholesterol: {data['chol']} mg/dl (Ideal: 125-200)")
    
    # Check Max Heart Rate
    if data['thalach'] < 60 or data['thalach'] > 100:
        if data['thalach'] <= 40:
            abnormalities.append(f"Critical Low Heart Rate: {data['thalach']} bpm")
        else:
            abnormalities.append(f"Abnormal Max Heart Rate: {data['thalach']} bpm (Normal: 60-100)")
    
    # Check ST Depression
    if data['oldpeak'] > 6.5:
        abnormalities.append(f"Abnormal ST Depression: {data['oldpeak']} mm")
    
    # Check combinations
    if data['fbs'] == 1:  # Fasting Blood Sugar > 120
        abnormalities.append("High Fasting Blood Sugar (>120 mg/dl)")
    
    if data['exang'] == 1:  # Exercise Induced Angina
        abnormalities.append("Exercise-Induced Angina detected")
    
    if data['cp'] == 0:  # Typical Angina
        abnormalities.append("Typical Angina symptoms")
    
    return abnormalities

def train_model():
    """Train the Naive Bayes model"""
    global model, le_dict
    
    # Read the file
    df = pd.read_csv('heart.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Label Encoding
    le = LabelEncoder()
    columns_to_encode = ['target', 'sex', 'cp', 'restecg', 'slope', 'thal', 'ca']
    
    for col in columns_to_encode:
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    print(f"\nEncoded columns: {columns_to_encode}")
    
    # Input - K (Feature Set)
    X = df.drop(columns=['target'])
    # Target - Ans (Label)
    y = df['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    # Display the shape of the datasets
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Model Training - Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n===== Model Performance =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Class distribution - 0 (Healthy): {(y_test==0).sum()}, 1 (Disease): {(y_test==1).sum()}")
    print("=============================\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle prediction requests"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # Extract data from request
        age = float(data['age'])
        sex = int(data['sex'])
        cp = int(data['cp'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        fbs = int(data['fbs'])
        restecg = int(data['restecg'])
        thalach = float(data['thalach'])
        exang = int(data['exang'])
        oldpeak = float(data['oldpeak'])
        slope = int(data['slope'])
        ca = int(data['ca'])
        thal = int(data['thal'])
        
        # Detect abnormalities
        abnormalities = detect_abnormalities({
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'fbs': fbs,
            'exang': exang,
            'cp': cp
        })
        
        # Create feature array in correct order
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Get probability scores
        healthy_prob = prediction_proba[0] * 100
        disease_prob = prediction_proba[1] * 100
        
        # If abnormalities detected, recommend doctor consultation
        if abnormalities:
            recommendation = "Please consult a Doctor immediately for proper evaluation."
            severity = "HIGH" if disease_prob > 50 or len(abnormalities) > 2 else "MODERATE"
        else:
            recommendation = "Your health metrics look normal. Maintain regular checkups."
            severity = "LOW"
        
        if prediction == 1 or disease_prob > 50:
            result = "⚠️ Patient is having Heart Disease Risk. Please consult the Doctor!"
            status = "HEART_DISEASE"
        else:
            result = "✓ Patient is Healthy."
            status = "HEALTHY"
        
        # If abnormalities exist, override to recommend doctor
        if abnormalities:
            status = "ABNORMAL"
            result = "⚠️ Abnormal Medical Values Detected. Please consult the Doctor!"
            recommendation = "These abnormal values require immediate medical attention."
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'status': status,
            'message': result,
            'healthy_probability': round(healthy_prob, 2),
            'disease_probability': round(disease_prob, 2),
            'abnormalities': abnormalities,
            'recommendation': recommendation,
            'severity': severity
        })
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f"Error in prediction: {str(e)}"
        }), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Gaussian Naive Bayes',
        'features': feature_names,
        'total_features': len(feature_names)
    })

if __name__ == '__main__':
    print("Training model...")
    train_model()
    print("Model trained successfully!")
    print("Flask server running at http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)
