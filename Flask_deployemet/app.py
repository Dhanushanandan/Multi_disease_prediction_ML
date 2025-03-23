from flask import Flask, render_template, request
import joblib
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load models for each disease
models = {
    "Chronic Kidney Disease": {
        "Logistic Regression": joblib.load("models/ckd_LogisticRegression_model.pkl"),
        "Random Forest": joblib.load("models/ckd_Random_model.pkl"),
        "Decision Tree": joblib.load("models/ckd_tree_model.pkl"),
        "K-Nearest Neighbors": joblib.load("models/ckd_KNN_model.pkl"),
        "SVM": joblib.load("models/ckd_svm_model.pkl")
    },
    "Diabetes": {
        "Logistic Regression": joblib.load("models/dia_LogisticRegression_model.pkl"),
        "Random Forest": joblib.load("models/dia_RandomForest_model.pkl"),
        "Decision Tree": joblib.load("models/dia_tree_model.pkl"),
        "K-Nearest Neighbors": joblib.load("models/dia_KNN_model.pkl"),
        "SVM": joblib.load("models/dia_svm_model.pkl")
    },
    "Stroke": {
        "Logistic Regression": joblib.load("models/str_LogisticRegression_model.pkl"),
        "Random Forest": joblib.load("models/str_Random_model.pkl"),
        "Decision Tree": joblib.load("models/str_tree_model.pkl"),
        "K-Nearest Neighbors": joblib.load("models/str_KNN_model.pkl"),
        "SVM": joblib.load("models/str_svm_model.pkl")
    },
    "Heart Disease": {
        "Logistic Regression": joblib.load("models/heart_LogisticRegression_model.pkl"),
        "Random Forest": joblib.load("models/heart_Random_model.pkl"),
        "Decision Tree": joblib.load("models/heart_tree_model.pkl"),
        "K-Nearest Neighbors": joblib.load("models/heart_KNN_model.pkl"),
        "SVM": joblib.load("models/heart_svm_model.pkl")
    }
}

# Mapping disease names to correct HTML file names
disease_templates = {
    "Chronic Kidney Disease": "ckd.html",
    "Diabetes": "diabetes.html",
    "Stroke": "stroke.html",
    "Heart Disease": "heart.html"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    if request.method == 'POST':
        try:
            # Collect input values
            features = [float(request.form[key]) for key in request.form.keys()]
            input_data = np.array(features).reshape(1, -1)
            
            # Predict using all models
            predictions = []
            for model in models[disease].values():
                prediction = model.predict(input_data)
                predictions.append(prediction[0])
            
            # Majority voting for final prediction
            final_prediction = Counter(predictions).most_common(1)[0][0]
            result = "Disease Detected" if final_prediction == 1 else "No Disease Detected"
            
            return render_template('result.html', disease=disease, result=result)
        except Exception as e:
            return render_template('result.html', disease=disease, error=f"Error: {str(e)}")
    
    # Render the correct HTML file based on disease selection
    return render_template(disease_templates.get(disease, "index.html"))

if __name__ == "__main__":
    app.run(debug=True)