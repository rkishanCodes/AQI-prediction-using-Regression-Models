import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Define the base directory based on the current file's location
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

print(f"Base Directory: {base_dir}")
print(f"Models Directory: {models_dir}")

model_paths = {
    "Decision Tree": os.path.join(models_dir, "Decision_tree_model.pkl"),
    "Random Forest": os.path.join(models_dir, "random_forest_model.pkl"),
    "KNN": os.path.join(models_dir, "KNN_model.pkl"),
    "Linear Regression": os.path.join(models_dir, "linear_regression_model.pkl"),
    "Gradient Boost": os.path.join(models_dir, "Gradient_boost_model.pkl"),
}

models = {}
for model_name, model_path in model_paths.items():
    print(f"Loading model '{model_name}' from: {model_path}")
    try:
        with open(model_path, 'rb') as file:
            models[model_name] = pickle.load(file)
        print(f"Loaded model '{model_name}' successfully.")
    except FileNotFoundError:
        print(f"Warning: Model file not found: {model_path}")

print(f"Available models: {list(models.keys())}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_aqi', methods=['POST'])
def predict_aqi():
    model_name = request.form.get('model')
    if not model_name:
        return "Error: Model not specified", 400

    # Extract and validate input data
    try:
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2 = float(request.form['no2'])
        nh3 = float(request.form['nh3'])
        so2 = float(request.form['so2'])
        co = float(request.form['co'])
        ozone = float(request.form['ozone'])
    except (ValueError, KeyError) as e:
        return f"Error: Invalid input data ({e})", 400

    if model_name not in models:
        return f"Error: Model '{model_name}' not found", 400

    model = models[model_name]
    input_data = np.array([[pm25, pm10, no2, nh3, so2, co, ozone]])
    try:
        predicted_aqi = model.predict(input_data)
    except Exception as e:
        return f"Error during prediction: {e}", 500

    return render_template('result.html', predicted_aqi=predicted_aqi[0])

if __name__ == '__main__':
    app.run(debug=True)