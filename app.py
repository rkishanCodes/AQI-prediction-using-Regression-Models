import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Define the base directory for models
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

model_paths = {
    "Decision Tree": os.path.join(models_dir, "Decision_tree_model.pkl"),
    "Random Forest": os.path.join(models_dir, "random_forest_model.pkl"),
    "KNN": os.path.join(models_dir, "KNN_model.pkl"),
    "Linear Regression": os.path.join(models_dir, "linear_regression_model.pkl"),
    "Gradient Boost": os.path.join(models_dir, "Gradient_boost_model.pkl"),
}

models = {}
for model_name, model_path in model_paths.items():
    with open(model_path, 'rb') as file:
        models[model_name] = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_aqi', methods=['POST'])
def predict_aqi():
    model_name = request.form['model']
    pm25 = float(request.form['pm25'])
    pm10 = float(request.form['pm10'])  
    no2 = float(request.form['no2'])
    nh3 = float(request.form['nh3'])
    so2 = float(request.form['so2'])
    co = float(request.form['co'])
    ozone = float(request.form['ozone'])

    model = models[model_name]
    input_data = np.array([[pm25, pm10, no2, nh3, so2, co, ozone]])
    predicted_aqi = model.predict(input_data)

    return render_template('result.html', predicted_aqi=predicted_aqi[0])

if __name__ == '__main__':
    app.run(debug=True)