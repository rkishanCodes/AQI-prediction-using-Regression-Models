# Air Quality Index (AQI) Prediction Project

## Overview

This project focuses on predicting Air Quality Index (AQI) values using various machine learning regression models. The goal is to provide accurate AQI forecasts to support informed decision-making and public health initiatives.

## Dataset

- **Entries**: 9783 data entries with 8 key air quality parameters:
  - **AQI Value** (target variable)
  - **PM2.5, PM10, NO2, NH3, SO2, CO, OZONE** (independent variables)

## Methodology
1. **Data Preprocessing**:
   - Handled missing values
   - Removed irrelevant columns
   - Normalized data

2. **Exploratory Data Analysis**:
   - Correlation analysis
   - Feature importance evaluation
   - Data visualization (scatter plots, heatmaps)

3. **Model Implementation**:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Support Vector Regression (SVR)
   - K-Nearest Neighbors (KNN) Regressor
   - XGBoost Regression

4. **Model Evaluation**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R-squared (R²)

## Results

**Top performing models**:
1. **Random Forest Regressor** (R²: 0.9876)
2. **XGBoost Regression** (R²: 0.9873)

## Technologies Used

- Python
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for data visualization
- Flask for web application development

## Web Application

A Flask-based web application is included for easy AQI prediction:
- User interface for input of air quality parameters
- Model selection option
- Display of predicted AQI value and corresponding air quality condition

## Future Work

- Incorporate real-time data through web scraping
- Extend the analysis to compare AQI predictions across different regions
- Explore additional machine learning algorithms and ensemble methods

## How to Run

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Flask application: `python app.py`
4. Access the web interface at `http://localhost:5000`
5. 
