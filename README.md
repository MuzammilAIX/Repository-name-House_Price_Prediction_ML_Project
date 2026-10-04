# House Price Prediction

A Machine Learning project that predicts residential house prices using property location, area, bedrooms, bathrooms, and balconies.

The project covers data preprocessing, feature engineering, model training, evaluation, and deployment with an interactive Streamlit dashboard.

## Project Architecture

```text
House_Price_Prediction_ML_Project/
│
├── app/
│   └── app.py
├── data/
│   ├── src/
│   │   └── housing.csv
│   └── Cleaned_data.csv
├── models/
│   └── House_prediction_model.pkl
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── notebooks/
│   └── house_prediction.ipynb
├── images/
├── requirements.txt
└── README.md
```

## Machine Learning Workflow

```text
Data
  ↓
Cleaning & Feature Engineering
  ↓
Preprocessing
  ↓
Train/Test Split
  ↓
Linear Regression
  ↓
Model Evaluation
  ↓
Saved Model
  ↓
Streamlit Dashboard
```

## Model

* Algorithm: Linear Regression
* Problem: Regression
* Target: House Price
* Categorical preprocessing: One-Hot Encoding
* Numerical preprocessing: StandardScaler
* Model pipeline: Scikit-learn Pipeline

### Input Features

```text
location
total_sqft
bath
balcony
bedrooms
```

## Dashboard

The Streamlit application provides:

* House price prediction
* Price and area analysis
* Dataset statistics
* Model information
* Interactive visualizations

## Technologies

Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit

## Run Locally

```bash
git clone <your-repository-url>
cd House_Price_Prediction_ML_Project
pip install -r requirements.txt
python -m streamlit run app/app.py
```

## Project Purpose

This project demonstrates an end-to-end Machine Learning workflow, from raw property data to a deployed prediction application.

## Author

**Enginner Muzammil**
