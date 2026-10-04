import os
import pickle as pk
import pandas as pd


# ==========================================
# 1. Project Path
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    'models',
    'House_prediction_model.pkl'
)


# ==========================================
# 2. Load Model
# ==========================================

with open(MODEL_PATH, 'rb') as file:
    model = pk.load(file)


# ==========================================
# 3. Prediction Function
# ==========================================

def predict_price(
    location,
    total_sqft,
    bath,
    balcony,
    bedrooms
):

    input_data = pd.DataFrame(
        [[
            location,
            total_sqft,
            bath,
            balcony,
            bedrooms
        ]],
        columns=[
            'location',
            'total_sqft',
            'bath',
            'balcony',
            'bedrooms'
        ]
    )

    prediction = model.predict(input_data)

    return prediction[0]