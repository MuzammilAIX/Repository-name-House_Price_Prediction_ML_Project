import os
import pickle as pk

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from preprocessing import preprocessing_pipeline


# ==========================================
# 1. File Paths
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'src',
    'housing.csv'
)

CLEANED_DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'Cleaned_data.csv'
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    'models',
    'House_prediction_model.pkl'
)


# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Original Shape:", df.shape)


# ==========================================
# 3. Handle Missing Values
# ==========================================

df['location'] = df['location'].fillna(
    df['location'].mode()[0]
)

df['size'] = df['size'].fillna(
    df['size'].mode()[0]
)

df['bath'] = df['bath'].fillna(
    df['bath'].mean()
)

df['balcony'] = df['balcony'].fillna(
    df['balcony'].median()
)


# ==========================================
# 4. Drop Unnecessary Columns
# ==========================================

df.drop(
    columns=[
        'area_type',
        'availability',
        'society'
    ],
    inplace=True
)


# ==========================================
# 5. Remove Duplicates
# ==========================================

df.drop_duplicates(
    inplace=True
)


# ==========================================
# 6. Clean Location
# ==========================================

df['location'] = df['location'].apply(
    lambda x: x.strip()
)


location_stats = (
    df.groupby('location')['location']
    .agg('count')
    .sort_values(ascending=True)
)


location_less_than_ten_entries = (
    location_stats[
        location_stats <= 10
    ]
)


df['location'] = df['location'].apply(
    lambda x:
    'other'
    if x in location_less_than_ten_entries
    else x
)


# ==========================================
# 7. Convert Size to Bedrooms
# ==========================================

df['bedrooms'] = df['size'].apply(
    lambda x: int(x.split(' ')[0])
)


# ==========================================
# 8. Clean Total Sqft
# ==========================================

def clean_sqft(sqft):

    tokens = str(sqft).split('-')

    if len(tokens) == 2:

        try:
            return (
                float(tokens[0]) +
                float(tokens[1])
            ) / 2

        except:
            return None

    try:
        return float(sqft)

    except:
        return None


df['total_sqft'] = df['total_sqft'].apply(
    clean_sqft
)


# Remove rows where sqft could not be converted

df.dropna(
    subset=['total_sqft'],
    inplace=True
)


# ==========================================
# 9. Feature Engineering
# ==========================================

df['sqft_per_bed'] = (
    df['total_sqft'] /
    df['bedrooms']
)


# ==========================================
# 10. Remove Unrealistic Sqft
# ==========================================

df = df[
    df['sqft_per_bed'] >= 300
]


# ==========================================
# 11. Price Per Sqft
# ==========================================

df['price_per_sqft'] = round(
    df['price'] * 100000 /
    df['total_sqft'],
    2
)


# ==========================================
# 12. Remove Low Price Per Sqft
# ==========================================

df = df[
    df['price_per_sqft'] >= 2000
]


# ==========================================
# 13. Remove Temporary Columns
# ==========================================

df.drop(
    columns=[
        'size',
        'sqft_per_bed',
        'price_per_sqft'
    ],
    inplace=True
)


# ==========================================
# 14. Save Cleaned Dataset
# ==========================================

df.to_csv(
    CLEANED_DATA_PATH,
    index=False
)

print("Cleaned Shape:", df.shape)


# ==========================================
# 15. Features and Target
# ==========================================

X = df.drop(
    columns=['price']
)

y = df['price']


# ==========================================
# 16. Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 17. Create Model Pipeline
# ==========================================

model = make_pipeline(
    preprocessing_pipeline,
    LinearRegression()
)


# ==========================================
# 18. Train Model
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 19. Evaluate Model
# ==========================================

score = model.score(
    X_test,
    y_test
)

print(
    "Model R2 Score:",
    score
)


# ==========================================
# 20. Save Model
# ==========================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

with open(
    MODEL_PATH,
    'wb'
) as file:

    pk.dump(
        model,
        file
    )


print(
    "Model saved successfully!"
)

print(
    "Model path:",
    MODEL_PATH
)