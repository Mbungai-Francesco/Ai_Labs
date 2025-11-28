import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Load dataset
dataset_path = 'kaggle/input/mental_health_dataset.csv'
df = pd.read_csv(dataset_path)

# Display dataset
# print(f"Dataset shape: {df.shape}")
# print(df.head())
# print(df.describe())
# print(df.columns.tolist())

# Data preprocessing
## No missing values detected
# print(df.isnull().sum())

## Check for duplicates
#print(f"Number of duplicate rows: {df.duplicated().sum()}")

## Converte gender Non-Binary/PreferNotToSay to Male/Female
gender_nums = df[df['gender'].isin(['Non-binary', 'Prefer not to say'])].index

total = len(gender_nums)
half = total // 2
remainder = total % 2  

df.loc[gender_nums[:half], 'gender'] = 'Male'
df.loc[gender_nums[half:], 'gender'] = 'Female'

print(df['gender'].value_counts())

# Feature selection
target_cols = ["stress_level", "depression_score", "anxiety_score"]
categorical_cols = ["gender", "employment_status", "work_environment",
                    "mental_health_history", "seeks_treatment"]
numeric_cols = ["age", "sleep_hours", "physical_activity_days",
                "social_support_score", "productivity_score"]

X = df.drop(columns=target_cols + ["mental_health_risk"])
y = df[target_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ]
)

# Create models
## For multi-output regression, we can use RandomForestRegressor wrapped in MultiOutputRegressor
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))


# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring='neg_mean_absolute_error'
)

print("MAE CV mean:", -scores.mean())
print("MAE CV std:", scores.std())

# Split data for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Fit the model
pipeline.fit(X_train, y_train)
# Predict
y_pred = pipeline.predict(X_test)

# Calculate MAE and RMSE for each target
for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    mse = mean_squared_error(y_test[col], y_pred[:, i])
    print(f"{col} - MAE: {mae}, RMSE: {mse}")