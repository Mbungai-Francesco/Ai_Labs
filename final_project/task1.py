import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Load dataset
DATASET_PATH = 'kaggle/input/mental_health_dataset.csv'
df = pd.read_csv(DATASET_PATH)
#print(f"Dataset shape: {df.shape}")
# print(df.head())
# print(df.describe())
# print(df.columns.tolist())
# print(df.info())

# Data preprocessing

## No missing values detected
# print(df.isnull().sum())

## Check for duplicates
#print(f"Number of duplicate rows: {df.duplicated().sum()}")

## Convert gender Non-Binary/PreferNotToSay to Male/Female
gender_nums = df[df['gender'].isin(['Non-binary', 'Prefer not to say'])].index

total = len(gender_nums)
half = total // 2
remainder = total % 2  

df.loc[gender_nums[:half], 'gender'] = 'Male'
df.loc[gender_nums[half:], 'gender'] = 'Female'
#print(df['gender'].value_counts())

## Feature selection
target_cols = ["stress_level", "depression_score", "anxiety_score"]

X = df.drop(columns=target_cols + ["mental_health_risk"])
y = df[target_cols]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
# print("Num cols:", numeric_cols)
# print("Cat cols:", categorical_cols)

## Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ]
)

# Create models
model_lr = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])
model_rf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, random_state=42))
])
model_gb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", MultiOutputRegressor(GradientBoostingRegressor()))
])
model_xgb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", MultiOutputRegressor(XGBRegressor()))
])

models = {
    "LinearRegression": model_lr,
    "RandomForest": model_rf,
    "GradientBoosting": model_gb,    
    "XGBoost": model_xgb
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
# model_lr.fit(X_train, y_train)
# y_pred = model_lr.predict(X_test)

# Evaluate the model
def evaluate(y_test, y_pred):
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R²  :", r2_score(y_test, y_pred))

# evaluate(y_test, y_pred)

# Compare models
def compare_models(models,X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        }
    return results
results = compare_models(models, X_train, X_test, y_train, y_test)
print(results)
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")

