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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
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

# ----------------------------
# 8. Visualisation : Scatter plots avec lignes de régression
# ----------------------------

# Sélectionner les features numériques à visualiser
# numeric_features = ['age', 'sleep_hours', 'physical_activity_days', "social_support_score", "productivity_score"]

# # Créer une grille de sous-graphiques
# fig, axes = plt.subplots(3, 5, figsize=(15, 9))
# axes = axes.flatten()

# target_vars = ['stress_level', 'depression_score', 'anxiety_score']

# for i, target in enumerate(target_vars):
#     for j, feature in enumerate(numeric_features):
#         ax = axes[i * 5 + j]
#         sns.scatterplot(data=df, x=feature, y=target, ax=ax, alpha=0.6)
#         sns.regplot(data=df, x=feature, y=target, ax=ax, scatter=False, color='red', line_kws={'linewidth': 2})
#         ax.set_title(f'{target} vs {feature}')
#         ax.set_xlabel(feature)
#         ax.set_ylabel(target)

# plt.tight_layout()
# plt.show()

# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()


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
    ("regressor", RandomForestRegressor(n_estimators=200,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    bootstrap=True))
])
model_gb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    random_state=42)))
])
model_xgb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", MultiOutputRegressor(XGBRegressor(n_estimators=200,
    learning_rate=0.2,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2,
    reg_lambda=2,
    random_state=42)))
])
model_dt = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", MultiOutputRegressor(DecisionTreeRegressor(
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
    )))
])

models = {
    "LinearRegression": model_lr,
    "RandomForest": model_rf,
    "GradientBoosting": model_gb,    
    "XGBoost": model_xgb,
    "DecisionTree": model_dt
}

def cross_validate_multioutput(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        mae_scores.append(mean_absolute_error(y_val, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, pred)))
        r2_scores.append(r2_score(y_val, pred))

    return {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores)
    }

# cv_results = cross_validate_multioutput(model_rf, X, y)
# print("Cross-validation results for RandomForest:")
# for metric, value in cv_results.items():
#     print(f"{metric}: {value}")

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
            "R²": r2_score(y_test, pred),
        }
    return results
results = compare_models(models, X_train, X_test, y_train, y_test)
# print(results)
# for model_name, metrics in results.items():
#     print(f"Model: {model_name}")
#     for metric_name, value in metrics.items():
#         print(f"  {metric_name}: {value}")

# ----------------------------
# 7. Visualisation des résultats (Bar plot comparatif)
# ----------------------------

# Extraire les noms des modèles et les métriques
model_names = list(results.keys())
mae_values = [results[model]['MAE'] for model in model_names]
rmse_values = [results[model]['RMSE'] for model in model_names]
r2_values = [results[model]['R²'] for model in model_names]

x = np.arange(len(model_names))  # positions des barres
width = 0.2  # largeur des barres

fig, ax = plt.subplots(figsize=(10, 6))

# Créer les barres
rects1 = ax.bar(x - width, r2_values, width, label='R²', color='#4C72B0')  # bleu foncé
rects2 = ax.bar(x, rmse_values, width, label='RMSE', color='#55A868')        # vert
rects3 = ax.bar(x + width, mae_values, width, label='MAE', color='#9467BD')   # violet

# Ajouter les titres et étiquettes
ax.set_title('Comparing Regression Model Performance')
ax.set_xlabel('Regression Models')
ax.set_ylabel('Metric Value')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Ajouter les valeurs au-dessus des barres (optionnel, mais utile pour la présentation)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()

# Optionnel : sauvegarder l'image
# fig.savefig('regression_model_comparison.png', dpi=300, bbox_inches='tight')
