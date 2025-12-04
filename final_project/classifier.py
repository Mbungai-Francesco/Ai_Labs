import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. Chargement des données
# ----------------------------
DATASET_PATH = 'kaggle/input/mental_health_dataset.csv'
df = pd.read_csv(DATASET_PATH)

# ----------------------------
# 2. Prétraitement des données
# ----------------------------
# Fusion des genres non-binaires dans les genres majoritaires
gender_mask = df['gender'].isin(['Non-binary', 'Prefer not to say'])
total = gender_mask.sum()
half = total // 2
indices = df[gender_mask].index
df.loc[indices[:half], 'gender'] = 'Male'
df.loc[indices[half:], 'gender'] = 'Female'

# Cible : mental_health_risk
target = 'mental_health_risk'
score = ['stress_level', 'depression_score', 'anxiety_score']
X = df.drop(columns=[target, score[0], score[1], score[2]])
y = df[target]

# Encodage de la cible en entiers
le = LabelEncoder()
y = le.fit_transform(y)  # 0: High, 1: Low, 2: Medium (ordre alphabétique)

# Séparation features numériques et catégorielles
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pipelines de prétraitement
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# ----------------------------
# 3. Définition des modèles
# ----------------------------
models = {
    "RandomForest": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ))
    ]),
    "XGBoost": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42,
            eval_metric='mlogloss'
        ))
    ]),
    "SVM": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])
}

# ----------------------------
# 4. Évaluation par validation croisée
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = {}
for name, model in models.items():
    print(f"\n[+] Entraînement du modèle : {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # utile pour classes déséquilibrées
    
    results[name] = {'Accuracy': acc, 'F1-Score (weighted)': f1}
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------------------
# 5. Analyse visuelle (matrice de confusion)
# ----------------------------
best_model_name = max(results, key=lambda k: results[k]['F1-Score (weighted)'])
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
label_order = ['High', 'Medium','Low']
label_order_indices = [le.transform([label])[0] for label in label_order]
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred_best, labels=label_order_indices)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_order, yticklabels=label_order)
plt.title(f'Confusion Matrix – {best_model_name}')
plt.ylabel('True label')
plt.xlabel('predicted label')
plt.show()

# --- Graphique comparatif des performances ---

model_names = list(results.keys())
accuracies = [results[m]['Accuracy'] for m in model_names]
f1_scores = [results[m]['F1-Score (weighted)'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#4C72B0')
bars2 = ax.bar(x + width/2, f1_scores, width, label='Weighted F1-Score', color='#55A868')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Performance Comparison — Classification on mental_health_risk')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Ajouter les valeurs au-dessus des barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()