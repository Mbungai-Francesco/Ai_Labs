# ----------------------------
# Imports
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, f1_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Pour UMAP
import umap

# Autres
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. Chargement et prétraitement de base
# ----------------------------
DATASET_PATH = 'kaggle/input/mental_health_dataset.csv'
df = pd.read_csv(DATASET_PATH)

# Fusion des genres minoritaires
gender_mask = df['gender'].isin(['Non-binary', 'Prefer not to say'])
indices = df[gender_mask].index
half = len(indices) // 2
df.loc[indices[:half], 'gender'] = 'Male'
df.loc[indices[half:], 'gender'] = 'Female'

# ----------------------------
# 2. Discrétisation des scores → classification
# ----------------------------
def discretize_stress(score):
    if score <= 4:
        return "Low"
    elif score <= 7:
        return "Medium"
    else:
        return "High"

def discretize_depression(score):
    if score <= 10:
        return "Low"
    elif score <= 20:
        return "Medium"
    else:
        return "High"

def discretize_anxiety(score):
    if score <= 10:
        return "Low"
    elif score <= 14:
        return "Medium"
    else:
        return "High"

# Nouvelles cibles catégorielles
df['stress_cat'] = df['stress_level'].apply(discretize_stress)
df['depression_cat'] = df['depression_score'].apply(discretize_depression)
df['anxiety_cat'] = df['anxiety_score'].apply(discretize_anxiety)

# Features et cibles
target_cols = ['stress_cat', 'depression_cat', 'anxiety_cat']
X = df.drop(columns=[
    'stress_level', 'depression_score', 'anxiety_score',
    'mental_health_risk',  
] + target_cols)
y = df[target_cols]

# ----------------------------
# 3. Prétraitement des features
# ----------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Transformer X (pour UMAP et modèles)
X_processed = preprocessor.fit_transform(X)

# Encoder les cibles
y_encoder = OrdinalEncoder()
y_encoded = y_encoder.fit_transform(y)

# ----------------------------
# 4. Division des données
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded[:, 0]  # stratify sur stress
)

# ----------------------------
# 5. Modèles multi-output
# ----------------------------
models = {
    "RandomForest": MultiOutputClassifier(RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )),
    "XGBoost": MultiOutputClassifier(XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )),
    "LogisticRegression": MultiOutputClassifier(LogisticRegression(
        max_iter=1000, random_state=42
    )),
    "DecisionTree": MultiOutputClassifier(RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=3, 
        max_depth=5, 
        random_state=42
    )),
}

print("=== Évaluation des modèles multi-output ===")
results_multi = {}
for name, model in models.items():
    print(f"\n[+] {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ✅ Calcul manuel du Hamming Loss pour multiclass-multioutput
    # y_test et y_pred sont de forme (n_samples, n_outputs)
    n_samples, n_outputs = y_test.shape
    hamming = np.mean(y_test != y_pred)  # proportion totale de prédictions incorrectes

    # Exact match accuracy (tous les labels doivent être corrects)
    exact_acc = np.mean(np.all(y_test == y_pred, axis=1))

    results_multi[name] = {"Hamming Loss": hamming, "Exact Match Accuracy": exact_acc}
    print(f"  Hamming Loss (manuelle) : {hamming:.4f}")
    print(f"  Exact Match Accuracy    : {exact_acc:.4f}")

    # Rapport par cible
    target_names = y_encoder.categories_
    for i, target_name in enumerate(['Stress', 'Depression', 'Anxiety']):
        print(f"\n  → {target_name} Report:")
        print(classification_report(y_test[:, i], y_pred[:, i],
                                    target_names=target_names[i],
                                    zero_division=0))
# ----------------------------
# 6. Visualisation UMAP (sur subset pour rapidité)
# ----------------------------
print("\n=== Génération de la visualisation UMAP ===")
subset_size = min(3000, X_processed.shape[0])
indices_subset = np.random.choice(X_processed.shape[0], subset_size, replace=False)

X_umap_input = X_processed[indices_subset]
y_umap = y_encoded[indices_subset]  # shape: (N, 3)

# Réduction UMAP
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = reducer.fit_transform(X_umap_input)

# Visualisation pour chaque cible
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
target_labels = ['Stress', 'Depression', 'Anxiety']
for i in range(3):
    scatter = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], 
                              c=y_umap[:, i], cmap='viridis', s=10, alpha=0.7)
    axes[i].set_title(f'UMAP — Colorred by {target_labels[i]}')
    plt.colorbar(scatter, ax=axes[i])
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Visualisation des performances des modèles (Approche 2)
# ----------------------------

# Extraire les métriques
model_names = list(results_multi.keys())
hamming_losses = [results_multi[model]["Hamming Loss"] for model in model_names]
exact_accuracies = [results_multi[model]["Exact Match Accuracy"] for model in model_names]

# Configurer le graphique
x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Barres
bars1 = ax.bar(x - width/2, hamming_losses, width, label='Hamming Loss', color='#E74C3C')      # rouge
bars2 = ax.bar(x + width/2, exact_accuracies, width, label='Exact Match Accuracy', color='#2ECC71')  # vert

# Titres et labels
ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Models Performance — Multi-Output Classification')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Ajouter les valeurs au-dessus des barres
def autolabel(bars, offset=0):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

autolabel(bars1, offset=0.01)
autolabel(bars2, offset=0.01)

fig.tight_layout()
plt.show()

# ----------------------------
# 8. Visualisation des F1-scores par cible et par modèle
# ----------------------------

f1_scores_by_model = {name: [] for name in model_names}
target_names = ['Stress', 'Depression', 'Anxiety']

for name in model_names:
    model = models[name]
    y_pred = model.predict(X_test)
    f1s = []
    for i in range(3):  # 3 cibles
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='weighted')
        f1s.append(f1)
    f1_scores_by_model[name] = f1s

# Créer le graphique
x = np.arange(len(target_names))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

for i, model_name in enumerate(model_names):
    ax.bar(x + i*width, f1_scores_by_model[model_name], width, label=model_name)

ax.set_xlabel('Targets')
ax.set_ylabel('F1-score (weighted)')
ax.set_title('F1-score per target and model')
ax.set_xticks(x + width * (len(model_names)-1) / 2)
ax.set_xticklabels(target_names)
ax.set_ylim(0, 1)
ax.legend(title="Models")
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Ajouter les valeurs au-dessus des barres (optionnel)
for i, model_name in enumerate(model_names):
    for j, f1 in enumerate(f1_scores_by_model[model_name]):
        ax.text(x[j] + i*width, f1 + 0.01, f'{f1:.2f}', 
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()