# ========================= train.py =========================

# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ================== LOAD DATA ==================
df = pd.read_csv("covertype.csv")


# ================== BASIC INFO ==================
print(df.shape)
print(df.info())
print(df.describe())


# ================== CLEANING ==================

# Missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)


# ================== OUTLIER HANDLING (IQR METHOD) ==================
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    if col != "Cover_Type":

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]


# ================== SKEWNESS TREATMENT ==================
skew_cols = [
    'Elevation',
    'Horizontal_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Horizontal_Distance_To_Fire_Points'
]

for col in skew_cols:
    df[col] = np.log1p(df[col])


# ================== FEATURE ENGINEERING ================== Existing columns
df['Hydrology_Distance'] = (
    df['Horizontal_Distance_To_Hydrology'] +
    df['Vertical_Distance_To_Hydrology']
)

df['Hillshade_mean'] = (
    df['Hillshade_9am'] +
    df['Hillshade_Noon'] +
    df['Hillshade_3pm']
) / 3


# ================== SPLIT X, y ==================
X = df.drop('Cover_Type', axis=1)       # to create new meaningful feature
y = df['Cover_Type']


# ================== LABEL ENCODING ==================
le = LabelEncoder()     
y = le.fit_transform(y)


# ================== TRAIN TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================== FEATURE SCALING ==================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================== HANDLE CLASS IMBALANCE ==================
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# ================== MODELS ==================
models = {

    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "KNN": KNeighborsClassifier(),

    "XGBoost": XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
}


# ================== TRAIN & EVALUATE ==================
print("\nStarting model training on the selected interpreter...")
results = {}

for name, model in models.items():

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"Accuracy: {accuracy}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# ================== BEST MODEL ==================
print("\n==============================")
print("BEST MODEL: Random Forest")
print("==============================")

# ================== HYPERPARAMETER TUNING ==================
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_pipeline = grid.best_estimator_

# extract the final estimator from the pipeline for feature importance
if hasattr(best_pipeline, 'named_steps') and 'model' in best_pipeline.named_steps:
    final_model = best_pipeline.named_steps['model']
else:
    # fallback: assume last step is the estimator
    try:
        final_model = best_pipeline.steps[-1][1]
    except Exception:
        final_model = best_pipeline

# ================== FEATURE IMPORTANCE ==================
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    feature_df = feature_df.sort_values(
        by='Importance',
        ascending=False
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_df
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
else:
    print("Final estimator does not expose feature_importances_. Skipping plot.")

# ================== SAVE MODEL ==================
joblib.dump(scaler, "scaler.pkl")
joblib.dump(best_pipeline, "forest_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel & Encoder saved successfully ✅")
