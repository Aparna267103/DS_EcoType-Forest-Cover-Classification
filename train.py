# ========================= train.py =========================

# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
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


# ================== FEATURE ENGINEERING ==================
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
X = df.drop('Cover_Type', axis=1)
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


# ================== HANDLE CLASS IMBALANCE ==================
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)


# ================== MODELS ==================
models = {

    "Random Forest": RandomForestClassifier(random_state=42),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Logistic Regression": LogisticRegression(max_iter=500),

    "KNN": KNeighborsClassifier(),

    "XGBoost": XGBClassifier(
        eval_metric='mlogloss',
        random_state=42
    )
}


# ================== TRAIN & EVALUATE ==================
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
best_model_name = max(results, key=results.get)

print("\n==============================")
print("BEST MODEL:", best_model_name)
print("==============================")

best_model = models[best_model_name]


# ================== HYPERPARAMETER TUNING ==================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)


# ================== FEATURE IMPORTANCE ==================
importances = best_model.feature_importances_

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


# ================== SAVE MODEL ==================
joblib.dump(best_model, "forest_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel & Encoder saved successfully ✅")