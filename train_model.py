"""
Builds the heart disease model: all preprocessing + Logistic Regression in one Pipeline.

The Streamlit app imports build_model() and trains on heart.csv when it starts.
You can also run this file directly to see accuracy:  python train_model.py
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns where 0 means "missing" -> replaced with the mean of the non-zero training values
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]
NUMERIC = ["Age", "MaxHR", "Oldpeak"]
BINARY = ["FastingBS"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def make_pipeline():
    preprocess = ColumnTransformer([
        ("zero_fix", Pipeline([
            ("impute", SimpleImputer(missing_values=0, strategy="mean")),
            ("scale", StandardScaler()),
        ]), ZERO_AS_MISSING),
        ("num", StandardScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def load_data(csv_path="heart.csv"):
    df = pd.read_csv(csv_path)
    return df.drop(columns="HeartDisease"), df["HeartDisease"]


def build_model(csv_path="heart.csv"):
    """Train on the full dataset and return (model, test_accuracy, test_f1)."""
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model = make_pipeline().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model = make_pipeline().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 score: {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
