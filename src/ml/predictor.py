"""
Disease Predictor - ML Models for Healthcare Analytics
Models:
  1. Diabetes Classifier (Random Forest)
  2. Heart Disease Classifier (Gradient Boosting)
  3. Cardiovascular Risk Score (Linear Regression)
  4. General Symptom-Based Risk (Logistic Regression)
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Synthetic Dataset Generators
# ─────────────────────────────────────────────

def generate_diabetes_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate realistic synthetic diabetes dataset.
    Based on Pima Indians Diabetes Dataset structure.
    Features: Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DiabetesPedigree, Age
    """
    np.random.seed(42)

    data = {
        "Pregnancies": np.random.randint(0, 17, n_samples),
        "Glucose": np.clip(np.random.normal(120, 30, n_samples), 0, 200).astype(int),
        "BloodPressure": np.clip(np.random.normal(70, 12, n_samples), 0, 122).astype(int),
        "SkinThickness": np.clip(np.random.normal(20, 15, n_samples), 0, 99).astype(int),
        "Insulin": np.clip(np.random.exponential(80, n_samples), 0, 846).astype(int),
        "BMI": np.clip(np.random.normal(32, 7, n_samples), 0, 67).round(1),
        "DiabetesPedigreeFunction": np.clip(np.random.exponential(0.47, n_samples), 0.08, 2.42).round(3),
        "Age": np.random.randint(21, 81, n_samples),
    }
    df = pd.DataFrame(data)

    # Simulate outcome based on risk factors (realistic correlation)
    risk_score = (
        0.02 * df["Glucose"]
        + 0.015 * df["BMI"]
        + 0.01 * df["Age"]
        + 0.2 * df["DiabetesPedigreeFunction"]
        + 0.005 * df["Pregnancies"]
        - 2.5
    )
    prob = 1 / (1 + np.exp(-risk_score))
    df["Outcome"] = (np.random.random(n_samples) < prob).astype(int)
    return df


def generate_heart_disease_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic heart disease dataset.
    Based on Cleveland Heart Disease Dataset structure.
    """
    np.random.seed(123)

    data = {
        "age": np.random.randint(29, 77, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),          # chest pain type
        "trestbps": np.clip(np.random.normal(131, 18, n_samples), 94, 200).astype(int),
        "chol": np.clip(np.random.normal(246, 52, n_samples), 126, 564).astype(int),
        "fbs": np.random.randint(0, 2, n_samples),          # fasting blood sugar > 120
        "restecg": np.random.randint(0, 3, n_samples),      # resting ECG
        "thalach": np.clip(np.random.normal(150, 23, n_samples), 71, 202).astype(int),
        "exang": np.random.randint(0, 2, n_samples),        # exercise induced angina
        "oldpeak": np.clip(np.random.exponential(1.04, n_samples), 0, 6.2).round(1),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 4, n_samples),           # vessels colored by flourosopy
        "thal": np.random.choice([0, 1, 2, 3], n_samples),
    }
    df = pd.DataFrame(data)

    risk_score = (
        0.03 * df["age"]
        + 0.02 * df["trestbps"]
        + 0.01 * df["chol"]
        - 0.02 * df["thalach"]
        + 0.5 * df["exang"]
        + 0.3 * df["oldpeak"]
        + 0.2 * df["ca"]
        - 2.0
    )
    prob = 1 / (1 + np.exp(-risk_score))
    df["target"] = (np.random.random(n_samples) < prob).astype(int)
    return df


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────

def train_diabetes_model() -> Pipeline:
    """Train Random Forest classifier for diabetes prediction."""
    logger.info("Training Diabetes Classifier...")
    df = generate_diabetes_dataset()
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X, y = df[features], df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    logger.info(f"Diabetes Model — Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    with open(MODEL_DIR / "diabetes_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


def train_heart_disease_model() -> Pipeline:
    """Train Gradient Boosting classifier for heart disease."""
    logger.info("Training Heart Disease Classifier...")
    df = generate_heart_disease_dataset()
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    logger.info(f"Heart Disease Model — Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    with open(MODEL_DIR / "heart_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


def train_risk_score_model() -> Pipeline:
    """Train Ridge Regression model for cardiovascular risk score (0–100)."""
    logger.info("Training Risk Score Regressor...")
    df = generate_heart_disease_dataset(n_samples=3000)
    features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "exang"]
    X = df[features]

    # Create continuous risk score target
    y = (
        0.4 * df["age"] / 77
        + 0.3 * df["trestbps"] / 200
        + 0.2 * df["chol"] / 564
        - 0.2 * df["thalach"] / 202
        + 0.3 * df["oldpeak"] / 6.2
        + 0.2 * df["ca"] / 4
        + 0.1 * df["exang"]
    ) * 100

    y = np.clip(y, 0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0))
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    # NOTE: `squared=False` was removed from scikit-learn's mean_squared_error
    # (deprecated in 1.4, removed in 1.6+). Compute RMSE manually so this stays
    # compatible across the pinned scikit-learn version and future upgrades.
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"Risk Score Model — RMSE: {rmse:.2f}")

    with open(MODEL_DIR / "risk_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline


# ─────────────────────────────────────────────
# Predictor Class (API-facing)
# ─────────────────────────────────────────────

class DiseasePredictor:
    """
    Loads trained ML models and exposes prediction methods.
    Trains models on first run if model files don't exist.
    """

    DIABETES_FEATURES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    HEART_FEATURES = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    RISK_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "exang"]

    def __init__(self):
        self.diabetes_model = self._load_or_train(
            MODEL_DIR / "diabetes_model.pkl", train_diabetes_model
        )
        self.heart_model = self._load_or_train(
            MODEL_DIR / "heart_model.pkl", train_heart_disease_model
        )
        self.risk_model = self._load_or_train(
            MODEL_DIR / "risk_model.pkl", train_risk_score_model
        )
        logger.info("All ML models loaded.")

    def _load_or_train(self, path: Path, train_fn):
        """Load model from disk or train if not present."""
        if path.exists():
            with open(path, "rb") as f:
                logger.info(f"Loaded model: {path.name}")
                return pickle.load(f)
        else:
            logger.info(f"Model not found, training: {path.name}")
            return train_fn()

    def _extract_features(self, data: Dict, feature_list: List[str]) -> np.ndarray:
        """Extract and order features from request dict."""
        values = []
        for feat in feature_list:
            val = data.get(feat)
            if val is None:
                val = data.get(feat.lower())
            values.append(float(val) if val is not None else np.nan)
        return np.array(values).reshape(1, -1)

    def predict_diabetes(self, data: Dict) -> Dict[str, Any]:
        """Predict diabetes risk from patient vitals."""
        X = self._extract_features(data, self.DIABETES_FEATURES)
        prob = self.diabetes_model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)
        return {
            "condition": "Diabetes",
            "prediction": "High Risk" if pred else "Low Risk",
            "probability": round(float(prob) * 100, 1),
            "confidence": round(float(max(prob, 1 - prob)) * 100, 1)
        }

    def predict_heart_disease(self, data: Dict) -> Dict[str, Any]:
        """Predict heart disease risk from clinical parameters."""
        X = self._extract_features(data, self.HEART_FEATURES)
        prob = self.heart_model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)
        return {
            "condition": "Heart Disease",
            "prediction": "High Risk" if pred else "Low Risk",
            "probability": round(float(prob) * 100, 1),
            "confidence": round(float(max(prob, 1 - prob)) * 100, 1)
        }

    def calculate_risk_score(self, data: Dict) -> Dict[str, Any]:
        """Calculate continuous cardiovascular risk score (0-100)."""
        X = self._extract_features(data, self.RISK_FEATURES)
        score = float(np.clip(self.risk_model.predict(X)[0], 0, 100))

        if score < 20:
            level, color = "Low", "green"
        elif score < 40:
            level, color = "Moderate", "yellow"
        elif score < 60:
            level, color = "High", "orange"
        else:
            level, color = "Very High", "red"

        factors = self._identify_risk_factors(data)

        return {
            "risk_percentage": round(score, 1),
            "risk_level": level,
            "risk_color": color,
            "factors": factors
        }

    def predict_all(self, data: Dict) -> List[Dict]:
        """Run all available prediction models on patient data."""
        results = []

        # Diabetes prediction (if glucose data available)
        if any(data.get(f) is not None for f in ["Glucose", "glucose", "BMI", "bmi"]):
            try:
                results.append(self.predict_diabetes(data))
            except Exception as e:
                logger.warning(f"Diabetes prediction skipped: {e}")

        # Heart disease prediction (if cardiovascular data available)
        if any(data.get(f) is not None for f in ["age", "chol", "trestbps"]):
            try:
                results.append(self.predict_heart_disease(data))
            except Exception as e:
                logger.warning(f"Heart disease prediction skipped: {e}")

        # Risk score
        if any(data.get(f) is not None for f in ["age", "chol"]):
            try:
                risk = self.calculate_risk_score(data)
                results.append({
                    "condition": "Cardiovascular Risk Score",
                    "prediction": risk["risk_level"],
                    "probability": risk["risk_percentage"],
                    "confidence": None
                })
            except Exception as e:
                logger.warning(f"Risk score skipped: {e}")

        if not results:
            raise ValueError("Insufficient patient data provided for predictions.")

        return results

    def _identify_risk_factors(self, data: Dict) -> List[str]:
        """Identify and explain contributing risk factors."""
        factors = []
        mappings = {
            ("age", 55): "Age above 55",
            ("trestbps", 140): "High blood pressure (>140 mmHg)",
            ("chol", 240): "High cholesterol (>240 mg/dL)",
            ("BMI", 30): "Obesity (BMI > 30)",
            ("Glucose", 126): "Elevated fasting glucose",
        }
        for (key, threshold), label in mappings.items():
            val = data.get(key) or data.get(key.lower())
            if val is not None and float(val) > threshold:
                factors.append(label)

        if data.get("exang") == 1 or data.get("exang") == "1":
            factors.append("Exercise-induced angina")
        if data.get("fbs") == 1 or data.get("fbs") == "1":
            factors.append("Fasting blood sugar > 120 mg/dL")

        return factors if factors else ["No significant isolated risk factors identified"]
