"""
Unit Tests for MedAI ML Predictor
Run: pytest tests/ -v
"""

import pytest
import numpy as np
from src.ml.predictor import (
    DiseasePredictor,
    generate_diabetes_dataset,
    generate_heart_disease_dataset,
    train_diabetes_model,
    train_heart_disease_model,
    train_risk_score_model,
)


class TestDataGenerators:
    """Tests for synthetic dataset generators."""

    def test_diabetes_dataset_shape(self):
        df = generate_diabetes_dataset(n_samples=100)
        assert len(df) == 100
        assert "Outcome" in df.columns
        assert "Glucose" in df.columns
        assert "BMI" in df.columns

    def test_diabetes_dataset_ranges(self):
        df = generate_diabetes_dataset(n_samples=500)
        assert df["Glucose"].between(0, 200).all()
        assert df["BMI"].between(0, 67).all()
        assert df["Outcome"].isin([0, 1]).all()

    def test_heart_dataset_shape(self):
        df = generate_heart_disease_dataset(n_samples=100)
        assert len(df) == 100
        assert "target" in df.columns
        assert "age" in df.columns

    def test_heart_dataset_binary_target(self):
        df = generate_heart_disease_dataset(n_samples=200)
        assert df["target"].isin([0, 1]).all()


class TestModelTraining:
    """Tests for model training pipelines."""

    def test_diabetes_model_trains(self):
        model = train_diabetes_model()
        assert model is not None
        X_test = np.array([[2, 120, 70, 25, 80, 28.5, 0.5, 35]])
        pred = model.predict(X_test)
        assert pred[0] in [0, 1]

    def test_heart_model_trains(self):
        model = train_heart_disease_model()
        assert model is not None
        X_test = np.array([[55, 1, 1, 130, 240, 0, 1, 150, 0, 1.5, 1, 1, 2]])
        pred = model.predict(X_test)
        assert pred[0] in [0, 1]

    def test_risk_model_trains(self):
        model = train_risk_score_model()
        assert model is not None
        X_test = np.array([[55, 140, 240, 150, 1.5, 1, 0]])
        pred = model.predict(X_test)
        assert 0 <= pred[0] <= 100


class TestDiseasePredictor:
    """Tests for the DiseasePredictor API class."""

    @staticmethod
    @pytest.fixture(scope="class")
    def predictor():
        return DiseasePredictor()

    def test_predictor_initializes(self, predictor):
        assert predictor.diabetes_model is not None
        assert predictor.heart_model is not None
        assert predictor.risk_model is not None

    def test_predict_diabetes_high_risk(self, predictor):
        data = {
            "Glucose": 180, "BMI": 38.5, "Age": 52,
            "Pregnancies": 6, "BloodPressure": 90,
            "SkinThickness": 35, "Insulin": 200,
            "DiabetesPedigreeFunction": 0.8
        }
        result = predictor.predict_diabetes(data)
        assert "prediction" in result
        assert "probability" in result
        assert result["probability"] >= 0
        assert result["probability"] <= 100

    def test_predict_diabetes_low_risk(self, predictor):
        data = {
            "Glucose": 85, "BMI": 22.0, "Age": 28,
            "Pregnancies": 0, "BloodPressure": 70,
            "SkinThickness": 15, "Insulin": 60,
            "DiabetesPedigreeFunction": 0.2
        }
        result = predictor.predict_diabetes(data)
        assert result["prediction"] in ["Low Risk", "High Risk"]

    def test_predict_heart_disease(self, predictor):
        data = {
            "age": 60, "sex": 1, "cp": 2, "trestbps": 145,
            "chol": 280, "fbs": 1, "restecg": 1, "thalach": 140,
            "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 3
        }
        result = predictor.predict_heart_disease(data)
        assert "prediction" in result
        assert "probability" in result

    def test_risk_score_returns_level(self, predictor):
        data = {
            "age": 50, "trestbps": 135, "chol": 230,
            "thalach": 155, "oldpeak": 1.0, "ca": 1, "exang": 0
        }
        result = predictor.calculate_risk_score(data)
        assert "risk_percentage" in result
        assert "risk_level" in result
        assert result["risk_level"] in ["Low", "Moderate", "High", "Very High"]
        assert 0 <= result["risk_percentage"] <= 100

    def test_predict_all_with_full_data(self, predictor):
        data = {
            "age": 55, "sex": 1, "Glucose": 150, "BMI": 32.0,
            "BloodPressure": 85, "SkinThickness": 28, "Insulin": 150,
            "DiabetesPedigreeFunction": 0.65, "Pregnancies": 3,
            "cp": 1, "trestbps": 140, "chol": 260, "fbs": 1,
            "restecg": 1, "thalach": 145, "exang": 0,
            "oldpeak": 1.8, "slope": 1, "ca": 1, "thal": 2
        }
        results = predictor.predict_all(data)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert "condition" in r
            assert "prediction" in r

    def test_predict_all_raises_on_empty(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_all({})

    def test_risk_factors_identified(self, predictor):
        data = {
            "age": 65, "trestbps": 150, "chol": 280,
            "thalach": 130, "oldpeak": 2.0, "ca": 2,
            "exang": 1, "fbs": 1
        }
        result = predictor.calculate_risk_score(data)
        factors = result["factors"]
        assert isinstance(factors, list)
        assert len(factors) > 0


class TestPreprocessing:
    """Tests for preprocessing pipeline."""

    def test_validate_normal_data(self):
        from src.pipelines.preprocessing import validate_patient_data
        data = {"age": 45, "Glucose": 110, "BMI": 27.5}
        is_valid, warnings = validate_patient_data(data)
        assert is_valid is True
        assert len(warnings) == 0

    def test_validate_out_of_range(self):
        from src.pipelines.preprocessing import validate_patient_data
        data = {"age": 200, "Glucose": 700}  # Both out of range
        is_valid, warnings = validate_patient_data(data)
        assert len(warnings) >= 2

    def test_feature_engineering_bmi(self):
        from src.pipelines.preprocessing import engineer_features
        data = {"BMI": 35.0}
        result = engineer_features(data)
        assert result["is_obese"] == 1

    def test_normalize_field_names(self):
        from src.pipelines.preprocessing import normalize_field_names
        data = {"blood_pressure": 120, "bmi": 25.0}
        result = normalize_field_names(data)
        assert "BloodPressure" in result
        assert "BMI" in result
