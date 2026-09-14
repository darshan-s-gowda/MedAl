"""
Data Preprocessing Pipeline for MedAI
Handles patient data validation, normalization, and feature engineering.
"""

import logging
from typing import Dict, Any, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Valid Ranges (clinical reference values)
# ─────────────────────────────────────────────

CLINICAL_RANGES = {
    "age":          (1,   120,   "years"),
    "Glucose":      (40,  500,   "mg/dL"),
    "BloodPressure":(40,  200,   "mmHg"),
    "BMI":          (10,  70,    "kg/m²"),
    "chol":         (80,  600,   "mg/dL"),
    "trestbps":     (60,  250,   "mmHg"),
    "thalach":      (60,  250,   "bpm"),
    "Insulin":      (0,   900,   "μU/mL"),
    "SkinThickness":(0,   100,   "mm"),
    "Pregnancies":  (0,   20,    "count"),
    "oldpeak":      (0,   10,    "mm"),
    "ca":           (0,   4,     "count"),
    "Age":          (1,   120,   "years"),
}


def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate patient data against clinical reference ranges.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []

    for field, (min_val, max_val, unit) in CLINICAL_RANGES.items():
        value = data.get(field) or data.get(field.lower())
        if value is not None:
            try:
                val = float(value)
                if not (min_val <= val <= max_val):
                    warnings.append(
                        f"⚠️ {field}={val} is outside normal clinical range "
                        f"({min_val}–{max_val} {unit})"
                    )
            except (ValueError, TypeError):
                warnings.append(f"⚠️ {field} has invalid non-numeric value: {value}")

    is_valid = len([w for w in warnings if "invalid" in w]) == 0
    return is_valid, warnings


def engineer_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add derived features commonly used in medical ML.
    """
    enriched = dict(data)

    # Pulse pressure
    sbp = data.get("trestbps") or data.get("BloodPressure")
    dbp = data.get("dbp")
    if sbp and dbp:
        enriched["pulse_pressure"] = float(sbp) - float(dbp)

    # BMI category flag
    bmi = data.get("BMI") or data.get("bmi")
    if bmi:
        bmi = float(bmi)
        enriched["is_obese"] = 1 if bmi >= 30 else 0
        enriched["is_overweight"] = 1 if 25 <= bmi < 30 else 0

    # Age group
    age = data.get("age") or data.get("Age")
    if age:
        age = int(float(age))
        enriched["age_group"] = (
            "young" if age < 40
            else "middle" if age < 60
            else "senior"
        )

    # Glucose category
    glucose = data.get("Glucose") or data.get("glucose")
    if glucose:
        glucose = float(glucose)
        enriched["glucose_category"] = (
            "normal" if glucose < 100
            else "pre-diabetic" if glucose < 126
            else "diabetic_range"
        )

    return enriched


def normalize_field_names(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names to handle both camelCase and snake_case inputs.
    """
    aliases = {
        "blood_pressure": "BloodPressure",
        "bloodpressure": "BloodPressure",
        "skin_thickness": "SkinThickness",
        "diabetes_pedigree": "DiabetesPedigreeFunction",
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "bmi": "BMI",
        "insulin": "Insulin",
    }
    normalized = {}
    for key, value in data.items():
        canonical = aliases.get(key.lower(), key)
        normalized[canonical] = value
    return normalized


def preprocess_patient_data(raw_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Full preprocessing pipeline:
    1. Normalize field names
    2. Validate ranges
    3. Engineer features
    
    Returns:
        (processed_data, warnings)
    """
    data = normalize_field_names(raw_data)
    is_valid, warnings = validate_patient_data(data)
    if not is_valid:
        logger.warning(f"Patient data validation errors: {warnings}")
    data = engineer_features(data)
    return data, warnings
