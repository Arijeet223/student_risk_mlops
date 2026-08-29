import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel


# ==============================
# LOAD MODEL
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "student_risk_model.pkl"
)

label_encoder = joblib.load(
    BASE_DIR / "label_encoder.pkl"
)


# ==============================
# CREATE API
# ==============================

app = FastAPI(
    title="Student Risk Prediction API",
    description="API for predicting student academic outcome",
    version="1.0"
)


# ==============================
# INPUT DATA
# ==============================

class StudentData(BaseModel):

    Marital_Status: int
    Application_mode: int
    Application_order: int
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Nacionality: int
    Mothers_qualification: int
    Fathers_qualification: int
    Mothers_occupation: int
    Fathers_occupation: int
    Displaced: int
    Educational_special_needs: int
    Debtor: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    International: int

    Previous_qualification_grade: float
    Admission_grade: float
    Age_at_enrollment: int

    Curricular_units_1st_sem_credited: int
    Curricular_units_1st_sem_enrolled: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: int

    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: int

    Unemployment_rate: float
    Inflation_rate: float
    GDP: float


# ==============================
# HOME ROUTE
# ==============================

@app.get("/")
def home():

    return {
        "message": "Student Risk Prediction API is running"
    }


# ==============================
# PREDICTION
# ==============================

@app.post("/predict")
def predict(student: StudentData):

    data = {

        "Marital Status": student.Marital_Status,
        "Application mode": student.Application_mode,
        "Application order": student.Application_order,
        "Course": student.Course,
        "Daytime/evening attendance": student.Daytime_evening_attendance,
        "Previous qualification": student.Previous_qualification,
        "Nacionality": student.Nacionality,
        "Mother's qualification": student.Mothers_qualification,
        "Father's qualification": student.Fathers_qualification,
        "Mother's occupation": student.Mothers_occupation,
        "Father's occupation": student.Fathers_occupation,
        "Displaced": student.Displaced,
        "Educational special needs": student.Educational_special_needs,
        "Debtor": student.Debtor,
        "Tuition fees up to date": student.Tuition_fees_up_to_date,
        "Gender": student.Gender,
        "Scholarship holder": student.Scholarship_holder,
        "International": student.International,

        "Previous qualification (grade)": student.Previous_qualification_grade,
        "Admission grade": student.Admission_grade,
        "Age at enrollment": student.Age_at_enrollment,

        "Curricular units 1st sem (credited)": student.Curricular_units_1st_sem_credited,
        "Curricular units 1st sem (enrolled)": student.Curricular_units_1st_sem_enrolled,
        "Curricular units 1st sem (evaluations)": student.Curricular_units_1st_sem_evaluations,
        "Curricular units 1st sem (approved)": student.Curricular_units_1st_sem_approved,
        "Curricular units 1st sem (grade)": student.Curricular_units_1st_sem_grade,
        "Curricular units 1st sem (without evaluations)": student.Curricular_units_1st_sem_without_evaluations,

        "Curricular units 2nd sem (credited)": student.Curricular_units_2nd_sem_credited,
        "Curricular units 2nd sem (enrolled)": student.Curricular_units_2nd_sem_enrolled,
        "Curricular units 2nd sem (evaluations)": student.Curricular_units_2nd_sem_evaluations,
        "Curricular units 2nd sem (approved)": student.Curricular_units_2nd_sem_approved,
        "Curricular units 2nd sem (grade)": student.Curricular_units_2nd_sem_grade,
        "Curricular units 2nd sem (without evaluations)": student.Curricular_units_2nd_sem_without_evaluations,

        "Unemployment rate": student.Unemployment_rate,
        "Inflation rate": student.Inflation_rate,
        "GDP": student.GDP
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    prediction_label = label_encoder.inverse_transform(
        prediction
    )[0]

    probabilities = model.predict_proba(df)[0]

    probability_dict = {
        label_encoder.classes_[i]: round(
            float(probabilities[i]) * 100,
            2
        )
        for i in range(len(probabilities))
    }

    return {
        "prediction": prediction_label,
        "probabilities": probability_dict
    }