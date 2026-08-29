from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

def test_home():
    response=client.get("/")

    assert response.status_code==200

    data = response.json()

    assert data["message"] == "Student Risk Prediction API is running"


def test_predict():

    student_data = {

        "Marital_Status": 1,
        "Application_mode": 17,
        "Application_order": 1,
        "Course": 9238,
        "Daytime_evening_attendance": 1,
        "Previous_qualification": 1,
        "Nacionality": 1,
        "Mothers_qualification": 13,
        "Fathers_qualification": 10,
        "Mothers_occupation": 6,
        "Fathers_occupation": 8,
        "Displaced": 1,
        "Educational_special_needs": 0,
        "Debtor": 0,
        "Tuition_fees_up_to_date": 1,
        "Gender": 1,
        "Scholarship_holder": 0,
        "International": 0,

        "Previous_qualification_grade": 130,
        "Admission_grade": 130,
        "Age_at_enrollment": 20,

        "Curricular_units_1st_sem_credited": 0,
        "Curricular_units_1st_sem_enrolled": 6,
        "Curricular_units_1st_sem_evaluations": 6,
        "Curricular_units_1st_sem_approved": 5,
        "Curricular_units_1st_sem_grade": 13,
        "Curricular_units_1st_sem_without_evaluations": 0,

        "Curricular_units_2nd_sem_credited": 0,
        "Curricular_units_2nd_sem_enrolled": 6,
        "Curricular_units_2nd_sem_evaluations": 6,
        "Curricular_units_2nd_sem_approved": 5,
        "Curricular_units_2nd_sem_grade": 13,
        "Curricular_units_2nd_sem_without_evaluations": 0,

        "Unemployment_rate": 10,
        "Inflation_rate": 1.5,
        "GDP": 2.0
    }

    response = client.post(
        "/predict",
        json=student_data
    )

    assert response.status_code == 200

    data = response.json()

    # Check prediction exists
    assert "prediction" in data

    # Check probabilities exist
    assert "probabilities" in data

    # Check prediction is valid
    assert data["prediction"] in [
        "Dropout",
        "Enrolled",
        "Graduate"
    ]

    # Check all classes exist
    assert "Dropout" in data["probabilities"]
    assert "Enrolled" in data["probabilities"]
    assert "Graduate" in data["probabilities"]

    # Check probabilities are reasonable
    total_probability = sum(
        data["probabilities"].values()
    )

    assert 99 <= total_probability <= 101

def test_invalid_input():

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422

# ==============================
# TEST PROBABILITY VALUES
# ==============================

def test_probability_values():

    student_data = {

        "Marital_Status": 1,
        "Application_mode": 17,
        "Application_order": 1,
        "Course": 9238,
        "Daytime_evening_attendance": 1,
        "Previous_qualification": 1,
        "Nacionality": 1,
        "Mothers_qualification": 13,
        "Fathers_qualification": 10,
        "Mothers_occupation": 6,
        "Fathers_occupation": 8,
        "Displaced": 1,
        "Educational_special_needs": 0,
        "Debtor": 0,
        "Tuition_fees_up_to_date": 1,
        "Gender": 1,
        "Scholarship_holder": 0,
        "International": 0,

        "Previous_qualification_grade": 130,
        "Admission_grade": 130,
        "Age_at_enrollment": 20,

        "Curricular_units_1st_sem_credited": 0,
        "Curricular_units_1st_sem_enrolled": 6,
        "Curricular_units_1st_sem_evaluations": 6,
        "Curricular_units_1st_sem_approved": 5,
        "Curricular_units_1st_sem_grade": 13,
        "Curricular_units_1st_sem_without_evaluations": 0,

        "Curricular_units_2nd_sem_credited": 0,
        "Curricular_units_2nd_sem_enrolled": 6,
        "Curricular_units_2nd_sem_evaluations": 6,
        "Curricular_units_2nd_sem_approved": 5,
        "Curricular_units_2nd_sem_grade": 13,
        "Curricular_units_2nd_sem_without_evaluations": 0,

        "Unemployment_rate": 10,
        "Inflation_rate": 1.5,
        "GDP": 2.0
    }

    response = client.post(
        "/predict",
        json=student_data
    )

    assert response.status_code == 200

    probabilities = response.json()["probabilities"]

    for probability in probabilities.values():

        assert 0 <= probability <= 100