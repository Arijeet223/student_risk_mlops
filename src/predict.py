import joblib
import pandas as pd

model=joblib.load("student_risk_model.pkl")
label_encoder=joblib.load("label_encoder.pkl")

student = pd.DataFrame([{

    "Marital Status": 1,
    "Application mode": 17,
    "Application order": 1,
    "Course": 9238,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 13,
    "Father's qualification": 10,
    "Mother's occupation": 6,
    "Father's occupation": 8,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "International": 0,

    "Previous qualification (grade)": 130,
    "Admission grade": 130,
    "Age at enrollment": 20,

    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 13,
    "Curricular units 1st sem (without evaluations)": 0,

    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 6,
    "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 13,
    "Curricular units 2nd sem (without evaluations)": 0,

    "Unemployment rate": 10,
    "Inflation rate": 1.5,
    "GDP": 2.0

}])

prediction=model.predict(student)
prediction_label=label_encoder.inverse_transform(prediction)
probabilities = model.predict_proba(student)[0]

print("\n==============================")
print("STUDENT RISK PREDICTION")
print("==============================")

print("Prediction:", prediction_label[0])
print("\nPrediction Probabilities:")

for label, probability in zip(
    label_encoder.classes_,
    probabilities
):
    print(f"{label}: {probability:.2%}")