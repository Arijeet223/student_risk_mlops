import pandas as pd

def load_data():
    df=pd.read_csv("data/students_dropout_academic_success.csv")
    return df

def prepare_data(df):
    target="target"

    features = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",

    # Academic performance
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",

    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",

    # Economic indicators
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]
    x = df[features]
    y = df[target]

    return x,y


if __name__=="__main__":
    df=load_data()
    x,y=prepare_data(df)

    print("feature shape",x.shape)
    print("target shape",y.shape)

    print("features")
    print(x.columns.tolist())

    print("\nTarget:")
    print(y.value_counts())

