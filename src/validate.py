import pandas as pd

from preprocess import load_data


required_columns = [
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
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "target"
]


def validate_data(df):

    print("Starting data validation...")

    # 1. Check columns

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("Missing columns:")
        print(missing_columns)

        return False

    print("Column check: PASS")


    # 2. Check missing values

    missing_values = df.isnull().sum().sum()

    if missing_values > 0:

        print(
            "Missing values found:",
            missing_values
        )

        return False

    print("Missing value check: PASS")


    # 3. Check dataset size

    if len(df) < 1000:

        print(
            "Dataset is too small:",
            len(df)
        )

        return False

    print(
        "Dataset size check: PASS",
        len(df),
        "rows"
    )


    # 4. Check target

    expected_targets = {
        "Dropout",
        "Enrolled",
        "Graduate"
    }

    actual_targets = set(
        df["target"].unique()
    )

    if not actual_targets.issubset(expected_targets):

        print(
            "Unexpected target values:",
            actual_targets
        )

        return False

    print("Target check: PASS")


    # 5. Check numeric values

    numeric_columns = [
        "Previous qualification (grade)",
        "Admission grade",
        "Age at enrollment",
        "Unemployment rate",
        "Inflation rate",
        "GDP"
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            print(
                "Invalid data type:",
                column
            )

            return False

    print("Numeric type check: PASS")


    print("\nData validation successful!")

    return True


if __name__ == "__main__":

    df = load_data()

    valid = validate_data(df)

    if not valid:

        raise ValueError(
            "Data validation failed!"
        )