import joblib
import yaml
import mlflow
from validate import validate_data
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_data, prepare_data

with open("config/config.yaml","r") as file:
    config=yaml.safe_load(file)

df = load_data()
if not validate_data(df):
    raise ValueError(
        "Data validation failed. Training stopped."
    )

X, y = prepare_data(df)

label_encoder = LabelEncoder()
y_xgb = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_xgb,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"],
    stratify=y_xgb
)


categorical_features = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]


numerical_features = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",

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
    "GDP"
]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

models = {

    "logistic_regression": LogisticRegression(
        max_iter=config["model"]["max_iter"],
        class_weight="balanced"
    ),
    

    "random_forest": RandomizedSearchCV(
    RandomForestClassifier(
        random_state=config["training"]["random_state"],
        class_weight="balanced"
    ),

    param_distributions={
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    },

    n_iter=15,
    scoring="f1_macro",
    cv=3,
    random_state=config["training"]["random_state"],
    n_jobs=-1
),
    "xgboost": RandomizedSearchCV(
    XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=config["training"]["random_state"],
        n_jobs=-1
    ),

    param_distributions={
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5]
    },

    n_iter=20,
    scoring="f1_macro",
    cv=3,
    random_state=config["training"]["random_state"],
    n_jobs=-1
)

}
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("student-dropout-prediction")

best_model = None
best_model_name = None
best_f1 = 0

min_macro_f1 = config["validation"]["min_macro_f1"]

for model_name, classifier in models.items():

    print("\nTraining:", model_name)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "test_size",
            config["training"]["test_size"]
        )

        mlflow.log_param(
            "random_state",
            config["training"]["random_state"]
        )
        if model_name in ["logistic_regression", "random_forest"]:
            mlflow.log_param(
                "class_weight",
                "balanced"
        )

        if model_name == "logistic_regression":
            mlflow.log_param(
                "max_iter",
                config["model"]["max_iter"]
            )


        model.fit(X_train, y_train)
        if model_name in ["random_forest", "xgboost"]:

            search = model.named_steps["classifier"]

            best_params = search.best_params_

            print("Best parameters:")
            print(best_params)

            mlflow.log_params({
                f"{model_name}_{key}": value
                for key, value in best_params.items()
            })

            cv_score = search.best_score_

            print("Best CV Macro F1:", cv_score)

            mlflow.log_metric(
                "cv_macro_f1",
                cv_score
            )
            

        y_pred = model.predict(X_test)


        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        macro_f1 = report["macro avg"]["f1-score"]

        mlflow.log_metric(
        "dropout_f1",
        report["Dropout"]["f1-score"]
        )

        mlflow.log_metric(
        "enrolled_f1",
        report["Enrolled"]["f1-score"]
        )

        mlflow.log_metric(
        "graduate_f1",
        report["Graduate"]["f1-score"]
        )

        print("\nClassification Report:")
        print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_

            )
        )
        if macro_f1 > best_f1:

            best_f1 = macro_f1
            best_model_name = model_name

            if model_name in ["random_forest", "xgboost"]:

                best_model = Pipeline(
                    steps=[
                        ("preprocessor", model.named_steps["preprocessor"]),
                        ("classifier", search.best_estimator_)
                    ]
                )

            else:
                best_model = model

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "macro_f1",
            macro_f1
        )

        model_to_log = model

        if model_name in ["random_forest", "xgboost"]:
            model_to_log = Pipeline(
                steps=[
            ("preprocessor", model.named_steps["preprocessor"]),
            ("classifier", search.best_estimator_)
            ]
        )

        if model_name == "xgboost":

            mlflow.xgboost.log_model(
                search.best_estimator_,
                name=f"{model_name}_model"
            )

        else:

            mlflow.sklearn.log_model(
                model_to_log,
                name=f"{model_name}_model"
            )

        print("Accuracy:", accuracy)
        print("Macro F1:", macro_f1)

print("\n==============================")
print("BEST MODEL")
print("==============================")

print("Model:", best_model_name)
print("Macro F1:", best_f1)

print("\nRequired Macro F1:", min_macro_f1)



if best_f1 >= min_macro_f1:

    print("Model passed quality gate!")

else:

    print("Model failed quality gate!")

    raise ValueError(
        f"Model Macro F1 {best_f1:.4f} "
        f"is below required threshold {min_macro_f1:.4f}"
    )



# ==============================
# REGISTER BEST MODEL
# ==============================

with mlflow.start_run():

    mlflow.log_param(
        "selected_model",
        best_model_name
    )

    mlflow.log_metric(
        "selected_macro_f1",
        best_f1
    )

    # Save complete pipeline
    joblib.dump(
        best_model,
        "student_risk_model.pkl"
    )

    mlflow.log_artifact(
        "student_risk_model.pkl"
    )

    # Save label encoder
    joblib.dump(
        label_encoder,
        "label_encoder.pkl"
    )

    mlflow.log_artifact(
        "label_encoder.pkl"
    )

    print("\nBest model saved successfully!")