import pandas as pd

df=pd.read_csv("data/students_dropout_academic_success.csv")

print(df.shape)

print("columns")
for columns in df.columns:
    print(columns)

print("missing values: ")
print(df.isnull().sum())

print("target")
print(df["target"].value_counts())