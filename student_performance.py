import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("student_performance_prediction.csv")
print("Dataset loaded successfully!")
print("Original shape:", df.shape)

if "Student ID" in df.columns:
    df=df.drop(columns=["Student ID"])
    print("Student ID removed.")

if "Passed" not in df.columns:
    raise ValueError("The column'Passed'was not found.")

missing_target = df["Passed"].isna().sum()
df=df.dropna(subset=["Passed"]).copy()
print("Rows removed because Passed was missing:",missing_target)

passed_clean=(
    df["Passed"]
    .astype(str)
    .str.strip()
    .str.lower()
)

passed_clean=passed_clean.replace({
    "pass":"1",
    "passed":"1",
    "yes":"1",
    "true":"1",

    "fail":"0",
    "failed":"0",
    "no":"0",
    "false":"0"
})

df["Passed"]=pd.to_numeric(
    passed_clean,
    errors="coerce"
)
df=df[df["Passed"].isin([0,1])].copy()
df["Passed"]=df["Passed"].astype(int)

if "Study Hours per Week" in df.columns:
    df.loc[df["Study Hours per Week"] < 0, "Study Hours per Week"] = pd.NA

if "Attendance Rate" in df.columns:
    df.loc[
        (df["Attendance Rate"] < 0) | (df["Attendance Rate"] > 100),
        "Attendance Rate"
    ] = pd.NA

if "Previous Grade" in df.columns:
    df.loc[
        (df["Previous Grade"] < 0) | (df["Previous Grade"] > 100),
        "Previous Grade"
    ] = pd.NA

print("Invalid values corrected.")

print("Passed converted to binary classification.")
print("Passed values:")
print(df["Passed"].value_counts())

duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().copy()
print("Duplicates removed:",duplicates_before)


numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object","category"]).columns.tolist()


if "Passed" in numeric_columns:
    numeric_columns.remove("Passed")

for column in numeric_columns:
    df[column]=df[column].fillna(
        df[column].median()
    )

for column in categorical_columns:   
    if not df[column].mode(). empty:
        df[column]=df[column].fillna(
            df[column].mode()[0]
        )

print("Missing values handled successfully.")

if "Parent Education Level" in df.columns:
    education_mapping = {
        "high school": 0,
        "associate": 1,
        "baccalaureate": 2,
        "bachelor": 2,
        "master": 3,
        "doctorate": 4
    }

    education = (
        df["Parent Education Level"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Parent Education Level"] = education.map(education_mapping)

    df["Parent Education Level"] = (
        df["Parent Education Level"]
        .fillna(df["Parent Education Level"].median())
    )

    print("Parent Education Level encoded successfully.")


categorical_columns = df.select_dtypes(
    include=["object","category"]
).columns.tolist()

for column in categorical_columns:
    df[column],_ = pd.factorize(df[column])
print("Categorical data encoded successfully.")

continuous_columns = [
    column for column in [
        "Study Hours per Week",
        "Attendance Rate",
        "Previous Grade"
    ]
    if column in df.columns
]

scaler = StandardScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])


print(" Feature standardization completed successfully.")

print("Remaining missing values:")
print(df.isnull().sum())

output_file = "Student Performance Processed.csv"

df.to_csv(output_file,index=False)

print("Data Preprocessing completed successfully!")
print("Final shape",df.shape)
print("Processed dataset saved as:" ,output_file)
print("First 5 rows of processed dataset:")
print (df.head())