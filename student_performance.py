import pandas as pd
df = pd.read_csv("student_performance_prediction.csv")
print("Dataset loaded successfully!")
print("Original shape:", df.shape)

numeric_columns = df.select_dtypes(include=["number"]).columns
categorical_columns = df.select_dtypes(include=["object","category"]).columns

for column in numeric_columns:
    if not df[column].mode().empty:
        df[column]=df[column].fillna(df[column].mode()[0])

print("Missing values handled successfully.")

duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
print("Duplicates removed:",duplicates_before)


categorical_columns = df.select_dtypes(
    include=["object","category"]
).columns

for column in categorical_columns:
    df[column] = pd.factorize(df[column])[0]



 
print("Categorical data encoded successfully.")
numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    mean = df[column].mean()
    std = df[column].std()
    if std !=0:
        df[column] =(df[column]-mean) /std

        

print("Standardization completed successfully.")

output_file = "Student Performance Processed.csv"

df.to_csv(output_file,index=False)

print("Data Preprocessing completed successfully!")
print("Final shape",df.shape)
print("Processed dataset saved as:" ,output_file)
print("First 5 rows of processed dataset:")
print(df.head())
 