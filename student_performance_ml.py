import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)




df = pd.read_csv("Student Performance Processed.csv")
print("Dataset loaded successfully!" )
print("Dataset shape:",df.shape)
print("Columns:",df.columns.tolist())

hours_column = None

possible_hours_columns=[
    "Hours",
    "Hours Studied",
    "Study Hours",
    "Study Hours Per Week",
    "Hours Per Week",
    "Study Hours Per Week"
]

for column in possible_hours_columns:
    if column in df.columns:
        hours_column = column 
        break

if hours_column is None:
    for column in df.columns:
        if "hour"in column.lower():
            hours_column = column
            break

if hours_column is None :
    raise ValueError ("could not find the study hours column.")   

if "Passed" not in df.columns:
    raise ValueError("could not find the Passed column.")   

print("Hours column:", hours_column) 
print("Target column:Passed") 





X=df[[hours_column]]
y=df["Passed"]
print("Features selected successfully!")
print("Target selected successfully!")

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    train_size=0.2,
    random_state=42,
    stratify=y

)


print("Data split completed successfully!")
print("Training samples:",len(X_train))
print("Testing samples:",len(X_test))


model = LogisticRegression()
model.fit(X_train,y_train)
print("Machine Learning model trained succcessfully!")


y_pred= model.predict(X_test)
print("Predictions complete successfully!")



accuracy= accuracy_score(y_test,y_pred)

precision=precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall=recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1=f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm= confusion_matrix(y_test,y_pred)

print("=====Model Performance=====")

print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)

print("Confusion Matrix:")
print(cm)




print("=====Confusion Matrix Details=====")

if cm.shape==(2,2):
    tn,fp,fn,tp=cm.ravel()
    print("True Negatives:",tn)
    print("False Positives:",fp)
    print("False Negative:",fn)
    print("True Positives:",tp)


print("completed successfully!")