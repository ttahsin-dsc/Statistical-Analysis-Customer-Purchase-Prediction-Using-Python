import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


#***Task 2(a)***

#!! step 1- importing data !!
file_path = Path(__file__).resolve().parent / "customers.xlsx"
df = pd.read_excel(file_path, engine="openpyxl") #read excel file

#to save all the figures in the same folder
output_folder = file_path.parent

print(df.head()) #first 5 rows
print(df.dtypes) #data types of columns
print(df.isnull().sum()) #count missing values
print(df.describe()) #statistical summary

#checking Gender and purchased for missing or inconsistent values and identifying duplicate rows 
print(df["Gender"].value_counts(dropna=False))
print(df["Purchased"].value_counts(dropna=False))
print("Duplicate rows:",df.duplicated().sum())

#!! step 2 - Data cleaning !!

df = df.replace(r'^\s*$', np.nan, regex=True) #replace blank values with missing values


#  converting columns to numeric format
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Income"] = pd.to_numeric (df["Income"], errors="coerce")
df["SpendingScore"] = pd.to_numeric(df["SpendingScore"], errors="coerce")
df["Purchased"] = pd.to_numeric(df["Purchased"], errors="coerce")

#Standardise Gender- cleaning gender column before encoding
df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
df["Gender"] = df["Gender"].replace({
    "M": "Male",
    "F": "Female",
    "Nan": np.nan
})

#Inspect missing values
print("Missing values before cleaning:")
print(df.isnull().sum())
print(df["Gender"].value_counts(dropna=False)) #count each Gender type before final cleaning


#Boxplots for checking outliers

plt.figure(figsize=(6,4))
sns.boxplot(x=df["Age"])
plt.title("Box Plot of Age(Outlier Detection)")
plt.xlabel("Age")
plt.tight_layout()
age_plot_path = output_folder / "boxplot_age.png"
plt.savefig(age_plot_path)
plt.close()
print("Age box plot saved at:", age_plot_path)

plt.figure(figsize=(6,4))
sns.boxplot(x=df["Income"])
plt.title("Box Plot of Income(Outlier Detection)")
plt.xlabel("Income")
plt.tight_layout()
income_plot_path = output_folder / "boxplot_income.png"
plt.savefig(income_plot_path)
plt.close()
print("Income box plot saved at:", income_plot_path)


plt.figure(figsize=(6,4))
sns.boxplot(x=df["SpendingScore"])
plt.title("Box Plot of Spending Score(Outlier Detection)")
plt.xlabel("Spending Score")
plt.tight_layout()
spending_plot_path = output_folder / "boxplot_spending_score.png"
plt.savefig(spending_plot_path)
plt.close()
print("Spending Score box plot saved at:", spending_plot_path)


#Checking outliers - IQR method 

def cap_outliers(column) :
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    #Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #counting how many outliers were detected
    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    print(
        f"{column}: lower bound = {lower_bound:.2f},"
        f"upper bound = {upper_bound:.2f},"
        f"outliers = {outlier_count}"

    )

    #capping outliers - replacing outliers with upper or lower threshold values
    df[column] = df[column].clip(lower_bound, upper_bound) 


cap_outliers("Age")
cap_outliers("Income")
cap_outliers("SpendingScore")



#Fill missing values using median after handling outliers
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Income"] = df["Income"].fillna(df["Income"].median())
df["SpendingScore"] = df["SpendingScore"].fillna(df["SpendingScore"].median())
df["Gender"] = df["Gender"].fillna("Unknown")


#Handling duplicates
print("Duplicate rows:", df.duplicated().sum()) #count and print duplicate rows
df.drop_duplicates(inplace=True) #remove exact duplicates




#Encoding Gender

df["Gender"] = df["Gender"].map({
    "Male":1,
    "Female":0,
    "Unknown": 2
})


#Drop CustomerID column
df = df.drop("CustomerID", axis=1, errors="ignore")

#double check for missing values before data modelling - final check

print("Missing values after cleaning:")
print(df.isnull().sum())
print("\n Cleaned dataset:")
print(df.head())




#!! Task - 2(b) - Model development !!

from sklearn.model_selection import train_test_split #to split dataset into training and testing data
from sklearn.linear_model import LogisticRegression #chosen ML model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #to evaluate model strength


X = df[["Gender", "Age", "Income", "SpendingScore"]] #independent variable - predictor
y = df["Purchased"] #dependent variable - outcome

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 50 #to reproduce train-test split
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Extracting the fitted logistic regression model
print("Intercept:", model.intercept_[0])

coefficients = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_[0]
})

print("Model coefficients:")
print(coefficients)

y_pred = model.predict(X_test) #predicting purchased outcome for test dataset

#Evaluating model strngth, data quality using classification metrics
print("Accuracy:", accuracy_score(y_test, y_pred)) #for correct predictions
print("Confusion Matrix:") #for finding out how many customers were correctly and incorrectly predicted
print(confusion_matrix(y_test, y_pred))
print("Classification Report:") #for precision, recall, f1 score, support
print(classification_report(y_test, y_pred))

#!! examples showing predictive capability !!

#creating two customers
new_customer = pd.DataFrame({
    "Gender": [1,0],
    "Age": [35, 50],
    "Income": [60000, 45000],
    "SpendingScore": [80,30]
})

prediction = model.predict(new_customer) #for predicting binary class
probability = model.predict_proba(new_customer) #for output probability

print("New customer predictions:")
print(prediction)

print("Prediction probability")
print(probability)

