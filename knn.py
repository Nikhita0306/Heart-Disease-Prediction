import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load the Excel file into a Pandas DataFrame
data = pd.read_excel("healthcare_data.xlsx")

# One-hot encode the 'Gender' column and other categorical columns
data = pd.get_dummies(data, columns=['Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'], 
                      prefix=['Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'])

# Define the features (X) and target variable (y)
X = data[['Age', 'Systolic', 'Diastolic', 'Gender_Female', 'Gender_Male', 'Cholesterol_High', 'Cholesterol_Normal', 
          'Smoking_No', 'Smoking_Yes', 'Family History_No', 'Family History_Yes', 'Exercise_No', 'Exercise_Yes']]
y_heart_disease = data['Heart Disease']

# Convert 'No' and 'Yes' to 0 and 1
y_heart_disease = y_heart_disease.map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_heart_disease, test_size=0.2, random_state=42)

# Create and train the KNN model for Heart Disease prediction
knn_heart_disease = KNeighborsClassifier(n_neighbors=5)
knn_heart_disease.fit(X_train, y_train)

# Take user input
age = int(input("Enter Age: "))
systolic = int(input("Enter Systolic Blood Pressure: "))
diastolic = int(input("Enter Diastolic Blood Pressure: "))
gender = input("Enter Gender (Male or Female): ").lower()
cholesterol = input("Enter Cholesterol (Normal or High): ").lower()
smoking = input("Do you smoke (Yes or No): ").lower()
family_history = input("Family History of Heart Disease (Yes or No): ").lower()
exercise = input("Regular Exercise (Yes or No): ").lower()

# One-hot encode the user's input
gender_female = 1 if gender == 'female' else 0
gender_male = 1 if gender == 'male' else 0
cholesterol_high = 1 if cholesterol == 'high' else 0
cholesterol_normal = 1 if cholesterol == 'normal' else 0
smoking_yes = 1 if smoking == 'yes' else 0
smoking_no = 1 if smoking == 'no' else 0
family_history_yes = 1 if family_history == 'yes' else 0
family_history_no = 1 if family_history == 'no' else 0
exercise_yes = 1 if exercise == 'yes' else 0
exercise_no = 1 if exercise == 'no' else 0

# Create a feature vector from user input
user_input = pd.DataFrame([[age, systolic, diastolic, gender_female, gender_male, cholesterol_high, cholesterol_normal,
               smoking_yes, smoking_no, family_history_yes, family_history_no, exercise_yes, exercise_no]],
               columns=['Age', 'Systolic', 'Diastolic', 'Gender_Female', 'Gender_Male', 'Cholesterol_High', 'Cholesterol_Normal',
                        'Smoking_No', 'Smoking_Yes', 'Family History_No', 'Family History_Yes', 'Exercise_No', 'Exercise_Yes'])

# Predict Heart Disease based on user input
predicted_heart_disease = knn_heart_disease.predict(user_input)
print(f"Predicted Heart Disease: {'Yes' if predicted_heart_disease[0] else 'No'}")