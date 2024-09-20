# Climate-Change-Prediction-Model-ML-Model-to-detect-climate-change 

Building a climate change prediction model with 90% accuracy typically involves leveraging machine learning (ML) models and environmental data. Here, I'll guide you through the process of creating a model that detects patterns related to climate change, such as temperature rise, carbon emissions, or weather pattern shifts.
the datasets used is from kaggle  which is used to predict the climate change 

Tools and Libraries:
Python: Main programming language.
Scikit-learn: For machine learning.
Pandas: Data manipulation.
NumPy: Numerical computing.
Matplotlib/Seaborn: Data visualization.
Anaconda: To manage dependencies and environments.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset (Assuming 'climate_data.csv' has relevant features like temperature, CO2 levels, etc.)
df = pd.read_csv('climate_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Data Preprocessing: Checking for missing values
df = df.dropna()

# Features (X) and Target (y) Variables
# X could include various factors like CO2, temperature, sea level, etc.
X = df[['CO2', 'global_temp', 'sea_level', 'greenhouse_gas_concentration']]  # Example features
y = df['climate_change_label']  # Target (1 = Climate Change detected, 0 = No significant change)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection: Using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


 by using this code  and dataset from kaggle we will predict the climate change using ML




