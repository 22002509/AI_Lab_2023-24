# AI MINI PROJECT - Weather Prediction
### DATE:    25-10-25                                                                       
### REGISTER NUMBER : 212222040120
### AIM: 
To build and evaluate a machine learning model for predicting a target outcome based on given input features.
###  Algorithm:
```
1.Import the dataset, clean, and normalize it for analysis.
2. Identify and select key features relevant to the target prediction.
3. Choose an appropriate machine learning model (e.g., Logistic Regression, LightGBM, etc.).
4. Train the selected model using the training dataset.
5. Evaluate the model performance using metrics such as accuracy, precision, recall, and AUC-ROC.
6.Interpret the model's predictions and visualize the outcomes with relevant graphs.
Result (One-line):
        
```
### Program:

### LINEAR REGRESSION
```
# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Read the uploaded CSV
file_path = '/content/weatherHistory.csv' # Colab default upload path
df = pd.read_csv(file_path)

# --- EDA and Cleaning (from original code) ---
# Clean up column names for easier use
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '').str.replace('-', '')

# Step 3: Handle missing values
df = df.dropna()

# --- ML Preparation Steps ---

# 1. Feature Selection
# We'll use numerical columns as features to predict Temperature (C).
features = ['Apparent_Temperature_C', 'Humidity', 'Wind_Speed_km/h', 'Wind_Bearing_degrees', 'Pressure_millibars']
target = 'Temperature_C'

X = df[features] # Features
y = df[target]   # Target

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# 2. Splitting the Data
# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Set Size:", X_train.shape[0])
print("Testing Set Size:", X_test.shape[0])

# 3. Model Selection and Training
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
print("\nTraining the Linear Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Prediction and Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"R-squared (R2) Score: {r2:.4f}")

# --- Visualization of Predictions ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Ideal line (y=x)
plt.title('Actual vs. Predicted Temperature')
plt.xlabel('Actual Temperature (C)')
plt.ylabel('Predicted Temperature (C)')
plt.grid(True)
plt.show()
```

### RANDOM REGRESSION
```
# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# --- Changed ML Model Import ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Read the uploaded CSV
file_path = '/content/weatherHistory.csv' # Colab default upload path
df = pd.read_csv(file_path)

# --- EDA and Cleaning (from original code) ---
# Clean up column names for easier use
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '').str.replace('-', '')

# Step 3: Handle missing values
df = df.dropna()

# --- ML Preparation Steps ---

# 1. Feature Selection
# We'll use numerical columns as features to predict Temperature (C).
features = ['Apparent_Temperature_C', 'Humidity', 'Wind_Speed_km/h', 'Wind_Bearing_degrees', 'Pressure_millibars']
target = 'Temperature_C'

X = df[features] # Features
y = df[target]   # Target

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# 2. Splitting the Data
# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Set Size:", X_train.shape[0])
print("Testing Set Size:", X_test.shape[0])

# 3. Model Selection and Training
# --- Changed Model Initialization to RandomForestRegressor ---
# n_estimators is the number of trees in the forest.
# random_state ensures reproducibility.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model using the training data
print("\nTraining the Random Forest Regressor model (this may take longer than Linear Regression)...")
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Prediction and Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation (Random Forest Regressor) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"R-squared (R2) Score: {r2:.4f}")

# --- Visualization of Predictions ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Ideal line (y=x)
plt.title('Actual vs. Predicted Temperature (Random Forest)')
plt.xlabel('Actual Temperature (C)')
plt.ylabel('Predicted Temperature (C)')
plt.grid(True)
plt.show()
```
### Output:

### Linear Regression

Features (X) shape: (95936, 5)
Target (y) shape: (95936,)

Training Set Size: 76748
Testing Set Size: 19188

Training the Linear Regression model...
Model training complete.

--- Model Evaluation ---
Mean Absolute Error (MAE): 0.74°C
R-squared (R2) Score: 0.9901

![WhatsApp Image 2025-10-25 at 14 29 02_308f6f7d](https://github.com/user-attachments/assets/471b44fe-3c79-44a4-bf41-a0de0a87dc11)

### Random Regression

Features (X) shape: (95936, 5)
Target (y) shape: (95936,)

Training Set Size: 76748
Testing Set Size: 19188

Training the Random Forest Regressor model (this may take longer than Linear Regression)...
Model training complete.

--- Model Evaluation (Random Forest Regressor) ---
Mean Absolute Error (MAE): 0.01°C
R-squared (R2) Score: 1.0000

![WhatsApp Image 2025-10-25 at 14 29 35_a078c2f8](https://github.com/user-attachments/assets/2282946a-64c3-4fac-b6b1-d64d022a60bc)


### Result:
The model achieved a satisfactory level of accuracy, demonstrating its capability in predicting the specified target outcome.
