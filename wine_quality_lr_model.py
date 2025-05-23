import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Set paths
data_path = r"C:\Users\allib\OneDrive\Desktop\MS Data Science\ANA680\Week3"
flask_project_path = r"C:\Users\allib\flaskenv\wine-quality"

# load datasets
red_df = pd.read_csv(os.path.join(data_path, "winequality-red.csv"), sep=';')
white_df = pd.read_csv(os.path.join(data_path, "winequality-white.csv"), sep=';')

# Add wine_type column
red_df['wine_type'] = 'red'
white_df['wine_type'] = 'white'

# Combine datasets
df = pd.concat([red_df, white_df], ignore_index=True)

# Prepare features and target
X = df.drop(['quality'], axis=1)
y = df['quality']

# Encode wine_type
X = pd.get_dummies(X, columns=['wine_type'], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate
y_pred = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)
r2_lr = r2_score(y_test, y_pred)
print("MSE:", mse_lr)
print("R²:", r2_lr)

# Save model
model_path = os.path.join(flask_project_path, "wine_quality_lr_model.pkl")
joblib.dump(lr_model, model_path)