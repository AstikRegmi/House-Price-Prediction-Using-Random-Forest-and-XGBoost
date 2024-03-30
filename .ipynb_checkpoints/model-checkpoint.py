import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("C:/House Price Project/data_reduce.csv")  # Use forward slash or double backslashes in the path

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Assuming 'model_rf' is your trained RandomForestRegressor model
model_rf = RandomForestRegressor()
model_rf.fit(X_scaled, y)

# Assuming 'model_xgb' is your trained XGBoost model
model_xgb = xgboost.XGBRegressor()
model_xgb.fit(X_scaled, y)

# Save the combined models and scaler to a dictionary
combined_models = {
    'RandomForestRegressor': model_rf,
    'XGBoostRegressor': model_xgb,
    'StandardScaler': sc
}

# Save the combined models and scaler to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(combined_models, file)
