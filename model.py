import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("C:/House Price Project/data_reduce.csv")
X = df.drop("price", axis=1)
y = df["price"]

# Dummy encode availability and location
df = pd.get_dummies(df, columns=['availability', 'location'], prefix=['availability', 'location'], drop_first=True)

# Feature scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=51)

# XGBoost Model
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)

# RandomForest Model
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

# Save models and scaler
combined_models = {
    'XGBoostRegressor': xgb_reg,
    'RandomForestRegressor': rfr,
    'StandardScaler': sc,
    'X': X  # Save X to use in app.py
}

with open('model.pkl', 'wb') as file:
    pickle.dump(combined_models, file)
