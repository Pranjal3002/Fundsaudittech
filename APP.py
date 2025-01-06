# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'C:/Users/HP/Desktop/Fundsaudit/comprehensive_mutual_funds_data.csv'
mutual_funds_data = pd.read_csv(file_path)

# Overview of the dataset
print("Dataset Overview:")
print(mutual_funds_data.info())
print("\nFirst 5 rows:")
print(mutual_funds_data.head())

# Convert numerical-like columns to numeric
for col in ['sortino', 'alpha', 'sd', 'beta', 'sharpe']:
    mutual_funds_data[col] = pd.to_numeric(mutual_funds_data[col], errors='coerce')

# Check for missing values
print("\nMissing Values:")
print(mutual_funds_data.isnull().sum())

# Fill missing values with column means for numerical columns
mutual_funds_data.fillna(mutual_funds_data.mean(numeric_only=True), inplace=True)

scheme_names = mutual_funds_data['scheme_name']

# Drop irrelevant columns
irrelevant_cols = ['scheme_name', 'fund_manager']  # Columns not contributing to prediction
mutual_funds_data.drop(columns=irrelevant_cols, inplace=True)

# Check the dataset after preprocessing
print("\nDataset After Preprocessing:")
print(mutual_funds_data.info())

# Summary statistics
print("\nSummary Statistics:")
print(mutual_funds_data.describe())

# Visualize distributions of numerical columns
numerical_cols = mutual_funds_data.select_dtypes(include=['float64', 'int64']).columns
mutual_funds_data[numerical_cols].hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Numerical Features')
plt.show()

# Correlation heatmap
numeric_data = mutual_funds_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# One-hot encode categorical features
mutual_funds_data_encoded = pd.get_dummies(mutual_funds_data, columns=['category', 'sub_category', 'amc_name'], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
numerical_features = mutual_funds_data_encoded.select_dtypes(include=['float64', 'int64']).columns
mutual_funds_data_encoded[numerical_features] = scaler.fit_transform(mutual_funds_data_encoded[numerical_features])

# Check the transformed dataset
print("\nDataset After Feature Engineering:")
print(mutual_funds_data_encoded.head())

# Create target columns for 1 month, 3 months, and 6 months by dividing returns_1yr
mutual_funds_data_encoded['returns_1month'] = mutual_funds_data_encoded['returns_1yr'] / 12
mutual_funds_data_encoded['returns_3months'] = mutual_funds_data_encoded['returns_1yr'] / 4
mutual_funds_data_encoded['returns_6months'] = mutual_funds_data_encoded['returns_1yr'] / 2

# Check the dataset after adding new target columns
print("\nDataset After Adding New Target Columns:")
print(mutual_funds_data_encoded[['returns_1month', 'returns_3months', 'returns_6months']].head())



# Define features
X = mutual_funds_data_encoded.drop(columns=['returns_1yr', 'returns_3yr', 'returns_5yr'])

# ----------- 1 Month Prediction -----------

# Define target for 1 month
y_1month = mutual_funds_data_encoded['returns_1month']

# Split the dataset for 1 month prediction
X_train, X_test, y_train, y_test = train_test_split(X, y_1month, test_size=0.2, random_state=42)

# Random Forest Regressor with Cross-validation
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=4, min_samples_leaf=2)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, X, y_1month, cv=10, scoring='neg_mean_squared_error')
print(f"Random Forest - 1 Month - Cross-validated MSE: {-cv_scores_rf.mean():.2f}")

# Hyperparameter tuning for Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 4, 6]}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Evaluate Random Forest for 1 month
y_pred_rf = best_rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - 1 Month - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")

# ----------- 3 Months Prediction -----------

# Define target for 3 months
y_3months = mutual_funds_data_encoded['returns_3months']

# Split the dataset for 3 months prediction
X_train, X_test, y_train, y_test = train_test_split(X, y_3months, test_size=0.2, random_state=42)

# Random Forest Regressor with Cross-validation
rt_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=4, min_samples_leaf=2)
rt_model.fit(X_train, y_train)
y_pred_rt = rt_model.predict(X_test)

# Cross-validation for Random Forest
cv_scores_rt = cross_val_score(rt_model, X, y_3months, cv=10, scoring='neg_mean_squared_error')
print(f"Random Forest - 3 Month - Cross-validated MSE: {-cv_scores_rt.mean():.2f}")

# Hyperparameter tuning for Random Forest
param_grid_rt = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 4, 6]}
grid_search_rt = GridSearchCV(rt_model, param_grid_rt, cv=5, scoring='neg_mean_squared_error')
grid_search_rt.fit(X_train, y_train)
best_rt_model = grid_search_rt.best_estimator_

# Evaluate Random Forest for 3 month
y_pred_rt = best_rt_model.predict(X_test)
mse_rt = mean_squared_error(y_test, y_pred_rt)
r2_rt = r2_score(y_test, y_pred_rt)
print(f"Random Forest - 3 Month - MSE: {mse_rt:.2f}, R²: {r2_rt:.2f}")



# ----------- 6 Months Prediction -----------

# Define target for 6 months
y_6months = mutual_funds_data_encoded['returns_6months']

# Split the dataset for 6 months prediction
X_train, X_test, y_train, y_test = train_test_split(X, y_6months, test_size=0.2, random_state=42)

# Random Forest Regressor with Cross-validation
rz_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=4, min_samples_leaf=2)
rz_model.fit(X_train, y_train)
y_pred_rz = rz_model.predict(X_test)

# Cross-validation for Random Forest
cv_scores_rz = cross_val_score(rz_model, X, y_6months, cv=10, scoring='neg_mean_squared_error')
print(f"Random Forest - 6 Month - Cross-validated MSE: {-cv_scores_rz.mean():.2f}")

# Hyperparameter tuning for Random Forest
param_grid_rz = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 4, 6]}
grid_search_rz = GridSearchCV(rz_model, param_grid_rz, cv=5, scoring='neg_mean_squared_error')
grid_search_rz.fit(X_train, y_train)
best_rz_model = grid_search_rz.best_estimator_

# Evaluate Random Forest for 6 month
y_pred_rz = best_rz_model.predict(X_test)
mse_rz = mean_squared_error(y_test, y_pred_rz)
r2_rz = r2_score(y_test, y_pred_rz)
print(f"Random Forest - 6 Month - MSE: {mse_rz:.2f}, R²: {r2_rz:.2f}")


# ----------- 1 Year Prediction -----------

# Define target for 1 year
y_1year = mutual_funds_data_encoded['returns_1yr']

# Split the dataset for 1 year prediction
X_train, X_test, y_train, y_test = train_test_split(X, y_1year, test_size=0.2, random_state=42)

# Random Forest Regressor with Cross-validation
ry_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=4, min_samples_leaf=2)
ry_model.fit(X_train, y_train)
y_pred_ry = ry_model.predict(X_test)

# Cross-validation for Random Forest
cv_scores_ry = cross_val_score(ry_model, X, y_1year, cv=10, scoring='neg_mean_squared_error')
print(f"Random Forest - 1 Year - Cross-validated MSE: {-cv_scores_ry.mean():.2f}")

# Hyperparameter tuning for Random Forest
param_grid_ry = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 4, 6]}
grid_search_ry = GridSearchCV(ry_model, param_grid_ry, cv=5, scoring='neg_mean_squared_error')
grid_search_ry.fit(X_train, y_train)
best_ry_model = grid_search_ry.best_estimator_

# Evaluate Random Forest for 1 year
y_pred_ry = best_ry_model.predict(X_test)
mse_ry = mean_squared_error(y_test, y_pred_ry)
r2_ry = r2_score(y_test, y_pred_ry)
print(f"Random Forest - 1 Year - MSE: {mse_ry:.2f}, R²: {r2_ry:.2f}")

# Prepare predictions for comparison
df_predictions = X_test.copy()
df_predictions['actual_1month'] = y_test
df_predictions['predicted_1month'] = y_pred_rf
df_predictions['actual_3months'] = y_test
df_predictions['predicted_3months'] = y_pred_rt
df_predictions['actual_6months'] = y_test
df_predictions['predicted_6months'] = y_pred_rz
df_predictions['actual_1year'] = y_test
df_predictions['predicted_1year'] = y_pred_ry

# Merge the scheme names to the predictions
df_predictions['scheme_name'] = scheme_names.loc[X_test.index].values

# Sort and display the top 5 mutual funds based on predicted returns for each period
top_1month = df_predictions[['scheme_name', 'predicted_1month']].sort_values(by='predicted_1month', ascending=False).head(5)
top_3months = df_predictions[['scheme_name', 'predicted_3months']].sort_values(by='predicted_3months', ascending=False).head(5)
top_6months = df_predictions[['scheme_name', 'predicted_6months']].sort_values(by='predicted_6months', ascending=False).head(5)
top_1year = df_predictions[['scheme_name', 'predicted_1year']].sort_values(by='predicted_1year', ascending=False).head(5)

print("\nTop 5 Mutual Funds for 1 Month Returns:")
print(top_1month)

print("\nTop 5 Mutual Funds for 3 Months Returns:")
print(top_3months)

print("\nTop 5 Mutual Funds for 6 Months Returns:")
print(top_6months)

print("\nTop 5 Mutual Funds for 1 Year Returns:")
print(top_1year)
