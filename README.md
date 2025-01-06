Fundsaudittech
Analysis of Mutual Fund Returns Prediction
Project Overview
This project aims to predict mutual fund returns for different time periods (1 month, 3 months, 6 months, and 1 year) using machine learning models. The dataset includes various features related to mutual funds, such as expense ratios, fund size, ratings, and historical returns. The goal is to recommend the best mutual funds based on predicted returns for each period.

Data Preprocessing
1. Dataset Overview
The dataset was loaded and inspected for missing values and data types.
Columns with numerical-like data (sortino, alpha, sd, beta, sharpe) were converted to numeric types.
2. Handling Missing Values
Missing values in numerical columns were filled with the column mean.
3. Feature Selection
Irrelevant columns (scheme_name, fund_manager) were dropped as they do not contribute to the prediction task.
4. Feature Engineering
New target columns (returns_1month, returns_3months, returns_6months) were derived from the returns_1yr column to represent returns for shorter periods.
5. Categorical Encoding
Categorical features (category, sub_category, amc_name) were one-hot encoded to prepare the data for machine learning models.
6. Feature Scaling
Numerical features were standardized using StandardScaler to ensure all features have equal importance during model training.
Exploratory Data Analysis
1. Distribution of Numerical Features
Histograms were plotted for all numerical columns to understand their distributions.
2. Correlation Analysis
A heatmap of the correlation matrix was generated to identify relationships between features and target variables.
Model Training and Evaluation
1. Target Variables
Separate models were trained for predicting returns for 1 month, 3 months, 6 months, and 1 year.
2. Machine Learning Model
Random Forest Regressor was used as the primary model due to its ability to handle non-linear relationships and feature importance analysis.
3. Model Evaluation
Models were evaluated using:
Mean Squared Error (MSE): Measures the average squared difference between actual and predicted values.
R² Score: Indicates the proportion of variance in the target variable explained by the model.
Cross-validation was performed to ensure model robustness.
4. Hyperparameter Tuning
Grid search was used to optimize hyperparameters such as n_estimators, max_depth, and min_samples_split.
