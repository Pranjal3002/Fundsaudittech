# **Fundsaudittech**

## **Analysis of Mutual Fund Returns Prediction**

### **Project Overview**
This project predicts mutual fund returns for different time periods (1 month, 3 months, 6 months, and 1 year) using machine learning models. The dataset includes features such as expense ratios, fund size, ratings, and historical returns. The goal is to recommend the best mutual funds based on predicted returns for each period.

---

### **Features**
- **Dataset Columns:**  
  - `scheme_name`, `min_sip`, `min_lumpsum`, `expense_ratio`, `fund_size_cr`, `fund_age_yr`, `fund_manager`, `sortino`, `alpha`, `sd`, `beta`, `sharpe`, `risk_level`, `amc_name`, `rating`, `category`, `sub_category`, `returns_1yr`, `returns_3yr`, `returns_5yr`.  
- **Target Variables:**  
  - `returns_1month`, `returns_3months`, `returns_6months`, `returns_1yr`.

---

### **Technologies Used**
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas` for data manipulation  
  - `numpy` for numerical computations  
  - `matplotlib` and `seaborn` for data visualization  
  - `scikit-learn` for machine learning and model evaluation  

---

### **Data Preprocessing**
1. **Handling Missing Values:**  
   - Filled missing values in numerical columns with the column mean.  
2. **Feature Selection:**  
   - Dropped irrelevant columns like `scheme_name` and `fund_manager`.  
3. **Feature Engineering:**  
   - Derived target columns for shorter time periods from `returns_1yr`.  
4. **Categorical Encoding:**  
   - One-hot encoded categorical columns (`category`, `sub_category`, `amc_name`).  
5. **Feature Scaling:**  
   - Standardized numerical features using `StandardScaler`.  

---

### **Exploratory Data Analysis**
- Visualized the distribution of numerical features using histograms.
![p](https://github.com/user-attachments/assets/ddfe4342-ab6f-41f1-93d7-14091cb5ba4f)

- Analyzed correlations using a heatmap to identify relationships between features and target variables.
- ![q](https://github.com/user-attachments/assets/adee4a14-5924-4f79-9cbd-35b754f069f7)


---

### **Model Training and Evaluation**
1. **Machine Learning Model:**  
   - **Random Forest Regressor** was chosen for its robustness and feature importance analysis.  
2. **Evaluation Metrics:**  
   - **Mean Squared Error (MSE):** Measures prediction accuracy.  
   - **R² Score:** Indicates the proportion of variance explained by the model.  
3. **Hyperparameter Tuning:**  
   - Optimized model performance using Grid Search for parameters like `n_estimators`, `max_depth`, and `min_samples_split`.  

---




