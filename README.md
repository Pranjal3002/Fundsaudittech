### **Name: Pranjal**
### **College_name: University of Petroleum and Energy Studies**


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
## Conclusion

The mutual fund dataset was analyzed, preprocessed, and used to build predictive models for returns over various time periods (1 month, 3 months, 6 months, and 1 year). Below are the key takeaways:

### Dataset Quality
- The dataset initially contained missing values in columns such as `returns_3yr`, `returns_5yr`, `sortino`, `alpha`, `sd`, and `beta`. These were addressed during preprocessing to ensure data consistency.
- After preprocessing, all columns were free of missing values, and feature engineering added meaningful predictors for better model performance.

### Model Performance
- Random Forest Regressor was selected for its robustness and ability to handle non-linear relationships.
- The models demonstrated high accuracy with the following results:
  - **1 Month Returns**: R² = 0.91
  - **3 Month Returns**: R² = 0.92
  - **6 Month Returns**: R² = 0.92
  - **1 Year Returns**: R² = 0.92
- Cross-validation ensured that the models were robust and generalizable.
- ![Screenshot 2025-01-06 092716](https://github.com/user-attachments/assets/e60112a0-8662-44b4-97e5-2bc58fbc0c20)


### Recommendations
- The top mutual funds for each time period were identified based on predicted returns. These funds can guide investors in making informed decisions.
- For example:
  - **1 Month Returns**: Bank of India Short Term Income – Direct Growth topped the list with a predicted return of 0.56%.
  - **1 Year Returns**: The same fund had the highest predicted return of 6.54%.

### Future Improvements
- Incorporate more features, such as market trends or macroeconomic indicators, to improve prediction accuracy.
- Experiment with other machine learning models like Gradient Boosting or Neural Networks for comparison.

This project successfully demonstrates the use of machine learning in financial data analysis, providing actionable insights for mutual fund investments.




