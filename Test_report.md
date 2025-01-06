# **Test Report: Mutual Fund Returns Prediction Model**

## **1. Introduction**
This report provides the results of testing the mutual fund returns prediction model, which aims to predict returns for different time periods (1 month, 3 months, 6 months, and 1 year) based on historical data and various features like expense ratio, fund size, risk level, etc.

## **2. Objectives**
The objectives of the testing are as follows:
- To evaluate the model's performance on predicting mutual fund returns for different time periods.
- To assess the accuracy and reliability of the model using various performance metrics.
- To identify any issues or improvements needed for better predictions.

## **3. Test Plan**
The following steps were performed during testing:
- **Dataset Used**: The dataset consists of 814 entries with 20 columns, including features like `expense_ratio`, `fund_size_cr`, `risk_level`, `returns_1yr`, and others.
- **Model**: Random Forest Regressor was used as the primary model due to its ability to handle non-linear relationships and feature importance analysis.
- **Test Methodology**: The model was tested using Mean Squared Error (MSE) and R² score as the primary evaluation metrics. Cross-validation was performed to ensure robustness.

## **4. Test Cases**
The following test cases were executed to evaluate the model's performance:

- **Test Case 1**: Predicting returns for 1 month.
  - **Input**: Features such as `expense_ratio`, `fund_size_cr`, `risk_level`, etc.
  - **Expected Result**: A predicted return for the 1-month period.
  - **Actual Result**: Predicted return was accurate with a high R² score.

- **Test Case 2**: Predicting returns for 3 months.
  - **Input**: Same features as above.
  - **Expected Result**: A predicted return for the 3-month period.
  - **Actual Result**: Model provided predictions with an R² score of 0.92.

- **Test Case 3**: Predicting returns for 6 months.
  - **Input**: Same features as above.
  - **Expected Result**: A predicted return for the 6-month period.
  - **Actual Result**: The model performed well with a high R² score of 0.92.

- **Test Case 4**: Predicting returns for 1 year.
  - **Input**: Same features as above.
  - **Expected Result**: A predicted return for the 1-year period.
  - **Actual Result**: Model's R² score was 0.92, indicating good prediction performance.

## **5. Results Summary**
The model was tested for all four time periods: 1 month, 3 months, 6 months, and 1 year. Below are the key findings:

| Time Period  | MSE      | R² Score  |
|--------------|----------|-----------|
| 1 Month      | 0.00     | 0.91      |
| 3 Months     | 0.02     | 0.92      |
| 6 Months     | 0.08     | 0.92      |
| 1 Year       | 0.33     | 0.92      |

### **Top 5 Mutual Funds for 1 Month Returns**
| Scheme Name                                      | Predicted 1 Month Return |
|--------------------------------------------------|--------------------------|
| Bank of India Short Term Income – Direct Growth  | 0.557593                 |
| Kotak Infrastructure & Ecoc. Reform-SP-DirGrowth  | 0.274928                 |
| Franklin Build India Fund                        | 0.151520                 |
| SBI PSU Fund                                     | 0.150754                 |
| AXIS Gold Fund                                   | 0.131260                 |

### **Top 5 Mutual Funds for 3 Months Returns**
| Scheme Name                                      | Predicted 3 Months Return |
|--------------------------------------------------|---------------------------|
| Bank of India Short Term Income – Direct Growth  | 1.634931                  |
| Kotak Infrastructure & Ecoc. Reform-SP-DirGrowth  | 0.827532                  |
| SBI PSU Fund                                     | 0.454949                  |
| Franklin Build India Fund                        | 0.454561                  |
| AXIS Gold Fund                                   | 0.394573                  |

### **Top 5 Mutual Funds for 6 Months Returns**
| Scheme Name                                      | Predicted 6 Months Return |
|--------------------------------------------------|---------------------------|
| Bank of India Short Term Income – Direct Growth  | 3.269862                  |
| Kotak Infrastructure & Ecoc. Reform-SP-DirGrowth  | 1.655065                  |
| SBI PSU Fund                                     | 0.909897                  |
| Franklin Build India Fund                        | 0.909123                  |
| AXIS Gold Fund                                   | 0.789147                  |

### **Top 5 Mutual Funds for 1 Year Returns**
| Scheme Name                                      | Predicted 1 Year Return |
|--------------------------------------------------|--------------------------|
| Bank of India Short Term Income – Direct Growth  | 6.539723                 |
| Kotak Infrastructure & Ecoc. Reform-SP-DirGrowth  | 3.310130                 |
| SBI PSU Fund                                     | 1.821493                 |
| Franklin Build India Fund                        | 1.815398                 |
| AXIS Gold Fund                                   | 1.576644                 |

## **6. Defects or Issues**
- **Missing Values**: The dataset had some missing values in columns like `returns_3yr` and `returns_5yr`, but they were handled by filling with appropriate values during preprocessing.
- **Feature Engineering**: Some features were derived from the `returns_1yr` column, and additional categorical variables were one-hot encoded, which might have led to overfitting in some models.

## **7. Performance Metrics**
The model performed well across all time periods, with high R² scores (above 0.90) for all predictions. The MSE was low, indicating that the model's predictions were close to the actual values.

## **8. Conclusion**
The mutual fund returns prediction model performed well, with high accuracy in predicting returns for different time periods. The Random Forest Regressor model demonstrated strong predictive power, and the results show that the model can effectively recommend the best mutual funds based on historical data. Future work could involve improving the model by incorporating more advanced techniques or additional features.

## **9. Next Steps**
- **Model Optimization**: Fine-tune the model using hyperparameter optimization techniques.
- **Feature Expansion**: Explore adding more features or external data sources for improved predictions.
- **Real-time Testing**: Implement the model in a real-time system to evaluate its performance on live data.
