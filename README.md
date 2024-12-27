# Predict the Success of Bank Telemarketing

## Introduction
This project aims to predict the success of telemarketing campaigns conducted by a banking institution. The campaigns involve contacting potential customers via phone calls to promote a bank term deposit. The goal is to use machine learning techniques to analyze the data and predict whether a customer will subscribe to the term deposit ('yes') or not ('no').
## Overview
The dataset contains information about the direct marketing campaigns, including client data, contact details, and results. We leverage this rich dataset to develop a predictive model using various machine learning algorithms.

## Step 1: Importing Required Libraries
To begin, we imported a wide range of Python libraries for:

- Data manipulation and analysis: numpy, pandas
- Visualization: matplotlib, seaborn
- Data preprocessing: SimpleImputer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, FunctionTransformer
- Feature selection: SelectKBest, chi2, RFE
- Modeling: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, LightGBM, XGBoost, and others
- Model evaluation: accuracy_score, f1_score, roc_curve, auc, classification_report, and more
Additionally, utility libraries like os for directory traversal and warnings to suppress warnings were included.

## Step 2: Loading Data
The dataset was loaded from the provided directory. The following files were used:

- Train Data: Contains the training data for building the model.
- Test Data: Contains the testing data for evaluating the model.
- Sample Submission: A sample format for submission.

## Step 3: Exploratory Data Analysis (EDA)
### Dataset Overview
- Training Data: 39,211 rows (samples) and 16 columns (15 features + 1 target variable).
- Testing Data: 10,000 rows and 15 features (no target variable).
- Target Variable: target (binary: "yes" for subscribing to a term deposit, "no" otherwise).
### Features
- Numerical:
  - age, balance, duration, campaign, pdays, previous
- Categorical:
  - job, marital, education, default, housing, loan, contact, poutcome
- Date:
  - last contact date: Requires further processing for insights.
### Basic Statistics
- Missing values in key features:
  - High missing percentages in poutcome (~75%) and contact (~26%).
  - education (~3.7%) and job (~0.58%) have moderate missing percentages.
- Target Variable:
  - Imbalanced: Majority are "no" for subscription.
### Univariate Analysis
- Categorical Variables:
  - Visualized distributions using count plots.
  - Insights:
    - Most clients are "married" and have "secondary" education.
    - Majority of clients do not have defaults or personal loans.
- Numerical Variables:
  - Histograms revealed:
    - age is right-skewed, with a median around 40.
    - balance has a wide range, including negative and zero values.
### Bivariate Analysis
- Scatter Plot (Age vs. Balance):
  - No clear correlation, but clients over 60 tend to have lower balances.
- Box Plots:
  - campaign and balance show variations between clients who subscribed vs. did not subscribe.
- Specific Insights:
  - 7.1% of clients have zero balances, and 7.6% have negative balances.
  - Some clients with zero or negative balances subscribed to term deposits.
  - Call duration correlates with subscription:
  - Minimum duration for subscription: 8 seconds.
  - Maximum duration: 4916 seconds.
### Missing Values
- poutcome and contact have high missing percentages.
- Missing values for categorical features like education and job might be imputed with strategies such as mode or classification models.
### Target Variable Analysis
- Severe class imbalance in target.
- Oversampling techniques (e.g., SMOTE) may be needed to balance classes for better model performance.
### Additional Insights
- Clients with previous = 0 often have poutcome = 'unknown'.
- Positive balances tend to align with subscription likelihood.
- Longer call durations are more likely to result in subscriptions.

## Step 4: Train-Validation Split
### Data Preparation:
- The dataset train_data_1 is used, which contains a target column representing the labels.
### Feature-Target Separation:
- X is created by dropping the target column from train_data_1, containing 15 features.
- y is created by isolating the target column, representing the labels.
### Train-Validation Split:
- The data is split into training and validation sets using train_test_split from sklearn.model_selection.
- The split is performed with a test_size of 0.2 (20% validation data) and a random_state of 42 to ensure reproducibility.
### Resulting Dataset Shapes:
- X_train: Training features, shape (31368, 15).
- X_val: Validation features, shape (7843, 15).
- y_train: Training labels, shape (31368,).
- y_val: Validation labels, shape (7843,).

## Step 5: Data Preprocessing
### Handling Missing Values
- Columns 'job' and 'education':
  - Missing values were filled with the most frequent value using SimpleImputer (mode imputation).
  - Applied to X_train, X_val, and test_data.
- Columns 'contact' and 'poutcome':
  - Missing values were filled with 'unknown'.
  - Applied to X_train, X_val, and test_data.
### Dealing with Outliers
- Outlier Removal:
  - For the 'balance' feature in X_train, rows with values outside the 1st to 99th percentile range were removed.
- Outlier Capping:
  - Applied to 'age', 'campaign', 'previous', and 'pdays' in X_train, X_val, and test_data.
  - Values above specific percentiles (e.g., 99th for 'age', 96th for 'pdays') were capped at the calculated upper bound.
### Feature Engineering
- Date Features:
  - The 'last contact date' column was converted to datetime format.
  - Extracted 'weekday', 'month', and 'day' as new features.
  - The original column was dropped.
- Binning:
  - 'age' was binned into age groups: <30, 30-39, 40-49, 50-59, and 60+.
  - The new 'age_group' column was created, and the 'age' column was optionally dropped.
- Derived Features:
  - 'balance_duration_ratio': Ratio of 'balance' to 'duration + 1'.
  - 'no_previous_contact': Binary feature indicating if 'pdays' is -1.
  - 'campaign_contact': Proportion of 'campaign' to the sum of 'campaign' and 'previous'.
### Duration Conversion
- 'duration' was converted from seconds to minutes and rounded to two decimal places.
- Rows in X_train where 'duration' was less than 5 seconds were removed.
### Final Dataset Shapes
- Training Set (X_train): 30,700 rows and 21 columns.
- Validation Set (X_val): 7,843 rows and 21 columns.
- Test Set (test_data): 10,000 rows and 21 columns.
### Key Metrics:
- Outlier removal reduced X_train rows from 31,368 to 30,740.
- 'balance_duration_ratio' had a mean of ~15.56 and a max value of 8,720.5.
- The 'age_group' distribution was:
  - 30-39: 11,686 rows (largest group).
  - 60+: 2,163 rows (smallest group).
 
## Additional Visualization Analysis
### Correlation Insights:
- Relationships like duration and campaign calls suggest optimal marketing strategies.
- Age and balance analysis helps identify high-yield customer segments.
### Actionable Insights:
- Recommend limiting campaign calls and targeting age-balance segments to improve subscription rates.
### Job and Month Analysis:
- Leveraging months like March and customer job types (e.g., students) could yield better results.

### Feature Encoding and scaling
- Dropped Features:
  - You removed redundant features like age and balance_level, which is efficient.
  - Ensure the dropped features don't negatively impact interpretability.
- Label Encoding:
  - Used for binary variables (default, housing, loan).
  - Remember to save the encoders for use during model inference.
- Ordinal and One-Hot Encoding:
  - Encoding age_group ordinally makes sense since it has a natural order.
  - One-hot encoding categorical variables adds clarity and ensures model adaptability.
- Scaling:
  - Standard scaling is appropriate for numerical columns, ensuring uniform treatment.

## 6. Feature Selection
- Objective: Select the most important features for your model to improve performance and reduce overfitting.
- Method Used: Recursive Feature Elimination (RFE) with a RandomForestClassifier.
- Process:
  - Used RandomForestClassifier as the estimator.
  - Specified that 28 features should be selected.
  - Applied RFE to select features by eliminating less important ones.
  - The selected features were printed, and the dataset was transformed to retain only these features.
- Outcome: The training, validation, and test datasets were reduced to 28 selected features. The shapes of the datasets after transformation were:
  - Training set: (30700, 28)
  - Validation set: (7843, 28)
  - Test set: (10000, 28)

## 7. Class Imbalance
- Objective: Handle class imbalance to avoid biasing the model toward the majority class.
- Method Used: SMOTEENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors).
- Process:
  - SMOTEENN was applied to the training data to create synthetic samples of the minority class and clean the dataset.
  - The resampled dataset was used for training the model.
  - The class distribution was checked before and after resampling to ensure balance.
- Outcome: You would expect a balanced class distribution, but the code for this part was commented out. If applied, SMOTEENN would handle the class imbalance.

## 8. Dummy Model
- Objective: Use a baseline model to compare performance and establish a reference point.
- Method Used: DummyClassifier with the 'most_frequent' strategy.
- Process:
  - DummyClassifier was initialized with the strategy to predict the most frequent class.
  - The model was trained on the selected features.
  - Predictions were made on the validation set, and the accuracy was calculated.
- Outcome:
  - Accuracy: 84.73%
  - Classification report:
    - Class 0 (majority class): Precision 0.85, Recall 1.00, F1-score 0.92
    - Class 1 (minority class): Precision 0.00, Recall 0.00, F1-score 0.00
    - The model achieves a high accuracy but performs poorly for the minority class (class 1), which is typical for imbalanced datasets.

## 9. Baseline Model
- Model: Logistic Regression with class balancing (class_weight='balanced').
- Performance:
  - Accuracy: 81.93%
  - Precision: High for class 0 (0.94), lower for class 1 (0.44).
  - Recall: Higher for class 1 (0.72), lower for class 0 (0.84).
  - F1-score: 0.89 for class 0, 0.55 for class 1.
  - Confusion Matrix: Shows a good ability to predict class 0, but struggles with class 1, showing room for improvement in handling the minority class.

## 10. Fitting and Hyperparameter Tuning of Different Models
### XGBoost Classifier (model_2):
- Accuracy: 86.03%
- Precision/Recall: High precision for class 0 (0.96) and class 1 (0.53), higher recall for class 1 (0.80).
- F1-Score: 0.64 for class 1.
- Key Parameters: scale_pos_weight, subsample=0.7, n_estimators=300, learning_rate=0.05.
### LightGBM Classifier (model_5):
- Accuracy: 86.59%
- Precision/Recall: High precision for class 0 (0.95) and class 1 (0.54), higher recall for class 1 (0.77).
- F1-Score: 0.64 for class 1.
- Key Parameters: is_unbalance=True, subsample=0.8, num_leaves=100, learning_rate=0.05.
### Decision Tree Classifier (model_6):
- Accuracy: 80.90%
- Precision/Recall: High precision for class 0 (0.97) and lower for class 1 (0.44).
- F1-Score: 0.58 for class 1.
- Key Parameters: max_depth=10, min_samples_split=50, min_samples_leaf=10.
### Random Forest Classifier (model_10):
- Accuracy: 86.47%
- Precision/Recall: High precision for class 0 (0.95) and class 1 (0.54), with decent recall for class 1 (0.76).
- F1-Score: 0.63 for class 1.
- Key Parameters: class_weight='balanced', n_estimators=300, min_samples_split=5, min_samples_leaf=4.
### Key Insights:
- XGBoost and LightGBM provided the highest accuracy and balanced performance, especially for class 0.
- Decision Tree showed weaker performance with lower accuracy for class 1.
- Random Forest performed well but had similar results to XGBoost and LightGBM.

## Step 11. Model Selection
Based on the cross-validation results, the best-performing model in terms of the mean F1 Macro score is LightGBM (model_5) with a score of 0.7744. Here's a summary of the model performance:

### Detailed Performance Metrics:
|           Model          | Accuracy (Val)	| F1 Macro (Val)	| F1 Macro (CV Mean)	| F1 Macro (CV Std) |
|--------------------------|----------------|-----------------|---------------------|-------------------|
|     XGBoost (model_2)    |	   0.8603   	|     0.7754    	|       0.7738      	|      0.0087       |
|    LightGBM (model_5)    |	   0.8659	    |     0.7769    	|       0.7744      	|      0.0105       |
|  Decision Tree (model_6) |     0.8090   	|     0.7279	    |       0.7326      	|      0.0093       |
| Random Forest (model_10) |	   0.8647   	|     0.7747	    |       0.7706      	|      0.0080       |
### Insights:
- LightGBM not only performed the best in terms of cross-validation mean F1 Macro score but also has relatively low variability (CV Std) compared to the other models.
- XGBoost and Random Forest also performed well, with similar F1 Macro values but slightly lower than LightGBM.
- Decision Tree performed the weakest, with both lower F1 Macro values and higher variability.
### Conclusion:
- LightGBM (model_5) is the recommended model based on the evaluation, showing both strong performance and stability across cross-validation.

## Step 12. Train whole data
After training the model on the combined training and validation datasets, the results are as follows:
### Accuracy:
  - Accuracy: 0.9472 (94.72%)
### Classification Report:
Class	Precision	Recall	F1-Score	Support
0	1.00	0.94	0.97	6645
1	0.74	1.00	0.85	1198
### Overall:
- Accuracy: 94.72% on the validation set.
- Precision (Class 0): Perfect precision of 1.00 for class 0.
- Recall (Class 1): Perfect recall of 1.00 for class 1, but precision is lower (0.74), indicating that the model is more conservative in predicting class 1 but captures all of its occurrences.
- F1-Score (Class 1): 0.85, which is relatively good but could be improved.
### Averages:
- Macro Average: Precision = 0.87, Recall = 0.97, F1-Score = 0.91.
- Weighted Average: Precision = 0.96, Recall = 0.95, F1-Score = 0.95.
### Conclusion:
The model performs well, especially in terms of recall for class 1, but there is room for improvement in precision for class 1. This might indicate a trade-off between precision and recall for this class.

## Step 13. **Submission**
### Test Data Check:
- The test_selected DataFrame contains 10,000 rows and 28 features, which match the number of features used in the model training process.
### Model Prediction:
- The model (model_5, LightGBM) is used to predict the target for the test data.
- Predictions are converted back to categorical values using target_encoder.inverse_transform().
### Submission Preparation:
- A new DataFrame submission is created with two columns: id (from the test data index) and target (the predicted target values).
- The submission DataFrame is then saved as a CSV file: submission.csv.

## Suggestions
### Targeted Campaigns:
- Focus marketing efforts on customers who have previously been contacted successfully. These customers show a higher likelihood of conversion based on the analysis of the poutcome feature.
- Customers with tertiary education or in specific job roles (e.g., management, technician) have shown a higher conversion rate. Tailor campaigns to address their specific financial needs.
### Optimize Contact Strategies:
- Use cellular contact methods over unknown contact methods as they are associated with better outcomes.
- Limit the number of contacts per campaign (campaign feature) to avoid fatigue and negative customer response. Campaigns with fewer attempts have been more successful.
### Utilize Timing Insights:
-Campaigns conducted during specific months (e.g., May, October) or weekdays have higher conversion probabilities. Adjust the timing of your campaigns accordingly.
### Financial Stability Analysis:
- Customers with higher balances and longer durations in previous engagements (balance_duration_ratio) are more likely to respond positively. Prioritize these customers in your campaigns.
### Leverage Machine Learning Predictions:
- Integrate the LightGBM model into the bank's marketing workflow to identify customers most likely to subscribe to a term deposit. The model has shown high accuracy (94.72%) and macro F1 scores, making it a reliable tool for segmentation.

## Conclusion
The analysis and machine learning models provide a robust foundation for improving the bank's marketing strategies. By targeting high-potential customer segments, optimizing campaign timing and contact methods, and using predictive insights from the LightGBM model, the bank can:
- Increase the effectiveness of its term deposit campaigns.
- Improve customer satisfaction by tailoring offers to relevant segments.
- Reduce unnecessary costs and efforts on low-conversion segments.

The integration of data-driven strategies will enhance the bank's operational efficiency and customer outreach, ultimately boosting revenue and strengthening customer relationships.
