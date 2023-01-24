# Customer Lifetime Value Predictions
This repository contains solution for the analytic vidya's jobathon January 2023 **Customer Lifetime Value Prediction**.

## 1. Problem Statement

VahanBima is one of the leading insurance companies in India. It provides motor vehicle insurance at the best prices with 24/7 claim settlement. It offers different types of policies for both personal and commercial vehicles. It has established its brand across different regions in India.

Around 90% of businesses today use personalized services. The company wants to launch different personalized experience programs for customers of VahanBima. The personalized experience can be dedicated resources for claim settlement, different kinds of services at the doorstep, etc. To do so, they would like to segment the customers into different tiers based on their customer lifetime value (CLTV).
To do it, they would like to predict the customer lifetime value based on the activity and interaction of the customer with the platform.

The challenge objective is to build a high-performance and interpretable machine learning model to predict the customer lifetime value (CLTV) based on the user and policy data.

## 2. Dataset
The training dataset has a total 89392 of rows and 11 columns of customer details such as unique customer id, gender, area, qualification of customer, income, marital status, number of policies and type of policy with claim amount, year of investment, and output variable cltv for customer lifetime value.

This is a regression problem, where we will predict the customer lifetime value `cltv`.

## 3. Approach

### 3.1 Exploratory Data Analysis

Basic data analysis is performed on a dataset to find if data contain any missing values, outliers, or duplicate records. After removing duplicates from the data, the dataset was split into training, validation, and test data in a ratio of 70:20:10 respectively for further analysis.

The output variable `cltv` is highly right-skewed with outliers. After removing outliers from the data, following hypothesis ANOVA test performed on features.

#### 3.1.1 ANOVA Hypothesis Test

One-way ANOVA has been performed, to find the correlation between categorical features and continuous output variables.

- H0 (null hypothesis): The group of means of each category of feature variable is the same.
  
- H1(alternate hypothesis): There is a statistically significant difference in the distribution of means of output variable for the groups of categories of feature variable.
  

The result shows that there is no statistically significant difference in means of a group of gender, and qualification, while there is a statistically significant difference in means for column `income`, `vintage`, `marital status`, `area`, and `type_of_policy`.

#### 3.1.2 Data Preprocessing

- The feature variable `claim_amount` is highly right-skewed, so log transformation is performed on both `claim_amount` and `cltv` features to make distribution Gaussian.
  
- Data Normalization is perform on data using `Min-Max-Scaler`.
  

### 3.2 Model Evaluation

- First, the base model `LinearRegression` algorithm is used to test the overall performance of data. Next, 7-fold cross-validation is performed and evaluated using `r2-score` on several machine learning regression algorithms with a default configuration which includes `AdaBoostRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, and `XGBRegressor`. The `RandomForestRegressor` is selected for final model submission as other models perform poorly on test data by overfitting the model.
  
- Hyperparameter tuning is performed for `RandomForestRegressor` to find the best values for number of estimators and depth of the tree.
  
- The final model of `RandomForestRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_forest=42)` gives a high test score of 0.14392. Hence it is selected for final submission.
