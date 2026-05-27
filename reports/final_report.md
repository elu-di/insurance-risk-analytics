# Unlocking Profitability: Analytics-Driven Risk and Pricing at AlphaCare Insurance Solutions

**By: Marketing Analytics Engineer**  
**Date: May 27, 2026**

## 1. Executive Summary

AlphaCare Insurance Solutions (ACIS) is at a pivotal point in its growth strategy within the South African auto-insurance market. To remain competitive and optimize profitability, relying on intuition-based pricing is no longer viable. Over the past week, we conducted a rigorous end-to-end analysis of 18 months of historical insurance claim data (Feb 2014 – Aug 2015). Our mission was clear: identify low-risk customer segments to attract with tailored pricing, uncover hidden profitability metrics, and build machine-learning models to optimize our risk-based pricing.

The results are highly promising. By switching from static pricing to a dynamic, risk-based ML framework, we can significantly improve profit margins while remaining fair to low-risk customers. 

## 2. Our Analytical Approach

We divided the project into four progressive stages:
1. **Exploratory Data Analysis (EDA):** Immersed ourselves in the data, mapping out distributions, detecting outliers, and discovering initial relationships between geographic regions, vehicle types, and profit margins.
2. **Data Version Control (DVC):** Established a robust, auditable data pipeline using DVC. This ensures all our transformations are perfectly reproducible for regulatory compliance and debugging.
3. **Hypothesis Testing (A/B Testing):** Tested assumptions statistically rather than relying on gut feeling. We investigated differences in claim frequency and severity across provinces, zip codes, and demographics.
4. **Statistical Modeling:** Built powerful predictive models (Linear Regression, Random Forests, XGBoost) to predict both the *probability* of a claim occurring and its potential *severity*.

## 3. Key Insights

### 3.1 Exploratory Data Analysis
- **Loss Ratios:** Certain provinces exhibit noticeably higher loss ratios compared to the national average, indicating geographic hotspots for claims.
- **Outliers:** A small fraction of claims account for a massive chunk of our financial liability. Properly accounting for these high-value claims in our premium pricing is crucial.
- **Vehicle Profiles:** Specific vehicle makes and older models inherently carry higher frequencies of low-severity claims, while newer, higher-end vehicles carry low frequencies but massive severity.

### 3.2 Hypothesis Testing Results
We rigidly tested several hypotheses using statistical methods (Chi-Squared and T-tests):
- **Geography & Risk:** We tested risk (claim occurrence) and margin differences across our top zip codes and provinces. 
- **Gender & Risk:** We evaluated the risk disparity between Men and Women. Statistical testing demonstrated no statistically significant difference in claim occurrence between genders, meaning any gender-based pricing model would not be mathematically sound for claim frequency.
- **Decision:** As none of the targeted A/B tests generated a p-value < 0.05 on the given sample, we failed to reject the null hypotheses for the targeted baseline variables. This indicates that univariate adjustments on singular variables like ZipCode or Gender may not be enough to segment risk effectively—multivariate modeling is strictly necessary.

### 3.3 Predictive Modeling & Machine Learning
We moved past simple segmentation and built complex predictive models:
- **Severity Models:** The Random Forest and XGBoost regressors were optimized to predict the `TotalClaims` amount. XGBoost emerged as highly capable at identifying the complex nonlinear relationships between vehicle type, age, and policy details.
- **Probability Models:** We used Logistic Regression and Random Forest Classifiers to identify the likelihood of a client making a claim. 
- **Feature Importance (SHAP):** SHAP values revealed that vehicle characteristics (age, value) and client features heavily dictate claim probability and severity, much more so than generic demographic features.

## 4. Recommendations for Marketing & Pricing

1. **Deploy Dynamic Risk-Based Pricing:** We strongly recommend replacing the naive `CalculatedPremiumPerTerm` baseline with the output of our ML Pricing Framework. The premium should be calculated dynamically: `Premium = (P(claim) * Predicted Severity) + Expense Loading + Profit Margin`.
2. **Geographic Targeting:** Use marketing spend to aggressively target provinces and zip codes that naturally fall into low-probability and low-severity brackets based on the ML models.
3. **Discontinue Unjustified Segmentation:** Stop utilizing simple univariate cuts (like solely relying on gender or simple zip code buckets) as our A/B tests proved they lack strict statistical significance in isolation. 

## 5. Limitations and Future Work

- **Data Imbalance:** Claims data is heavily skewed (most people don't claim). We recommend utilizing SMOTE or advanced oversampling techniques to improve classification accuracy for the minority class in the future.
- **Temporal Blind Spots:** The 18-month window is adequate, but insurance models are notoriously susceptible to seasonality (e.g., more accidents in winter). We require at least 36 months of data to properly capture long-term seasonal impacts.
- **Next Steps:** Future work will involve putting the XGBoost models into a live CI/CD shadow deployment to track their performance against real-world incoming policies without impacting actual business financials.
