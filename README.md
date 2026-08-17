# Flipkart-Order-Intelligence-and-Support-Assistant
IITP Capstone project


# Part 1 -- Return-Risk Scoring Pipeline

This section outlines the data verification, preprocessing, modeling, and evaluation steps for the return-risk scoring model, as executed in `part1_return_risk.ipynb`.

## 1. Dataset Verification
* **Data Dimensions:** The generated `orders_dataset.csv` contains exactly 6,000 rows and 13 columns.
* **Overall Return Rate:** The dataset exhibits an overall return rate of **22.75%**.
* **Missing Data:** The `rating_given` column is missing in 783 rows, which accounts for **13.05%** of the data. 

**Missingness Classification (MAR):**
The missingness pattern for `rating_given` is **Missing At Random (MAR)**. This classification is made because the missingness explicitly depends on the observed `payment_method` column. Specifically, there is a measured missing-rate gap where `COD` orders have a ~22% chance of missing a rating, whereas non-COD orders only have a 6% chance of missing a rating. It is not MCAR (because a real dependency exists) nor MNAR (because missingness does not depend on the unobserved rating itself).

## 2. Baseline Model Trap
A `DummyClassifier` (using the `most_frequent` strategy) was trained to establish a baseline.
* **Accuracy:** 77.25%
* **F1-Score (returned=1):** 0.0
* **Precision / Recall:** 0.0 / 0.0

**The Trap:** 
The high accuracy (77.25%) is highly misleading. This is a classic "high accuracy, zero recall" trap in imbalanced datasets. Because the model simply predicts "Not Returned" for every single order, it successfully captures the majority class but misses 100% of actual returns. For the real business problem, this renders the baseline entirely useless, as no proactive interventions can be made if the model never flags an order.

## 3. Logistic Regression & Threshold Sweep
A Logistic Regression model (with `class_weight="balanced"`) was trained on the preprocessed data.
* **Default Threshold (0.5):** 
  * **ROC-AUC:** 0.6240
  * **F1-Score:** 0.3940
  * **Recall:** 57.88%
  * **Precision:** 29.87%

* **Optimized Threshold (0.42):** 
  Sweeping the decision threshold revealed that a threshold of **0.42** maximizes the F1 score.
  * **Max F1-Score:** 0.4067
  * **Recall at 0.42:** 80.59% *(an increase of 22.71 percentage points over the default)*
  * **Precision at 0.42:** 27.19% *(a precision drop of 2.68 percentage points)*

**Business Trade-off:**
Lowering the threshold to 0.42 successfully increases recall (catching more real returns before they happen) at the expense of precision (generating more false alarms). In a business context, this threshold represents the sweet spot between spending too much money on unnecessary interventions (e.g., calling customers, holding shipments due to false positives) and wasting money on reverse-logistics fees and damaged inventory from unprevented returns (False Negatives).

## 4. Random Forest Tuning & Evaluation
A `RandomForestClassifier` was tuned using `GridSearchCV` over 5-fold StratifiedKFold cross-validation.
* **Best Parameters:** `n_estimators` = 200, `max_depth` = 6
* **Cross-Validated ROC-AUC:** 0.6152
* **Held-out Test ROC-AUC:** 0.6176
*(The test ROC-AUC is within 0.05 of the CV score, providing strong evidence against severe overfitting).*

## 5. Feature Importance & Explanation
Extracting the impurity-based `.feature_importances_` from the winning Random Forest yielded the following top 5 features:
1. `payment_method_COD` (0.1645)
2. `price_inr` (0.1211)
3. `delivery_distance_km` (0.0933)
4. `order_id` (0.0911)
5. `customer_tenure_days` (0.0900)

**Permutation Importance Comparison:**
When computing `sklearn.inspection.permutation_importance` on the held-out test split, the rankings shifted significantly. Notably, **`delivery_distance_km`** dropped entirely from the top 5, plummeting to a near-zero/negative importance mean (-0.0009). 

*Why impurity overrates noisy columns:* Impurity-based importance metrics heavily overrate noisy, high-cardinality continuous columns because they offer the tree algorithm far more unique split points to artificially reduce training node impurity, regardless of true predictive signal.

## 6. Subgroup / Root-Cause Analysis
Test-set recall and precision were broken down by categorical subgroups:

**By Product Category:**
* **Apparel:** Recall: 0.5000 | Precision: 0.3333
* **Beauty:** Recall: 0.5806 | Precision: 0.4390
* **Electronics:** Recall: 0.3846 | Precision: 0.3448
* **Footwear:** Recall: 0.5000 | Precision: 0.3373
* **Home:** Recall: 0.6471 | Precision: 0.2366

**By Payment Method:**
* **COD:** Recall: 0.8903 | Precision: 0.3262
* **Prepaid_Card:** Recall: 0.0000 | Precision: 0.0000
* **Prepaid_UPI:** Recall: 0.0000 | Precision: 0.0000
* **Wallet:** Recall: 0.0000 | Precision: 0.0000

*Note: The model performs meaningfully worse on non-COD payment methods (Wallet, Prepaid_Card, Prepaid_UPI), suffering from 0.0000 recall, as well as lower performance distributions in specific sub-categories like Beauty.*

**Proposed Fix:** 
Instead of collecting more data, we should implement a **category-specific decision threshold** rather than a single global cutoff. Because categories like Beauty have a much lower baseline return probability, the global threshold is too strict. By shifting the decision threshold down specifically for the Beauty category and/or non-COD orders, we can flag high-risk orders in these groups and recover recall without cratering the model's overall precision across larger buckets.

## 7. Saved Artifacts
The tuned Random Forest pipeline (including preprocessing) has been persisted.
* **Model Path:** `models/return_risk_model.pkl` (loads without error via `joblib`).
* **Random Forest Optimal Threshold (`t*_rf`):** A secondary threshold sweep specifically on the Random Forest's `predict_proba` output determined that the F1-maximising threshold is **`t*_rf` = 0.44** (achieving a Max F1 of 0.4110). This specific value is what the LangGraph agent in Part 3 will use to anchor its return-risk buckets.