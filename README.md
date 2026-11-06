# Online Shoppers Purchasing Intention Classifier

### a. Problem Statement
A large proportion of online shopping sessions fail to convert into purchases. The objective of this project is to predict whether an online shopper will complete a transaction (make a purchase) during a browsing session. By analyzing session features—such as the duration of page visits, the type of visitor, bounce rates, and timing—we aim to classify the session outcome as `Revenue = True` or `Revenue = False`. This binary classification problem is crucial for e-commerce platforms to optimize marketing interventions, personalize user experiences, and maximize conversion rates.

### b. Dataset Description
* **Source:** UCI Machine Learning Repository (Online Shoppers Purchasing Intention Dataset).
* **Instances:** 12,330 sessions.
* **Features:** 17 raw features capturing page views (Administrative, Informational, Product Related), Google Analytics metrics (BounceRates, ExitRates, PageValues), and session contexts. The pipeline incorporates engineered features (`Avg_Duration_Per_Page` and `Product_Ratio`), properly categorizes integer-coded IDs, and applies power transformations to handle right-skewed numerical data.
* **Target Variable:** `Revenue` (Boolean: True/False).
* **Characteristics:** The dataset is imbalanced, with approximately 84.5% of sessions resulting in the negative class (False) and 15.5% in the positive class (True).

### c. Github Repository Link
**Repository:** [https://github.com/thomas037/wilp-ml-online-shoppers-intention]

### d. Models Used and Evaluation Metrics Comparison
*Note: The metrics below reflect the models' performance on the test set after applying class balancing techniques, feature engineering, and power transformations.*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8654 | 0.9236 | 0.5421 | 0.8429 | 0.6598 | 0.6028 |
| **Decision Tree** | 0.8682 | 0.9280 | 0.5501 | 0.8194 | 0.6583 | 0.5982 |
| **kNN** | 0.9039 | 0.9095 | 0.7458 | 0.5759 | 0.6499 | 0.6020 |
| **Naive Bayes** | 0.2178 | 0.7677 | 0.1633 | 0.9817 | 0.2800 | 0.0851 |
| **Support Vector Machine** | 0.8670 | 0.9295 | 0.5453 | 0.8508 | 0.6646 | 0.6091 |

### e. Observations on Model Performance

| **ML Model Name** | **Observation about model performance** |
| :--- | :--- |
| **Logistic Regression** | Serves as a very strong, interpretable baseline. Benefiting from the power-transformed skewed numerical features and balanced class weights, it achieved an excellent AUC (0.9236) and a high recall, successfully identifying the majority of actual purchasers. |
| **Decision Tree** | Performed consistently well across the board. By constraining leaf splits and balancing class weights, it managed the dataset's imbalance effectively, yielding strong AUC (0.9280) and F1 scores without overfitting to the majority class. |
| **kNN** | Achieved the highest overall Accuracy (0.9039) and Precision (0.7458) by utilizing distance-weighted Manhattan metrics. However, this came at a significant cost to Recall compared to the parametric models, meaning it misses more actual purchasers. |
| **Naive Bayes** | Suffered heavily from its feature independence assumption. Because e-commerce features (like bounce rates and exit rates) are highly correlated, the model overestimated positive signals, resulting in extreme Recall (0.9817) but terrible Precision (0.1633) and Accuracy (0.2178) due to massive false positives. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)**. SVM provided the best overall balance, yielding the highest F1 Score (0.6646), the highest AUC (0.9295), and the highest MCC (0.6091). The RBF kernel successfully navigated the non-linear boundaries of the engineered features, making it the most reliable model for identifying true buyers. |