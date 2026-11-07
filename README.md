# Online Shoppers Purchasing Intention Classifier

### a. Problem Statement
A large proportion of online shopping sessions fail to convert into purchases. The objective of this project is to predict whether an online shopper will complete a transaction (make a purchase) during a browsing session. By analyzing session features such as the duration of page visits, the type of visitor, bounce rates, and timing, we aim to classify the session outcome as `Revenue = True` or `Revenue = False`. This binary classification problem is crucial for e-commerce platforms to optimize marketing interventions, personalize user experiences, and maximize conversion rates.

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

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | 0.8682 | 0.9120 | 0.5545 | 0.7592 | 0.6409 | 0.5730 | 
| **Decision Tree** | 0.8625 | 0.9404 | 0.5334 | 0.8979 | 0.6693 | 0.6213 | 
| **kNN** | 0.8933 | 0.9139 | 0.7742 | 0.4398 | 0.5609 | 0.5317 | 
| **Naive Bayes** | 0.6959 | 0.8185 | 0.3164 | 0.8298 | 0.4581 | 0.3692 | 
| **Support Vector Machine** | 0.8946 | 0.9386 | 0.6178 | 0.8377 | 0.7111 | 0.6597 | 

### e. Observations on Model Performance

| **ML Model** | **Observation about model performance** | 
| ----- | ----- | 
| **Logistic Regression** | Serves as a solid baseline. It achieved a strong AUC (0.9120) and a good balance of precision and recall, but struggles to capture more complex, non-linear relationships compared to the tree-based and SVM models. | 
| **Decision Tree** | Achieved the highest AUC (0.9404) and the highest Recall (0.8979) among all models. It successfully identifies the vast majority of actual purchasers, though this aggressive identification results in a slightly lower precision (0.5334). | 
| **kNN** | Yielded the highest Precision (0.7742) but the lowest Recall (0.4398). While it is very accurate when it predicts a purchase, it misses more than half of the actual buyers, indicating it is too conservative on this dataset. | 
| **Naive Bayes** | Maintained high Recall (0.8298) but suffered from the lowest Accuracy (0.6959) and Precision (0.3164). The feature independence assumption causes it to overestimate positive signals in correlated e-commerce features, leading to many false positives. | 
| **Overall, Winner for the dataset?** | **Support Vector Machine (SVM)**. SVM provided the absolute best overall balance, dominating with the highest Accuracy (0.8946), F1 Score (0.7111), and MCC (0.6597). It successfully captures non-linear boundaries while maintaining a healthy balance between identifying buyers (Recall: 0.8377) and keeping false alarms in check (Precision: 0.6178). | 