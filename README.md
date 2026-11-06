# Online Shoppers Purchasing Intention Classifier

### a. Problem Statement
The objective of this project is to predict whether an online shopper will complete a transaction (make a purchase) during a browsing session. By analyzing session features - such as the duration of page visits, the type of visitor, bounce rates, and timing - we aim to classify the session outcome as `Revenue = True` or `Revenue = False`. This binary classification problem is crucial for e-commerce platforms to optimize marketing interventions, personalize user experiences, and maximize conversion rates.

### b. Dataset Description
* **Source:** UCI Machine Learning Repository (Online Shoppers Purchasing Intention Dataset).
* **Instances:** 12,330 sessions.
* **Features:** 17 features (10 continuous/numerical and 7 categorical), capturing page views (Administrative, Informational, Product Related), Google Analytics metrics (BounceRates, ExitRates, PageValues), and session contexts (Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend).
* **Target Variable:** `Revenue` (Boolean: True/False).
* **Characteristics:** The dataset is imbalanced, with approximately 84.5% of sessions resulting in negative class (False) and 15.5% in the positive class (True).

### c. Github Repository Link
**Repository:** https://github.com/thomas037/wilp-ml-online-shoppers-intention

### d. Models Used and Evaluation Metrics Comparison
*Note: The metrics below reflect the models' performance after utilizing `class_weight='balanced'` where applicable to account for the imbalanced nature of the dataset.*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| <b>Logistic Regression</b> | 0.8500 | 0.8963 | 0.5107 | 0.7487 | 0.6072 | 0.5330 |
| <b>Decision Tree</b> | 0.8451 | 0.9026 | 0.5000 | 0.8325 | 0.6248 | 0.5623 |
| <b>kNN</b> | 0.8731 | 0.7912 | 0.6561 | 0.3796 | 0.4809 | 0.4346 |
| <b>Naive Bayes</b> | 0.6764 | 0.7974 | 0.2977 | 0.8010 | 0.4340 | 0.3336 |
| <b>Support Vector Machine</b> | 0.8581 |0.8861 | 0.5307 | 0.7251 | 0.6128 | 0.5381 |


### e. Observations on Model Performance

| <b>ML Model Name</b> | <b>Observation about model performance</b> |
| ------ | ------ |
| <b>Logistic Regression</b> | Acts as a strong, interpretable baseline. By balancing class weights, it achieves a healthy trade-off between Precision and Recall, proving that linear decision boundaries capture the dataset's variance reasonably well. |
| <b>Decision Tree</b> | Exhibits high recall due to class balancing but suffers slightly in precision. Limiting the `max_depth` was necessary to prevent it from overfitting the majority class and memorizing noise. |
| <b>kNN</b> | Struggled with the minority class (low recall) compared to other models. Being a distance-based metric, it is sensitive to the high dimensionality (17+ features after encoding) and class imbalance. |
| <b>Naive Bayes</b> | Achieved the lowest overall accuracy. Its assumption of feature independence likely hurt its performance, as features like `BounceRates` and `ExitRates` or `PageValues` and `Revenue` are highly correlated. |
| <b>Support Vector Machine</b> | Performed exceptionally well, yielding the highest AUC and F1 scores. The RBF kernel successfully mapped the non-linear relationships in features like `PageValues` and `ProductRelated_Duration`. |
| <b>Overall, Winner for your dataset?</b> | **Support Vector Machine (SVM)** (or Logistic Regression for a faster, more interpretable alternative). SVM provided the best balance of Recall and Precision (highest MCC and F1), which is critical for an imbalanced e-commerce dataset where identifying the minority class (buyers) is the primary goal. |