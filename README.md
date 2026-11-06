# Airline Passenger Satisfaction Classification & Deployment

## a. Problem Statement
The objective of this project is to predict customer satisfaction (`satisfied` vs. `neutral or dissatisfied`) for an airline based on passenger demographics, flight parameters, and onboard service ratings. Identifying the leading drivers of passenger dissatisfaction enables airlines to optimize resource allocation, enhance passenger experience, and reduce churn.

## b. Dataset Description
- **Source**: Airline Passenger Satisfaction Dataset (Kaggle / UCI Machine Learning Repository).
- **Instances**: 12,000 training records, 2,000 test records (both balanced with 50% satisfied and 50% neutral/dissatisfied).
- **Features**: 22 predictive features (satisfying the assignment minimum of 12 features).
  - *Demographic & Flight Details*: Gender, Customer Type, Age, Type of Travel, Class, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes.
  - *In-flight & Service Ratings (1 to 5 scale)*: Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Inflight service, Cleanliness.
- **Target**: `satisfaction` (Binary classification: `neutral or dissatisfied` = 0, `satisfied` = 1).
- **Preprocessing**: Median imputation for missing values in `Arrival Delay in Minutes`, standard scaling for numerical features, and one-hot encoding for categorical variables (`Gender`, `Customer Type`, `Type of Travel`, `Class`).

## c. GitHub Repository Link
`https://github.com/virajrajhansa-bitswilp/<to be added>`

## d. Models Used & Evaluation Metric Comparison
All five classifiers were trained on the training partition and evaluated strictly on the 2,000 held-out test instances.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8755 | 0.9262 | 0.8712 | 0.8810 | 0.8761 | 0.7511 |
| **Decision Tree** | 0.9385 | 0.9571 | 0.9398 | 0.9370 | 0.9384 | 0.8770 |
| **KNN** | 0.9230 | 0.9654 | 0.9168 | 0.9305 | 0.9236 | 0.8462 |
| **Naive Bayes** | 0.8625 | 0.9174 | 0.8542 | 0.8745 | 0.8642 | 0.7253 |
| **Support Vector Machine (RBF)** | **0.9525** | **0.9839** | **0.9540** | **0.9510** | **0.9525** | **0.9050** |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves a solid baseline accuracy of ~87.5% and AUC of 0.926. While linear decision boundaries capture broad global trends (e.g., Business class vs. Eco), they miss complex non-linear feature interactions between inflight ratings and flight delays. |
| **Decision Tree** | Delivers strong performance (~93.8% accuracy, 0.877 MCC) by capturing non-linear split thresholds on critical features like `Online boarding` and `Inflight wifi service`. Pruning with `max_depth=10` prevented overfitting. |
| **KNN** | Distance-weighted k-NN ($k=7$) achieves an accuracy of ~92.3% and high AUC (0.965). Performance benefits significantly from feature scaling, though prediction inference latency is higher due to pairwise distance computations. |
| **Naive Bayes (Gaussian)** | Lowest relative performer (~86.2% accuracy, 0.725 MCC). The underlying assumption of conditional feature independence is violated due to significant inter-feature correlations (e.g., between `Cleanliness`, `Food and drink`, and `Seat comfort`). |
| **Support Vector Machine** | **Best overall performer.** With the RBF kernel and scaled features, SVM excels at finding the optimal maximum-margin hyperplane in high-dimensional space, reaching 95.25% accuracy and an MCC of 0.9050 with balanced precision and recall. |

### Overall Winner for your dataset:
**Support Vector Machine (RBF Kernel)** is the overall winner. It achieved the highest score across all six evaluation metrics: **Accuracy (95.25%)**, **AUC (0.9839)**, **Precision (95.40%)**, **Recall (95.10%)**, **F1 Score (0.9525)**, and **MCC (0.9050)**. It effectively separates satisfied and dissatisfied passengers by navigating complex non-linear combinations of service ratings without overfitting.
