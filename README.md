# Airline Passenger Satisfaction Classification & Deployment

## a. Problem Statement
The objective of this project is to predict customer satisfaction (`satisfied` vs. `neutral or dissatisfied`) for an airline based on passenger demographics, flight parameters, and onboard service ratings. Identifying the leading drivers of passenger dissatisfaction enables airlines to optimize resource allocation, enhance passenger experience, and reduce churn.

## b. Dataset Description
- **Source**: Airline Passenger Satisfaction Dataset (Kaggle).
- **Instances**: 12,000 training records, 2,000 test records (both balanced with 50% satisfied and 50% neutral/dissatisfied).
- **Features**: 22 predictive features (satisfying the assignment minimum of 12 features).
  - *Demographic & Flight Details*: Gender, Customer Type, Age, Type of Travel, Class, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes.
  - *In-flight & Service Ratings (1 to 5 scale)*: Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Inflight service, Cleanliness.
- **Target**: `satisfaction` (Binary classification: `neutral or dissatisfied` = 0, `satisfied` = 1).
- **Preprocessing**: Median imputation for missing values in `Arrival Delay in Minutes`, standard scaling for numerical features, and one-hot encoding for categorical variables (`Gender`, `Customer Type`, `Type of Travel`, `Class`).

## c. GitHub Repository Link
`https://github.com/virajrajhansa-bitswilp/2026AJ05729_ML`

## d. Models Used & Evaluation Metric Comparison
All five classifiers were trained on the training partition and evaluated strictly on the 2,000 held-out test instances.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8610 | 0.9191 | 0.8776 | 0.8390 | 0.8579 | 0.7227 |
| **Decision Tree** | 0.9350 | 0.9733 | 0.9315 | 0.9390 | 0.9353 | 0.8700 |
| **KNN** | 0.9060 | 0.9627 | 0.9256 | 0.8830 | 0.9038 | 0.8129 |
| **Naive Bayes** | 0.8525 | 0.9198 | 0.8852 | 0.8100 | 0.8460 | 0.7076 |
| **Support Vector Machine (RBF)** | **0.9350** | **0.9821** | **0.9412** | **0.9280** | **0.9345** | **0.8701** |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves a baseline accuracy of 86.10% and an AUC of 0.9191. While linear decision boundaries capture broad high-level trends (e.g., travel class and booking ease), they struggle with complex non-linear interactions across service rating dimensions. |
| **Decision Tree** | Strong non-linear performance (93.50% accuracy, 0.9733 AUC, and the highest recall at 93.90%). Hierarchical threshold splits on high-impact features such as Online boarding and Inflight wifi service capture dissatisfaction drivers well. |
| **KNN** | Distance-weighted k-NN ($k=7$) reaches 90.60% accuracy and 0.9627 AUC. Feature scaling makes distance computations reliable, though inference latency remains higher due to pairwise comparison against training instances. |
| **Naive Bayes (Gaussian)** | Lowest overall performer (85.25% accuracy, 0.7076 MCC). The core assumption of conditional independence between features is violated because passenger ratings (such as Cleanliness, Food and drink, and Seat comfort) exhibit strong mutual collinearity. |
| **Support Vector Machine** | **Best overall performer.** With the RBF kernel and scaled continuous/categorical features, SVM constructs an optimal separating margin, achieving the highest AUC (0.9821), highest precision (0.9412), highest MCC (0.8701), and joint-highest accuracy (93.50%). |

### Overall Winner for your dataset:
**Support Vector Machine (RBF Kernel)** is the overall winner. It achieved the highest discriminative power **(AUC = 0.9821)**, the highest quality predictions **(Precision = 0.9412)**, the top overall correlation metric **(MCC = 0.8701)**, and tied for the highest **Accuracy (93.50%)**. While the Decision Tree scored slightly higher on recall, SVM demonstrates superior class separability and generalizability without the risk of overfitting.
