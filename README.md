# PayPredict – Customer Churn Prediction

## 📌 Project Overview

**PayPredict** is a Machine Learning project that predicts whether a telecom customer is likely to **churn (leave the service)** based on their demographic information, account details, and service usage.

The project uses a **Random Forest Classifier** to identify customers who are at higher risk of leaving.

## 🎯 Objective

The main objectives of PayPredict are to:

* Predict customer churn using Machine Learning.
* Identify the factors that influence customer churn.
* Help businesses identify high-risk customers.
* Support customer-retention strategies using data-driven predictions.

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and preprocessing
* **Scikit-learn** – Machine Learning
* **Random Forest Classifier** – Churn prediction
* **Git & GitHub** – Version control

## 📂 Project Structure

```text
PayPredict/
│
├── paypredict.py
├── customer chur.csv
├── README.md
└── requirements.txt
```

## 📊 Dataset

The project uses a telecom customer churn dataset containing customer information such as:

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Contract
* Payment Method
* Monthly Charges
* Total Charges
* Churn status

The **Churn** column is used as the target variable.

## 🔄 Machine Learning Workflow

The project follows these steps:

1. Load the customer dataset.
2. Remove unnecessary columns such as `customerID`.
3. Convert numerical values into appropriate data types.
4. Handle missing values.
5. Encode categorical variables.
6. Separate features and target variable.
7. Split the dataset into training and testing sets.
8. Train a Random Forest Classifier.
9. Predict customer churn.
10. Evaluate the model using accuracy and classification metrics.
11. Display important features influencing predictions.

## 🤖 Model

The project uses the **Random Forest Classifier** with:

* 100 decision trees
* `random_state = 42`
* Balanced class weights

Random Forest is suitable for this problem because it can handle multiple features and capture nonlinear relationships between customer characteristics and churn behavior.

## 📈 Evaluation

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Classification Report**

The program also displays the most important features used by the Random Forest model.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd PayPredict
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the model

```bash
python paypredict.py
```

Make sure `customer chur.csv` is present in the same directory as `paypredict.py`.

## 📌 Example Output

```text
Dataset loaded successfully!

PAYPREDICT - MODEL RESULTS
-----------------------------
Accuracy: 0.XXXX

Classification Report:

              precision    recall    f1-score
...
```

The program also displays the **Top Important Features** contributing to customer churn predictions.

## 🚀 Future Improvements

Possible future improvements include:

* Add a user-friendly web interface using Streamlit.
* Compare Random Forest with Logistic Regression, XGBoost, and other models.
* Perform hyperparameter tuning.
* Add visualizations for customer churn analysis.
* Deploy the model as a web application.
* Provide individual customer churn-risk predictions.

## 👩‍💻 Author

**Mullapudi S Kamaniya Kanaka Mahalakshmi**

---

⭐ If you find this project useful, consider giving the repository a star!
.
