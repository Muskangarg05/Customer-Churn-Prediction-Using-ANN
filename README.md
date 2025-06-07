# Customer Churn Prediction Using ANN

This project predicts whether a customer will leave a bank (churn) using an Artificial Neural Network (ANN). It uses the `Churn_Modelling.csv` dataset and performs preprocessing, model training, and evaluation.

---

## 📂 Dataset
The dataset contains 10,000 entries with features like:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target variable: 1 = churn, 0 = retain)

---

## 🧠 Model
A simple Feedforward Neural Network (ANN) built using TensorFlow/Keras with:
- Input layer with 16 neurons (ReLU)
- Hidden layer with 8 neurons (ReLU)
- Output layer with 1 neuron (Sigmoid)
- Dropout regularization (0.2)

---

## 🛠 Features
- Data preprocessing (dropping unnecessary columns, encoding)
- Feature scaling using `StandardScaler`
- ANN model creation and training
- Early stopping to prevent overfitting
- Evaluation using confusion matrix and classification report
- Model saving as `.h5` file
- Plot generation:
  - Training vs validation accuracy/loss
  - Confusion matrix

---

## 📊 Output
- `training_history.png`: Shows model performance during training
- `confusion_matrix.png`: Visual representation of classification accuracy
- `customer_churn_model.h5`: Saved trained model

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
