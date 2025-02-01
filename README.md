# Credit Risk Assessment

This repository contains a Jupyter Notebook (`credit-risk-assessment.ipynb`) that performs a comprehensive analysis of a credit risk dataset. The notebook includes data exploration, feature engineering, and machine learning modeling to predict loan defaults.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Neural Network Model](#neural-network-model)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction
The goal of this project is to assess credit risk by analyzing a dataset containing various features related to loan applicants. The notebook walks through the process of data cleaning, exploratory data analysis, feature engineering, and building a machine learning model to predict loan defaults.

## Dataset
The dataset used in this project is `credit_risk_dataset.csv`, which contains information about loan applicants, including their age, income, employment length, loan intent, loan grade, loan amount, interest rate, loan status, and more.

## Features
The dataset includes the following features:
- `person_age`: Age of the individual applying for the loan.
- `person_income`: Annual income of the individual.
- `person_home_ownership`: Type of home ownership of the individual.
- `person_emp_length`: Employment length of the individual in years.
- `loan_intent`: The intent behind the loan application.
- `loan_grade`: The grade assigned to the loan based on the creditworthiness of the borrower.
- `loan_amnt`: The loan amount requested by the individual.
- `loan_int_rate`: The interest rate associated with the loan.
- `loan_status`: Loan status, where 0 indicates non-default and 1 indicates default.
- `loan_percent_income`: The percentage of income represented by the loan amount.
- `cb_person_default_on_file`: Historical default of the individual as per credit bureau records.
- `cb_person_cred_hist_length`: The length of credit history for the individual.

## Exploratory Data Analysis (EDA)
The notebook includes a detailed exploratory data analysis to understand the distribution of the data, identify missing values, and detect outliers. Key visualizations include:
- Distribution of loan statuses.
- Correlation matrix to identify relationships between features.
- Histograms and box plots for numerical features.

## Feature Engineering
Several new features are created to improve the predictive power of the model:
- `loan_to_income_ratio`: The ratio of the loan amount to the individual's income.
- `loan_to_emp_length_ratio`: The ratio of employment length to the loan amount.
- `int_rate_to_loan_amt_ratio`: The ratio of the interest rate to the loan amount.

## Modeling
The notebook uses a Random Forest classifier to predict loan defaults. The model is trained on the engineered features, and its performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Results
The model's performance is summarized, and key insights are drawn from the analysis. Feature importance is also analyzed to understand which features contribute most to the prediction.

## Neural Network Model

### Model Architecture
The neural network model is built using Keras with the following architecture:
- **Input Layer**: The input shape is determined by the number of features in the dataset.
- **Hidden Layers**:
  - Dense layer with 32 units and ReLU activation.
  - Dense layer with 16 units and ReLU activation.
  - Dense layer with 8 units and ReLU activation.
- **Output Layer**: Dense layer with 1 unit and sigmoid activation for binary classification.

### Model Compilation
The model is compiled with the following parameters:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy (since this is a binary classification problem)
- **Metrics**: Accuracy

### Training
The model is trained with the following parameters:
- **Epochs**: 100
- **Batch Size**: 1000
- **Class Weights**: Applied to handle class imbalance.
- **Early Stopping**: Implemented with a patience of 10 epochs, monitoring validation accuracy, and restoring the best weights.

### Results
The model achieved the following performance metrics on the test set:

| Metric     | Class 0 | Class 1 | Weighted Avg |
|------------|---------|---------|--------------|
| Precision  | 0.93    | 0.64    | 0.87         |
| Recall     | 0.88    | 0.77    | 0.86         |
| F1-Score   | 0.90    | 0.70    | 0.86         |
| Accuracy   |         |         | 0.86         |

### Confusion Matrix

|                | Predicted Non-Default | Predicted Default |
|----------------|-----------------------|-------------------|
| **Actual Non-Default** | 4472                  | 611               |
| **Actual Default**     | 330                   | 1102              |

### ROC Curve
The ROC curve shows the trade-off between true positive rate (TPR) and false positive rate (FPR) at different thresholds. The area under the ROC curve (AUC) is a measure of the model's ability to distinguish between the two classes.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-risk-assessment.git
   cd credit-risk-assessment
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook credit-risk-assessment.ipynb
   ```
3. Run the cells sequentially to perform data analysis and model training.

## Dependencies
To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, if you are using Jupyter Notebook, ensure you have the necessary libraries such as:
- TensorFlow
- Keras
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
