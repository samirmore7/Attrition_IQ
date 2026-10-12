# AttritionIQ | Workforce Insights

### HR Analytics, Employee Attrition Prediction & Workforce Intelligence

<p align="center">
  <a href="https://attritioniq-7.streamlit.app/#hr-attrition-intelligence">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Streamlit App"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white" alt="Snowflake"/>
  <img src="https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black" alt="Power BI"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
</p>

**AttritionIQ** is an end-to-end HR analytics and employee attrition prediction project that combines data analytics, machine learning, cloud data warehousing, and business intelligence.

The project uses **Snowflake, Python, Random Forest, Streamlit, and Microsoft Power BI** to explore employee attrition patterns, investigate workforce trends, estimate attrition probabilities, and identify employee segments that may benefit from further HR review.

## Live Demo

### [Launch AttritionIQ — Live Streamlit Application](https://attritioniq-7.streamlit.app/#hr-attrition-intelligence)

**GitHub Repository:** [Attrition_IQ](https://github.com/samirmore7/Attrition_IQ)

## Table of Contents

- [Project Overview](#project-overview)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Dataset Overview](#dataset-overview)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Model](#machine-learning-model)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)
- [Power BI Dashboard](#power-bi-dashboard)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Run the Application](#run-the-application)
- [Security and Responsible Use](#security-and-responsible-use)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

Employee attrition can increase recruitment costs, disrupt teams, and result in the loss of organizational knowledge. Organizations can use workforce analytics to understand historical turnover patterns and investigate the factors associated with employees leaving.

AttritionIQ combines three analytical approaches:

1. **Descriptive analytics:** Understand workforce composition and historical attrition patterns.
2. **Predictive analytics:** Estimate attrition probability using a trained Random Forest classification pipeline.
3. **Business intelligence:** Explore attrition drivers and workforce segments through interactive Power BI dashboards.

The project demonstrates an end-to-end workflow, from retrieving workforce data and preparing features to building a machine learning model and presenting results through interactive dashboards.

Predictions are estimates intended to support human-led analysis. They do not guarantee that an employee will leave.

## Dashboard Screenshots

The following screenshots show the three pages of the Power BI report.

**Important:** Upload these images to a folder named `screenshots` in the root of your GitHub repository, using the exact filenames shown below.

### 1. Executive Overview

![AttritionIQ Executive Overview](screenshots/01_Executive_Overview.png)

The Executive Overview provides a consolidated view of workforce metrics, including:

- Total employees.
- Employees retained and employees who left.
- Overall attrition rate.
- Average monthly income.
- Attrition rate by department.
- Employees leaving by job role.
- Employee distribution and attrition rate by tenure band.
- Employee attrition split.
- Decomposition tree for exploring attrition drivers.
- Interactive employee and department filters.

### 2. Understanding Employee Attrition

![AttritionIQ Attrition Drivers](screenshots/02_Attrition_Drivers.png)

This page investigates attrition patterns across multiple employee attributes:

- Overtime status.
- Distance from home.
- Age group.
- Monthly income band.
- Performance rating.
- Job satisfaction.
- Department.
- Key Influencers analysis.

These comparisons help users investigate patterns associated with attrition. They do not establish causation.

### 3. Risk Segmentation & Action List

![AttritionIQ Risk Segmentation](screenshots/03_Risk_Segmentation.png)

This page provides a rule-based employee segmentation view with:

- Attrition rate by number of defined risk factors.
- High-risk employee count.
- Active high-risk employee count.
- Employee action-list table.
- Department and job-role filters.
- Selected employee attributes for further review.

The Power BI risk-factor score is based on manually defined business rules and is separate from the Random Forest model's predicted probability.

### Live Prediction Application

Explore the interactive application:

**[Open AttritionIQ on Streamlit](https://attritioniq-7.streamlit.app/#hr-attrition-intelligence)**

The Streamlit application provides workforce analytics and an employee attrition prediction interface. The Power BI report is a separate deliverable and is not hosted at the Streamlit URL.

---

## Key Features

- Snowflake integration for workforce data retrieval.
- Exploratory data analysis and data-quality checks.
- Random Forest classification pipeline.
- Numerical and categorical missing-value imputation.
- One-hot encoding for categorical features.
- SMOTE for addressing class imbalance during model training.
- Classification threshold evaluation.
- Interactive Streamlit analytics and prediction interface.
- Power BI executive dashboard and attrition-driver analysis.
- Rule-based employee risk segmentation.
- Model evaluation metrics and visualization outputs.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Cloud data warehouse | Snowflake |
| Machine learning | Scikit-learn, Random Forest |
| Class imbalance handling | imbalanced-learn, SMOTE |
| Data visualization | Matplotlib, Seaborn, Plotly |
| Web application | Streamlit |
| Business intelligence | Microsoft Power BI |
| Model persistence | Joblib / serialized scikit-learn pipeline |
| Development tools | VS Code, Git, GitHub |

## Dataset Overview

The project uses the Snowflake table:

`HR_ATTRITION.RAW.HR_ATTRITION_FINAL`

### Dataset Summary

| Metric | Value |
|---|---:|
| Total employee records | 4,700 |
| Total columns | 13 |
| Employees who stayed | 3,341 |
| Employees who left | 1,359 |
| Overall attrition rate | 28.91% |
| Duplicate rows identified during EDA | 0 |

### Dataset Features

| Column | Description |
|---|---|
| `EMP_ID` | Employee identifier |
| `AGE` | Employee age |
| `DEPARTMENT` | Department |
| `JOB_ROLE` | Job role |
| `GENDER` | Gender |
| `MARITAL_STATUS` | Marital status |
| `PERFORMANCE_RATING` | Performance rating |
| `OVERTIME` | Overtime status |
| `DISTANCE_FROM_HOME` | Distance from home |
| `ATTRITION` | Target indicating whether an employee left |
| `MONTHLY_INCOME` | Monthly income |
| `YEARS_AT_COMPANY` | Employee tenure |
| `JOB_SATISFACTION` | Job satisfaction rating |

### Data Quality

Exploratory analysis identified missing values in:

- `PERFORMANCE_RATING`: 377 records, approximately 8.02%.
- `DISTANCE_FROM_HOME`: 379 records, approximately 8.06%.

The machine learning pipeline handles missing numerical values using median imputation and missing categorical values using most-frequent imputation.

The employee identifier, `EMP_ID`, is excluded from the model's predictive features.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand workforce characteristics and investigate patterns associated with attrition.

The analysis includes:

- Overall attrition distribution.
- Attrition by department.
- Attrition by overtime status.
- Salary comparisons between attrition groups.
- Missing-value analysis.
- Numerical summary statistics.
- Visualizations for reporting and interpretation.

Example output files:

```text
outputs/
├── attrition_by_department.png
├── attrition_by_overtime.png
└── salary_vs_attrition.png
```

The analysis helps describe relationships in the dataset but does not prove that any single employee attribute causes attrition.

## Machine Learning Model

### Objective

The objective is to estimate the probability that an employee belongs to the attrition class using the available workforce attributes.

The target variable is encoded as:

- `No` → `0`
- `Yes` → `1`

### Model Pipeline

The workflow consists of the following steps:

1. Retrieve workforce data from Snowflake.
2. Exclude the employee identifier from the predictive features.
3. Split the dataset into training and testing subsets using an 80/20 stratified split.
4. Impute missing numerical values using the median.
5. Impute missing categorical values using the most frequent category.
6. Encode categorical variables using one-hot encoding.
7. Apply SMOTE within the training pipeline to address class imbalance.
8. Train a Random Forest classifier.
9. Evaluate classification performance.
10. Compare classification probability thresholds.
11. Save the trained pipeline and evaluation outputs.
12. Integrate the model into the Streamlit prediction application.

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

The trained preprocessing and classification pipeline is saved as:

`models/attrition_model.pkl`

The application must load a compatible saved pipeline to generate predictions.

## Model Evaluation

The initial Random Forest evaluation at the default classification threshold produced these recorded results:

| Metric | Score |
|---|---:|
| Accuracy | 68.62% |
| Precision | 40.00% |
| Recall | 16.91% |
| F1-score | 23.77% |
| ROC-AUC | 0.6418 |

### Classification Threshold Optimization

Different classification thresholds were evaluated to study the trade-off between detecting employees who left and generating false-positive predictions.

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.20 | 42.02% | 32.34% | 91.91% | 47.85% |
| **0.25** | **45.96%** | **33.79%** | **90.44%** | **49.20%** |
| 0.30 | 50.32% | 34.93% | 83.09% | 49.18% |
| 0.35 | 56.91% | 36.94% | 69.12% | 48.14% |
| 0.40 | 65.96% | 42.55% | 50.37% | 46.13% |
| 0.45 | 71.06% | 50.00% | 37.50% | 42.86% |
| 0.50 | 70.74% | 48.92% | 25.00% | 33.09% |

A threshold of **0.25** was selected for the Streamlit application's binary classification decision because it achieved the highest F1-score among the tested thresholds.

At this threshold, recall is high, meaning the model identifies a large proportion of employees who left in the evaluation data. However, precision and accuracy are lower than at some higher thresholds, so some employees who stayed may also be flagged.

The threshold should be selected according to the business cost of missed attrition versus unnecessary follow-up. For reliable final reporting, threshold selection should use validation data and final metrics should be measured on a separate held-out test set.

The results above reflect the recorded project evaluation and should not be described as independent final test performance unless a separate final test set was used.

The Streamlit risk categories and the Power BI risk-factor score are separate from the Random Forest's binary classification threshold.

## Streamlit Application

The deployed Streamlit application provides an interactive interface for workforce analytics and prediction.

Its features include:

### Executive Overview
- Workforce KPIs.
- Employee retention and attrition summaries.
- Department and job-role visualizations.

### Attrition Analytics
- Attrition comparisons by employee attributes.
- Interactive filters.
- Workforce trend visualizations.

### Predict Attrition
- Employee attribute inputs.
- Model-generated attrition probability.
- Binary classification using the configured threshold.
- Risk indicators to support further review.

### Employee Explorer
- Filtered employee data exploration.
- Employee attribute analysis.
- Data export where supported by the application.

### Model Intelligence
- Model performance metrics.
- Feature-importance information.
- Threshold and model insights.

**Live application:** [https://attritioniq-7.streamlit.app/](https://attritioniq-7.streamlit.app/)

Database-dependent features require the configured Snowflake connection, and prediction functionality requires a compatible saved model.

## Power BI Dashboard

The Power BI report is titled **AttritionIQ | Workforce Insights** and contains three pages.

### Page 1: Executive Overview

Provides a consolidated view of workforce metrics, department-level attrition, employees leaving by job role, tenure analysis, attrition distribution, and interactive filters.

### Page 2: Understanding Employee Attrition

Investigates attrition patterns across overtime, distance from home, age group, income band, performance rating, job satisfaction, and department. It also includes Key Influencers analysis.

### Page 3: Risk Segmentation & Action List

Segments employees using a manually defined risk-factor score and displays risk-group attrition rates, employee counts, and a filtered employee action list.

### Business-Defined Risk Factors

The Power BI score adds one point for each of the following conditions:

```dax
Risk Factors =
IF(HR_ATTRITION_FINAL[OVERTIME] = "Yes", 1, 0)
+ IF(HR_ATTRITION_FINAL[MONTHLY_INCOME] < 45000, 1, 0)
+ IF(HR_ATTRITION_FINAL[DISTANCE_FROM_HOME] > 20, 1, 0)
+ IF(HR_ATTRITION_FINAL[YEARS_AT_COMPANY] < 2, 1, 0)
+ IF(HR_ATTRITION_FINAL[JOB_SATISFACTION] <= 2, 1, 0)
```

The score ranges from 0 to 5 and represents the number of configured conditions met.

This is a business-rule-based score, not the Random Forest's learned prediction score. Meeting several conditions does not prove that an employee will leave.

The Power BI report is saved as:

`HR_Attrition_Dashboard.pbix`

It can be opened in Power BI Desktop. Online publication and data refresh require separate configuration and credentials.

## Project Structure

The local development setup includes the following files. The GitHub repository may contain a subset, depending on which files have been committed.

```text
Attrition_IQ/
├── data/
├── models/
│   └── attrition_model.pkl
├── outputs/
│   ├── attrition_by_department.png
│   ├── attrition_by_overtime.png
│   ├── salary_vs_attrition.png
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_metrics.csv
│   ├── threshold_optimization_results.csv
│   ├── final_threshold_config.csv
│   ├── final_random_forest_confusion_matrix.png
│   └── threshold_vs_f1.png
├── screenshots/
│   ├── 01_Executive_Overview.png
│   ├── 02_Attrition_Drivers.png
│   └── 03_Risk_Segmentation.png
├── src/
│   ├── connect_snowflake.py
│   ├── eda.py
│   └── train_model.py
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

Do not commit `.env`, credentials, private employee data, or the virtual environment. Include the Power BI `.pbix` file only if you intend to share it.

## Installation and Setup

### Prerequisites

- Python compatible with the dependencies listed in `requirements.txt`.
- A Snowflake account with access to the required database and table.
- Git.
- Power BI Desktop if you want to open the Power BI report.

### 1. Clone the Repository

```bash
git clone https://github.com/samirmore7/Attrition_IQ.git
cd Attrition_IQ
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Snowflake

Create a local `.env` file using the exact variable names expected by `src/connect_snowflake.py`.

For example, if your connection script uses these variable names:

```env
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=HR_ATTRITION
SNOWFLAKE_SCHEMA=RAW
SNOWFLAKE_ROLE=your_role
```

Replace the placeholders with your own Snowflake configuration. Never commit this file or publish credentials.

### 5. Run Exploratory Data Analysis

```bash
python src/eda.py
```

### 6. Train the Machine Learning Model

```bash
python src/train_model.py
```

This regenerates the saved model and evaluation outputs. Retraining is not necessary to launch the application if a compatible trained model is already available.

### 7. Launch Streamlit

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

The application requires valid Snowflake configuration for database-dependent features.

## Security and Responsible Use

- Keep credentials and `.env` files out of source control.
- Use least-privilege access to Snowflake.
- Avoid publishing personally identifiable employee information.
- Use synthetic or appropriately anonymized examples in public screenshots.
- Treat model probabilities as estimates, not certainties.
- Assess fairness, privacy, and model performance before operational use.
- Do not use predictions or risk scores as the sole basis for employment decisions.

## Limitations and Future Improvements

- Validate the classification threshold on a dedicated validation set.
- Evaluate final performance on a separate held-out test set.
- Improve the precision–recall balance.
- Assess probability calibration and prediction stability.
- Evaluate performance across relevant employee groups.
- Add automated data-quality checks and software tests.
- Monitor model performance as new data becomes available.
- Document Power BI refresh and publication configuration.
- Add a screenshot of the Streamlit prediction interface.

## Author

**Samir More**

- GitHub: [@samirmore7](https://github.com/samirmore7)
- Project Repository: [Attrition_IQ](https://github.com/samirmore7/Attrition_IQ)
- Live Application: [AttritionIQ on Streamlit](https://attritioniq-7.streamlit.app/)

---

*AttritionIQ — Turning workforce data into actionable insights.*
