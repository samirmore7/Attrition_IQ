# AttritionIQ | Workforce Insights

### Employee Attrition Analytics & Machine Learning Prediction Platform

<p align="center">
  <a href="https://attritioniq-7.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Streamlit App"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white" alt="Snowflake"/>
  <img src="https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black" alt="Power BI"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
</p>

**AttritionIQ** is an end-to-end HR analytics and employee attrition prediction project that combines business intelligence, data engineering, exploratory data analysis, and machine learning to transform workforce data into actionable insights.

The project integrates a **Snowflake data warehouse, Python-based Random Forest prediction pipeline, Streamlit web application, and interactive Power BI dashboards** to help users understand attrition trends, investigate workforce patterns, and identify employees who may benefit from further HR review.

## Live Application

### [Launch AttritionIQ — Streamlit App](https://attritioniq-7.streamlit.app/#hr-attrition-intelligence)

Explore the deployed application for interactive workforce analytics, employee attrition predictions, risk indicators, and model insights.

**GitHub Repository:** [Attrition_IQ](https://github.com/samirmore7/Attrition_IQ)

## Project Highlights

- Analyzed a workforce dataset containing **4,700 employee records and 13 columns**.
- Integrated Snowflake as the source for HR analytics.
- Developed a Random Forest classification pipeline with preprocessing and SMOTE.
- Built an interactive Streamlit application for workforce analysis and attrition prediction.
- Created a three-page Power BI report for executive insights, attrition drivers, and risk segmentation.
- Evaluated different classification thresholds to study the precision–recall trade-off.
- Implemented employee segmentation using configurable business-defined risk factors.

## Dashboard Previews

The following images should be uploaded to the repository under the `screenshots/` folder. The README will display them once the filenames match the files you upload.

### 1. Streamlit — HR Attrition Intelligence

![Streamlit HR Attrition Dashboard](screenshots/streamlit-dashboard.png)

[Open the live Streamlit dashboard](https://attritioniq-7.streamlit.app/)

### 2. Power BI — Executive Overview

![Power BI Executive Overview](screenshots/powerbi-executive-overview.png)

Provides a high-level view of employee headcount, attrition rates, department comparisons, job-role trends, and workforce composition.

### 3. Power BI — Attrition Drivers

![Power BI Attrition Drivers](screenshots/powerbi-attrition-drivers.png)

Explores attrition patterns across overtime, age, income, distance from home, performance ratings, and job satisfaction.

### 4. Power BI — Risk Segmentation & Action List

![Power BI Risk Segmentation](screenshots/powerbi-risk-segmentation.png)

Segments employees based on defined risk factors and provides an action-list view for further HR review.

### 5. Streamlit — Employee Attrition Prediction

![Employee Attrition Prediction](screenshots/attrition-prediction.png)

Shows the employee prediction interface and its outputs. Use a synthetic example or appropriately anonymized information in screenshots.

> If any screenshot is not uploaded yet, upload it using the exact filename above or update the corresponding Markdown path.

---

## 1. Business Problem

Employee attrition can increase recruitment and onboarding costs, disrupt teams, and result in the loss of organizational knowledge.

HR teams need ways to explore workforce data, understand where attrition is concentrated, and investigate the characteristics associated with employees leaving an organization.

AttritionIQ addresses this challenge through three connected analytical capabilities:

1. **Descriptive analytics:** Understand workforce composition and historical attrition patterns.
2. **Predictive analytics:** Estimate attrition probability using a trained machine learning model.
3. **Business intelligence:** Explore attrition drivers and employee segments through interactive Power BI dashboards.

The objective is to support informed workforce planning and retention discussions—not to make automatic employment decisions.

## 2. Project Architecture

```text
                  HR Workforce Data
                         |
                         v
                 Snowflake Warehouse
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Python Data Analysis      Power BI Reports
             |
             v
      Data Preprocessing
             |
             v
      SMOTE + Random Forest
             |
             v
      Trained ML Pipeline
             |
             v
       Streamlit Application
             |
             v
      Attrition Predictions,
      Analytics & Risk Insights
```

Snowflake is the common data source for the analytics workflows. The Streamlit application hosts the interactive Python application, while Power BI provides a separate business intelligence report.

The Power BI report is a separate deliverable and is not automatically published through the Streamlit URL.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Data warehouse | Snowflake |
| Exploratory analysis | Pandas, Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Class imbalance | imbalanced-learn, SMOTE |
| Interactive visualization | Plotly |
| Web application | Streamlit |
| Business intelligence | Microsoft Power BI |
| Model persistence | Joblib / Pickle-compatible pipeline |
| Development and version control | VS Code, Git, GitHub |

## 4. Dataset Overview

The project uses the `HR_ATTRITION.RAW.HR_ATTRITION_FINAL` table in Snowflake.

### Dataset Summary

| Metric | Value |
|---|---:|
| Total employee records | 4,700 |
| Total columns | 13 |
| Employees who stayed | 3,341 |
| Employees who left | 1,359 |
| Overall attrition rate | 28.91% |
| Duplicate rows identified | 0 |

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
| `ATTRITION` | Target variable indicating whether an employee left |
| `MONTHLY_INCOME` | Monthly income |
| `YEARS_AT_COMPANY` | Employee tenure |
| `JOB_SATISFACTION` | Job satisfaction rating |

### Data Quality

Exploratory analysis identified missing values in:

- `PERFORMANCE_RATING`: 377 records, approximately 8.02%.
- `DISTANCE_FROM_HOME`: 379 records, approximately 8.06%.

The model pipeline handles missing numerical values through median imputation and missing categorical values through most-frequent imputation. The employee identifier is excluded from the model's input features.

## 5. Exploratory Data Analysis

The EDA workflow investigates workforce characteristics and their relationships with attrition.

Key analyses include:

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

These analyses describe patterns and associations in the available dataset. They do not establish that a particular characteristic causes an employee to leave.

## 6. Machine Learning — Random Forest

### Objective

The prediction model estimates the probability that an employee belongs to the attrition class based on the available employee attributes.

The target variable is `ATTRITION`, encoded as:

- `No` → `0`
- `Yes` → `1`

### Model Pipeline

The machine learning workflow consists of:

1. Loading the employee dataset from Snowflake.
2. Excluding `EMP_ID` from the predictive features.
3. Splitting the data into training and testing sets using an 80/20 stratified split.
4. Imputing missing numerical values with the median.
5. Imputing missing categorical values with the most frequent category.
6. Encoding categorical variables using one-hot encoding.
7. Applying SMOTE to the training data to address class imbalance.
8. Training a Random Forest classifier.
9. Evaluating predictions using classification metrics.
10. Comparing probability thresholds.
11. Saving the trained pipeline and evaluation outputs.

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

The preprocessing steps and classifier are packaged into a pipeline so that the same transformations can be applied during prediction.

### Model Evaluation

The initial Random Forest model, evaluated at the default classification threshold, achieved the following recorded results:

| Metric | Score |
|---|---:|
| Accuracy | 68.62% |
| Precision | 40.00% |
| Recall | 16.91% |
| F1-score | 23.77% |
| ROC-AUC | 0.6418 |

### Threshold Optimization

A separate threshold evaluation explored how changing the classification threshold affects attrition detection.

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.20 | 42.02% | 32.34% | 91.91% | 47.85% |
| **0.25** | **45.96%** | **33.79%** | **90.44%** | **49.20%** |
| 0.30 | 50.32% | 34.93% | 83.09% | 49.18% |
| 0.35 | 56.91% | 36.94% | 69.12% | 48.14% |
| 0.40 | 65.96% | 42.55% | 50.37% | 46.13% |
| 0.45 | 71.06% | 50.00% | 37.50% | 42.86% |
| 0.50 | 70.74% | 48.92% | 25.00% | 33.09% |

The threshold of **0.25** was selected for the Streamlit application's binary attrition classification because it produced the highest F1-score among the tested thresholds.

At this threshold, recall is high, meaning the model identifies a large proportion of employees who left in the evaluation data. However, precision and accuracy are lower than at some higher thresholds, so more employees who stay may also be flagged.

The choice of threshold should depend on the business cost of missed attrition versus unnecessary follow-up. These results should be confirmed using a held-out evaluation dataset before operational use.

**Evaluation note:** The threshold-optimization figures are reported from the project's recorded evaluation output. They should not be described as independent final test performance unless the threshold was selected on validation data and evaluated on a separate test set.

## 7. Streamlit Application

The Streamlit application provides an interactive interface for exploring workforce data and using the trained prediction pipeline.

### Executive Overview

- Total employee count.
- Employee attrition metrics.
- Workforce summary charts.
- Department and job-role insights.

### Attrition Analytics

- Attrition by overtime.
- Attrition by department and job role.
- Demographic and workforce comparisons.
- Interactive filters and visualizations.

### Predict Attrition

- Enter or select employee attributes.
- Generate an attrition probability using the trained pipeline.
- Classify the prediction using the configured threshold.
- Display a risk category and relevant risk indicators.

### Employee Explorer

- Filter employee records.
- Explore selected employee attributes.
- Export available filtered data where supported.

### Model Intelligence

- View model evaluation metrics.
- Review feature-importance information.
- Explore threshold and model insights.

The application uses a separate descriptive risk categorization from the binary classification threshold. A risk category shown in the dashboard should therefore not be interpreted as the same thing as the model's binary prediction.

**Live application:** [https://attritioniq-7.streamlit.app/](https://attritioniq-7.streamlit.app/)

## 8. Power BI Dashboard

The Power BI report is named **AttritionIQ | Workforce Insights** and contains three pages.

### Page 1 — Executive Overview

Provides a high-level view of workforce composition and employee attrition.

Includes:

- Total employee count.
- Employees who stayed and left.
- Overall attrition rate.
- Attrition by department.
- Employees leaving by job role.
- Attrition distribution.
- Tenure-band analysis.
- Decomposition tree for exploring attrition drivers.
- Interactive workforce slicers.

### Page 2 — Attrition Drivers

Investigates how attrition rates differ across selected employee attributes.

Includes:

- Attrition by overtime.
- Attrition by distance-from-home band.
- Attrition by age group.
- Attrition by income band.
- Attrition by performance rating.
- Attrition by job satisfaction.
- Department-level comparison.
- Key Influencers analysis.

### Page 3 — Risk Segmentation & Action List

Uses a defined set of business rules to segment employees for further review.

The configured risk-factor score adds one point for each of the following conditions:

```dax
Risk Factors =
IF(HR_ATTRITION_FINAL[OVERTIME] = "Yes", 1, 0)
+ IF(HR_ATTRITION_FINAL[MONTHLY_INCOME] < 45000, 1, 0)
+ IF(HR_ATTRITION_FINAL[DISTANCE_FROM_HOME] > 20, 1, 0)
+ IF(HR_ATTRITION_FINAL[YEARS_AT_COMPANY] < 2, 1, 0)
+ IF(HR_ATTRITION_FINAL[JOB_SATISFACTION] <= 2, 1, 0)
```

The score ranges from 0 to 5 and represents the number of configured conditions met.

The page includes:

- Attrition rate by risk-factor count.
- High-risk employee counts based on the configured criteria.
- Active employees meeting selected risk criteria.
- An employee action-list table.
- Department and job-role filters.

These rules are descriptive business criteria, not the Random Forest model's learned feature weights. Meeting several conditions does not prove that an employee will leave.

**Power BI file:** `HR_Attrition_Dashboard.pbix`

The report can be opened in Power BI Desktop. Online publication and refresh require separate configuration.

## 9. Project Structure

The following represents the local development structure. Only files actually committed to GitHub will be available to repository visitors.

```text
Attrition_IQ/
│
├── data/
│
├── models/
│   └── attrition_model.pkl
│
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
│
├── screenshots/
│   ├── streamlit-dashboard.png
│   ├── powerbi-executive-overview.png
│   ├── powerbi-attrition-drivers.png
│   ├── powerbi-risk-segmentation.png
│   └── attrition-prediction.png
│
├── src/
│   ├── connect_snowflake.py
│   ├── eda.py
│   └── train_model.py
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `.env` file, virtual environment, private datasets, and other sensitive or unnecessary files should be excluded from GitHub.

## 10. Run Locally

### Prerequisites

- Python compatible with the installed dependencies.
- Snowflake access to the required database and table.
- Git.
- Power BI Desktop for the Power BI report.

### Step 1 — Clone the repository

```bash
git clone https://github.com/samirmore7/Attrition_IQ.git
cd Attrition_IQ
```

### Step 2 — Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Configure Snowflake

Create a local `.env` file in the project root using the exact variable names expected by `src/connect_snowflake.py`.

Example configuration:

```env
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=HR_ATTRITION
SNOWFLAKE_SCHEMA=RAW
SNOWFLAKE_ROLE=your_role
```

Replace these placeholders with your own configuration. Do not commit credentials to GitHub.

### Step 5 — Run exploratory data analysis

```bash
python src/eda.py
```

### Step 6 — Train the model

```bash
python src/train_model.py
```

This regenerates the model and supported evaluation outputs. Retraining is not required simply to launch the application if a compatible trained model is already available.

### Step 7 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open the local address printed in the terminal, typically `http://localhost:8501`.

The application requires valid Snowflake configuration for database-dependent functionality.

## 11. Deployment

The Streamlit application is deployed at:

**[https://attritioniq-7.streamlit.app/](https://attritioniq-7.streamlit.app/)**

The deployed application requires its configured Python dependencies, compatible saved model, and database secrets.

Deployment configuration should keep Snowflake credentials in the hosting platform's secrets manager rather than in source code.

The Power BI report is maintained separately. Its online availability and refresh behavior depend on its publication method, connection mode, credentials, and any required gateway or licensing configuration.

## 12. Security, Privacy & Responsible AI

This project uses employee-related data and predictive outputs, so appropriate safeguards are essential.

- Never commit passwords, tokens, or the `.env` file.
- Avoid publishing personally identifiable employee information.
- Use least-privilege permissions for Snowflake access.
- Use anonymized or synthetic examples in public screenshots.
- Treat predicted probabilities as estimates rather than certainty.
- Evaluate model fairness and performance across relevant employee groups before operational use.
- Use risk scores to support constructive human review, not as the sole basis for employment decisions.

## 13. Limitations & Future Enhancements

Potential improvements include:

- Evaluate the model using a dedicated validation set and an untouched final test set.
- Tune hyperparameters and improve precision–recall balance.
- Assess probability calibration and prediction stability.
- Investigate potential bias across employee groups.
- Add automated data-quality checks.
- Add automated testing and continuous deployment.
- Monitor model performance as new data becomes available.
- Improve dashboard documentation and add more visual examples.
- Configure and document Power BI refresh behavior.
- Evaluate the business impact of retention interventions.

## 14. Key Learnings

This project demonstrates practical experience in:

- Connecting Python applications to a cloud data warehouse.
- Performing exploratory data analysis.
- Building a machine learning preprocessing and classification pipeline.
- Handling class imbalance with SMOTE.
- Evaluating a classifier using multiple metrics.
- Investigating classification threshold trade-offs.
- Developing interactive applications with Streamlit.
- Building business intelligence reports in Power BI.
- Combining predictive analytics with business-defined risk indicators.
- Documenting and deploying a data science project.

## Author

**Samir More**

- GitHub: [samirmore7](https://github.com/samirmore7)
- Project repository: [Attrition_IQ](https://github.com/samirmore7/Attrition_IQ)
- Live application: [AttritionIQ — Streamlit](https://attritioniq-7.streamlit.app/)

---

*AttritionIQ — Turning workforce data into actionable insights.*
