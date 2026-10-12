import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from connect_snowflake import get_connection


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

TABLE_NAME = "HR_ATTRITION.RAW.HR_ATTRITION_FINAL"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs"
)


# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load data from Snowflake
# --------------------------------------------------

def load_data():
    """Load HR Attrition data from Snowflake."""

    conn = None
    cursor = None

    try:
        conn = get_connection()

        query = f"""
            SELECT *
            FROM {TABLE_NAME}
        """

        cursor = conn.cursor()

        cursor.execute(query)

        data = cursor.fetchall()

        columns = [column[0] for column in cursor.description]

        df = pd.DataFrame(data, columns=columns)

        return df

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


# --------------------------------------------------
# Basic Data Quality Analysis
# --------------------------------------------------

def data_quality_report(df):
    """Print basic data quality information."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    # Missing values
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    missing_report = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_percentage
    })

    print(missing_report)

    # Duplicate rows
    print("\n" + "=" * 60)
    print("DUPLICATES")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    # Data types
    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes)

    # Numerical statistics
    print("\n" + "=" * 60)
    print("NUMERICAL SUMMARY")
    print("=" * 60)

    print(df.describe())


# --------------------------------------------------
# Attrition Analysis
# --------------------------------------------------

def attrition_analysis(df):
    """Analyze target variable."""

    print("\n" + "=" * 60)
    print("ATTRITION DISTRIBUTION")
    print("=" * 60)

    attrition_counts = df["ATTRITION"].value_counts()

    print(attrition_counts)

    print("\nAttrition percentage:")

    attrition_percentage = (
        df["ATTRITION"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(attrition_percentage)


# --------------------------------------------------
# Chart 1: Attrition by Department
# --------------------------------------------------

def plot_attrition_by_department(df):

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="DEPARTMENT",
        hue="ATTRITION"
    )

    plt.title("Employee Attrition by Department")
    plt.xlabel("Department")
    plt.ylabel("Number of Employees")

    plt.xticks(rotation=30)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "attrition_by_department.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# Chart 2: Attrition by Overtime
# --------------------------------------------------

def plot_attrition_by_overtime(df):

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=df,
        x="OVERTIME",
        hue="ATTRITION"
    )

    plt.title("Employee Attrition by Overtime")
    plt.xlabel("Overtime")
    plt.ylabel("Number of Employees")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "attrition_by_overtime.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# Chart 3: Salary vs Attrition
# --------------------------------------------------

def plot_salary_vs_attrition(df):

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="ATTRITION",
        y="MONTHLY_INCOME"
    )

    plt.title("Monthly Income vs Employee Attrition")
    plt.xlabel("Attrition")
    plt.ylabel("Monthly Income")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "salary_vs_attrition.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# Main EDA Pipeline
# --------------------------------------------------

def main():

    print("=" * 60)
    print("HR ATTRITION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data from Snowflake...")

    df = load_data()

    print("Data loaded successfully!")

    # Basic information
    data_quality_report(df)

    # Attrition analysis
    attrition_analysis(df)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_attrition_by_department(df)

    plot_attrition_by_overtime(df)

    plot_salary_vs_attrition(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()