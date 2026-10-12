import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Create and return a Snowflake connection."""

    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

    return conn


if __name__ == "__main__":

    conn = None
    cursor = None

    try:
        # --------------------------------------------------
        # 1. Connect to Snowflake
        # --------------------------------------------------

        conn = get_connection()

        print("Successfully connected to Snowflake!")

        cursor = conn.cursor()

        # --------------------------------------------------
        # 2. Check Snowflake version
        # --------------------------------------------------

        cursor.execute("SELECT CURRENT_VERSION()")

        result = cursor.fetchone()

        print("\nSnowflake version:", result[0])

        # --------------------------------------------------
        # 3. Show tables in HR_ATTRITION.RAW
        # --------------------------------------------------

        cursor.execute("SHOW TABLES")

        print("\nTables in HR_ATTRITION.RAW:\n")

        tables = cursor.fetchall()

        for row in tables:

            if row[2] == "HR_ATTRITION" and row[3] == "RAW":
                print(row[1])

        # --------------------------------------------------
        # 4. Describe HR Attrition Final table
        # --------------------------------------------------

        table_name = "HR_ATTRITION.RAW.HR_ATTRITION_FINAL"

        cursor.execute(f"DESCRIBE TABLE {table_name}")

        print(f"\nColumns in {table_name}:\n")

        columns = cursor.fetchall()

        for row in columns:

            print(
                f"Column: {row[0]} | "
                f"Type: {row[1]} | "
                f"Nullable: {row[3]}"
            )

        # --------------------------------------------------
        # 5. Preview sample records
        # --------------------------------------------------

        cursor.execute(f"""
            SELECT *
            FROM {table_name}
            LIMIT 10
        """)

        print("\nSample records:\n")

        rows = cursor.fetchall()

        for row in rows:
            print(row)

        # --------------------------------------------------
        # 6. Success message
        # --------------------------------------------------

        print("\nDatabase inspection completed successfully!")

    except Exception as e:

        print("\nConnection or query failed:")
        print(e)

    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()

        print("\nSnowflake connection closed.")