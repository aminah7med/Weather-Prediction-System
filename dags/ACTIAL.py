from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2

CSV_FILE = "/opt/airflow/dags/aiii_model/middle_east_weather_last_20_days.csv"

DB_CONN = {
    "host": "postgres",
    "port": 5432,
    "dbname": "airflow",
    "user": "airflow",
    "password": "airflow"
}
def load_csv_to_postgres():
    df = pd.read_csv(CSV_FILE)

    df = df.rename(columns={
        "avgtemperature": "temp",
        "maxtemperature": "temp_max",
        "mintemperature": "temp_min"
    })

    df = df[["city", "date", "temp", "temp_max", "temp_min"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date

    conn = psycopg2.connect(**DB_CONN)
    cur = conn.cursor()

    # 1️⃣ CREATE TABLE (لو مش موجود)
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS weather_actuals (
        city TEXT,
        date DATE,
        temp DOUBLE PRECISION,
        temp_max DOUBLE PRECISION,
        temp_min DOUBLE PRECISION,
        fetched_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (city, date)
    );
    """
    cur.execute(create_table_sql)

    # 2️⃣ INSERT / UPSERT
    insert_sql = """
        INSERT INTO weather_actuals (city, date, temp, temp_max, temp_min)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (city, date)
        DO UPDATE SET
            temp = EXCLUDED.temp,
            temp_max = EXCLUDED.temp_max,
            temp_min = EXCLUDED.temp_min,
            fetched_at = NOW();
    """

    for _, row in df.iterrows():
        cur.execute(
            insert_sql,
            (
                row["city"],
                row["date"],
                row["temp"],
                row["temp_max"],
                row["temp_min"]
            )
        )

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Data loaded into weather_actuals successfully")

with DAG(
    dag_id="load_weather_actuals_csv",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["weather", "postgres"]
) as dag:

    load_task = PythonOperator(
        task_id="load_csv_to_weather_actuals",
        python_callable=load_csv_to_postgres
    )

    load_task
