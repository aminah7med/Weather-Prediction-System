
# """
# Weather Prediction Pipeline - FINAL VERSION
# Author: Ahmed Sami
# Improvements:
# 1. Data Cleaning (IQR outliers removal)
# 2. XGBoost Model (better than RF)
# 3. Enhanced Features (7-day lags, statistics)
# 4. API Retry for failed cities (1 retry after 5s)
# 5. Fixed bugs in daily_predict_api
# """

# import os
# import time
# from datetime import datetime, timedelta
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import joblib
# import numpy as np
# import pandas as pd
# import psycopg2
# import requests
# import xgboost as xgb

# from airflow import DAG
# from airflow.hooks.base import BaseHook
# from airflow.models import Variable
# from airflow.operators.python import PythonOperator, BranchPythonOperator
# from airflow.operators.dummy import DummyOperator

# # =========================================
# # CONFIG
# # =========================================

# BASE_PATH = "/opt/airflow/dags/aiii_model"
# RAW_CSV = os.path.join(BASE_PATH, "city_temperature.csv")
# CLEAN_CSV = os.path.join(BASE_PATH, "clean_data.csv")
# TRANSFORMED_CSV = os.path.join(BASE_PATH, "transformed_data.csv")
# MODELS_DIR = "/opt/airflow/models"

# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(BASE_PATH, exist_ok=True)

# POSTGRES_CONN_ID = "ml_con"
# WEATHER_API_KEY = "bd75bd2d767f720269c389aa29da6be9"
# MAX_WORKERS = 10


# def _get_pg_conn():
#     """Get PostgreSQL connection parameters"""
#     conn = BaseHook.get_connection(POSTGRES_CONN_ID)
#     return {
#         "host": conn.host,
#         "dbname": conn.schema,
#         "user": conn.login,
#         "password": conn.password,
#         "port": conn.port or 5432
#     }


# # =========================================
# # BRANCH DECISION
# # =========================================

# def check_day_one():
#     """Check if Day 1 training already completed"""
#     flag = Variable.get("day_one_done", default_var="no")
    
#     if flag == "no":
#         print("🎯 Running DAY 1 - Initial Training")
#         return "extract_csv"
#     else:
#         print("✅ DAY 1 completed - Running DAY 2 - Daily Prediction")
#         return "skip_day1"


# # =========================================
# # DAY 1 — TRAINING TASKS (WITH CLEANING)
# # =========================================

# def extract_csv(**context):
#     """Extract and clean raw temperature data WITH OUTLIER REMOVAL"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 1: EXTRACTING & CLEANING DATA")
#     print("=" * 70)
    
#     if not os.path.exists(RAW_CSV):
#         raise FileNotFoundError(f"❌ Raw CSV not found: {RAW_CSV}")
    
#     print(f"📂 Reading: {RAW_CSV}")
#     df = pd.read_csv(RAW_CSV)
#     initial_count = len(df)
#     print(f"📊 Initial shape: {df.shape}")
    
#     # Clean column names
#     df.columns = df.columns.str.lower().str.replace(" ", "_")
    
#     # Convert Fahrenheit → Celsius
#     print("🌡️  Converting Fahrenheit to Celsius...")
#     df["avgtemperature"] = (df["avgtemperature"] - 32) * 5/9
    
#     # Remove invalid temperatures
#     df["avgtemperature"] = df["avgtemperature"].replace([-99], np.nan)
#     df = df.dropna(subset=["avgtemperature"])
    
#     # Drop state column
#     if "state" in df.columns:
#         df = df.drop(columns=["state"])
    
#     # Create date column
#     df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors='coerce')
#     df = df.dropna(subset=["date"])
    
#     # Sort and deduplicate
#     df = df.sort_values(["city", "date"]).drop_duplicates(subset=["city", "date"])
    
#     # ===== DATA CLEANING: Remove outliers per city =====
#     print("🧹 Cleaning outliers using IQR method...")
    
#     # 1. Remove impossible temperatures
#     df = df[(df['avgtemperature'] >= -50) & (df['avgtemperature'] <= 50)]
#     removed_range = initial_count - len(df)
    
#     # 2. Remove outliers per city using IQR
#     clean_list = []
    
#     for city, group in df.groupby('city'):
#         Q1 = group['avgtemperature'].quantile(0.25)
#         Q3 = group['avgtemperature'].quantile(0.75)
#         IQR = Q3 - Q1
        
#         # Use 3 * IQR (more conservative for temperature)
#         lower_bound = Q1 - 3 * IQR
#         upper_bound = Q3 + 3 * IQR
        
#         group_clean = group[
#             (group['avgtemperature'] >= lower_bound) & 
#             (group['avgtemperature'] <= upper_bound)
#         ]
        
#         clean_list.append(group_clean)
    
#     df = pd.concat(clean_list, ignore_index=True)
#     removed_outliers = initial_count - removed_range - len(df)
    
#     # Save
#     df.to_csv(CLEAN_CSV, index=False)
    
#     print(f"✅ Cleaning complete:")
#     print(f"   ❌ Out of range: {removed_range:,} ({removed_range/initial_count*100:.1f}%)")
#     print(f"   ❌ Outliers: {removed_outliers:,} ({removed_outliers/initial_count*100:.1f}%)")
#     print(f"   ✅ Remaining: {len(df):,} ({len(df)/initial_count*100:.1f}%)")
#     print(f"💾 Saved to: {CLEAN_CSV}")
#     print("=" * 70 + "\n")


# def transform_data(**context):
#     """Create enhanced features with 7-day window"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 1: ENHANCED FEATURE ENGINEERING")
#     print("=" * 70)
    
#     df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
#     print(f"📊 Input shape: {df.shape}")
    
#     final_list = []
    
#     for city, g in df.groupby("city"):
#         g = g.sort_values("date").reset_index(drop=True)
        
#         if len(g) < 14:
#             continue
        
#         # Enhanced Features
#         g["temp_lag1"] = g["avgtemperature"].shift(1)
#         g["temp_lag2"] = g["avgtemperature"].shift(2)
#         g["temp_lag3"] = g["avgtemperature"].shift(3)
#         g["temp_lag7"] = g["avgtemperature"].shift(7)
        
#         g["temp_ma3"] = g["avgtemperature"].shift(1).rolling(3).mean()
#         g["temp_ma7"] = g["avgtemperature"].shift(1).rolling(7).mean()
#         g["temp_std7"] = g["avgtemperature"].shift(1).rolling(7).std()
        
#         g["temp_change"] = g["avgtemperature"].shift(1) - g["avgtemperature"].shift(2)
        
#         g["temp_min7"] = g["avgtemperature"].shift(1).rolling(7).min()
#         g["temp_max7"] = g["avgtemperature"].shift(1).rolling(7).max()
#         g["temp_range7"] = g["temp_max7"] - g["temp_min7"]
        
#         g["day_of_week"] = g["date"].dt.dayofweek
#         g["month_num"] = g["date"].dt.month
#         g["year_num"] = g["date"].dt.year
#         g["day_of_year"] = g["date"].dt.dayofyear
        
#         g["season"] = g["month_num"].apply(lambda m: 
#             1 if m in [12, 1, 2] else 2 if m in [3, 4, 5] else 3 if m in [6, 7, 8] else 4
#         )
        
#         g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        
#         g = g.dropna().reset_index(drop=True)
        
#         if len(g) > 0:
#             final_list.append(g)
    
#     final_df = pd.concat(final_list, ignore_index=True)
#     final_df.to_csv(TRANSFORMED_CSV, index=False)
    
#     print(f"✅ Transformed shape: {final_df.shape}")
#     print(f"💾 Saved to: {TRANSFORMED_CSV}")
#     print("=" * 70 + "\n")


# def train_and_evaluate(**context):
#     """Train XGBoost model (best accuracy from tests)"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 1: TRAINING XGBOOST MODEL")
#     print("=" * 70)
    
#     import xgboost as xgb
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
#     df = pd.read_csv(TRANSFORMED_CSV, parse_dates=["date"])
    
#     features = [
#         "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7",
#         "temp_ma3", "temp_ma7", "temp_std7",
#         "temp_change", "temp_min7", "temp_max7", "temp_range7",
#         "day_of_week", "month_num", "year_num", "day_of_year",
#         "season", "is_weekend"
#     ]
#     target = "avgtemperature"
    
#     print(f"🎯 Total Features: {len(features)}")
    
#     # Train/Test split per city
#     train_list, test_list = [], []
    
#     for city, g in df.groupby("city"):
#         g = g.sort_values("date")
#         split = int(len(g) * 0.8)
        
#         if split < 1:
#             continue
        
#         train_list.append(g.iloc[:split])
#         test_list.append(g.iloc[split:])
    
#     train = pd.concat(train_list)
#     test = pd.concat(test_list)
    
#     X_train, y_train = train[features], train[target]
#     X_test, y_test = test[features], test[target]
    
#     print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
#     # Train XGBoost (based on test results: MAE 1.831°C, 67.2% accuracy)
#     print("\n🤖 Training XGBoost...")
    
#     xgb_model = xgb.XGBRegressor(
#         n_estimators=500,
#         max_depth=8,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         min_child_weight=3,
#         gamma=0.1,
#         reg_alpha=0.1,
#         reg_lambda=1.0,
#         random_state=42,
#         n_jobs=-1,
#         verbosity=0
#     )
    
#     xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
#     xgb_pred = xgb_model.predict(X_test)
#     xgb_mae = mean_absolute_error(y_test, xgb_pred)
#     xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
#     xgb_r2 = r2_score(y_test, xgb_pred)
#     within_2 = np.sum(np.abs(y_test - xgb_pred) <= 2) / len(y_test) * 100
    
#     print(f"   MAE:  {xgb_mae:.3f}°C")
#     print(f"   RMSE: {xgb_rmse:.3f}°C")
#     print(f"   R²:   {xgb_r2:.4f}")
#     print(f"   Accuracy (±2°C): {within_2:.1f}%")
    
#     # Feature Importance
#     print(f"\n🎯 Top 10 Features:")
#     importances = pd.DataFrame({
#         'feature': features,
#         'importance': xgb_model.feature_importances_
#     }).sort_values('importance', ascending=False)
    
#     for idx, row in importances.head(10).iterrows():
#         print(f"   {row['feature']:<20} {row['importance']:.4f}")
    
#     # Save model
#     ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     model_path = f"{MODELS_DIR}/xgboost_model_{ts}.pkl"
#     joblib.dump(xgb_model, model_path)
    
#     Variable.set("latest_rf_model", model_path)
    
#     print(f"\n🏆 Winner: XGBoost")
#     print(f"💾 Model saved: {model_path}")
#     print("=" * 70 + "\n")


# def load_to_postgres(**context):
#     """Load transformed data to PostgreSQL"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 1: LOADING TO DATABASE")
#     print("=" * 70)
    
#     df = pd.read_csv(TRANSFORMED_CSV, parse_dates=["date"])
#     df = df.drop_duplicates(subset=["city", "date"])
    
#     print(f"📊 Rows to load: {len(df):,}")
    
#     tmp = "/tmp/weather_ml_load.csv"
#     df.to_csv(tmp, index=False, header=False)
    
#     params = _get_pg_conn()
    
#     with psycopg2.connect(**params) as conn:
#         cur = conn.cursor()
        
#         cur.execute("DROP TABLE IF EXISTS weather_ml CASCADE")
        
#         cur.execute("""
#             CREATE TABLE weather_ml (
#                 region TEXT,
#                 country TEXT,
#                 city TEXT,
#                 month INT,
#                 day INT,
#                 year INT,
#                 avgtemperature DOUBLE PRECISION,
#                 date DATE,
#                 temp_lag1 DOUBLE PRECISION,
#                 temp_lag2 DOUBLE PRECISION,
#                 temp_lag3 DOUBLE PRECISION,
#                 temp_lag7 DOUBLE PRECISION,
#                 temp_ma3 DOUBLE PRECISION,
#                 temp_ma7 DOUBLE PRECISION,
#                 temp_std7 DOUBLE PRECISION,
#                 temp_change DOUBLE PRECISION,
#                 temp_min7 DOUBLE PRECISION,
#                 temp_max7 DOUBLE PRECISION,
#                 temp_range7 DOUBLE PRECISION,
#                 day_of_week INT,
#                 month_num INT,
#                 year_num INT,
#                 day_of_year INT,
#                 season INT,
#                 is_weekend INT,
#                 PRIMARY KEY (city, date)
#             );
#         """)
        
#         with open(tmp, "r") as f:
#             cur.copy_expert("COPY weather_ml FROM STDIN WITH CSV", f)
        
#         conn.commit()
        
#         cur.execute("SELECT COUNT(*) FROM weather_ml")
#         count = cur.fetchone()[0]
        
#         print(f"✅ Loaded {count:,} rows")
    
#     os.remove(tmp)
#     print("=" * 70 + "\n")


# def mark_day_one_done(**context):
#     """Mark Day 1 as complete"""
#     print("=" * 70)
#     print(" " * 25 + "✅ DAY 1 COMPLETE")
#     print("=" * 70)
    
#     Variable.set("day_one_done", "yes")
    
#     print("\n🚀 Next run will execute Day 2 (Daily Prediction)")
#     print("=" * 70 + "\n")


# # =========================================
# # DAY 2 — DAILY PREDICTION (WITH RETRY)
# # =========================================

# def fetch_city_weather_with_retry(city, retry_count=2):
#     """
#     Fetch weather with retry logic
#     Args:
#         city: City name
#         retry_count: Number of retries (default 2 = initial + 1 retry)
#     """
#     url = f"http://api.openweathermap.org/data/2.5/weather"
#     params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    
#     for attempt in range(retry_count):
#         try:
#             resp = requests.get(url, params=params, timeout=15)
#             resp.raise_for_status()
            
#             data = resp.json()
            
#             if "main" not in data or "temp" not in data["main"]:
#                 if attempt < retry_count - 1:
#                     print(f"⚠️  {city}: Invalid response, retrying in 5s...")
#                     time.sleep(5)
#                     continue
#                 return None
            
#             temp = float(data["main"]["temp"])
            
#             if attempt > 0:
#                 print(f"✓ {city}: {temp:.1f}°C (retry #{attempt})")
#             else:
#                 print(f"✓ {city}: {temp:.1f}°C")
            
#             return {
#                 "city": city,
#                 "temp": temp,
#                 "temp_max": float(data["main"].get("temp_max", temp)),
#                 "temp_min": float(data["main"].get("temp_min", temp))
#             }
        
#         except Exception as e:
#             if attempt < retry_count - 1:
#                 print(f"❌ {city}: {str(e)[:40]}, retrying in 5s...")
#                 time.sleep(5)
#             else:
#                 print(f"💔 {city}: Failed after {retry_count} attempts")
#                 return None
    
#     return None


# def weather_api(**context):
#     """Parallel API calls with retry for failed cities"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 2: FETCHING WEATHER DATA")
#     print("=" * 70)
    
#     if not os.path.exists(CLEAN_CSV):
#         raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    
#     df = pd.read_csv(CLEAN_CSV)[["city", "country"]].drop_duplicates()
#     cities = df["city"].tolist()
    
#     print(f"📍 Cities: {len(cities)} | Workers: {MAX_WORKERS}\n")
    
#     results = []
#     failed = []
    
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_city = {
#             executor.submit(fetch_city_weather_with_retry, city): city 
#             for city in cities
#         }
        
#         for future in as_completed(future_to_city):
#             city = future_to_city[future]
            
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#                 else:
#                     failed.append(city)
#             except Exception as e:
#                 print(f"❌ Exception {city}: {e}")
#                 failed.append(city)
    
#     print(f"\n✅ Success: {len(results)}/{len(cities)} ({len(results)/len(cities)*100:.1f}%)")
#     print(f"❌ Failed: {len(failed)}/{len(cities)}")
    
#     if failed and len(failed) <= 20:
#         print(f"Failed cities: {', '.join(failed[:20])}")
    
#     print("=" * 70 + "\n")
    
#     context["ti"].xcom_push(key="api_rows", value=results)


# def build_daily_features(**context):
#     """Build features using 7-day window"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 2: BUILDING FEATURES")
#     print("=" * 70)
    
#     rows = context["ti"].xcom_pull(task_ids="weather_api", key="api_rows")
#     if not rows:
#         raise ValueError("No API data received!")

#     today = datetime.utcnow().date()
#     predict_date = today + timedelta(days=1)
    
#     print(f"Today: {today} | Predicting for: {predict_date}\n")

#     params = _get_pg_conn()
#     features = []
#     fallback_count = 0

#     with psycopg2.connect(**params) as conn:
#         cur = conn.cursor()
        
#         for row in rows:
#             city = row["city"]
#             api_temp = row["temp"]

#             cur.execute("""
#                 SELECT temp 
#                 FROM weather_actuals 
#                 WHERE city = %s 
#                   AND date < %s
#                 ORDER BY date DESC 
#                 LIMIT 7
#             """, (city, today))

#             recent_temps = [float(r[0]) for r in cur.fetchall()]

#             if len(recent_temps) < 7:
#                 cur.execute("""
#                     SELECT AVG(avgtemperature) 
#                     FROM weather_ml 
#                     WHERE city = %s AND month_num = %s
#                 """, (city, predict_date.month))
                
#                 avg_row = cur.fetchone()
#                 historical_avg = float(avg_row[0]) if avg_row and avg_row[0] else 15.0

#                 while len(recent_temps) < 7:
#                     recent_temps.append(historical_avg)
                
#                 fallback_count += 1

#             # Calculate features
#             lag1 = recent_temps[0]
#             lag2 = recent_temps[1] if len(recent_temps) > 1 else lag1
#             lag3 = recent_temps[2] if len(recent_temps) > 2 else lag1
#             lag7 = recent_temps[6] if len(recent_temps) > 6 else lag1
            
#             ma3 = np.mean(recent_temps[:3])
#             ma7 = np.mean(recent_temps)
#             std7 = np.std(recent_temps)
            
#             temp_change = lag1 - lag2
#             min7 = np.min(recent_temps)
#             max7 = np.max(recent_temps)
#             range7 = max7 - min7

#             features.append({
#                 "city": city,
#                 "predict_date": predict_date,
#                 "temp_lag1": lag1,
#                 "temp_lag2": lag2,
#                 "temp_lag3": lag3,
#                 "temp_lag7": lag7,
#                 "temp_ma3": ma3,
#                 "temp_ma7": ma7,
#                 "temp_std7": std7,
#                 "temp_change": temp_change,
#                 "temp_min7": min7,
#                 "temp_max7": max7,
#                 "temp_range7": range7,
#                 "day_of_week": predict_date.weekday(),
#                 "month_num": predict_date.month,
#                 "year_num": predict_date.year,
#                 "day_of_year": predict_date.timetuple().tm_yday,
#                 "season": 1 if predict_date.month in [12,1,2] else 
#                           2 if predict_date.month in [3,4,5] else 
#                           3 if predict_date.month in [6,7,8] else 4,
#                 "is_weekend": 1 if predict_date.weekday() >= 5 else 0,
#                 "api_temp": api_temp
#             })

#     print(f"✅ Features: {len(features)} | Fallback: {fallback_count}")
#     print("=" * 70 + "\n")
    
#     context["ti"].xcom_push(key="daily_features", value=features)


# def validate_prediction(city, prediction, lag1, month, conn):
#     """Validate predictions"""
#     if not -50 < prediction < 60:
#         return False, "Out of range"
    
#     if abs(prediction - lag1) > 25:
#         return False, f"Large jump: {abs(prediction - lag1):.1f}°C"
    
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT AVG(avgtemperature) as avg, STDDEV(avgtemperature) as std
#         FROM weather_ml
#         WHERE city = %s AND month_num = %s
#     """, (city, month))
    
#     row = cur.fetchone()
    
#     if row and row[0] and row[1]:
#         avg, std = float(row[0]), float(row[1])
        
#         if std > 0:
#             z_score = abs(prediction - avg) / std
            
#             if z_score > 3.5:
#                 return False, f"Anomaly: {z_score:.1f}σ"
    
#     return True, "Valid"


# def daily_predict_api(**context):
#     """Predict with XGBoost model (FIXED)"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 2: PREDICTING")
#     print("=" * 70)
    
#     features_data = context["ti"].xcom_pull(key="daily_features", task_ids="build_daily_features")
    
#     if not features_data:
#         raise ValueError("No features!")
    
#     model_path = Variable.get("latest_rf_model")
    
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found: {model_path}")
    
#     print(f"🤖 Loading: {model_path}\n")
#     model = joblib.load(model_path)
    
#     # MUST match training features
#     FEATURES = [
#         "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7",
#         "temp_ma3", "temp_ma7", "temp_std7",
#         "temp_change", "temp_min7", "temp_max7", "temp_range7",
#         "day_of_week", "month_num", "year_num", "day_of_year",
#         "season", "is_weekend"
#     ]
    
#     predictions = []
#     invalid = []
    
#     params = _get_pg_conn()
    
#     with psycopg2.connect(**params) as conn:
#         for f in features_data:
#             city = f["city"]
            
#             # Build feature vector (FIXED: ensure all features present)
#             X = np.array([[f[k] for k in FEATURES]])
#             pred = float(model.predict(X)[0])
            
#             is_valid, reason = validate_prediction(
#                 city, pred, f["temp_lag1"], f["month_num"], conn
#             )
            
#             if is_valid:
#                 predictions.append({
#                     "city": city,
#                     "date": f["predict_date"],
#                     "prediction": pred,
#                     "predicted_at": datetime.utcnow(),
#                     "lag1": f["temp_lag1"],
#                     "ma7": f["temp_ma7"],
#                     "api_temp": f.get("api_temp", None)
#                 })
                
#                 print(f"✓ {city}: {pred:.1f}°C")
#             else:
#                 invalid.append({"city": city, "prediction": pred, "reason": reason})
#                 print(f"❌ {city}: {pred:.1f}°C - {reason}")
    
#     print(f"\n✅ Valid: {len(predictions)} | ❌ Invalid: {len(invalid)}")
#     print("=" * 70 + "\n")
    
#     context["ti"].xcom_push(key="preds", value=predictions)



# def store_daily_predictions(**context):
#     """حفظ التنبؤات بسرعة صاروخية باستخدام Bulk Insert - آمنة من Zombie Tasks"""
#     print("=" * 70)
#     print(" " * 20 + "DAY 2: STORING PREDICTIONS (BULK MODE)")
#     print("=" * 70)
    
#     preds = context["ti"].xcom_pull(key="preds", task_ids="daily_predict_api")
    
#     if not preds or len(preds) == 0:
#         print("تحذير: مفيش تنبؤات لحفظها")
#         return
    
#     print(f"جاري حفظ أو تحديث {len(preds)} تنبؤ باستخدام Bulk Insert...")

#     params = _get_pg_conn()
    
#     try:
#         with psycopg2.connect(**params) as conn:
#             cur = conn.cursor()
            
#             # إنشاء الجدول لو مش موجود (مرة واحدة بس)
#             cur.execute("""
#                 CREATE TABLE IF NOT EXISTS weather_predictions (
#                     city TEXT,
#                     date DATE,
#                     prediction DOUBLE PRECISION,
#                     predicted_at TIMESTAMP,
#                     lag1 DOUBLE PRECISION,
#                     ma7 DOUBLE PRECISION,
#                     api_temp DOUBLE PRECISION,
#                     PRIMARY KEY (city, date)
#                 );
#             """)
            
#             # التأكد من وجود الأعمدة
#             cur.execute("""
#                 ALTER TABLE weather_predictions 
#                 ADD COLUMN IF NOT EXISTS lag1 DOUBLE PRECISION,
#                 ADD COLUMN IF NOT EXISTS ma7 DOUBLE PRECISION,
#                 ADD COLUMN IF NOT EXISTS api_temp DOUBLE PRECISION;
#             """)
            
#             # تحضير البيانات للـ Bulk Insert
#             data_to_insert = [
#                 (
#                     p["city"],
#                     p["date"],
#                     p["prediction"],
#                     p["predicted_at"],
#                     p.get("lag1"),
#                     p.get("ma7"),
#                     p.get("api_temp")
#                 )
#                 for p in preds
#             ]
            
#             # BULK INSERT السحري - أسرع 50 مرة ولا يسبب timeout أبدًا
#             from psycopg2.extras import execute_values
            
#             execute_values(
#                 cur,
#                 """
#                 INSERT INTO weather_predictions 
#                 (city, date, prediction, predicted_at, lag1, ma7, api_temp)
#                 VALUES %s
#                 ON CONFLICT (city, date) DO UPDATE SET
#                     prediction = EXCLUDED.prediction,
#                     predicted_at = EXCLUDED.predicted_at,
#                     lag1 = EXCLUDED.lag1,
#                     ma7 = EXCLUDED.ma7,
#                     api_temp = EXCLUDED.api_temp;
#                 """,
#                 data_to_insert
#             )
            
#             conn.commit()
#             print(f"تم حفظ {len(preds)} تنبؤ بنجاح في أقل من ثانية!")
            
#     except Exception as e:
#         print(f"خطأ أثناء الحفظ: {e}")
#         raise
    
#     print("=" * 70 + "\n")
    





# def fetch_city_actual(city, date):
#     """
#     Fetch actual temperature for a single city
#     """
#     url = f"http://api.openweathermap.org/data/2.5/weather"
#     params = {
#         "q": city,
#         "appid": WEATHER_API_KEY,
#         "units": "metric"
#     }
    
#     try:
#         resp = requests.get(url, params=params, timeout=10)
#         resp.raise_for_status()
        
#         data = resp.json()
        
#         if "main" not in data:
#             return None
        
#         temp = float(data["main"]["temp"])
#         temp_max = float(data["main"].get("temp_max", temp))
#         temp_min = float(data["main"].get("temp_min", temp))
        
#         return {
#             "city": city,
#             "date": date,
#             "temp": temp,
#             "temp_max": temp_max,
#             "temp_min": temp_min
#         }
    
#     except Exception as e:
#         print(f"❌ {city}: {str(e)[:50]}")
#         return None


# def fetch_end_of_day_actuals(**context):
#     """
#     Fetch actual temperatures for all cities at end of day
#     """
#     print("=" * 70)
#     print(" " * 15 + "FETCHING END-OF-DAY ACTUAL TEMPERATURES")
#     print("=" * 70)
    
#     today = datetime.utcnow().date()
    
#     print(f"📅 Date: {today}")
#     print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n")
    
#     # Get cities list
#     if not os.path.exists(CLEAN_CSV):
#         raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    
#     df = pd.read_csv(CLEAN_CSV)[["city"]].drop_duplicates()
#     cities = df["city"].tolist()
    
#     print(f"📍 Cities to fetch: {len(cities)}")
    
#     # Parallel fetch
#     results = []
#     failed = []
    
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_city = {
#             executor.submit(fetch_city_actual, city, today): city
#             for city in cities
#         }
        
#         for future in as_completed(future_to_city):
#             city = future_to_city[future]
            
#             try:
#                 result = future.result()
                
#                 if result:
#                     results.append(result)
#                     print(f"✓ {city}: {result['temp']:.1f}°C")
#                 else:
#                     failed.append(city)
            
#             except Exception as e:
#                 print(f"❌ {city}: {e}")
#                 failed.append(city)
            
#             # Small delay to respect rate limits
#             time.sleep(0.1)
    
#     # Store results
#     if results:
#         print(f"\n💾 Storing {len(results)} actuals to database...")
        
#         params = _get_pg_conn()
        
#         with psycopg2.connect(**params) as conn:
#             cur = conn.cursor()
            
#             # Create table if not exists
#             cur.execute("""
#                 CREATE TABLE IF NOT EXISTS weather_actuals(
#                     city TEXT,
#                     date DATE,
#                     temp DOUBLE PRECISION,
#                     temp_max DOUBLE PRECISION,
#                     temp_min DOUBLE PRECISION,
#                     fetched_at TIMESTAMP DEFAULT NOW(),
#                     PRIMARY KEY(city, date)
#                 );
#             """)
            
#             # Create index
#             cur.execute("""
#                 CREATE INDEX IF NOT EXISTS idx_actuals_city_date 
#                 ON weather_actuals(city, date DESC);
#             """)
            
#             # Insert actuals
#             for r in results:
#                 cur.execute("""
#                     INSERT INTO weather_actuals(city, date, temp, temp_max, temp_min, fetched_at)
#                     VALUES(%s, %s, %s, %s, %s, NOW())
#                     ON CONFLICT(city, date) DO UPDATE
#                     SET temp = EXCLUDED.temp,
#                         temp_max = EXCLUDED.temp_max,
#                         temp_min = EXCLUDED.temp_min,
#                         fetched_at = EXCLUDED.fetched_at
#                 """, (
#                     r["city"],
#                     r["date"],
#                     r["temp"],
#                     r["temp_max"],
#                     r["temp_min"]
#                 ))
            
#             conn.commit()
            
#             # Cleanup old data (keep only last 30 days)
#             print("🧹 Cleaning old data (keeping last 30 days)...")
#             cur.execute("""
#                 DELETE FROM weather_actuals
#                 WHERE date < CURRENT_DATE - INTERVAL '30 days'
#             """)
            
#             deleted = cur.rowcount
#             conn.commit()
            
#             if deleted > 0:
#                 print(f"   Deleted {deleted} old records")
    
#     # Summary
#     print("\n" + "=" * 70)
#     print(f"✅ Success: {len(results)} cities")
#     print(f"❌ Failed: {len(failed)} cities")
    
#     if failed and len(failed) <= 10:
#         print(f"Failed cities: {', '.join(failed)}")
    
#     print("=" * 70 + "\n")
    
#     # Push to XCom for monitoring
#     context["ti"].xcom_push(key="actuals_count", value=len(results))
#     context["ti"].xcom_push(key="failed_count", value=len(failed))


# # =========================================
# # TASK: CALCULATE ACCURACY
# # ========
# # 
# # =================================

# def calculate_daily_accuracy(**context):
#     """
#     Calculate prediction accuracy for today (now that we have actuals)
#     """
#     print("=" * 70)
#     print(" " * 20 + "CALCULATING PREDICTION ACCURACY")
#     print("=" * 70)
    
#     today = datetime.utcnow().date()
#     # target_date = (datetime.utcnow().date() - timedelta(days=1))
    
#     print(f"📅 Evaluating predictions for: {today}\n")
    
#     params = _get_pg_conn()
    
#     with psycopg2.connect(**params) as conn:
#         cur = conn.cursor()
        
#         # Get predictions vs actuals for today
#         cur.execute("""
#             SELECT 
#                 p.city,
#                 p.prediction,
#                 a.temp as actual,
#                 ABS(p.prediction - a.temp) as absolute_error
#             FROM weather_predictions p
#             INNER JOIN weather_actuals a 
#                 ON p.city = a.city AND p.date = a.date
#             WHERE p.date = %s
#         """, (today,))
        
#         results = cur.fetchall()
        
#         if not results:
#             print("⚠️  No predictions found for today")
#             return
        
#         # Calculate metrics
#         errors = [r[3] for r in results]
        
#         mae = sum(errors) / len(errors)
#         rmse = (sum([e**2 for e in errors]) / len(errors)) ** 0.5
        
#         within_2 = sum([1 for e in errors if e <= 2])
#         within_5 = sum([1 for e in errors if e <= 5])
        
#         print(f"📊 Metrics for {len(results)} cities:")
#         print(f"   MAE:  {mae:.2f}°C")
#         print(f"   RMSE: {rmse:.2f}°C")
#         print(f"   Within ±2°C: {within_2}/{len(results)} ({within_2/len(results)*100:.1f}%)")
#         print(f"   Within ±5°C: {within_5}/{len(results)} ({within_5/len(results)*100:.1f}%)")
        
#         # Show best and worst predictions
#         sorted_results = sorted(results, key=lambda x: x[3])
        
#         print("\n🏆 Best predictions (lowest error):")
#         for city, pred, actual, error in sorted_results[:5]:
#             print(f"   {city}: predicted {pred:.1f}°C, actual {actual:.1f}°C (error: {error:.2f}°C)")
        
#         print("\n😓 Worst predictions (highest error):")
#         for city, pred, actual, error in sorted_results[-5:]:
#             print(f"   {city}: predicted {pred:.1f}°C, actual {actual:.1f}°C (error: {error:.2f}°C)")
        
#         # Store metrics
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS model_metrics(
#                 date DATE PRIMARY KEY,
#                 mae DOUBLE PRECISION,
#                 rmse DOUBLE PRECISION,
#                 total_predictions INT,
#                 within_2_degrees INT,
#                 within_5_degrees INT,
#                 created_at TIMESTAMP DEFAULT NOW()
#             );
#         """)
        
#         cur.execute("""
#             INSERT INTO model_metrics(date, mae, rmse, total_predictions, within_2_degrees, within_5_degrees)
#             VALUES(%s, %s, %s, %s, %s, %s)
#             ON CONFLICT(date) DO UPDATE
#             SET mae = EXCLUDED.mae,
#                 rmse = EXCLUDED.rmse,
#                 total_predictions = EXCLUDED.total_predictions,
#                 within_2_degrees = EXCLUDED.within_2_degrees,
#                 within_5_degrees = EXCLUDED.within_5_degrees
#         """, (today, mae, rmse, len(results), within_2, within_5))
        
#         conn.commit()
    
#     print("\n" + "=" * 70 + "\n")




# # =========================================
# # DAG DEFINITION
# # =========================================

# default_args = {
#     "owner": "ahmed_sami",
#     "retries": 2,
#     "retry_delay": timedelta(minutes=5),
# }

# with DAG(
#     dag_id="weather_prediction_pipeline",
#     description="Complete pipeline: Training + Prediction (XGBoost + Cleaning)",
#     start_date=datetime(2025, 12, 1),
#     schedule_interval="5 18 * * *",  # Daily at 08:00 AM
#     catchup=False,
#     default_args=default_args,
#     tags=["weather", "ml", "xgboost", "final"]
# ) as dag:
    
#     # ============ BRANCH DECISION ============
#     check = BranchPythonOperator(
#         task_id="check_day_one",
#         python_callable=check_day_one
#     )
    
#     # ============ DAY 1 TASKS ============
#     extract = PythonOperator(
#         task_id="extract_csv",
#         python_callable=extract_csv
#     )
    
#     transform = PythonOperator(
#         task_id="transform_data",
#         python_callable=transform_data
#     )
    
#     train = PythonOperator(
#         task_id="train_and_evaluate",
#         python_callable=train_and_evaluate
#     )
    
#     load = PythonOperator(
#         task_id="load_to_postgres",
#         python_callable=load_to_postgres
#     )
    
#     done = PythonOperator(
#         task_id="mark_day_one_done",
#         python_callable=mark_day_one_done
#     )
    
#     # ============ SKIP DAY 1 ============
#     skip = DummyOperator(task_id="skip_day1")
    
#     # ============ JOIN POINT ============
#     join = DummyOperator(
#         task_id="join",
#         trigger_rule="none_failed_min_one_success"
#     )
    
#     # ============ DAY 2 TASKS ============
#     weather = PythonOperator(
#         task_id="weather_api",
#         python_callable=weather_api
#     )
    
#     features = PythonOperator(
#         task_id="build_daily_features",
#         python_callable=build_daily_features
#     )
    
#     predict = PythonOperator(
#         task_id="daily_predict_api",
#         python_callable=daily_predict_api
#     )
    
#     store = PythonOperator(
#         task_id="store_daily_predictions",
#         python_callable=store_daily_predictions
#     )
    
#     # ============ TASK DEPENDENCIES ============
    
#     # Branch decision
#     check >> [extract, skip]
    
#     # Day 1 path
#     extract >> transform >> train >> load >> done >> join
    
#     # Day 2 path (skip Day 1)
#     skip >> join
    
#     # Day 2 tasks (always run after join)
#     join >> weather >> features >> predict >> store



# with DAG(
#     dag_id="weather_actuals_pipeline",
#     description="End-of-day: Collect actual temperatures",
#     start_date=datetime(2025, 12, 1),
#     schedule_interval="0 18 * * *",  # 23:50 PM daily
#     catchup=False,
#     default_args=default_args,
#     tags=["weather", "actuals", "daily"]
# ) as dag:
    
#     start = DummyOperator(task_id="start")
    
#     fetch = PythonOperator(
#         task_id="fetch_end_of_day_actuals",
#         python_callable=fetch_end_of_day_actuals
#     )
    
#     accuracy = PythonOperator(
#         task_id="calculate_daily_accuracy",
#         python_callable=calculate_daily_accuracy
#     )
    
#     end = DummyOperator(task_id="end")
    
#     # Task dependencies
#     start >> fetch >> accuracy >> end









"""
Weather Prediction Pipeline - UPDATED FOR NEW DATA FORMAT
Author: Ahmed Sami
Updates:
1. New data format with MinTemperature and MaxTemperature
2. Using only temperature data (removed wind and precipitation)
3. Enhanced features with min/max temperatures
4. Data cleaning with IQR outliers removal
5. XGBoost Model with improved features
"""

import os
import time
from datetime import datetime, timedelta, time as dtime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

    

import joblib
import numpy as np
import pandas as pd
import psycopg2
import requests
import xgboost as xgb

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator

# =========================================
# CONFIG
# =========================================

BASE_PATH = "/opt/airflow/dags/aiii_model"
RAW_CSV = os.path.join(BASE_PATH, "city_temperature.csv")
CLEAN_CSV = os.path.join(BASE_PATH, "clean_data.csv")
TRANSFORMED_CSV = os.path.join(BASE_PATH, "transformed_data.csv")
MODELS_DIR = "/opt/airflow/models"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BASE_PATH, exist_ok=True)

POSTGRES_CONN_ID = "ml_con"
WEATHER_API_KEY = "bd75bd2d767f720269c389aa29da6be9"
MAX_WORKERS = 10


def _get_pg_conn():
    """Get PostgreSQL connection parameters"""
    conn = BaseHook.get_connection(POSTGRES_CONN_ID)
    return {
        "host": conn.host,
        "dbname": conn.schema,
        "user": conn.login,
        "password": conn.password,
        "port": conn.port or 5432
    }


# =========================================
# BRANCH DECISION
# =========================================

def check_day_one():
    """Check if Day 1 training already completed"""
    flag = Variable.get("day_one_done", default_var="no")
    
    if flag == "no":
        print("🎯 Running DAY 1 - Initial Training")
        return "extract_csv"
    else:
        print("✅ DAY 1 completed - Running DAY 2 - Daily Prediction")
        return "skip_day1"


# =========================================
# DAY 1 — TRAINING TASKS (WITH CLEANING)
# =========================================

def extract_csv(**context):
    """Extract and clean raw temperature data WITH OUTLIER REMOVAL"""
    print("=" * 70)
    print(" " * 20 + "DAY 1: EXTRACTING & CLEANING DATA")
    print("=" * 70)
    
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"❌ Raw CSV not found: {RAW_CSV}")
    
    print(f"📂 Reading: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    initial_count = len(df)
    print(f"📊 Initial shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Expected columns in new format
    expected_cols = ['region', 'country', 'state', 'city', 'year', 'month', 'day', 
                    'avgtemperature', 'mintemperature', 'maxtemperature']
    
    # Check if columns exist
    for col in expected_cols[:10]:  # Check main temperature columns
        if col not in df.columns:
            print(f"⚠️  Warning: Column '{col}' not found")
    
    # Remove invalid temperatures
    print("🧹 Cleaning temperature data...")
    df["avgtemperature"] = df["avgtemperature"].replace([-99, 999, -999], np.nan)
    df["mintemperature"] = df["mintemperature"].replace([-99, 999, -999], np.nan)
    df["maxtemperature"] = df["maxtemperature"].replace([-99, 999, -999], np.nan)
    
    # Drop rows with missing temperature data
    df = df.dropna(subset=["avgtemperature"])
    
    # Create date column
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors='coerce')
    df = df.dropna(subset=["date"])
    
    # Sort and deduplicate
    df = df.sort_values(["city", "date"]).drop_duplicates(subset=["city", "date"])
    
    # ===== DATA CLEANING: Remove outliers per city =====
    print("🧹 Cleaning outliers using IQR method...")
    
    # 1. Remove impossible temperatures
    df = df[(df['avgtemperature'] >= -50) & (df['avgtemperature'] <= 60)]
    df = df[(df['mintemperature'] >= -60) & (df['mintemperature'] <= 60)]
    df = df[(df['maxtemperature'] >= -50) & (df['maxtemperature'] <= 70)]
    removed_range = initial_count - len(df)
    
    # 2. Remove outliers per city using IQR
    clean_list = []
    
    for city, group in df.groupby('city'):
        # IQR for average temperature
        Q1 = group['avgtemperature'].quantile(0.25)
        Q3 = group['avgtemperature'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Use 3 * IQR (more conservative for temperature)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        group_clean = group[
            (group['avgtemperature'] >= lower_bound) & 
            (group['avgtemperature'] <= upper_bound)
        ]
        
        clean_list.append(group_clean)
    
    df = pd.concat(clean_list, ignore_index=True)
    removed_outliers = initial_count - removed_range - len(df)
    
    # Keep only necessary columns
    keep_cols = ['region', 'country', 'city', 'year', 'month', 'day', 
                'avgtemperature', 'mintemperature', 'maxtemperature', 'date']
    df = df[keep_cols]
    
    # Save
    df.to_csv(CLEAN_CSV, index=False)
    
    print(f"✅ Cleaning complete:")
    print(f"   ❌ Out of range: {removed_range:,} ({removed_range/initial_count*100:.1f}%)")
    print(f"   ❌ Outliers: {removed_outliers:,} ({removed_outliers/initial_count*100:.1f}%)")
    print(f"   ✅ Remaining: {len(df):,} ({len(df)/initial_count*100:.1f}%)")
    print(f"💾 Saved to: {CLEAN_CSV}")
    print("=" * 70 + "\n")


def transform_data(**context):
    """Create enhanced features with 7-day window and min/max temperatures"""
    print("=" * 70)
    print(" " * 20 + "DAY 1: ENHANCED FEATURE ENGINEERING")
    print("=" * 70)
    
    df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
    print(f"📊 Input shape: {df.shape}")
    
    final_list = []
    
    for city, g in df.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)
        
        if len(g) < 14:
            continue
        
        # Temperature lags (avg, min, max)
        g["temp_lag1"] = g["avgtemperature"].shift(1)
        g["temp_lag2"] = g["avgtemperature"].shift(2)
        g["temp_lag3"] = g["avgtemperature"].shift(3)
        g["temp_lag7"] = g["avgtemperature"].shift(7)
        
        # Min/Max lags
        g["min_temp_lag1"] = g["mintemperature"].shift(1)
        g["max_temp_lag1"] = g["maxtemperature"].shift(1)
        g["temp_range_lag1"] = g["max_temp_lag1"] - g["min_temp_lag1"]
        
        # Moving averages
        g["temp_ma3"] = g["avgtemperature"].shift(1).rolling(3).mean()
        g["temp_ma7"] = g["avgtemperature"].shift(1).rolling(7).mean()
        g["temp_std7"] = g["avgtemperature"].shift(1).rolling(7).std()
        
        # Min/Max moving averages
        g["min_temp_ma7"] = g["mintemperature"].shift(1).rolling(7).mean()
        g["max_temp_ma7"] = g["maxtemperature"].shift(1).rolling(7).mean()
        
        # Temperature changes
        g["temp_change"] = g["avgtemperature"].shift(1) - g["avgtemperature"].shift(2)
        g["temp_change_3d"] = g["avgtemperature"].shift(1) - g["avgtemperature"].shift(3)
        
        # Rolling statistics
        g["temp_min7"] = g["avgtemperature"].shift(1).rolling(7).min()
        g["temp_max7"] = g["avgtemperature"].shift(1).rolling(7).max()
        g["temp_range7"] = g["temp_max7"] - g["temp_min7"]
        
        # Temperature volatility
        g["temp_volatility"] = g["temp_std7"] / (g["temp_ma7"] + 0.1)  # Coefficient of variation
        
        # Date features
        g["day_of_week"] = g["date"].dt.dayofweek
        g["month_num"] = g["date"].dt.month
        g["year_num"] = g["date"].dt.year
        g["day_of_year"] = g["date"].dt.dayofyear
        
        # Season
        g["season"] = g["month_num"].apply(lambda m: 
            1 if m in [12, 1, 2] else 2 if m in [3, 4, 5] else 3 if m in [6, 7, 8] else 4
        )
        
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
        
        # Drop rows with NaN values
        g = g.dropna().reset_index(drop=True)
        
        if len(g) > 0:
            final_list.append(g)
    
    final_df = pd.concat(final_list, ignore_index=True)
    final_df.to_csv(TRANSFORMED_CSV, index=False)
    
    print(f"✅ Transformed shape: {final_df.shape}")
    print(f"📊 Features created: {final_df.shape[1]}")
    print(f"💾 Saved to: {TRANSFORMED_CSV}")
    print("=" * 70 + "\n")


def train_and_evaluate(**context):
    """Train XGBoost model with enhanced features"""
    print("=" * 70)
    print(" " * 20 + "DAY 1: TRAINING XGBOOST MODEL")
    print("=" * 70)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    df = pd.read_csv(TRANSFORMED_CSV, parse_dates=["date"])
    
    features = [
        "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7",
        "min_temp_lag1", "max_temp_lag1", "temp_range_lag1",
        "temp_ma3", "temp_ma7", "temp_std7",
        "min_temp_ma7", "max_temp_ma7",
        "temp_change", "temp_change_3d",
        "temp_min7", "temp_max7", "temp_range7",
        "temp_volatility",
        "day_of_week", "month_num", "year_num", "day_of_year",
        "season", "is_weekend"
    ]
    target = "avgtemperature"
    
    print(f"🎯 Total Features: {len(features)}")
    
    # Train/Test split per city
    train_list, test_list = [], []
    
    for city, g in df.groupby("city"):
        g = g.sort_values("date")
        split = int(len(g) * 0.8)
        
        if split < 1:
            continue
        
        train_list.append(g.iloc[:split])
        test_list.append(g.iloc[split:])
    
    train = pd.concat(train_list)
    test = pd.concat(test_list)
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Train XGBoost
    print("\n🤖 Training XGBoost...")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    within_2 = np.sum(np.abs(y_test - xgb_pred) <= 2) / len(y_test) * 100
    
    print(f"   MAE:  {xgb_mae:.3f}°C")
    print(f"   RMSE: {xgb_rmse:.3f}°C")
    print(f"   R²:   {xgb_r2:.4f}")
    print(f"   Accuracy (±2°C): {within_2:.1f}%")
    
    # Feature Importance
    print(f"\n🎯 Top 10 Features:")
    importances = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importances.head(10).iterrows():
        print(f"   {row['feature']:<20} {row['importance']:.4f}")
    
    # Save model
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = f"{MODELS_DIR}/xgboost_model_{ts}.pkl"
    joblib.dump(xgb_model, model_path)
    
    Variable.set("latest_rf_model", model_path)
    
    print(f"\n🏆 Model: XGBoost")
    print(f"💾 Model saved: {model_path}")
    print("=" * 70 + "\n")


def load_to_postgres(**context):
    """Load transformed data to PostgreSQL"""
    print("=" * 70)
    print(" " * 20 + "DAY 1: LOADING TO DATABASE")
    print("=" * 70)
    
    df = pd.read_csv(TRANSFORMED_CSV, parse_dates=["date"])
    df = df.drop_duplicates(subset=["city", "date"])
    
    print(f"📊 Rows to load: {len(df):,}")
    
    tmp = "/tmp/weather_ml_load.csv"
    df.to_csv(tmp, index=False, header=False)
    
    params = _get_pg_conn()
    
    with psycopg2.connect(**params) as conn:
        cur = conn.cursor()
        
        cur.execute("DROP TABLE IF EXISTS weather_ml CASCADE")
        
        cur.execute("""
            CREATE TABLE weather_ml (
                region TEXT,
                country TEXT,
                city TEXT,
                year INT,
                month INT,
                day INT,
                avgtemperature DOUBLE PRECISION,
                mintemperature DOUBLE PRECISION,
                maxtemperature DOUBLE PRECISION,
                date DATE,
                temp_lag1 DOUBLE PRECISION,
                temp_lag2 DOUBLE PRECISION,
                temp_lag3 DOUBLE PRECISION,
                temp_lag7 DOUBLE PRECISION,
                min_temp_lag1 DOUBLE PRECISION,
                max_temp_lag1 DOUBLE PRECISION,
                temp_range_lag1 DOUBLE PRECISION,
                temp_ma3 DOUBLE PRECISION,
                temp_ma7 DOUBLE PRECISION,
                temp_std7 DOUBLE PRECISION,
                min_temp_ma7 DOUBLE PRECISION,
                max_temp_ma7 DOUBLE PRECISION,
                temp_change DOUBLE PRECISION,
                temp_change_3d DOUBLE PRECISION,
                temp_min7 DOUBLE PRECISION,
                temp_max7 DOUBLE PRECISION,
                temp_range7 DOUBLE PRECISION,
                temp_volatility DOUBLE PRECISION,
                day_of_week INT,
                month_num INT,
                year_num INT,
                day_of_year INT,
                season INT,
                is_weekend INT,
                PRIMARY KEY (city, date)
            );
        """)
        
        with open(tmp, "r") as f:
            cur.copy_expert("COPY weather_ml FROM STDIN WITH CSV", f)
        
        conn.commit()
        
        cur.execute("SELECT COUNT(*) FROM weather_ml")
        count = cur.fetchone()[0]
        
        print(f"✅ Loaded {count:,} rows")
    
    os.remove(tmp)
    print("=" * 70 + "\n")


def mark_day_one_done(**context):
    """Mark Day 1 as complete"""
    print("=" * 70)
    print(" " * 25 + "✅ DAY 1 COMPLETE")
    print("=" * 70)
    
    Variable.set("day_one_done", "yes")
    
    print("\n🚀 Next run will execute Day 2 (Daily Prediction)")
    print("=" * 70 + "\n")


# =========================================
# DAY 2 — DAILY PREDICTION (WITH RETRY)
# =========================================
def fetch_city_weather_with_retry(city, retry_count=2):
    """
    Fetch weather with retry logic
    Returns a FIXED daily reading using 06:00 UTC anchor time
    """

    ANCHOR_HOUR = 6  # 🔒 FIXED: 06:00 UTC
    
    url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    
    for attempt in range(retry_count):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            if "list" not in data or not data["list"]:
                if attempt < retry_count - 1:
                    time.sleep(5)
                    continue
                return None
            
            today = datetime.now(timezone.utc).date()
            today_readings = []
            daily_temps = {"min": None, "max": None}
            
            for forecast in data["list"]:
                forecast_time = datetime.fromtimestamp(
                    forecast["dt"], tz=timezone.utc
                )
                if forecast_time.date() != today:
                    continue
                
                temp = float(forecast["main"]["temp"])
                today_readings.append({
                    "time": forecast_time,
                    "temp": temp
                })
                
                if daily_temps["min"] is None or temp < daily_temps["min"]:
                    daily_temps["min"] = temp
                if daily_temps["max"] is None or temp > daily_temps["max"]:
                    daily_temps["max"] = temp
            
            if not today_readings:
                if attempt < retry_count - 1:
                    time.sleep(5)
                    continue
                return None
            
            # ─────────────────────────────────────────
            # 🔒 FIXED DAILY READING (ANCHOR = 06:00 UTC)
            # ─────────────────────────────────────────
            anchor_time = dtime(ANCHOR_HOUR, 0)
            
            after_anchor = [
                r for r in today_readings
                if r["time"].time() >= anchor_time
            ]
            
            if after_anchor:
                fixed_reading = min(after_anchor, key=lambda x: x["time"])
            else:
                fixed_reading = min(today_readings, key=lambda x: x["time"])
            
            temp = fixed_reading["temp"]
            
            return {
                "city": city,
                "temp": temp,
                "temp_max": daily_temps["max"],
                "temp_min": daily_temps["min"],
                "reading_time": fixed_reading["time"].strftime("%H:%M UTC")
            }
        
        except Exception:
            if attempt < retry_count - 1:
                time.sleep(5)
            else:
                return None
    
    return None














def weather_api(**context):
    """Parallel API calls with retry for failed cities"""
    print("=" * 70)
    print(" " * 20 + "DAY 2: FETCHING WEATHER DATA")
    print("=" * 70)
    
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    
    df = pd.read_csv(CLEAN_CSV)[["city", "country"]].drop_duplicates()
    cities = df["city"].tolist()
    
    print(f"📍 Cities: {len(cities)} | Workers: {MAX_WORKERS}\n")
    
    results = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_city = {
            executor.submit(fetch_city_weather_with_retry, city): city 
            for city in cities
        }
        
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed.append(city)
            except Exception as e:
                print(f"❌ Exception {city}: {e}")
                failed.append(city)
    
    print(f"\n✅ Success: {len(results)}/{len(cities)} ({len(results)/len(cities)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)}/{len(cities)}")
    
    if failed and len(failed) <= 20:
        print(f"Failed cities: {', '.join(failed[:20])}")
    
    print("=" * 70 + "\n")
    
    context["ti"].xcom_push(key="api_rows", value=results)


def build_daily_features(**context):
    """Build features using 7-day window with min/max temperatures"""
    print("=" * 70)
    print(" " * 20 + "DAY 2: BUILDING FEATURES")
    print("=" * 70)
    
    rows = context["ti"].xcom_pull(task_ids="weather_api", key="api_rows")
    if not rows:
        raise ValueError("No API data received!")

    today = datetime.utcnow().date()
    predict_date = today + timedelta(days=1)
    
    print(f"Today: {today} | Predicting for: {predict_date}\n")

    params = _get_pg_conn()
    features = []
    fallback_count = 0

    with psycopg2.connect(**params) as conn:
        cur = conn.cursor()
        
        for row in rows:
            city = row["city"]
            api_temp = row["temp"]
            api_temp_max = row["temp_max"]
            api_temp_min = row["temp_min"]

            # Get recent 7 days from actuals
            cur.execute("""
                SELECT temp, temp_max, temp_min
                FROM weather_actuals 
                WHERE city = %s 
                    AND date < %s
                ORDER BY date DESC 
                LIMIT 7
            """, (city, today))

            recent_data = cur.fetchall()
            recent_temps = [float(r[0]) for r in recent_data]
            recent_max = [float(r[1]) for r in recent_data]
            recent_min = [float(r[2]) for r in recent_data]

            # Fill with historical averages if not enough data
            if len(recent_temps) < 7:
                cur.execute("""
                    SELECT AVG(avgtemperature), AVG(maxtemperature), AVG(mintemperature)
                    FROM weather_ml 
                    WHERE city = %s AND month_num = %s
                """, (city, predict_date.month))
                
                avg_row = cur.fetchone()
                historical_avg = float(avg_row[0]) if avg_row and avg_row[0] else 15.0
                historical_max = float(avg_row[1]) if avg_row and avg_row[1] else 20.0
                historical_min = float(avg_row[2]) if avg_row and avg_row[2] else 10.0

                while len(recent_temps) < 7:
                    recent_temps.append(historical_avg)
                    recent_max.append(historical_max)
                    recent_min.append(historical_min)
                
                fallback_count += 1

            # Calculate features
            lag1 = recent_temps[0]
            lag2 = recent_temps[1] if len(recent_temps) > 1 else lag1
            lag3 = recent_temps[2] if len(recent_temps) > 2 else lag1
            lag7 = recent_temps[6] if len(recent_temps) > 6 else lag1
            
            min_temp_lag1 = recent_min[0] if recent_min else lag1 - 2
            max_temp_lag1 = recent_max[0] if recent_max else lag1 + 2
            temp_range_lag1 = max_temp_lag1 - min_temp_lag1
            
            ma3 = np.mean(recent_temps[:3])
            ma7 = np.mean(recent_temps)
            std7 = np.std(recent_temps)
            
            min_temp_ma7 = np.mean(recent_min) if recent_min else ma7 - 2
            max_temp_ma7 = np.mean(recent_max) if recent_max else ma7 + 2
            
            temp_change = lag1 - lag2
            temp_change_3d = lag1 - lag3
            
            min7 = np.min(recent_temps)
            max7 = np.max(recent_temps)
            range7 = max7 - min7
            
            temp_volatility = std7 / (ma7 + 0.1)

            features.append({
                "city": city,
                "predict_date": predict_date,
                "temp_lag1": lag1,
                "temp_lag2": lag2,
                "temp_lag3": lag3,
                "temp_lag7": lag7,
                "min_temp_lag1": min_temp_lag1,
                "max_temp_lag1": max_temp_lag1,
                "temp_range_lag1": temp_range_lag1,
                "temp_ma3": ma3,
                "temp_ma7": ma7,
                "temp_std7": std7,
                "min_temp_ma7": min_temp_ma7,
                "max_temp_ma7": max_temp_ma7,
                "temp_change": temp_change,
                "temp_change_3d": temp_change_3d,
                "temp_min7": min7,
                "temp_max7": max7,
                "temp_range7": range7,
                "temp_volatility": temp_volatility,
                "day_of_week": predict_date.weekday(),
                "month_num": predict_date.month,
                "year_num": predict_date.year,
                "day_of_year": predict_date.timetuple().tm_yday,
                "season": 1 if predict_date.month in [12,1,2] else 
                            2 if predict_date.month in [3,4,5] else 
                            3 if predict_date.month in [6,7,8] else 4,
                "is_weekend": 1 if predict_date.weekday() >= 5 else 0,
                "api_temp": api_temp,
                "api_temp_max": api_temp_max,
                "api_temp_min": api_temp_min
            })

    print(f"✅ Features: {len(features)} | Fallback: {fallback_count}")
    print("=" * 70 + "\n")
    
    context["ti"].xcom_push(key="daily_features", value=features)


def validate_prediction(city, prediction, lag1, month, conn):
    """Validate predictions"""
    if not -50 < prediction < 60:
        return False, "Out of range"
    
    if abs(prediction - lag1) > 25:
        return False, f"Large jump: {abs(prediction - lag1):.1f}°C"
    
    cur = conn.cursor()
    cur.execute("""
        SELECT AVG(avgtemperature) as avg, STDDEV(avgtemperature) as std
        FROM weather_ml
        WHERE city = %s AND month_num = %s
    """, (city, month))
    
    row = cur.fetchone()
    
    if row and row[0] and row[1]:
        avg, std = float(row[0]), float(row[1])
        
        if std > 0:
            z_score = abs(prediction - avg) / std
            
            if z_score > 3.5:
                return False, f"Anomaly: {z_score:.1f}σ"
    
    return True, "Valid"


def daily_predict_api(**context):
    """Predict with XGBoost model"""
    print("=" * 70)
    print(" " * 20 + "DAY 2: PREDICTING")
    print("=" * 70)
    
    features_data = context["ti"].xcom_pull(key="daily_features", task_ids="build_daily_features")
    
    if not features_data:
        raise ValueError("No features!")
    
    model_path = Variable.get("latest_rf_model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"🤖 Loading: {model_path}\n")
    model = joblib.load(model_path)
    
    # MUST match training features
    FEATURES = [
        "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7",
        "min_temp_lag1", "max_temp_lag1", "temp_range_lag1",
        "temp_ma3", "temp_ma7", "temp_std7",
        "min_temp_ma7", "max_temp_ma7",
        "temp_change", "temp_change_3d",
        "temp_min7", "temp_max7", "temp_range7",
        "temp_volatility",
        "day_of_week", "month_num", "year_num", "day_of_year",
        "season", "is_weekend"
    ]
    
    predictions = []
    invalid = []
    
    params = _get_pg_conn()
    
    with psycopg2.connect(**params) as conn:
        for f in features_data:
            city = f["city"]
            
            # Build feature vector
            X = np.array([[f[k] for k in FEATURES]])
            pred = float(model.predict(X)[0])
            
            is_valid, reason = validate_prediction(
                city, pred, f["temp_lag1"], f["month_num"], conn
            )
            
            if is_valid:
                predictions.append({
                    "city": city,
                    "date": f["predict_date"],
                    "prediction": pred,
                    "predicted_at": datetime.utcnow(),
                    "lag1": f["temp_lag1"],
                    "ma7": f["temp_ma7"],
                    "api_temp": f.get("api_temp", None),
                    "api_temp_max": f.get("api_temp_max", None),
                    "api_temp_min": f.get("api_temp_min", None)
                })
                
                print(f"✓ {city}: {pred:.1f}°C")
            else:
                invalid.append({"city": city, "prediction": pred, "reason": reason})
                print(f"❌ {city}: {pred:.1f}°C - {reason}")
    
    print(f"\n✅ Valid: {len(predictions)} | ❌ Invalid: {len(invalid)}")
    print("=" * 70 + "\n")
    
    context["ti"].xcom_push(key="preds", value=predictions)
    
    
def store_daily_predictions(**context):

    print("=" * 70)
    print(" " * 20 + "DAY 2: STORING PREDICTIONS (OPTIMIZED)")
    print("=" * 70)
    
    preds = context["ti"].xcom_pull(key="preds", task_ids="daily_predict_api")
    
    if not preds or len(preds) == 0:
        print("⚠️  No predictions to store")
        return
    
    print(f"💾 Storing {len(preds)} predictions...")
    print(f"📅 Target date: {preds[0]['date']}")

    params = _get_pg_conn()
    params['connect_timeout'] = 10
    
    conn = None
    try:
        print("🔌 Connecting...")
        conn = psycopg2.connect(**params)
        conn.set_session(autocommit=False)
        print("✅ Connected!")
        
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 60000;")
        
        # ════════════════════════════════════════════
        # Create table if not exists
        # ════════════════════════════════════════════
        print("📋 Ensuring table exists...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS weather_predictions (
                city TEXT NOT NULL,
                date DATE NOT NULL,
                prediction DOUBLE PRECISION,
                predicted_at TIMESTAMP,
                lag1 DOUBLE PRECISION,
                ma7 DOUBLE PRECISION,
                api_temp DOUBLE PRECISION,
                api_temp_max DOUBLE PRECISION,
                api_temp_min DOUBLE PRECISION,
                PRIMARY KEY (city, date)
            );
        """)
        print("✅ Table ready")
        
        # ════════════════════════════════════════════
        # Check existing records
        # ════════════════════════════════════════════
        target_date = preds[0]['date']
        print(f"🔍 Checking for existing records on {target_date}...")
        
        cur.execute("""
            SELECT city, date 
            FROM weather_predictions 
            WHERE date = %s
        """, (target_date,))
        
        existing_records = cur.fetchall()
        existing_cities = {row[0] for row in existing_records}
        
        if existing_records:
            print(f"⚠️  Found {len(existing_records)} existing records for {target_date}")
            print(f"   Cities: {', '.join(existing_cities)}")
        else:
            print(f"✅ No existing records for {target_date}")
        
        # ════════════════════════════════════════════
        # Separate UPDATE and INSERT operations
        # ════════════════════════════════════════════
        records_to_update = []
        records_to_insert = []
        
        for p in preds:
            record_data = (
                p["city"], p["date"], p["prediction"], 
                p["predicted_at"], p.get("lag1"), p.get("ma7"),
                p.get("api_temp"), p.get("api_temp_max"), p.get("api_temp_min")
            )
            
            if p["city"] in existing_cities:
                records_to_update.append(record_data)
            else:
                records_to_insert.append(record_data)
        
        # ════════════════════════════════════════════
        # UPDATE existing records
        # ════════════════════════════════════════════
        if records_to_update:
            print(f"🔄 Updating {len(records_to_update)} existing records...")
            from psycopg2.extras import execute_values
            
            execute_values(
                cur,
                """
                UPDATE weather_predictions SET
                    prediction = data.prediction,
                    predicted_at = data.predicted_at,
                    lag1 = data.lag1,
                    ma7 = data.ma7,
                    api_temp = data.api_temp,
                    api_temp_max = data.api_temp_max,
                    api_temp_min = data.api_temp_min
                FROM (VALUES %s) AS data(
                    city, date, prediction, predicted_at, lag1, ma7, 
                    api_temp, api_temp_max, api_temp_min
                )
                WHERE weather_predictions.city = data.city 
                    AND weather_predictions.date = data.date::DATE
                """,
                records_to_update,
                page_size=500
            )
            print(f"✅ Updated {len(records_to_update)} records")
        
        # ════════════════════════════════════════════
        # INSERT new records
        # ════════════════════════════════════════════
        if records_to_insert:
            print(f"➕ Inserting {len(records_to_insert)} new records...")
            from psycopg2.extras import execute_values
            
            execute_values(
                cur,
                """
                INSERT INTO weather_predictions 
                (city, date, prediction, predicted_at, lag1, ma7, 
                    api_temp, api_temp_max, api_temp_min)
                VALUES %s
                """,
                records_to_insert,
                page_size=500
            )
            print(f"✅ Inserted {len(records_to_insert)} records")
        
        conn.commit()
        print(f"\n🎉 Success! Total processed: {len(preds)} predictions")
        print(f"   - Updated: {len(records_to_update)}")
        print(f"   - Inserted: {len(records_to_insert)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
    
    print("=" * 70 + "\n")
    
    
def fetch_city_actual(city, target_date):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        temp = data["main"]["temp"]
        temp_min = data["main"]["temp_min"]
        temp_max = data["main"]["temp_max"]
        return {
            "city": city,
            "date": target_date,
            "temp": temp,
            "temp_min": temp_min,
            "temp_max": temp_max
        }
    except:
        return None
    
    
    
    
def fetch_end_of_day_actuals(**context):
    """
    Fetch actual temperatures for all cities
    Runs ONLY after fixed End-Of-Day time (23:00 UTC)
    """
    
    from datetime import datetime, timezone, time as dtime
    import os
    import time
    import pandas as pd
    import psycopg2
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    EOD_HOUR = 23  # 🔒 FIXED END-OF-DAY (23:00 UTC)
    
    print("=" * 70)
    print(" " * 15 + "FETCHING END-OF-DAY ACTUAL TEMPERATURES")
    print("=" * 70)
    
    now = datetime.now(timezone.utc)
    
    # 🔒 HARD FIX: do not run before EOD
    if now.time() < dtime(EOD_HOUR, 0):
        print(f"⏳ Current UTC time {now.strftime('%H:%M')} < {EOD_HOUR}:00")
        print("❌ Not end-of-day yet — skipping actuals fetch")
        print("=" * 70 + "\n")
        return
    
    today = now.date()
    
    print(f"📅 Date (UTC): {today}")
    print(f"⏰ Time (UTC): {now.strftime('%H:%M:%S')}\n")
    
    # ════════════════════════════════════════════
    # Load cities list
    # ════════════════════════════════════════════
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    
    df = pd.read_csv(CLEAN_CSV)[["city"]].drop_duplicates()
    cities = df["city"].tolist()
    
    print(f"📍 Cities to fetch: {len(cities)}")
    
    # ════════════════════════════════════════════
    # Parallel fetch with ThreadPoolExecutor
    # ════════════════════════════════════════════
    results = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_city = {
            executor.submit(fetch_city_actual, city, today): city
            for city in cities
        }
        
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            
            try:
                result = future.result()
                
                if result:
                    results.append(result)
                    print(f"✓ {city}: {result['temp']:.1f}°C")
                else:
                    failed.append(city)
            
            except Exception as e:
                print(f"❌ {city}: {e}")
                failed.append(city)
            
            # Small delay to respect API rate limits
            time.sleep(0.1)
    
    # ════════════════════════════════════════════
    # Store results to database
    # ════════════════════════════════════════════
    if results:
        print(f"\n💾 Storing {len(results)} actuals to database...")
        
        params = _get_pg_conn()
        
        with psycopg2.connect(**params) as conn:
            cur = conn.cursor()
            
            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS weather_actuals(
                    city TEXT,
                    date DATE,
                    temp DOUBLE PRECISION,
                    temp_max DOUBLE PRECISION,
                    temp_min DOUBLE PRECISION,
                    fetched_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY(city, date)
                );
            """)
            
            # Create index for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_actuals_city_date 
                ON weather_actuals(city, date DESC);
            """)
            
            # Insert actuals using ON CONFLICT (upsert)
            for r in results:
                cur.execute("""
                    INSERT INTO weather_actuals(
                        city, date, temp, temp_max, temp_min, fetched_at
                    )
                    VALUES(%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT(city, date) DO UPDATE
                    SET temp = EXCLUDED.temp,
                        temp_max = EXCLUDED.temp_max,
                        temp_min = EXCLUDED.temp_min,
                        fetched_at = EXCLUDED.fetched_at
                """, (
                    r["city"],
                    r["date"],
                    r["temp"],
                    r["temp_max"],
                    r["temp_min"]
                ))
            
            conn.commit()
            
            # ════════════════════════════════════════════
            # Cleanup old data (keep only last 30 days)
            # ════════════════════════════════════════════
            print("🧹 Cleaning old data (keeping last 30 days)...")
            cur.execute("""
                DELETE FROM weather_actuals
                WHERE date < CURRENT_DATE - INTERVAL '30 days'
            """)
            
            deleted = cur.rowcount
            conn.commit()
            
            if deleted > 0:
                print(f"   Deleted {deleted} old records")
    
    # ════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"✅ Success: {len(results)} cities")
    print(f"❌ Failed: {len(failed)} cities")
    
    if failed and len(failed) <= 10:
        print(f"Failed cities: {', '.join(failed)}")
    
    print("=" * 70 + "\n")
    
    # Push to XCom for monitoring
    context["ti"].xcom_push(key="actuals_count", value=len(results))
    context["ti"].xcom_push(key="failed_count", value=len(failed))
    



def calculate_daily_accuracy(**context):
    """Calculate prediction accuracy for today"""
    print("=" * 70)
    print(" " * 20 + "CALCULATING PREDICTION ACCURACY")
    print("=" * 70)
    
    today = datetime.utcnow().date()
    
    print(f"📅 Evaluating predictions for: {today}\n")
    
    params = _get_pg_conn()
    
    with psycopg2.connect(**params) as conn:
        cur = conn.cursor()
        
        # Get predictions vs actuals for today
        cur.execute("""
            SELECT 
                p.city,
                p.prediction,
                a.temp as actual,
                ABS(p.prediction - a.temp) as absolute_error
            FROM weather_predictions p
            INNER JOIN weather_actuals a 
                ON p.city = a.city AND p.date = a.date
            WHERE p.date = %s
        """, (today,))
        
        results = cur.fetchall()
        
        if not results:
            print("⚠️  No predictions found for today")
            return
        
        # Calculate metrics
        errors = [r[3] for r in results]
        
        mae = sum(errors) / len(errors)
        rmse = (sum([e**2 for e in errors]) / len(errors)) ** 0.5
        
        within_2 = sum([1 for e in errors if e <= 2])
        within_5 = sum([1 for e in errors if e <= 5])
        
        print(f"📊 Metrics for {len(results)} cities:")
        print(f"   MAE:  {mae:.2f}°C")
        print(f"   RMSE: {rmse:.2f}°C")
        print(f"   Within ±2°C: {within_2}/{len(results)} ({within_2/len(results)*100:.1f}%)")
        print(f"   Within ±5°C: {within_5}/{len(results)} ({within_5/len(results)*100:.1f}%)")
        
        # Show best and worst predictions
        sorted_results = sorted(results, key=lambda x: x[3])
        
        print("\n🏆 Best predictions (lowest error):")
        for city, pred, actual, error in sorted_results[:5]:
            print(f"   {city}: predicted {pred:.1f}°C, actual {actual:.1f}°C (error: {error:.2f}°C)")
        
        print("\n😓 Worst predictions (highest error):")
        for city, pred, actual, error in sorted_results[-5:]:
            print(f"   {city}: predicted {pred:.1f}°C, actual {actual:.1f}°C (error: {error:.2f}°C)")
        
        # Store metrics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics(
                date DATE PRIMARY KEY,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                total_predictions INT,
                within_2_degrees INT,
                within_5_degrees INT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        cur.execute("""
            INSERT INTO model_metrics(date, mae, rmse, total_predictions, within_2_degrees, within_5_degrees)
            VALUES(%s, %s, %s, %s, %s, %s)
            ON CONFLICT(date) DO UPDATE
            SET mae = EXCLUDED.mae,
                rmse = EXCLUDED.rmse,
                total_predictions = EXCLUDED.total_predictions,
                within_2_degrees = EXCLUDED.within_2_degrees,
                within_5_degrees = EXCLUDED.within_5_degrees
        """, (today, mae, rmse, len(results), within_2, within_5))
        
        conn.commit()
    
    print("\n" + "=" * 70 + "\n")


# =========================================
# DAG DEFINITION
# =========================================

default_args = {
    "owner": "ahmed_sami",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="weather_prediction_pipeline_v2",
    description="Updated pipeline for new data format with min/max temperatures",
    start_date=datetime(2025, 12, 1),
    schedule_interval="5 18 * * *",  # Daily at 08:05 AM
    catchup=False,
    default_args=default_args,
    tags=["weather", "ml", "xgboost", "v2"]
) as dag:
    
    # ============ BRANCH DECISION ============
    check = BranchPythonOperator(
        task_id="check_day_one",
        python_callable=check_day_one
    )
    
    # ============ DAY 1 TASKS ============
    extract = PythonOperator(
        task_id="extract_csv",
        python_callable=extract_csv
    )
    
    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data
    )
    
    train = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=train_and_evaluate
    )
    
    load = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres
    )
    
    done = PythonOperator(
        task_id="mark_day_one_done",
        python_callable=mark_day_one_done
    )
    
    # ============ SKIP DAY 1 ============
    skip = DummyOperator(task_id="skip_day1")
    
    # ============ JOIN POINT ============
    join = DummyOperator(
        task_id="join",
        trigger_rule="none_failed_min_one_success"
    )
    
    # ============ DAY 2 TASKS ============
    weather = PythonOperator(
        task_id="weather_api",
        python_callable=weather_api
    )
    
    features = PythonOperator(
        task_id="build_daily_features",
        python_callable=build_daily_features
    )
    
    predict = PythonOperator(
        task_id="daily_predict_api",
        python_callable=daily_predict_api
    )
    
    store = PythonOperator(
        task_id="store_daily_predictions",
        python_callable=store_daily_predictions
    )
    
    # ============ TASK DEPENDENCIES ============
    
    # Branch decision
    check >> [extract, skip]
    
    # Day 1 path
    extract >> transform >> train >> load >> done >> join
    
    # Day 2 path (skip Day 1)
    skip >> join
    
    # Day 2 tasks (always run after join)
    join >> weather >> features >> predict >> store


# =========================================
# SECOND DAG: ACTUALS COLLECTION
# =========================================

with DAG(
    dag_id="weather_actuals_pipeline_v2",
    description="End-of-day: Collect actual temperatures with min/max",
    start_date=datetime(2025, 12, 1),
    schedule_interval="0 18 * * *",  # 23:50 PM daily
    catchup=False,
    default_args=default_args,
    tags=["weather", "actuals", "daily", "v2"]
) as dag:
    
    start = DummyOperator(task_id="start")
    
    fetch = PythonOperator(
        task_id="fetch_end_of_day_actuals",
        python_callable=fetch_end_of_day_actuals
    )
    
    accuracy = PythonOperator(
        task_id="calculate_daily_accuracy",
        python_callable=calculate_daily_accuracy
    )
    
    end = DummyOperator(task_id="end")
    
    # Task dependencies
    start >> fetch >> accuracy >> end
    
    
    
    

