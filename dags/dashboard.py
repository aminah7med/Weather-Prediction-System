
# # def _get_pg_conn():
# #     """Get PostgreSQL connection parameters"""
# #     # conn = BaseHook.get_connection(POSTGRES_CONN_ID)
# #     return {
# #         "host": 'postgres',
# #         "dbname": 'airflow',
# #         "user": 'airflow',
# #         "password":'airflow',
# #         "port":  5432
# #     }
    
# # print('sdcsdvsd')
# # if(_get_pg_conn()):
# #     print ('hwfwefwef')










# dashboard/app.py
# Weather ML Dashboard - Fixed Final Production Version
# Author: Ahmed Sami

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from datetime import datetime, timedelta
import numpy as np


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="🌤️ Weather Prediction Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# DATABASE CONNECTION (FIXED)
# =========================================
@st.cache_resource(show_spinner="🔌 Connecting to database...")
def get_connection():
    try:
        conn = psycopg2.connect(
            host="postgres",      # docker-compose service name
            dbname="airflow",
            user="airflow",
            password="airflow",
            port=5432
        )
        return conn
    except Exception as e:
        st.error("❌ Failed to connect to PostgreSQL")
        st.error(str(e))
        st.stop()

conn = get_connection()



@st.cache_data(ttl=60, show_spinner=False)
def run_query(query, params=None):
    """Execute query with caching and error handling"""
    try:
        df = pd.read_sql(query, conn, params=params)
        return df
    except psycopg2.Error as e:
        st.error(f"❌ Query Error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        return pd.DataFrame()

# =========================================
# HEADER
# =========================================
st.markdown('<h1 class="main-title">🌤️ Weather Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Machine Learning Weather Forecasting System powered by XGBoost</p>', unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    # Date selector
    today = datetime.now().date()
    selected_date = st.date_input(
        "📅 Select Date",
        value=today,
        max_value=today + timedelta(days=1),
        key="date_selector"
    )
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Data refreshed!")
        st.rerun()
    
    st.markdown("---")
    
    # System Info
    st.markdown("### 📊 System Info")
    st.caption("🤖 Model: XGBoost")
    st.caption("📈 Features: 17")
    st.caption("⏰ Predictions: Daily @ 8:00 AM UTC")
    st.caption("🌡️ Actuals: Daily @ 00:05 AM UTC")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    
    # Total cities
    cities_df = run_query("SELECT COUNT(DISTINCT city) as count FROM weather_ml")
    if not cities_df.empty:
        st.metric("Cities Tracked", cities_df.iloc[0]['count'])
    
    # Total predictions
    total_preds = run_query("SELECT COUNT(*) as count FROM weather_predictions")
    if not total_preds.empty:
        st.metric("Total Predictions", f"{total_preds.iloc[0]['count']:,}")
    
    # Latest MAE
    latest_mae = run_query("""
        SELECT mae 
        FROM model_metrics 
        ORDER BY date DESC 
        LIMIT 1
    """)
    if not latest_mae.empty and latest_mae.iloc[0]['mae']:
        st.metric("Latest MAE", f"{latest_mae.iloc[0]['mae']:.2f}°C")

# =========================================
# KEY METRICS ROW
# =========================================
st.markdown("### 📊 Today's Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# Last prediction time
last_pred = run_query("SELECT MAX(predicted_at) as time FROM weather_predictions")
last_time = None
if not last_pred.empty and pd.notna(last_pred.iloc[0]['time']):
    last_time = pd.to_datetime(last_pred.iloc[0]['time'])

with col1:
    if last_time:
        st.metric("⏰ Last Prediction", last_time.strftime("%H:%M"))
    else:
        st.metric("⏰ Last Prediction", "N/A")

# Today's predictions count
today_count = run_query("""
    SELECT COUNT(*) as count 
    FROM weather_predictions 
    WHERE date = %s
""", (selected_date,))

with col2:
    count = today_count.iloc[0]['count'] if not today_count.empty else 0
    st.metric("🔮 Predictions", count)

# Today's MAE
mae_query = """
    SELECT ROUND(AVG(ABS(p.prediction - a.temp))::numeric, 2) as mae
    FROM weather_predictions p
    INNER JOIN weather_actuals a 
        ON p.city = a.city AND p.date = a.date
    WHERE p.date = %s
"""
mae_df = run_query(mae_query, (selected_date,))

with col3:
    if not mae_df.empty and pd.notna(mae_df.iloc[0]['mae']):
        mae = float(mae_df.iloc[0]['mae'])
        delta_color = "normal" if mae < 2.5 else "inverse"
        st.metric("📉 MAE", f"{mae:.2f}°C", 
                delta="Excellent" if mae < 2.0 else "Good" if mae < 3.0 else "Fair",
                delta_color=delta_color)
    else:
        st.metric("📉 MAE", "Waiting...", help="Actuals not collected yet")

# Accuracy ±2°C
with col4:
    if not mae_df.empty and pd.notna(mae_df.iloc[0]['mae']):
        acc_query = """
            SELECT ROUND(
                COUNT(CASE WHEN ABS(p.prediction - a.temp) <= 2 THEN 1 END)::numeric * 100.0 /
                NULLIF(COUNT(a.temp), 0), 1
            ) as accuracy
            FROM weather_predictions p
            INNER JOIN weather_actuals a 
                ON p.city = a.city AND p.date = a.date
            WHERE p.date = %s
        """
        acc_df = run_query(acc_query, (selected_date,))
        
        if not acc_df.empty and pd.notna(acc_df.iloc[0]['accuracy']):
            accuracy = float(acc_df.iloc[0]['accuracy'])
            st.metric("🎯 Accuracy (±2°C)", f"{accuracy:.1f}%")
        else:
            st.metric("🎯 Accuracy", "N/A")
    else:
        st.metric("🎯 Accuracy", "Waiting...")

# Verified count
with col5:
    if not mae_df.empty and pd.notna(mae_df.iloc[0]['mae']):
        verified_query = """
            SELECT COUNT(*) as count
            FROM weather_predictions p
            INNER JOIN weather_actuals a 
                ON p.city = a.city AND p.date = a.date
            WHERE p.date = %s
        """
        verified_df = run_query(verified_query, (selected_date,))
        
        if not verified_df.empty:
            st.metric("✅ Verified", verified_df.iloc[0]['count'])
        else:
            st.metric("✅ Verified", "0")
    else:
        st.metric("✅ Verified", "0")

st.markdown("---")

# =========================================
# TABS
# =========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔮 Today's Predictions",
    "📈 Model Performance",
    "🏆 City Rankings"
])

# =========================================
# TAB 1: OVERVIEW
# =========================================
with tab1:
    st.header("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Last 7 Days - Prediction Volume")
        
        trend_query = """
            SELECT date, COUNT(*) as predictions
            FROM weather_predictions
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY date 
            ORDER BY date
        """
        trend_df = run_query(trend_query)
        
        if not trend_df.empty:
            fig = px.bar(
                trend_df,
                x='date',
                y='predictions',
                title="Daily Prediction Count",
                labels={'date': 'Date', 'predictions': 'Predictions'},
                color='predictions',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ No predictions in the last 7 days")
    
    with col2:
        st.subheader("📈 Model Accuracy Trend (30 Days)")
        
        metrics_query = """
            SELECT date, 
                   mae,
                   (within_2_degrees::float / NULLIF(total_predictions, 0) * 100) as accuracy_pct
            FROM model_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY date
        """
        metrics_df = run_query(metrics_query)
        
        if not metrics_df.empty and len(metrics_df) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['mae'],
                name="MAE (°C)",
                line=dict(color='#f5576c', width=3),
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['accuracy_pct'],
                name="Accuracy %",
                line=dict(color='#11998e', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                yaxis=dict(title="MAE (°C)", side="left"),
                yaxis2=dict(title="Accuracy %", side="right", overlaying="y", range=[0, 100]),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Accumulating accuracy metrics...")
    
    st.markdown("---")
    
    # System Status
    st.subheader("💚 System Status")
    
    # Check recent activity
    recent_check = run_query("""
        SELECT COUNT(*) as count
        FROM weather_predictions
        WHERE predicted_at >= NOW() - INTERVAL '24 hours'
    """)
    
    if not recent_check.empty and recent_check.iloc[0]['count'] > 0:
        st.success("✅ **System Operational** - Pipeline running normally")
    else:
        st.warning("⚠️ **No Recent Activity** - No predictions in the last 24 hours")

# =========================================
# TAB 2: TODAY'S PREDICTIONS
# =========================================
with tab2:
    st.header(f"🔮 Predictions for {selected_date.strftime('%Y-%m-%d')}")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_verified_only = st.checkbox("Show verified only", value=False)
    
    with col2:
        sort_option = st.selectbox(
            "Sort by",
            ["Error (High-Low)", "Error (Low-High)", "City (A-Z)", "Prediction (High-Low)"]
        )
    
    # Build query
    if show_verified_only:
        preds_query = """
            SELECT 
                p.city,
                ROUND(p.prediction::numeric, 2) as prediction,
                ROUND(a.temp::numeric, 2) as actual,
                ROUND(ABS(p.prediction - a.temp)::numeric, 2) as error,
                p.predicted_at::time as time
            FROM weather_predictions p
            INNER JOIN weather_actuals a 
                ON p.city = a.city AND p.date = a.date
            WHERE p.date = %s
        """
    else:
        preds_query = """
            SELECT 
                p.city,
                ROUND(p.prediction::numeric, 2) as prediction,
                ROUND(a.temp::numeric, 2) as actual,
                ROUND(ABS(p.prediction - a.temp)::numeric, 2) as error,
                p.predicted_at::time as time
            FROM weather_predictions p
            LEFT JOIN weather_actuals a 
                ON p.city = a.city AND p.date = a.date
            WHERE p.date = %s
        """
    
    # Add sorting
    if sort_option == "Error (High-Low)":
        preds_query += " ORDER BY error DESC NULLS LAST"
    elif sort_option == "Error (Low-High)":
        preds_query += " ORDER BY error ASC NULLS LAST"
    elif sort_option == "City (A-Z)":
        preds_query += " ORDER BY p.city"
    else:  # Prediction (High-Low)
        preds_query += " ORDER BY p.prediction DESC"
    
    preds_df = run_query(preds_query, (selected_date,))
    
    if preds_df.empty:
        st.info("⏳ No predictions available for this date yet")
    else:
        # Color coding function
        def color_error(val):
            if pd.isna(val):
                return ''
            if val <= 2:
                return 'background-color: #90EE90'  # Green
            elif val <= 5:
                return 'background-color: #FFD700'  # Yellow
            else:
                return 'background-color: #FFB6C1'  # Red
        
        # Apply styling
        styled_df = preds_df.style.applymap(color_error, subset=['error'])
        
        # Display
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        st.caption("🟢 Green: Error ≤ 2°C | 🟡 Yellow: Error ≤ 5°C | 🔴 Red: Error > 5°C")
        
        # Download button
        csv = preds_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"predictions_{selected_date}.csv",
            mime="text/csv"
        )

# =========================================
# TAB 3: MODEL PERFORMANCE
# =========================================
with tab3:
    st.header("📈 Model Performance Analysis")
    
    metrics_query = """
        SELECT 
            date, 
            mae, 
            rmse, 
            within_2_degrees, 
            within_5_degrees,
            total_predictions,
            ROUND((within_2_degrees::numeric / NULLIF(total_predictions, 0) * 100), 1) as acc_2c,
            ROUND((within_5_degrees::numeric / NULLIF(total_predictions, 0) * 100), 1) as acc_5c
        FROM model_metrics
        ORDER BY date DESC 
        LIMIT 30
    """
    
    metrics_df = run_query(metrics_query)
    
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values('date')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 MAE Trend")
            
            fig1 = px.line(
                metrics_df,
                x='date',
                y='mae',
                title="Mean Absolute Error Over Time",
                labels={'date': 'Date', 'mae': 'MAE (°C)'},
                markers=True
            )
            fig1.add_hline(y=2.5, line_dash="dash", line_color="green", 
                          annotation_text="Target: 2.5°C")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Accuracy Trend")
            
            fig2 = px.line(
                metrics_df,
                x='date',
                y='acc_2c',
                title="Accuracy (±2°C) Over Time",
                labels={'date': 'Date', 'acc_2c': 'Accuracy (%)'},
                markers=True
            )
            fig2.add_hline(y=70, line_dash="dash", line_color="green",
                          annotation_text="Target: 70%")
            fig2.update_layout(height=400, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Summary Stats
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_mae = metrics_df['mae'].mean()
            st.metric("Average MAE", f"{avg_mae:.2f}°C")
        
        with col2:
            avg_rmse = metrics_df['rmse'].mean()
            st.metric("Average RMSE", f"{avg_rmse:.2f}°C")
        
        with col3:
            avg_acc = metrics_df['acc_2c'].mean()
            st.metric("Avg Accuracy (±2°C)", f"{avg_acc:.1f}%")
        
        with col4:
            avg_acc_5 = metrics_df['acc_5c'].mean()
            st.metric("Avg Accuracy (±5°C)", f"{avg_acc_5:.1f}%")
        
        st.markdown("---")
        
        # Recent Performance Table
        st.subheader("📋 Recent Performance (Last 14 Days)")
        
        recent_df = metrics_df.tail(14).copy()
        recent_df['date'] = pd.to_datetime(recent_df['date']).dt.strftime('%Y-%m-%d')
        
        display_df = recent_df[[
            'date', 'mae', 'rmse', 'acc_2c', 'acc_5c', 'total_predictions'
        ]].rename(columns={
            'date': 'Date',
            'mae': 'MAE (°C)',
            'rmse': 'RMSE (°C)',
            'acc_2c': 'Accuracy ±2°C (%)',
            'acc_5c': 'Accuracy ±5°C (%)',
            'total_predictions': 'Predictions'
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("⏳ Accumulating performance metrics...")

# =========================================
# TAB 4: CITY RANKINGS
# =========================================
with tab4:
    st.header("🏆 City Performance Rankings (Last 7 Days)")
    
    city_perf_query = """
        SELECT 
            p.city,
            COUNT(*) as predictions,
            ROUND(AVG(p.prediction)::numeric, 2) as avg_prediction,
            ROUND(AVG(a.temp)::numeric, 2) as avg_actual,
            ROUND(AVG(ABS(p.prediction - a.temp))::numeric, 2) as avg_error
        FROM weather_predictions p
        LEFT JOIN weather_actuals a 
            ON p.city = a.city AND p.date = a.date
        WHERE p.date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY p.city
        HAVING COUNT(*) >= 3
        ORDER BY avg_error ASC NULLS LAST
    """
    
    city_perf_df = run_query(city_perf_query)
    
    if not city_perf_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥇 Top 10 Best Cities (Lowest Error)")
            
            best_cities = city_perf_df.head(10).copy()
            
            fig = px.bar(
                best_cities,
                x='avg_error',
                y='city',
                orientation='h',
                title="Most Accurate Predictions",
                labels={'avg_error': 'Average Error (°C)', 'city': 'City'},
                color='avg_error',
                color_continuous_scale='Greens_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                best_cities.rename(columns={
                    'city': 'City',
                    'predictions': 'Predictions',
                    'avg_prediction': 'Avg Predicted (°C)',
                    'avg_actual': 'Avg Actual (°C)',
                    'avg_error': 'Avg Error (°C)'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.subheader("🔻 Top 10 Worst Cities (Highest Error)")
            
            worst_cities = city_perf_df.tail(10).copy()
            worst_cities = worst_cities.sort_values('avg_error', ascending=False)
            
            fig = px.bar(
                worst_cities,
                x='avg_error',
                y='city',
                orientation='h',
                title="Least Accurate Predictions",
                labels={'avg_error': 'Average Error (°C)', 'city': 'City'},
                color='avg_error',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                worst_cities.rename(columns={
                    'city': 'City',
                    'predictions': 'Predictions',
                    'avg_prediction': 'Avg Predicted (°C)',
                    'avg_actual': 'Avg Actual (°C)',
                    'avg_error': 'Avg Error (°C)'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("⏳ Need at least 7 days of data for city rankings")

# =========================================
# FOOTER
# =========================================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🤖 Powered by XGBoost ML")

with col2:
    st.caption(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.caption("👨‍💻 Built by Ahmed Sami")

st.markdown(
    f"<p style='text-align:center; color:gray; font-size:0.9rem;'>"
    f"Weather ML System v2.0 • © {datetime.now().year}</p>",
    unsafe_allow_html=True
)
















