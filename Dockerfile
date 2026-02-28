# =========================================
# Dockerfile للمشروع - نسخة مبسطة وجاهزة
# Base: Apache Airflow 2.10.2 مع Python 3.10
# =========================================

FROM apache/airflow:2.10.2-python3.10

# =========================================
# معلومات المشروع
# =========================================
LABEL maintainer="Ahmed Sami"
LABEL version="2.0"
LABEL description="Weather Prediction ML Pipeline"

# =========================================
# تثبيت أدوات النظام (كـ root)
# =========================================
USER root

# تثبيت الأدوات الأساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libpq-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# إنشاء المجلدات المطلوبة
RUN mkdir -p /opt/airflow/models \
    /opt/airflow/dags/aiii_model \
    /opt/airflow/plugins \
    /opt/airflow/logs \
    && chown -R airflow:root /opt/airflow

# =========================================
# تثبيت مكتبات Python (كـ airflow user)
# =========================================
USER airflow

# ترقية pip
RUN pip install --no-cache-dir --upgrade pip==24.3.1

# نسخ ملف المتطلبات
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt

# تثبيت المكتبات من requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# =========================================
# إعدادات البيئة
# =========================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIRFLOW_HOME=/opt/airflow

# =========================================
# مجلد العمل
# =========================================
WORKDIR /opt/airflow

# =========================================
# الـ Entrypoint موجود في الصورة الأساسية
# =========================================