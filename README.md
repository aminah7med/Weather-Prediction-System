🌦 Weather Prediction System
🚀 End-to-End Data Engineering & Machine Learning Pipeline

A production-grade weather prediction system that processes historical and real-time data through a fully automated, orchestrated pipeline.

📌 Project Overview

This project implements a complete Data Engineering + Machine Learning workflow, including:

📥 Data Ingestion (API / CSV Sources)

🧹 Data Cleaning & Transformation

🧠 Feature Engineering

🤖 Model Training & Evaluation

📊 Daily Prediction Storage

🔄 Apache Airflow Orchestration

🐳 Dockerized Deployment Environment

The system is designed to simulate a real-world production data pipeline used in modern data platforms.

🏗 System Architecture

The pipeline is orchestrated using Apache Airflow and containerized via Docker, ensuring modularity and scalability.

🔁 Workflow Steps

Extract weather data

Transform & clean data

Load processed data into PostgreSQL

Train & evaluate ML model

Store daily predictions

🛠 Technology Stack
Layer	Technology Used
Programming	Python
Orchestration	Apache Airflow
Containerization	Docker & Docker Compose
Database	PostgreSQL
Data Processing	Pandas & NumPy
Machine Learning	Scikit-learn
Integration	REST APIs

📂 Project Structure
weather-prediction/
│
├── dags/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
