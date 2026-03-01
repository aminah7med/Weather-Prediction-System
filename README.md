
<br>

<h1 align="center">Weather Prediction System</h1>
<h3 align="center">End-to-End Data Engineering & Machine Learning Pipeline</h3>

<p align="center">
A production-grade weather prediction system that processes historical and real-time data through a fully automated, orchestrated pipeline.
</p>

<hr>

<h2 align="center">Project Overview</h2>

This project implements a complete Data Engineering + Machine Learning workflow, including:
<br>
*   Data Ingestion (API / CSV Sources)
*   Data Cleaning & Transformation
*   Feature Engineering
*   Model Training & Evaluation
*   Daily Prediction Storage
*   Apache Airflow Orchestration
*   Dockerized Deployment Environment

<br>

<p align="center">The system is designed to simulate a real-world production data pipeline used in modern data platforms.</p>

<hr>

<h2 align="center">System Architecture</h2>

<p align="center">The pipeline is orchestrated using Apache Airflow and containerized via Docker, ensuring modularity and scalability.</p>

<br>

<h3 align="center">Workflow Steps</h3>
<p align="center">
1. Extract weather data<br>
2. Transform & clean data<br>
3. Load processed data into PostgreSQL<br>
4. Train & evaluate ML model<br>
5. Store daily predictions
</p>

<hr>

<h2 align="center">Technology Stack</h2>

<br>

| Layer | Technology |
| :--- | :--- |
| **Programming** | Python |
| **Orchestration** | Apache Airflow |
| **Containerization** | Docker & Docker Compose |
| **Database** | PostgreSQL |
| **Data Processing** | Pandas & NumPy |
| **Machine Learning** | Scikit-learn |
| **Integration** | REST APIs |

<hr>

<h2 align="center">Project Structure</h2>

<pre>
weather-prediction/
│
├── dags/               # Apache Airflow DAG files
├── docker-compose.yml  # Docker services definition
├── Dockerfile          # Docker image build instructions
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
</pre>
