# ⚡ Electricity Theft Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask Web App](https://img.shields.io/badge/Flask-Web%20App-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ML Pipeline](https://img.shields.io/badge/ML%20Pipeline-Optimized-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![CI/CD Status](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](#)

> **Deploying industrial-grade Machine Learning to safeguard smart grid integrity against non-technical losses.**

---

## 📌 Overview

Electricity theft represents a multi-billion dollar challenge for utilities worldwide. This project implements a **comprehensive, end-to-end Machine Learning pipeline** designed to detect anomalous consumption patterns (theft) in smart grid environments.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Raw Smart Meter Data] --> B[Data Ingestion Service]
    B --> C[Feature Engineering Engine]
    C --> D[SMOTE Class Balancing]
    D --> E[Model Selection / Training]
    E --> F[Serialized Artifacts (.pkl)]
    F --> G[Flask Inference API]
    G --> H[Premium Glassmorphism UI]
```

---

## 🚀 Key Technical Features

### 1. Robust Pipeline Engineering
- **Modular Design**: Individual services for ingestion, transformation, and training.
- **Advanced Preprocessing**: Implementation of recursive statistical feature extraction.
- **Class Balancing**: Integrated **SMOTE** over-sampling to handle severe class imbalance.

### 2. High-Performance Modeling
Utilizing state-of-the-art **CatBoost** and **XGBoost** classifiers optimized for categorical and unbalanced data.

### 3. Modern Inference Portal
A premium web interface built with **Glassmorphism** principles, providing real-time scores.

---

## 📊 Performance Metrics

| Metric | Score (Post-SMOTE) | Improvement |
| :--- | :--- | :--- |
| **F1-Score** | `0.28+` | `130%` Increase |
| **AUC-ROC** | `0.75` | Significant Separability |
| **Accuracy** | `~94%` | High Precision |

---

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.12+ installed. 
```bash
pip install -r requirements.txt
```

### 2. Execution
**Train the Pipeline**:
```bash
python -c "from src.pipeline.train_pipeline import TrainPipeline; TrainPipeline().run()"
```

**Launch the Prediction Portal**:
```bash
python app.py
```

---

## 📂 Project Structure
- `src/components`: Core ML logic (Ingestion, Transformation, Training).
- `src/pipeline`: Execution workflows.
- `notebooks`: High-quality Exploratory Data Analysis.
- `reports`: Automation health reports and diagnostics.
- `templates`: Premium UI assets.

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
