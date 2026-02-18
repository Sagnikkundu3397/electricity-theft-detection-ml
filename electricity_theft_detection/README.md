# ⚡ Electricity Theft Detection: High-Resilience ML Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-green?logo=kaggle&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

> **Deploying industrial-grade Machine Learning to safeguard smart grid integrity against non-technical losses.**

---

## 🌟 Overview

Electricity theft represents a multi-billion dollar challenge for utilities worldwide. This project implements a **comprehensive, end-to-end Machine Learning pipeline** designed to detect anomalous consumption patterns (theft) in smart grid environments.

### Why This Matters
- **Grid Stability**: High non-technical losses can destabilize regional power grids.
- **Economic Impact**: Theft increases the cost of energy for honest consumers.
- **Operational Efficiency**: Automated detection enables targeted auditing, saving thousands in manual inspection costs.

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
- **Modular Design**: Individual services for ingestion, transformation, and training facilitate easy scaling and maintenance.
- **Advanced Preprocessing**: Implementation of recursive statistical feature extraction (Mean, Std Dev, Min/Max, Zero-readings).
- **Class Balancing**: Integrated **SMOTE** over-sampling to handle severe class imbalance (rare theft cases).

### 2. High-Performance Modeling
We utilize a state-of-the-art **CatBoost** classifier, optimized for categorical data and unbalanced sets, alongside a suite of traditional classifiers (XGBoost, Random Forest, Logistic Regression).

### 3. Modern Inference Portal
A premium web interface built with **Glassmorphism** principles, providing real-time theft probability scores and confidence metrics.

---

## 📊 Performance Metrics

| Metric | Score (Post-SMOTE) | Improvement |
| :--- | :--- | :--- |
| **F1-Score** | `0.28+` | `130%` Increase |
| **AUC-ROC** | `0.75` | Significant Separability |
| **Accuracy** | `~94%` | High Precision |

---

## 🧠 Technical Depth & Challenges

### Handling Extreme Imbalance
With theft cases making up less than 5% of the raw data, the model initially favored the majority class (Normal). By integrating **SMOTE** (Synthetic Minority Over-sampling Technique), we increased the model's awareness of theft patterns, resulting in a **130% boost in F1-Score**.

### Feature Engineering for Forensic Analysis
Instead of raw consumption values, we engineered statistical proxies to detect "human-like" vs "anomalous" usage:
- **Zero-Reading Frequency**: A key indicator of meter bypass or physical tampering.
- **Volatility (Std Dev)**: Captures irregular "spiky" consumption that often correlates with non-technical losses.

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
*Accessible at `http://localhost:5000`*

---

## 📂 Project Structure
- `src/components`: Core ML logic (Ingestion, Transformation, Training).
- `src/pipeline`: Execution workflows.
- `notebooks`: High-quality Exploratory Data Analysis.
- `reports`: Automation health reports and diagnostics.
- `templates`: Premium UI assets.

---

## 🤝 Contributing
Advancing grid resilience is a collective effort. Contributions in the form of PRs or issue reporting are highly valued.

## 📄 License
This project is licensed under the **MIT License**.
