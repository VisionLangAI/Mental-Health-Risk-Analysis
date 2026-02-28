# Psychological Wellbeing Prediction using Interpretable AI (Mental Health Risk)

This repository contains the complete implementation of a novel model for predicting students' psychological wellbeing using a hybrid FT-Transformer + LSTM ensemble architecture. The model incorporates explainability and statistical validation techniques.

## Overview

- **Objective**: Predict mental health risk levels (Low, Medium, High) based on multiple psychological, behavioral, and demographic factors.
- **Approach**: Combines advanced deep learning with explainable AI (XAI) techniques such as SHAP, LIME, and statistical tests.
- **Features Used**: Anxiety, Depression, Productivity, Social Support, Sleep Hours, Age, Stress Level, Activity, Employment, Gender

## Project Structure

- `data/` - Input dataset and preprocessed files
- `models/` - Baseline models and proposed ensemble model
- `explainability/` - SHAP, LIME, Feature Importance Visualizations
- `evaluation/` - Metrics, confusion matrix, calibration plots
- `notebooks/` - Jupyter notebooks for step-by-step execution

## How to Run

1. Install dependencies using `requirements.txt`
2. Run `main.py` to start training and evaluation
3. Visualizations and outputs will be saved in `outputs/`

## Techniques

- FT-Transformer + LSTM + Cross-Attention Attribution Layer (CAAL)
- Entropy-based Feature Engineering
- SHAP and LIME Explainability
- Statistical Testing: ANOVA, Chi-square, Friedman

## 📬 Contact

For questions, reach out via email or GitHub Issues.
