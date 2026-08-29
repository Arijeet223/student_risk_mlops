# Student Risk Prediction MLOps Project

## Project Overview

This project is an end-to-end Machine Learning and MLOps application designed to predict the academic outcome of a student.

The model predicts one of the following outcomes:

- Dropout
- Enrolled
- Graduate

The project includes the complete machine learning pipeline, a FastAPI prediction API, Docker containerization, automated testing, and Continuous Integration using GitHub Actions.

---

# Project Architecture

```text
Dataset
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Saved ML Model
   ↓
FastAPI Prediction API
   ↓
Docker Container
   ↓
Automated Testing
   ↓
GitHub Actions CI