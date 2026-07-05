# Yaqza Final Project Report

## 1. Project Overview
Yaqza is a predictive maintenance solution for aircraft engine health monitoring. The system combines a FastAPI backend, a lightweight dashboard frontend, and a machine-learning pipeline that predicts Remaining Useful Life (RUL) from engine sensor readings.

## 2. Architecture Summary
- Backend: FastAPI service with SQLite persistence and prediction endpoints.
- Frontend: Browser-based dashboard that calls the backend API.
- Machine Learning: A fallback regression model trained from the provided preprocessed dataset.
- Data Layer: Seeded engine history stored locally for demonstration and testing.

## 3. What Was Fixed
- Resolved the feature-schema mismatch between runtime input features and the saved model artifact.
- Ensured the prediction pipeline uses the same feature columns as the training data.
- Stabilized the API so health, status, and prediction routes return successful results.
- Added configurable public deployment origins via environment variables.

## 4. Validation Results
The backend was verified with end-to-end API calls for the health, status, and prediction routes.

Verified outcomes:
- Health endpoint: status ok
- Status endpoint: service metadata returned successfully
- Prediction endpoint: successful responses for seeded engines ENG001, ENG002, and ENG003
- Model artifact: retrained and saved successfully with validation metrics of approximately:
  - MAE: 13.35
  - RMSE: 18.62
  - R2: 0.80

## 5. Run Instructions
1. Create and activate the Python environment.
2. Install dependencies from requirements.txt and backend/requirements.txt.
3. Seed the database with sample engine data.
4. Start the backend server.
5. Open the dashboard and use the configured API endpoint.

## 6. Deployment Readiness
The application is now structured for local validation and can be exposed publicly with environment-based configuration for the public base URL and allowed origins. The backend is not yet hosted on a public cloud endpoint in this workspace session, but the codebase is prepared for that step.
