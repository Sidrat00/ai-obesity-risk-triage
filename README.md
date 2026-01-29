# ai-obesity-risk-triage
AI-powered obesity risk stratification and triage tool using ML on lifestyle data.
# AI-Powered Obesity Risk Stratification and Triage Tool

This project uses machine learning to predict multi-class obesity level from demographic and lifestyle data, then converts the predictions into risk tiers with suggested follow-up actions to help clinics prioritize obesity counseling and preventive care.

## Project Goal

- Support primary-care clinics and community health workers in identifying which patients are at highest obesity-related risk so limited counseling and follow-up resources can be focused where they matter most.

## Data

- Updated Obesity Dataset (~2K+ rows) with features such as Age, Gender, Height, Weight, family history of overweight, physical activity (FAF), screen time (TUE), eating habits (FAVC, CAEC, CALC), transport mode (MTRANS), and the target label `NObeyesdad` (Normal_Weight, Overweight levels, Obesity types).
- BMI is derived from Height and Weight, and missing values are handled with `SimpleImputer` (median for numeric features, most frequent for categorical features).

## Methods

- End-to-end ML pipeline in Python (pandas, scikit-learn, TensorFlow).
- Preprocessing with `ColumnTransformer` + `Pipeline` (imputation, standardization of numeric features, one-hot encoding of categorical features).
- Models: Logistic Regression (baseline), Random Forest (best-performing model), optional neural network.
- Evaluation metrics: accuracy, macro-precision, macro-recall, macro-F1, and macro multi-class ROC-AUC.

## Decision-Support Layer

- Map predicted obesity classes to risk tiers (Low / Medium / High).
- Attach suggested follow-up actions for each tier (e.g., intensive counseling vs standard counseling vs routine education).
- Provide a triage table and risk-tier distribution chart to illustrate how a clinic could use the model to prioritize patients.

## Results (Example)

- Random Forest outperforms Logistic Regression on accuracy and macro-F1 on the test set.
- Confusion matrices show that Normal_Weight and the most severe Obesity classes are well distinguished, while neighboring overweight/obesity categories are more frequently confused.
- Feature importance highlights BMI, family history of overweight, and physical activity as key drivers of predicted obesity level.

## Fairness & Limitations

- Simple subgroup checks (e.g., accuracy by gender or transport mode) are used to explore potential performance differences across groups.
- The dataset is not from Bangladeshi clinics or any specific local health system; real deployment would require retraining, calibration, and validation on local data.
- This is a prototype decision-support tool, not a medical device, and should not be used for clinical decisions without proper validation and oversight.

## Future Work

- Build a small web interface (Streamlit or Gradio) so staff can enter patient data and see risk tier + suggested action.
- Retrain and validate the model on local clinic data and calibrate thresholds before deployment.
- Extend fairness analysis and add uncertainty estimates for safer decision-making.
