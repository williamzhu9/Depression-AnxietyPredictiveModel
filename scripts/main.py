# main.py
import pandas as pd
import joblib
import os
import pickle
from student_depression_processor import preprocess_student_depression

MODEL_DIR = "../models/models_saved"

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path)
    
# Load trained models
model_depression_anxiety_rf = load_model("model_depression_anxiety_rf.pkl")
model_depression_anxiety_xg = load_model("model_depression_anxiety_xg.pkl")
model_student_depression_rf = load_model("model_student_depression_rf.pkl")
model_student_depression_xg = load_model("model_student_depression_xg.pkl")

feature_groups = [
    # features for depression_anxiety
    # [
    #     "school_year",
    #     "age",
    #     "bmi",
    #     "phq_score",
    #     "suicidal",
    #     "depression_diagnosis",
    #     "depression_treatment",
    #     "gad_score",
    #     "anxiousness",
    #     "anxiety_diagnosis",
    #     "anxiety_treatment",
    #     "epworth_score",
    #     "sleepiness",
    #     "gender_female",
    #     "gender_male",
    #     "who_bmi_Class I Obesity",
    #     "who_bmi_Class II Obesity",
    #     "who_bmi_Class III Obesity",
    #     "who_bmi_Normal",
    #     "who_bmi_Not Availble",
    #     "who_bmi_Overweight",
    #     "who_bmi_Underweight",
    #     "depression_severity_Mild",
    #     "depression_severity_Moderate",
    #     "depression_severity_Moderately severe",
    #     "depression_severity_None-minimal",
    #     "depression_severity_Severe",
    #     "depression_severity_none",
    #     "anxiety_severity_0",
    #     "anxiety_severity_Mild",
    #     "anxiety_severity_Moderate",
    #     "anxiety_severity_None-minimal",
    #     "anxiety_severity_Severe"
    # ],  
    # features for student_depression
    [      
        "gender",
        "age",
        "academic pressure",
        "work pressure",
        "cgpa",
        "study satisfaction",
        "job satisfaction",
        "sleep duration",
        "dietary habits",
        "education level",
        "work/study hours",
        "financial stress",
        "employment",
        "have you ever had suicidal thoughts ?",
        "family history of mental illness"
    ],  
]

raw_columns = [
    # Raw columns in depression_anxiety
    # [
    #     "school_year",
    #     "age",
    #     "bmi",
    #     "phq_score",
    #     "suicidal",
    #     "depression_diagnosis",
    #     "depression_treatment",
    #     "gad_score",
    #     "anxiousness",
    #     "anxiety_diagnosis",
    #     "anxiety_treatment",
    #     "epworth_score",
    #     "sleepiness",
    #     "gender_female",
    #     "gender_male",
    #     "who_bmi_Class I Obesity",
    #     "who_bmi_Class II Obesity",
    #     "who_bmi_Class III Obesity",
    #     "who_bmi_Normal",
    #     "who_bmi_Not Availble",
    #     "who_bmi_Overweight",
    #     "who_bmi_Underweight",
    #     "depression_severity_Mild",
    #     "depression_severity_Moderate",
    #     "depression_severity_Moderately severe",
    #     "depression_severity_None-minimal",
    #     "depression_severity_Severe",
    #     "depression_severity_none",
    #     "anxiety_severity_0",
    #     "anxiety_severity_Mild",
    #     "anxiety_severity_Moderate",
    #     "anxiety_severity_None-minimal",
    #     "anxiety_severity_Severe"
    # ],  
    # Raw columns in student_depression
    [      
        "gender",
        "age",
        "academic pressure",
        "work pressure",
        "cgpa",
        "study satisfaction",
        "job satisfaction",
        "sleep duration",
        "dietary habits",
        "degree",
        "work/study hours",
        "financial stress",
        "profession",
        "have you ever had suicidal thoughts ?",
        "family history of mental illness"
    ],  
]

# Load input
input_df = pd.read_csv("../raw/input/student.csv")
input_df.columns = input_df.columns.str.lower()

model_inputs = {}

for i, features in enumerate(raw_columns):
    # Keep only columns that exist in input_df
    cols_to_use = [c for c in features if c in input_df.columns]
    
    # Subset the dataframe
    model_inputs[f"model_{i}"] = input_df[cols_to_use].copy()

processed_inputs = {}
for key, df in model_inputs.items():
    processed_inputs[key] = preprocess_student_depression(df)

predictions = []

model = model_student_depression_rf
df_proc = processed_inputs["model_0"]  # student_depression

preds = model.predict(df_proc)
probs = model.predict_proba(df_proc).max(axis=1)  # confidence per row

pred_df = df_proc.copy()
pred_df["pred_class"] = preds
pred_df["pred_confidence"] = probs

print(pred_df)
pred_df.to_csv("ensemble_predictions.csv", index=False)
