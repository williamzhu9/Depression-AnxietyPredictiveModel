import pandas as pd

#importing file 
csv_path = "../raw/anxiety_depression_data.csv"
df = pd.read_csv(csv_path)

boolean_map = {
    1: True,
    2: False
}

#assess employment
def assess_employment(employment_status):
    if employment_status == "Student":
        return "Unemployed"
    else:
        return employment_status

#assess sleep quality
def assess_sleep_quality(sleep_hours):
    if sleep_hours < 5:
        return "poor"
    elif sleep_hours >= 5 and sleep_hours < 7:
        return "bad"
    elif sleep_hours >= 7 and sleep_hours < 9:
        return "good"
    elif sleep_hours >= 9: 
        return "excellent"
    else:
        return None

#assessing physical health: phys activity, substance use, chronic illness, work stress
#score out of 10 with different weightings on the categories
#sleep hours are 70%, other's are 10% each 
def assess_phys_health(activity_hours, substance_use, chronic_illness, medication_use, family_history_mental_illness, therapy, meditation):
    score = 4
    daily_hours = activity_hours/7

    # Daily activity factor on patient's health risks
    # 70 minutes a week, 10 minutes a day of excersise, bare minimum
    if daily_hours < 0.167:
        score += 1
    # Between bare minimum and recommended amount
    elif daily_hours >= 0.167 and daily_hours <= 0.357:
        score -= 0.5
    # 150 minutes a week, classified as healthy/good by WHO & HHS
    elif daily_hours > 0.357:
        score -= 1

    # Historical factors on patient's health risks
    if family_history_mental_illness:
        score += 2
    
    if chronic_illness == 1:
        score += 1

    # Lifestyle factors on patient's health risks
    if substance_use == "Occasional":
        score += 0.25
    elif substance_use == "Frequent":
        score += 0.5
    
    if medication_use == "Occasional":
        score += 0.25
    elif medication_use == "Regular":
        score += 0.5

    if therapy:
        score -=1

    if meditation:
        score -=1


    # Overall final classification of health risk score
    if score >= 6.5:
        return "high"
    if score > 4.5 and score < 6.5:
        return "moderate"
    if score <= 4.5:
        return "low"
    return score

df["Employment_Status"] = df["Employment_Status"].apply(assess_employment)

df["Sleep_Quality"] = df["Sleep_Hours"].apply(assess_sleep_quality)

df["Health_Risks"] = df.apply(
    lambda row: assess_phys_health(
        row["Physical_Activity_Hrs"],
        row["Substance_Use"],
        row["Chronic_Illnesses"],
        row["Medication_Use"],
        row["Family_History_Mental_Illness"],
        row["Therapy"],
        row["Meditation"]
    ),
    axis=1
)

# Drop old rolled up columns
df.drop(["Sleep_Hours", "Medication_Use", "Substance_Use", "Chronic_Illnesses", "Family_History_Mental_Illness", "Therapy", "Meditation", "physical_activity_hrs"], axis=1, inplace=True)

df.columns = df.columns.str.lower()

# Rename columns
df.rename(columns={"education": "education_level",
                   "employment": "employment_status"
        },inplace=True)
df["education_level"] = df["education_level"].str.lower()
df["employment_status"] = df["employment_status"].str.lower()

# Reorder columns
first_cols = ["gender", "age", "education_level", "employment_status", "sleep_quality", "health_risks"]
other_cols = [col for col in df.columns if col not in first_cols]
new_order = first_cols + other_cols
df = df[new_order]

df.to_csv("../staging/processed_anxiety_depression_data.csv", index=False)