import pandas as pd

df = pd.read_csv("../raw/student_depression_dataset.csv")

health_as_num = {
    "Unhealthy": 1,
    "Moderate": 2,
    "Healthy": 3
}

num_as_health = {
    1: "unhealthy",
    2: "moderate",
    3: "healthy"
}

boolean_map = {
    "Yes": True,
    "No": False
}

df["Profession"] = df["Profession"].apply(
    lambda x: "unemployed" if x == "student" else "employed"
)

df["Sleep Duration"] = df["Sleep Duration"].apply(
    lambda x: "poor" if x == "Less than 5 hours"
    else "fair" if x == "5-6 hours"
    else "good" if x == "7-8 hours"
    else "excellent"
)

df["Degree"] = df["Degree"].apply(
    lambda x: "high school" if x == "Class 12"
    else "bachelor's" if x.startswith('B') or x == "LLB"
    else "master's" if x.startswith('M') or x == "LLM"
    else "phd" if x == "PhD"
    else "other"
)

# Extract score mapping from categorical values
df["Dietary Habits"] = df["Dietary Habits"].map(health_as_num)
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(boolean_map)
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map(boolean_map)

# Calculate health risk score
df["Health_Risks"] = df["Dietary Habits"] - df["Family History of Mental Illness"].astype(int) - df["Have you ever had suicidal thoughts ?"].astype(int)
df["Health_Risks"] = df["Health_Risks"].clip(lower=1, upper=3)
df = df[df["Health_Risks"].notna()]

# Translate sscore back into readable values
df["Health_Risks"] = df["Health_Risks"].map(num_as_health)


# Drop old rolled up columns
df.drop(['id','City',"Dietary Habits", "Family History of Mental Illness", "Have you ever had suicidal thoughts ?"], axis=1,inplace=True)

# Rename columns
df.rename(columns={"Profession": "Employment_Status",
                   "Degree": "Education_Level",
                   "Sleep Duration": "Sleep_Quality"
        },inplace=True)

df['Gender'] = df['Gender'].str.lower()
df.columns = df.columns.str.lower()

# Reorder columns
first_cols = ["gender", "age", "education_level", "employment_status", "sleep_quality", "health_risks"]
other_cols = [col for col in df.columns if col not in first_cols]
new_order = first_cols + other_cols
df = df[new_order]

# Write processed data to csv
df.to_csv("../staging/processed_student_depression_dataset.csv",index=False)