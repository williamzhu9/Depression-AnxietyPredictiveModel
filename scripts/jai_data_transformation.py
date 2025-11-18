import pandas as pd

#importing file 
csv_path = "anxiety_depression_data.csv"
df = pd.read_csv(csv_path)

#assess employment
def assess_employment(employment_status):
    if employment_status == "Student":
        return "Unemployed"
    else:
        return employment_status

#assess sleep quality
def assess_sleep_quality(sleep_hours):
    if sleep_hours < 6:
        return "Poor"
    elif sleep_hours >= 6 and sleep_hours <= 7:
        return "Fair"
    elif sleep_hours > 7:
        return "Good"
    else:
        return "FAILED"

#assessing physical health: phys activity, substance use, chronic illness, work stress
#score out of 10 with different weightings on the categories
#sleep hours are 70%, other's are 10% each 
def assess_phys_health(activity_hours, substance_use, chronic_illness):
    score = 2
    daily_hours = activity_hours/7

    #physical activity builds up the rest of the score out of 10 (70%)
    if daily_hours < 0.18:
        score += 2.6
    elif daily_hours >= 0.18 and daily_hours <= 0.38:
        score += 5.2
    elif daily_hours > 0.38:
        score += 8

    #other factors will decrease the initial 30% accordingly 
    if substance_use == "Occasion":
        score -= 0.5
    elif substance_use == "Frequent":
        score -= 1 
    
    if chronic_illness == 1:
        score -= 1
    
    if score < 4:
        return "Unhealthy"
    if score >= 4 and score < 6:
        return "Moderate"
    if score > 6:
        return "Healthy"

df["Employment_Status"] = df["Employment_Status"].apply(assess_employment)

df["Sleep_Quality"] = df["Sleep_Hours"].apply(assess_sleep_quality)

df["Phys_Health"] = df.apply(
    lambda row: assess_phys_health(
        row["Physical_Activity_Hrs"],
        row["Substance_Use"],
        row["Chronic_Illnesses"],
    ),
    axis=1
)

df.drop(["Gender", "Sleep_Hours"], axis=1, inplace=True)

df.to_csv("processed_anxiety_depression_data.csv", index=False)