import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

INPUT_PATH = PROJECT_ROOT / "raw" / "depression_anxiety_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "staging" / "depression_anxiety_standardized.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)



def standardize_demographics(df: pd.DataFrame) -> pd.DataFrame:

    if "school_year" in df.columns:
        df["education_level"] = "bachelors degree"
        df["employment_status"] = "unemployed"
    else:
        df["education_level"] = None
        df["employment_status"] = None

    df = df.drop(columns=["school_year"], errors="ignore")

    return df

def standardize_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sleep quality must be derived directly from Epworth, with a 4-category scale.
    Categories aligned to clinical interpretation but mapped into the
    user's 4 discrete bins:
        - <5  → poor
        - 5-6 → fair
        - 7-8 → good
        - 9+  → excellent
    """

    def classify_sleep(row):
        epw = row.get("epworth_score")
        sleepy = row.get("sleepiness")

        try:
            epw = float(epw)
        except (TypeError, ValueError):
            epw = None

        # Ranges taken from official EPW score classification, scores above 10 indicate problems with sleep habits, scores below 10 are normal
        if epw is not None:
            if epw <= 5:
                return "excellent"      
            elif 6 <= epw <= 10:
                return "good"           
            elif 11 <= epw <= 15:
                return "bad"           
            elif epw > 15:
                return "poor"           
        return "unknown"

    df["sleep_quality"] = df.apply(classify_sleep, axis=1)
    return df

def standardize_bmi(df: pd.DataFrame) -> pd.DataFrame:
    if "who_bmi" in df.columns:
        health_score_map = {
            "Normal": 3,
            "Underweight": 1,
            "Overweight": 2,
            "Class I Obesity": 1,
            "Class II Obesity": 0,
            "Class III Obesity": 0,
            "Not Availble": None,
            "Not Available": None,
        }
        num_as_health = {
            1: "unhealthy",
            2: "moderate",
            3: "healthy"
        }
        df["health_risks"] = df["who_bmi"].map(health_score_map).map(num_as_health)
    return df


def _parse_bool_like(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def standardize_depression(df: pd.DataFrame) -> pd.DataFrame:
    if "phq_score" in df.columns:


        bins = [-1, 4, 9, 14, 19, 27]
        labels = [0, 1, 2, 3, 4]

        df["depression_severity_score"] = pd.cut(
            df["phq_score"], bins=bins, labels=labels
        ).astype("Int64")

        df["depression_any_symptoms"] = (df["phq_score"] >= 5).astype("Int64")

        if "depression_diagnosis" in df.columns:
            df["depression_diagnosed"] = _parse_bool_like(df["depression_diagnosis"]).astype("Int64")

        if "depression_treatment" in df.columns:
            df["depression_treated"] = _parse_bool_like(df["depression_treatment"]).astype("Int64")

    return df


def standardize_anxiety(df: pd.DataFrame) -> pd.DataFrame:
    if "gad_score" in df.columns:

        bins = [-1, 4, 9, 14, 21]
        labels = [0, 1, 2, 3]

        df["anxiety_severity_score"] = pd.cut(
            df["gad_score"], bins=bins, labels=labels
        ).astype("Int64")

        df["anxiety_any_symptoms"] = (df["gad_score"] >= 5).astype("Int64")

        if "anxiety_diagnosis" in df.columns:
            df["anxiety_diagnosed"] = _parse_bool_like(df["anxiety_diagnosis"]).astype("Int64")

        if "anxiety_treatment" in df.columns:
            df["anxiety_treated"] = _parse_bool_like(df["anxiety_treatment"]).astype("Int64")

    return df

def main():
    df_raw = pd.read_csv(INPUT_PATH)
    df = df_raw.copy()

    df = standardize_demographics(df)
    df = standardize_sleep(df)
    df = standardize_bmi(df)
    df = standardize_depression(df)
    df = standardize_anxiety(df)

    
    cols_to_drop = [
        "id",
        "bmi",
        "who_bmi",
        "depression_severity",
        "depressiveness",
        "suicidal",
        "depression_diagnosis",
        "depression_treatment",
        "anxiety_severity",
        "anxiousness",
        "anxiety_diagnosis",
        "anxiety_treatment",
        "sleepiness",
        "epworth_score",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df.dropna(inplace=True)
    first_cols = ["gender", "age", "education_level", "employment_status", "sleep_quality", "health_risks"]
    other_cols = [col for col in df.columns if col not in first_cols]
    new_order = first_cols + other_cols
    df = df[new_order]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved standardized file to: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
