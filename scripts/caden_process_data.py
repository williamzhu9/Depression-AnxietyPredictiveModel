import pandas as pd
from pathlib import Path

# ----- PATHS -----
BASE_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = BASE_DIR.parent                      

INPUT_PATH = PROJECT_ROOT / "raw" / "depression_anxiety_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "staging" / "depression_anxiety_standardized.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def standardize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    if "school_year" in df.columns:
        edu_map = {
            1: "university undergrad",
            2: "university undergrad",
            3: "university undergrad",
            4: "university undergrad",
        }
        df["education_current"] = df["school_year"].map(edu_map)
        df["education_level"] = "bachelors degree"

    df["employment_status_standard"] = "unemployed"
    return df


def standardize_sleep(df: pd.DataFrame) -> pd.DataFrame:
    def estimate_sleep_hours(row):
        score = row.get("epworth_score")
        sleepy = row.get("sleepiness")

        if pd.notna(score):
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

        if score is not None:
            est_hours = 9 - (score / 4.0)
            est_hours = max(3.0, min(10.0, est_hours))
            return int(est_hours)

        sleepy_str = str(sleepy).strip().lower()
        if sleepy_str == "true":
            return 5
        elif sleepy_str == "false":
            return 8
        return None

    if "epworth_score" in df.columns or "sleepiness" in df.columns:
        df["sleep_quality_score"] = df.apply(estimate_sleep_hours, axis=1)

        def classify_from_score(x):
            if pd.isna(x):
                return "unknown"
            if x < 6:
                return "poor"
            elif 6 <= x <= 7:
                return "fair"
            else:
                return "good"

        df["sleep_quality_cat"] = df["sleep_quality_score"].apply(classify_from_score)

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
        df["physical_health_score"] = df["who_bmi"].map(health_score_map)
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
            diag_true = _parse_bool_like(df["depression_diagnosis"])
            df["depression_diagnosed"] = diag_true.astype("Int64")

        if "depression_treatment" in df.columns:
            treated_true = _parse_bool_like(df["depression_treatment"])
            df["depression_treated"] = treated_true.astype("Int64")

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
            diag_true = _parse_bool_like(df["anxiety_diagnosis"])
            df["anxiety_diagnosed"] = diag_true.astype("Int64")

        if "anxiety_treatment" in df.columns:
            treated_true = _parse_bool_like(df["anxiety_treatment"])
            df["anxiety_treated"] = treated_true.astype("Int64")

    return df


def main():
    df_raw = pd.read_csv(INPUT_PATH)
    df = df_raw.copy()

    df = standardize_demographics(df)
    df = standardize_sleep(df)
    df = standardize_bmi(df)
    df = standardize_depression(df)
    df = standardize_anxiety(df)

    df = df.dropna()

    cols_to_drop = [
        "id",
        "bmi",
        "who_bmi",
        "depression_severity",
        "depressiveness",
        "suicidal",
        "depression_diagnosis",
        "depression_treatment",
        "gad_score",
        "anxiety_severity",
        "anxiousness",
        "anxiety_diagnosis",
        "anxiety_treatment",
        "epworth_score",
        "sleepiness",
        "phq_score",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved standardized file to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
