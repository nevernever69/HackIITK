import pandas as pd
import numpy as np
import os
from datetime import datetime


# Step 1: Load Raw Logs
def load_raw_logs(log_folder):
    """
    Load raw log files from a folder and combine them into a single DataFrame.
    Assumes logs are in CSV format.
    """
    log_files = [
        os.path.join(log_folder, file)
        for file in os.listdir(log_folder)
        if file.endswith(".csv")
    ]

    # Check if any CSV files were found
    if not log_files:
        print(
            f"Warning: No CSV files found in the folder: {os.path.abspath(log_folder)}"
        )
        return pd.DataFrame()

    raw_logs = pd.concat([pd.read_csv(file) for file in log_files], ignore_index=True)
    return raw_logs


# Step 2: Preprocess Logs
def preprocess_logs(raw_logs):
    """
    Clean and parse raw logs to extract relevant fields.
    """
    # Example: Parse timestamps and extract useful features
    if "date" in raw_logs.columns:
        raw_logs["date"] = pd.to_datetime(raw_logs["date"])
        raw_logs["hour"] = raw_logs["date"].dt.hour
        raw_logs["day_of_week"] = raw_logs["date"].dt.dayofweek  # Monday=0, Sunday=6

    # Handle missing values
    raw_logs.fillna(0, inplace=True)

    return raw_logs


# Step 3: Generate Features
def generate_features(preprocessed_logs):
    """
    Generate structured features from preprocessed logs.
    Handles missing columns gracefully.
    """
    # Initialize default values for features as empty DataFrames
    features = {
        "logon_count": pd.DataFrame(columns=["user", "pc", "logon_count"]),
        "after_hours": pd.DataFrame(columns=["user", "pc", "after_hours"]),
        "thumb_drive_usage": pd.DataFrame(columns=["user", "pc", "thumb_drive_usage"]),
        "url_count": pd.DataFrame(columns=["user", "pc", "url_count"]),
    }

    # Check if required columns exist and calculate features
    if "activity" in preprocessed_logs.columns:
        features["logon_count"] = (
            preprocessed_logs[preprocessed_logs["activity"] == "Logon"]
            .groupby(["user", "pc"])
            .size()
            .reset_index(name="logon_count")
        )
        features["after_hours"] = (
            preprocessed_logs[
                (preprocessed_logs["hour"] < 8) | (preprocessed_logs["hour"] > 18)
            ]
            .groupby(["user", "pc"])
            .size()
            .reset_index(name="after_hours")
        )
        features["thumb_drive_usage"] = (
            preprocessed_logs[preprocessed_logs["activity"] == "connect"]
            .groupby(["user", "pc"])
            .size()
            .reset_index(name="thumb_drive_usage")
        )

    if "url" in preprocessed_logs.columns:
        features["url_count"] = (
            preprocessed_logs.groupby(["user", "pc"])["url"]
            .count()
            .reset_index(name="url_count")
        )

    # Merge all features into a single DataFrame
    structured_data = features["logon_count"]
    for key, df in features.items():
        if key != "logon_count":
            structured_data = pd.merge(
                structured_data, df, on=["user", "pc"], how="left"
            )

    # Fill missing values with 0
    structured_data.fillna(0, inplace=True)

    # Add role information (if available)
    if "role" in preprocessed_logs.columns:
        structured_data = pd.merge(
            structured_data,
            preprocessed_logs[["user", "role"]].drop_duplicates(),
            on="user",
            how="left",
        )
    else:
        structured_data["role"] = 0  # Default role (e.g., regular user)

    return structured_data


# Step 4: Dataset Generation Pipeline
def dataset_generation_pipeline(log_folder, output_file):
    """
    Run the dataset generation pipeline:
    1. Load raw logs.
    2. Preprocess logs.
    3. Generate features.
    4. Save the structured dataset.
    """
    # Step 1: Load raw logs
    raw_logs = load_raw_logs(log_folder)
    if raw_logs.empty:
        print("No data to process. Exiting pipeline.")
        return

    print("Raw logs loaded successfully.")

    # Step 2: Preprocess logs
    preprocessed_logs = preprocess_logs(raw_logs)
    print("Logs preprocessed successfully.")

    # Step 3: Generate features
    structured_data = generate_features(preprocessed_logs)
    print("Features generated successfully.")

    # Step 4: Save the structured dataset
    structured_data.to_csv(output_file, index=False)
    print(f"Structured dataset saved to {output_file}.")


# Main Execution
if __name__ == "__main__":
    # Path to the folder containing raw log files
    log_folder = "data"  # Update this path if necessary
    print(f"Looking for log files in: {os.path.abspath(log_folder)}")

    # Path to save the structured dataset
    output_file = "structured_dataset.csv"

    # Run the dataset generation pipeline
    dataset_generation_pipeline(log_folder, output_file)
