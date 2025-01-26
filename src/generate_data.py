import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_raw_logs(log_folder):
    """
    Load raw log files from a folder and merge them into a single DataFrame.
    Assumes that all logs are in CSV format.

    Args:
        log_folder (str): Path to the folder containing log files.

    Returns:
        pd.DataFrame: Combined DataFrame containing data from all CSVs, or an empty DataFrame if no files are found.
    """
    # Get all CSV files in the specified folder
    log_files = [
        os.path.join(log_folder, file)
        for file in os.listdir(log_folder)
        if file.endswith(".csv")
    ]

    # Warn if no files are found
    if not log_files:
        print(
            f"Warning: No CSV files found in the folder: {os.path.abspath(log_folder)}"
        )
        # Return an empty DataFrame to avoid breaking later steps
        return pd.DataFrame()

    # Read all found CSV files and merge them into one DataFrame
    raw_logs = pd.concat([pd.read_csv(file) for file in log_files], ignore_index=True)

    return raw_logs


def preprocess_logs(raw_logs):
    """
    Clean up and process raw log data to extract useful fields and make it analysis-ready.

    Args:
        raw_logs (pd.DataFrame): The raw combined log data.

    Returns:
        pd.DataFrame: Processed DataFrame with cleaned and new features added.
    """
    # Check if 'date' column exists to parse timestamps
    if "date" in raw_logs.columns:
        # Convert 'date' to datetime and extract useful time features
        raw_logs["date"] = pd.to_datetime(raw_logs["date"])
        # Extract the hour of the event
        raw_logs["hour"] = raw_logs["date"].dt.hour
        # Monday=0, Sunday=6
        raw_logs["day_of_week"] = raw_logs["date"].dt.dayofweek

    # Handle missing values, fill missing values (maybe from incomplete logs) with 0
    raw_logs.fillna(0, inplace=True)

    return raw_logs


def generate_features(preprocessed_logs):
    """
    Generate structured features from preprocessed logs.
    Handles missing columns gracefully, so no crashes happen if some logs are incomplete.

    Args:
        preprocessed_logs (pd.DataFrame): The cleaned log data after preprocessing.

    Returns:
        pd.DataFrame: A structured DataFrame with features like logon counts, after-hours activity, etc.
    """
    # Initialize default features with empty DataFrames
    # These will hold the calculations for logon, after-hours activity, etc.
    features = {
        "logon_count": pd.DataFrame(columns=["user", "pc", "logon_count"]),
        "after_hours": pd.DataFrame(columns=["user", "pc", "after_hours"]),
        "thumb_drive_usage": pd.DataFrame(columns=["user", "pc", "thumb_drive_usage"]),
        "url_count": pd.DataFrame(columns=["user", "pc", "url_count"]),
    }

    # Calculate features if required columns exist
    if "activity" in preprocessed_logs.columns:
        # Count logons for each user on each PC
        features["logon_count"] = (
            preprocessed_logs[preprocessed_logs["activity"] == "Logon"]
            .groupby(["user", "pc"])
            .size()
            .reset_index(name="logon_count")
        )

        # Count after-hours activity (before 8 AM or after 6 PM)
        features["after_hours"] = (
            preprocessed_logs[
                (preprocessed_logs["hour"] < 8) | (preprocessed_logs["hour"] > 18)
            ]
            .groupby(["user", "pc"])
            .size()
            .reset_index(name="after_hours")
        )

        # Count thumb drive connections
        features["thumb_drive_usage"] = (
            preprocessed_logs[preprocessed_logs["activity"] == "connect"]
            .groupby(["user", "pc"])
            .size()
            .reset_index(name="thumb_drive_usage")
        )

    # If the 'url' column exists, count URL accesses for each user on each PC
    if "url" in preprocessed_logs.columns:
        features["url_count"] = (
            preprocessed_logs.groupby(["user", "pc"])["url"]
            .count()
            .reset_index(name="url_count")
        )

    # Start merging all the calculated features into one DataFrame
    structured_data = features["logon_count"]
    for key, df in features.items():
        # Skip the first one (already added)
        if key != "logon_count":
            structured_data = pd.merge(
                structured_data, df, on=["user", "pc"], how="left"
            )

    # Fill missing values (e.g., no after-hours logons or thumb drive usage)
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
        # Default role when it's not provided in the logs
        # Assuming 0 means regular user
        structured_data["role"] = 0

    return structured_data


def dataset_generation_pipeline(log_folder, output_file):
    """
    Run the full dataset generation pipeline. This takes the raw logs and processes them step-by-step:
    1. Load raw logs from the folder.
    2. Preprocess logs to clean and organize them.
    3. Generate structured features from the cleaned data.
    4. Save the final dataset to a CSV file.

    Args:
        log_folder (str): Path to the folder containing raw log files.
        output_file (str): Path to save the generated structured dataset.
    """
    # Step 1: Load raw logs
    raw_logs = load_raw_logs(log_folder)
    # If no data was found, exit early
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
    # Path to the folder where raw logs are stored
    log_folder = "data"  # Update this path if necessary
    print(f"Looking for log files in: {os.path.abspath(log_folder)}")

    # Path where the structured dataset will be saved
    output_file = "structured_dataset.csv"

    # Start the process for generating dataset
    dataset_generation_pipeline(log_folder, output_file)
