import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import xgboost as xgb  # Import XGBoost


def create_unified_dataset():
    """
    Generates a unified dataset by processing and merging logon, device, http, and LDAP data.

    Returns:
        pd.DataFrame: Unified dataset with aggregated features per user.
    """
    # Load and preprocess logon data
    logon_data = pd.read_csv("./data/logon.csv")
    logon_data["date"] = pd.to_datetime(logon_data["date"])
    logon_data["hour"] = logon_data["date"].dt.hour
    logon_data["after_hours"] = logon_data["hour"].apply(
        # Mark logons as after-hours if outside 8 AM - 6 PM
        lambda x: 1 if x < 8 or x > 18 else 0
    )
    logon_data["logon_count"] = logon_data.groupby("user")["user"].transform("count")

    # Load and preprocess device data
    device_data = pd.read_csv("data/device.csv")
    device_data["date"] = pd.to_datetime(device_data["date"])
    device_data["thumb_drive_usage"] = device_data["activity"].apply(
        lambda x: 1 if x == "connect" else 0
    )
    device_data["thumb_drive_count"] = device_data.groupby("user")[
        "thumb_drive_usage"
    ].transform("sum")  # Count thumb drive connection events per user

    # Load and preprocess http data
    http_data = pd.read_csv(
        "data/http.csv", header=None, names=["id", "date", "user", "pc", "url"]
    )
    http_data["date"] = pd.to_datetime(http_data["date"])
    http_data["url_count"] = http_data.groupby("user")["url"].transform("count")

    # Load and preprocess LDAP data
    ldap_folder = "data/LDAP"
    ldap_files = [
        os.path.join(ldap_folder, file)
        for file in os.listdir(ldap_folder)
        if file.endswith(".csv")
    ]
    ldap_data = pd.concat([pd.read_csv(file) for file in ldap_files], ignore_index=True)
    ldap_data["role"] = ldap_data["Role"].apply(lambda x: 1 if x == "IT Admin" else 0)
    ldap_data.rename(columns={"user_id": "user"}, inplace=True)
    ldap_data = ldap_data[["user", "role"]]
    ldap_data.drop_duplicates(subset=["user"], inplace=True)

    # Aggregate features per user
    logon_agg = (
        logon_data.groupby("user")
        .agg(
            total_logons=("logon_count", "max"),
            after_hours_logons=("after_hours", "sum"),
        )
        .reset_index()
    )

    # Aggregate device data to summarize thumb drive usage
    device_agg = (
        device_data.groupby("user")
        .agg(thumb_drive_count=("thumb_drive_count", "max"))
        .reset_index()
    )

    # Aggregate HTTP data to summarize URL access activity
    http_agg = (
        http_data.groupby("user").agg(url_count=("url_count", "max")).reset_index()
    )

    # Merge all datasets
    merged_data = logon_agg.merge(device_agg, on="user", how="outer")
    merged_data = merged_data.merge(http_agg, on="user", how="outer")
    merged_data = merged_data.merge(ldap_data, on="user", how="outer")

    # Fill missing values with 0
    merged_data.fillna(0, inplace=True)

    # Save the unified dataset to a CSV file
    merged_data.to_csv("unified_dataset.csv", index=False)

    return merged_data


# Load pre-trained model and scaler
# caching the model and scaler for better performance
@st.cache_resource  
def load_model_and_scaler():
    # Load the pre-trained XGBoost model
    model = joblib.load("./models/xgboost_model.pkl")
    # Load the scaler for feature normalization
    scaler = joblib.load("./models/scaler.pkl")

    return model, scaler


# Use Streamlit's caching feature to optimize performance and prevent redundant data loading
@st.cache_data
def load_data():
    """
    Loads the unified dataset from a CSV file.

    Returns:
        pd.DataFrame: The unified dataset.
    """
    # Read the CSV file with low_memory=False to handle large files efficiently and avoid DtypeWarnings
    data = pd.read_csv("unified_dataset.csv", low_memory=False)

    return data



def preprocess_data(data, scaler):
    """
    Prepares and scales the dataset features using the provided scaler.

    Args:
        data (pd.DataFrame): The unified dataset containing user features.
        scaler: The pre-loaded scaler object used for feature normalization.

    Returns:
        np.ndarray: Scaled feature matrix ready for model input.
    """
    # Define the features to be used for model prediction
    features = [
        "total_logons",
        "after_hours_logons",
        "thumb_drive_count",
        "url_count",
        "role",
    ]

    # Extract only the selected feature columns from the dataset
    X = data[features]

    # Scale the feature matrix using the provided scaler for normalization
    X_scaled = scaler.transform(X)

    return X_scaled


# Streamlit app
def main():
    st.title("Anomaly Detection Dashboard")

    # Load pre-trained model and scaler
    model, scaler = load_model_and_scaler()

    # Load data
    data = load_data()

    # Preprocess data
    X_scaled = preprocess_data(data, scaler)

    # Generate predictions
    data["anomaly_score"] = model.predict_proba(X_scaled)[:, 1]
    data["anomaly_prediction"] = model.predict(X_scaled)

    # Save results to CSV
    output_csv_path = "anomaly_results.csv"
    data[
        [
            "user",
            "total_logons",
            "after_hours_logons",
            "thumb_drive_count",
            "url_count",
            "role",
            "anomaly_prediction",
            "anomaly_score",
        ]
    ].to_csv(output_csv_path, index=False)

    # Display the results CSV (only a portion of the dataset)
    st.subheader("Anomaly Detection Results")
    st.write(
        data[
            [
                "user",
                "total_logons",
                "after_hours_logons",
                "thumb_drive_count",
                "url_count",
                "role",
                "anomaly_prediction",
                "anomaly_score",
            ]
        ].head(1000)
    )

    # Visualize anomaly scores
    st.subheader("Anomaly Score Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data["anomaly_score"], bins=50, kde=True, color="blue")
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # Visualize anomaly predictions
    st.subheader("Anomaly Predictions")
    anomaly_counts = data["anomaly_prediction"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=anomaly_counts.index,
        y=anomaly_counts.values,
        hue=anomaly_counts.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Anomaly Predictions (0 = Normal, 1 = Anomaly)")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    st.pyplot(plt)

    # Feature Importance (if available)
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, importance_type="weight", max_num_features=10)
        plt.title("Feature Importance")
        st.pyplot(plt)


if __name__ == "__main__":
    unified_dataset = create_unified_dataset()
    main()
