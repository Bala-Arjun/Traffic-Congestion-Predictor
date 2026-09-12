import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
import os

def generate_synthetic_dataset(n_samples=10000):
    """
    Generate a larger, more realistic synthetic traffic dataset
    with patterns matching real-world traffic behavior.
    """
    np.random.seed(42)
    records = []

    for _ in range(n_samples):
        hour = np.random.randint(0, 24)
        day = np.random.randint(1, 8)  # 1=Mon ... 7=Sun
        is_weekend = day >= 6

        # Free flow speed varies by road type
        free_speed = np.random.choice([30, 50, 60, 80, 100], p=[0.1, 0.2, 0.4, 0.2, 0.1])

        # Simulate realistic congestion patterns
        if is_weekend:
            if 10 <= hour <= 20:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            else:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        else:
            # Morning rush
            if 7 <= hour <= 9:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
            # Evening rush
            elif 17 <= hour <= 19:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.1, 0.25, 0.65])
            # Midday moderate
            elif 11 <= hour <= 14:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            # Night / early morning
            elif hour < 6 or hour >= 22:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03])
            else:
                congestion_factor = np.random.choice([0, 1, 2], p=[0.55, 0.3, 0.15])

        # Derive speed from congestion level + noise
        if congestion_factor == 0:
            speed = free_speed * np.random.uniform(0.75, 1.0)
        elif congestion_factor == 1:
            speed = free_speed * np.random.uniform(0.45, 0.74)
        else:
            speed = free_speed * np.random.uniform(0.1, 0.44)

        speed = round(max(2, speed), 1)

        # Additional features
        speed_ratio = speed / free_speed
        is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        is_night = 1 if (hour < 6 or hour >= 22) else 0

        records.append({
            "hour": hour,
            "day": day,
            "speed": speed,
            "free_speed": free_speed,
            "speed_ratio": speed_ratio,
            "is_weekend": int(is_weekend),
            "is_rush_hour": is_rush_hour,
            "is_night": is_night,
            "congestion": congestion_factor
        })

    return pd.DataFrame(records)


def train():
    csv_path = "traffic_data.csv"

    # Always generate a full synthetic dataset
    print("Generating synthetic dataset with 10,000 samples...")
    df_synthetic = generate_synthetic_dataset(10000)

    # If an existing CSV is present, merge it in (keeps your real data)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        print(f"Found existing traffic_data.csv ({len(df_existing)} rows) — merging with synthetic data.")
        # Only keep rows that have the required columns
        required = ["hour", "day", "speed", "free_speed", "congestion"]
        if all(c in df_existing.columns for c in required):
            df = pd.concat([df_synthetic, df_existing], ignore_index=True)
        else:
            print("  existing CSV missing required columns — using synthetic data only.")
            df = df_synthetic
    else:
        df = df_synthetic

    # Re-compute engineered features on the full merged dataset
    df["speed_ratio"] = df["speed"] / df["free_speed"].replace(0, 1)
    df["is_weekend"] = (df["day"] >= 6).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0)
    df["is_night"] = df["hour"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)

    # Save merged dataset back
    df.to_csv(csv_path, index=False)
    print(f"Dataset ready: {len(df)} rows, saved to {csv_path}")
    print(f"Congestion distribution:\n{df['congestion'].value_counts().sort_index().rename({0:'Low',1:'Medium',2:'High'})}\n")

    features = ["hour", "day", "speed", "free_speed", "speed_ratio", "is_weekend", "is_rush_hour", "is_night"]
    X = df[features]
    y = df["congestion"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use GradientBoosting for better accuracy than RandomForest
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Model Performance:")
    # Use labels present in test set to avoid mismatch errors
    present_labels = sorted(y_test.unique())
    label_names = {0: "Low", 1: "Medium", 2: "High"}
    target_names = [label_names[l] for l in present_labels]
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")


if __name__ == "__main__":
    train()