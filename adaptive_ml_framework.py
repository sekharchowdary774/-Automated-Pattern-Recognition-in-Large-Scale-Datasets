
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------
# 1. Synthetic Large-Scale Patterned Data Generation
# -----------------------------
def generate_large_scale_data(n_samples=5000):
    np.random.seed(42)
    cluster_1 = np.random.randn(n_samples//2, 2) + np.array([5, 5])
    cluster_2 = np.random.randn(n_samples//2, 2) + np.array([-5, -5])
    labels = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
    data = np.vstack([cluster_1, cluster_2])
    df = pd.DataFrame(data, columns=["feature_1", "feature_2"])
    df["target"] = labels
    return df


# -----------------------------
# 2. Data Loading & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_large_scale_data()
    df.to_csv("adaptive_pattern_data.csv", index=False)
    print("Dataset saved as adaptive_pattern_data.csv")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("adaptive_pattern_data.csv")
    print("\nDataset Head:\n", df.head())
    print("\nDataset Info:\n", df.info())
    print("\nDataset Description:\n", df.describe())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "adaptive_scaler.pkl")
    print("Scaler saved as adaptive_scaler.pkl")

    return X_scaled, y


# -----------------------------
# 5. Adaptive Learning Components
# -----------------------------
def build_autoencoder(input_dim):
    inp = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(inp)
    encoded = Dense(10, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    out = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inp, out)
    encoder = Model(inp, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


# -----------------------------
# 6. Training
# -----------------------------
def train_models(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    autoencoder, encoder = build_autoencoder(X_scaled.shape[1])
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training Autoencoder (Feature Extractor)...")
    autoencoder.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=30,
        batch_size=32,
        callbacks=[early]
    )

    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    print("Training Adaptive Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_encoded, y_train)

    encoder.save("adaptive_encoder.h5")
    joblib.dump(rf, "adaptive_rf.pkl")

    print("Models saved.")

    return rf, encoder, X_test_encoded, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_model(rf, X_test_encoded, y_test):
    y_pred = rf.predict(X_test_encoded)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Adaptive Pattern Recognition Confusion Matrix")
    plt.tight_layout()
    plt.savefig("adaptive_confusion_matrix.png")
    plt.show()


# -----------------------------
# 8. Inference Function
# -----------------------------
def adaptive_predict(input_features):
    scaler = joblib.load("adaptive_scaler.pkl")
    rf = joblib.load("adaptive_rf.pkl")
    encoder = Model.load_model("adaptive_encoder.h5")

    input_features = np.array(input_features).reshape(1, -1)
    scaled = scaler.transform(input_features)
    encoded = encoder.predict(scaled)
    return rf.predict(encoded)[0]


# -----------------------------
# 9. Main Execution
# -----------------------------
def main():
    if not os.path.exists("adaptive_pattern_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X_scaled, y = preprocess_data(df)
    rf, encoder, X_test_encoded, y_test = train_models(X_scaled, y)
    evaluate_model(rf, X_test_encoded, y_test)

    sample = [4.2, 5.1]
    print("\nSample Prediction:", adaptive_predict(sample))


if __name__ == "__main__":
    main()
