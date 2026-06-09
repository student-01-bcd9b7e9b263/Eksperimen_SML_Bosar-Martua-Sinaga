import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import shutil

def train_model():
    # Path disesuaikan karena dijalankan di dalam folder MLProject
    data_path = "ispu_dki1_preprocessing.csv"
    print(f"Membaca dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Memisahkan Fitur dan Target
    X = df.drop('categori', axis=1)
    y = df['categori']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Memastikan folder model_output bersih sebelum diisi model baru
    if os.path.exists("model_output"):
        shutil.rmtree("model_output")

    # MLflow Tracking (Lokal)
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Akurasi Model: {acc:.4f}")

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        
        # MENYIMPAN MODEL LOKAL UNTUK DOCKER (SYARAT ADVANCE)
        mlflow.sklearn.save_model(model, "model_output")
        
        print("Model berhasil ditraining dan disimpan ke folder 'model_output'!")

if __name__ == "__main__":
    train_model()
