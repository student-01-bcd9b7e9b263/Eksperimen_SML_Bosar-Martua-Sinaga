import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Inisialisasi DagsHub (GANTI DENGAN MILIK ANDA)
# Format: dagshub.init(repo_owner='username_anda', repo_name='nama_repo_anda', mlflow=True)
dagshub.init(repo_owner='sinagabosar16', repo_name='Eksperimen_SML_Bosar-Martua-Sinaga', mlflow=True)

def train_basic_model():
    # 2. Memuat Dataset Bersih
    # Sesuaikan path ini dengan lokasi file CSV bersih dari Kriteria 1
    data_path = "ispu_dki1_preprocessing.csv"
    print(f"Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)

    # 3. Memisahkan Fitur (X) dan Target (y)
    # Target kita adalah kolom 'categori' (yang sudah di-encode jadi angka di tahap 1)
    X = df.drop('categori', axis=1)
    y = df['categori']

    # Membagi data Train dan Test (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Memulai MLflow Run
    with mlflow.start_run(run_name="Basic_RandomForest_Model"):
        # Inisialisasi Model Dasar
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Log parameter model ke MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Melatih Model
        model.fit(X_train, y_train)

        # Prediksi dan Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Akurasi Model Dasar: {acc:.4f}")
        print(f"F1 Score Model Dasar: {f1:.4f}")

        # Log Metrics ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log Model sebagai Artifact di MLflow
        mlflow.sklearn.log_model(model, "basic_model")
        
        # Simpan model secara lokal (opsional tapi disarankan)
        joblib.dump(model, "basic_model.pkl")
        print("Model dasar berhasil dilatih dan dicatat di DagsHub MLflow!")

if __name__ == "__main__":
    train_basic_model()
