import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi DagsHub (Pastikan username dan repo sesuai)
dagshub.init(repo_owner='Lathif', repo_name='Eksperimen_SML_Lathif', mlflow=True)

def train_model():
    # Karena dijalankan via MLproject, path data cukup nama filenya saja 
    # asalkan berada di direktori yang sama (MLProject folder)
    data_path = "ispu_dki1_preprocessing.csv"
    print(f"Membaca dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Memisahkan Fitur dan Target
    X = df.drop('categori', axis=1)
    y = df['categori']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow Tracking
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Akurasi Model: {acc:.4f}")

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        print("Model berhasil ditraining dan dicatat di MLflow!")

if __name__ == "__main__":
    train_model()
