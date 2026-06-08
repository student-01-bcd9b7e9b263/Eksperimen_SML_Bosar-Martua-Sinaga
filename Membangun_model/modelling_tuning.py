import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Inisialisasi DagsHub (GANTI DENGAN MILIK ANDA)
dagshub.init(repo_owner='sinagabosar16', repo_name='Eksperimen_SML_Bosar-Martua-Sinaga', mlflow=True)

def train_tuned_model():
    # 2. Memuat Dataset
    data_path = "ispu_dki1_preprocessing.csv"
    df = pd.read_csv(data_path)

    # 3. Memisahkan Fitur (X) dan Target (y)
    X = df.drop('categori', axis=1)
    y = df['categori']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Setup Hyperparameter Tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    print("Memulai Hyperparameter Tuning... (Ini mungkin memakan waktu beberapa menit)")
    grid_search.fit(X_train, y_train)

    # 5. Memulai MLflow Run untuk Model Terbaik
    with mlflow.start_run(run_name="Tuned_RandomForest_Model"):
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print("Parameter Terbaik Ditemukan:", best_params)

        # Log parameter terbaik ke MLflow
        mlflow.log_param("model_type", "RandomForest_Tuned")
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Prediksi dengan model terbaik
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Akurasi Model Tuning: {acc:.4f}")
        print(f"F1 Score Model Tuning: {f1:.4f}")

        # Log Metrics ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log Model sebagai Artifact di MLflow
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        # Simpan model terbaik secara lokal untuk deployment nanti
        joblib.dump(best_model, "best_rf_model.pkl")
        print("Model hasil tuning berhasil dicatat di DagsHub MLflow!")

if __name__ == "__main__":
    train_tuned_model()
