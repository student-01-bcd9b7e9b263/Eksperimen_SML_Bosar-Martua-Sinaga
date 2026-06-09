# Menggunakan base image Python yang ringan
FROM python:3.9-slim

# Menyiapkan direktori kerja di dalam container
WORKDIR /app

# Menginstal dependensi yang dibutuhkan untuk training model
RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow dagshub

# (Catatan: Saat dijalankan via `mlflow run`, MLflow secara otomatis 
# akan me-mount folder MLProject ke dalam container, jadi kita tidak 
# perlu melakukan COPY file python di sini).
