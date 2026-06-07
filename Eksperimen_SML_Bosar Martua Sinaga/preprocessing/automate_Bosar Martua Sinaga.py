import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def run_preprocessing(input_path, output_path):
    print(f"Mulai memproses data dari: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Parsing waktu & Pemotongan Data (2021+)
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
    df = df[df['tanggal'].dt.year >= 2021].copy()
    
    # 2. Bersihkan nilai invalid awal
    df.replace('---', np.nan, inplace=True)
    
    kolom_numerik = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max']
    kolom_teks = ['critical', 'categori']
    
    # 3. Imputasi Numerik (Interpolate + Bfill + Ffill)
    for col in kolom_numerik:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
    # 4. Imputasi Teks (Ffill + Bfill)
    for col in kolom_teks:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            
    # 5. Ekstraksi Waktu
    df['bulan'] = df['tanggal'].dt.month
    df['hari_dalam_minggu'] = df['tanggal'].dt.dayofweek
    df.drop('tanggal', axis=1, inplace=True)
            
    # 6. Encoding Kategorikal
    cat_cols = ['stasiun', 'critical', 'categori']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    # 7. Scaling Numerik
    scaler = StandardScaler()
    num_cols = [col for col in kolom_numerik if col in df.columns]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 8. Simpan Hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"SUKSES! Data bersih berhasil disimpan ke: {output_path}")
    return df

if __name__ == "__main__":
    # Path relatif ini disesuaikan untuk dijalankan dari root GitHub Repository
    INPUT_FILE = "ispu_dki1_raw.csv"
    OUTPUT_FILE = "preprocessing/ispu_dki_preprocessing.csv"
    
    run_preprocessing(INPUT_FILE, OUTPUT_FILE)