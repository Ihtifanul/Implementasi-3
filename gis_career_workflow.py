import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. SIMULASI INTEGRASI DATA GEOSPASIAL / SPASIAL
# Mensimulasikan pengumpulan data atribut fisik dan spasial (Koordinat, Elevasi,
# Indeks Vegetasi/NDVI, dan Kepadatan Infrastruktur) untuk analisis industri.

def load_and_preprocess_spatial_data(num_samples=500):
    print("[1/4] Mengintegrasikan Data Atribut Spasial & Infrastruktur...")
    np.random.seed(42)
    
    # Generate data spasial tiruan
    elevation = np.random.uniform(10, 1500, num_samples)         # Ketinggian (meter)
    ndvi = np.random.uniform(-0.1, 0.9, num_samples)            # Indeks Vegetasi (-1 s/d 1)
    infra_density = np.random.uniform(0, 100, num_samples)      # Kepadatan Infrastruktur (%)
    distance_to_road = np.random.uniform(5, 5000, num_samples)   # Jarak ke Jalan Utama (meter)
    
    # Gabungkan ke dalam Pandas DataFrame
    df = pd.DataFrame({
        'Elevation': elevation,
        'NDVI': ndvi,
        'Infra_Density': infra_density,
        'Distance_Road': distance_to_road
    })
    
    # Aturan Klasifikasi Lahan Industri (0: Area Konservasi/Hutan, 1: Zona Pengembangan/Aset)
    # Aturan: NDVI > 0.4 dan Elevasi > 200 dianggap area vegetasi/konservasi, lainnya zona aset
    labels = np.where((df['NDVI'] > 0.4) & (df['Elevation'] > 200), 0, 1)
    
    return df, labels


# 2. PEMBAGIAN DATA (Train-Test Split)
# Membagi dataset spasial menjadi data latih (80%) dan data uji (20%).

def split_spatial_dataset(X, y):
    print("[2/4] Membagi Dataset Spasial (80% Training, 20% Testing)...")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 3. PELATIHAN MODEL CLASSIFICATION UNTUK GIS & ASET
# Melatih model Machine Learning (Random Forest) untuk klasifikasi spasial otomatis.

def train_gis_classifier(X_train, y_train):
    print("[3/4] Melatih Model Machine Learning untuk Otomatisasi GIS Industri...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# 4. EVALUASI MODEL & KALKULASI EFISIENSI INDUSTRI
# Mengukur akurasi prediksi serta kalkulasi estimasi otomatisasi waktu/biaya.

def evaluate_and_simulate_industry_impact(model, X_test, y_test):
    print("[4/4] Mengevaluasi Performa Model & Dampak Efisiensi Industri...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("      HASIL ANALISIS MODEL GIS INDUSTRI & KARIR")
    print("="*50)
    print(f"Akurasi Model Prediksi Spasial : {acc * 100:.2f}%")
    print("\nDetail Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Konservasi/Hutan', 'Zona Pengembangan/Aset']))
    
    # Simulasi perhitungan efisiensi waktu kerja
    manual_hours_per_sample = 0.5  # 30 menit per pemetaan manual
    total_samples = len(X_test)
    manual_total_time = total_samples * manual_hours_per_sample
    automated_time = 0.05  # waktu eksekusi skrip dalam detik
    
    time_saved = manual_total_time - (automated_time / 3600)
    
    print("-" * 50)
    print("SIMULASI MANFAAT AUTOMATION DI INDUSTRI GIS:")
    print(f"• Total Data Diuji           : {total_samples} sampel spasial")
    print(f"• Estimasi Pemetaan Manual   : {manual_total_time:.1f} jam kerja")
    print(f"• Estimasi Otomatisasi ML    : < 1 detik")
    print(f"• Efisiensi Waktu Terhemat   : {time_saved:.1f} jam kerja")
    print("="*50)


# MAIN PROGRAM

if __name__ == "__main__":
    print("=== WORKFLOW: CAREER AND COMPETITION IN INDUSTRIAL ERA WITH GIS ===")
    
    # Step 1: Integrasi Data
    X, y = load_and_preprocess_spatial_data(num_samples=1000)
    
    # Step 2: Split Data
    X_train, X_test, y_train, y_test = split_spatial_dataset(X, y)
    
    # Step 3: Pelatihan Model
    model = train_gis_classifier(X_train, y_train)
    
    # Step 4: Evaluasi & Simulasi Industri
    evaluate_and_simulate_industry_impact(model, X_test, y_test)
    
    print("\n=== ESEKUSI WORKFLOW SELESAI ===")