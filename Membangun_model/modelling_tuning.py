import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import json # Diperlukan untuk mencatat PARAM_GRID secara manual

### 1. Fungsi Memuat Data (load_data)
# Tetap sama karena tidak terkait dengan logging MLflow
def load_data(file_path="Preprocessed_Dry_Bean.csv"):
    """
    Memuat dataset Dry Bean yang sudah diproses sebelumnya.
    """
    df = pd.read_csv(file_path)
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    return X, y

### 2. Fungsi Tuning dan Logging Manual (tune_and_log_model)
def tune_and_log_model(X_train, X_test, y_train, y_test, param_grid):
    """Melakukan hyperparameter tuning KNN menggunakan GridSearchCV dan MANUAL MLflow Logging."""

    # Set nama eksperimen MLflow
    mlflow.set_experiment("Dry_Bean_Classification_KNN_Tuning_Manual")
    
    with mlflow.start_run() as run:
        print("Mulai Hyperparameter Tuning menggunakan GridSearchCV...")
        
        # Inisialisasi model dasar
        knn = KNeighborsClassifier()
        
        # Inisialisasi GridSearchCV
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Melatih dan mencari parameter terbaik
        grid_search.fit(X_train, y_train)
        
        # --- LOGGING MANUAL DIMULAI DI SINI ---
        
        # 1. LOG PARAMETER YANG DIUJI (PARAM_GRID)
        mlflow.log_param("param_grid", json.dumps(param_grid))
        mlflow.log_param("cv_folds", 5)
        
        # 2. LOG HASIL TERBAIK (Parameter dan Metrik CV Terbaik)
        best_params = grid_search.best_params_
        best_score_cv = grid_search.best_score_
        
        print(f"\nParameter Terbaik: {best_params}")
        print(f"Skor Akurasi CV Terbaik: {best_score_cv:.4f}")
        
        # Log parameter terbaik sebagai parameter run utama
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)

        # Log skor CV terbaik sebagai metrik
        mlflow.log_metric("best_cv_accuracy", best_score_cv)
        
        # Model Terbaik
        best_knn = grid_search.best_estimator_
        
        # 3. MENGHITUNG DAN LOG METRIK AKHIR PADA DATA UJI (TEST SET)
        y_pred = best_knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log metrik pengujian secara manual
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        
        # 4. LOG MODEL TERBAIK
        # log_model(model, artifact_path)
        mlflow.sklearn.log_model(best_knn, "best_knn_model")
        
        # --- LOGGING MANUAL SELESAI ---

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Akurasi Model Terbaik (pada data uji): {accuracy:.4f}")

### 3. Blok Utama Eksekusi
if __name__ == "__main__":
    FILE_PATH = "Preprocessed_Dry_Bean.csv" 
    
    PARAM_GRID = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }

    print(f"Memuat data dari {FILE_PATH}...")
    try:
        X, y = load_data() 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print("Melakukan Tuning dan mencatat model terbaik dengan MANUAL MLflow Logging...")
        tune_and_log_model(X_train, X_test, y_train, y_test, PARAM_GRID)
        
    except KeyError as e:
        print(f"Error: Kolom target {e} tidak ditemukan. Harap pastikan nama kolom target ('Class') sudah benar.")
    except FileNotFoundError:
        print(f"Error: File '{FILE_PATH}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")