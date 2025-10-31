import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def load_data(file_path="Preprocessed_Dry_Bean.csv"):
    """Memuat dataset Dry Bean yang sudah diproses sebelumnya."""
    df = pd.read_csv(file_path)
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    return X, y

def train_and_log_model(X_train, X_test, y_train, y_test, n_neighbors):
    """Melatih model KNN dan menggunakan MLflow autologging."""

    # Aktifkan autologging untuk scikit-learn
    mlflow.sklearn.autolog()

    # Buat atau pilih experiment
    mlflow.set_experiment("Dry_Bean_Classification_KNN_Autolog")

    # Jalankan run MLflow
    with mlflow.start_run() as run:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Hitung metrik (ditampilkan saja, tidak di-log manual)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print("\n=== HASIL EVALUASI MODEL ===")
        print(f"MLflow Run ID : {run.info.run_id}")
        print(f"Akurasi       : {accuracy:.4f}")
        print(f"Presisi       : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print(f"F1 Score      : {f1:.4f}")
        print("============================\n")

# Pastikan script ini bisa langsung dijalankan
if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Split data train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Jalankan training + autolog
    train_and_log_model(X_train, X_test, y_train, y_test, n_neighbors=5)

    print("Training selesai")
    print("Jalankan perintah berikut untuk melihat hasil di MLflow UI:")
    print("mlflow ui")
    print("Lalu buka browser ke: http://localhost:5000")