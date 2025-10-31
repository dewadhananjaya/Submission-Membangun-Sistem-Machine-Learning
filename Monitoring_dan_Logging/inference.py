import requests
import json
import random

# Tentukan jumlah fitur yang benar untuk dataset Dry Bean
NUM_FEATURES = 16 

# 📌 DAFTAR NAMA KOLOM WAJIB yang diambil dari header CSV
FEATURE_COLUMNS = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 
    'ShapeFactor4'
]

def infer(instances, url="http://127.0.0.1:5005/invocations", format="dataframe_split"):
    """
    Kirim data ke endpoint MLflow model serving dan kembalikan hasil prediksi.
    Dipaksa menggunakan format 'dataframe_split' untuk memenuhi persyaratan skema (schema enforcement).

    Args:
        instances (list of list): Data input. Harus memiliki 16 fitur.
        url (str): Endpoint MLflow model serving. Default port 5005.
        format (str): Format payload. Default diubah ke 'dataframe_split'.

    Returns:
        dict/str: Hasil prediksi dari model atau None jika gagal.
    """
    headers = {"Content-Type": "application/json"}
    payload = {}

    if format == "dataframe_split":
        # Solusi untuk skema yang hilang: Menggunakan format DataFrame Split
        # dan menyertakan NAMA KOLOM yang sesuai dengan yang diharapkan model.
        if len(FEATURE_COLUMNS) != NUM_FEATURES:
            print("Error internal: Jumlah kolom tidak cocok dengan NUM_FEATURES.")
            return None
            
        payload = {
            "dataframe_split": {
                "columns": FEATURE_COLUMNS, # <-- Menggunakan NAMA KOLOM ASLI
                "data": instances
            }
        }
    
    elif format == "instances":
        # Format 'instances' gagal karena model memerlukan nama kolom.
        print("Peringatan: Format 'instances' kemungkinan akan gagal karena skema model memerlukan nama kolom.")
        payload = {"instances": instances}
    
    else:
        print(f"Error: Format '{format}' tidak didukung.")
        return None
    
    # Serialisasi payload ke JSON
    data_json = json.dumps(payload)
    print("--- Payload JSON yang Dikirim ---")
    print(data_json)
    print("---------------------------------")
    
    try:
        response = requests.post(url, data=data_json, headers=headers, timeout=15)
        response.raise_for_status() 
        
        try:
            return response.json()
        except json.JSONDecodeError:
            print("Peringatan: Respon server BUKAN JSON. Mengembalikan teks mentah.")
            return response.text
            
    except requests.exceptions.ConnectionError:
        print(f"Request GAGAL: Gagal terhubung ke server di {url}. Pastikan 'mlflow models serve' sudah berjalan.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Request GAGAL: {e}")
        try:
            error_details = response.json()
            print("Detail Error dari Server:")
            print(json.dumps(error_details, indent=2))
        except:
            print("Detail Error (teks mentah):")
            print(response.text)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request gagal karena alasan tak terduga: {e}")
        return None

if __name__ == "__main__":
    # 🚨 PENTING: Sampel data TIRUAN (dummy) dengan 16 fitur.
    # Nilai-nilai ini HARUS sesuai dengan data yang sudah di pra-pemrosesan/scaling.
    
    # Mengambil sampel data yang sama dari input terakhir Anda
    sample_data = [0.2789, -0.95, -0.4499, -0.5536, 0.4729, 0.3534, 0.7844, -0.8261, -0.1562, -0.9404, -0.5627, 0.0107, -0.9469, -0.6023, 0.2998, 0.0899]
    sample = [sample_data]
    
    if len(sample[0]) != NUM_FEATURES:
         print(f"Peringatan: Jumlah fitur sampel ({len(sample[0])}) TIDAK SAMA dengan NUM_FEATURES ({NUM_FEATURES}).")
         exit()

    print(f"Mengirim sampel ke server (port 5005) ({NUM_FEATURES} fitur): {sample}")
    
    # ➡️ Dipaksa menggunakan format 'dataframe_split' untuk menyertakan nama kolom.
    # Karena fungsi infer sekarang default ke 'dataframe_split', Anda bisa memanggilnya tanpa parameter format:
    result = infer(sample) 
    
    if result:
        print("\n🎉 Inferensi Berhasil.")
        print("Hasil prediksi:", result)
    else:
        print("\n❌ Inferensi Gagal.")