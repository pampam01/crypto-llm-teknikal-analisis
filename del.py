import os
import tempfile

# Dapatkan path direktori sementara
temp_dir = tempfile.gettempdir()

# Iterasi melalui file di direktori sementara
for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)
    try:
        # Hapus file jika sesuai dengan pola nama file sementara Anda
        if filename.endswith(".png"):  # Sesuaikan dengan ekstensi file Anda
            os.remove(file_path)
            print(f"File {file_path} berhasil dihapus.")
    except Exception as e:
        print(f"Terjadi kesalahan saat menghapus file {file_path}: {e}")