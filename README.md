# MVP Demo Aplikasi Identifikasi Rugae

Ini adalah aplikasi demo Streamlit untuk memvisualisasikan hasil model AI identifikasi rugae palatina.

## 🏃‍♂️ Cara Menjalankan Demo (via `ngrok`)

Aplikasi ini akan dijalankan di server lokal (laptop yang kuat) dan di-tunneling menggunakan `ngrok` agar bisa diakses oleh klien dari mana saja.

### I. Setup Awal (Hanya dilakukan sekali)

1.  Struktur folder
    ```
    /rugae_demo_app/
    |
    |-- 📂 data/
    |   `-- 📂 test_set_npz/
    |       |-- 📂 K01A (Bernado Barus)/
    |       |   |-- K01A (Bernado Barus)_final.npz
    |       `-- 📂 ... (folder pasien test lainnya) ...
    |
    |-- 📂 models/
    |   `-- 📄 best_recognition_triplet_model.pth
    |
    |-- 📂 src/
    |   |-- 📄 app.py                     # <-- Skrip Streamlit
    |   |-- 📄 model_recognition.py       # <-- Wajib ada (definisi arsitektur)
    |   |-- 📄 extract_morphometry_utils.py # <-- Fungsi morfometrik
    |   `-- ... (file .py pendukung lainnya jika ada) ...
    |
    |-- 📄 .gitignore
    |-- 📄 requirements.txt
    `-- 📄 README.md
    ```
3.  Pastikan Anda memiliki environment Python yang sudah terinstal semua dependensi.
    ```bash
    pip install -r requirements.txt
    ```
4.  Pastikan semua data dan model sudah ada di folder yang benar (`/data/` dan `/models/`).
5.  Tunneling dengan `ngrok`:
    5a.  Instal `ngrok`:
    Download `ngrok.exe` dari [ngrok.com](https://ngrok.com/)
    5b.  (Penting) Autentikasi `ngrok` (hanya perlu sekali) dengan menjalankan perintah yang ada di dashboard `ngrok`:
    ```bash
    ngrok config add-authtoken [TOKEN_ANDA]
    ```

### II. Menjalankan Aplikasi

1.  Anda perlu membuka **dua terminal** di direktori *root* proyek ini.

2.  **Di Terminal 1 (Jalankan Server Streamlit):**
    ```bash
    # Pindah ke direktori 'src'
    cd src

    # Jalankan aplikasi streamlit
    streamlit run app.py
    ```
    Biarkan terminal ini tetap berjalan.

3.  **Di Terminal 2 (Jalankan Tunneling ngrok):**
    ```bash
    # Jalankan ngrok untuk mengekspos port 8501 (port default Streamlit)
    ngrok http 8501
    ```

### III. Bagikan Link
`ngrok` akan memberi Anda sebuah URL publik seperti ini: `Forwarding https://abc-123.ngrok-free.app -> http://localhost:8501`
Bagikan link `https://abc-123.ngrok-free.app` kepada siapa saja yang perlu melihat demo. Selama kedua terminal Anda tetap berjalan, aplikasi akan bisa diakses dari link tersebut.


### IV. Catatan untuk dipahami
1.  Koordinasi untuk Demo: Tentukan jam presentasi.
2.  Server: 15 menit sebelum jam presentasi, ulangi "Langkah 2: Menjalankan Aplikasi" di atas untuk menjalankan server.
3.  Server: Kirimkan link ngrok yang baru (misal: https://xyz.ngrok.io) kepada client.
4.  Client: Membuka link di browser dan menjalankan demo secara live, ditenagai oleh GPU dari Server.
5.  Server: Wajib menjaga laptop tetap menyala dan terhubung ke internet stabil selama demo berlangsung.
