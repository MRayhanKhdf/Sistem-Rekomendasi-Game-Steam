
# Laporan Proyek Machine Learning - Sistem Rekomendasi Game

## Project Overview

Game menjadi salah satu industri digital yang mengalami pertumbuhan pesat. Banyaknya pilihan game membuat pengguna sering merasa kesulitan memilih game baru yang sesuai preferensi. Oleh karena itu, dibutuhkan sistem rekomendasi yang mampu memberikan saran game berdasarkan kebiasaan atau preferensi pengguna.

Proyek ini bertujuan untuk membangun sistem rekomendasi game berbasis machine learning. Model yang dikembangkan memanfaatkan pendekatan **Collaborative Filtering** dan **Content-based Filtering**, dengan implementasi end-to-end dari pengumpulan data (scraping), preprocessing, pelatihan model, evaluasi, hingga deployment dasbor.

**Rubrik Tambahan (Opsional):**
- Permasalahan ini penting karena membantu pengguna menemukan game yang relevan dengan lebih cepat dan meningkatkan retensi pemain.
- Referensi:
  - R. Burke, "Hybrid Recommender Systems: Survey and Experiments," User Modeling and User-Adapted Interaction, 2002.
  - A. Sidiropoulos and Y. Manolopoulos, "Generalized comparison of graph-based ranking algorithms for recommendations," Information Systems, 2018.

## Business Understanding

### Problem Statements
- Bagaimana memberikan rekomendasi game untuk pengguna berdasarkan interaksi historis?
- Bagaimana memberikan rekomendasi game berdasarkan input genre dan tag dari pengguna baru (tanpa histori)?

### Goals
- Menghasilkan sistem rekomendasi berbasis **Collaborative Filtering** menggunakan ANN embedding.
- Menghasilkan sistem rekomendasi berbasis **Content-based Filtering** yang dapat digunakan oleh pengguna baru tanpa histori.

### Solution Statements
- Menggunakan ANN latent factor model (embedding) untuk collaborative filtering.
- Menggunakan vectorisasi genre dan tag (CountVectorizer + cosine similarity) untuk content-based filtering.

## Data Understanding

Dataset dikumpulkan dengan teknik web scraping dari Steam (https://store.steampowered.com) sebanyak 4000 game.
Dataset memiliki 4001 baris dan 8 kolom, baris awal ialah baris nama kolom/fitur dan 4000 baris dibawahnya ialah baris dataset
Isi Kolom/Fitur yang digunakan:
- `appid`: ID unik game di Steam
- `title`: nama game
- `genres`: genre game 
- `tags`: fitur/tag game 
- `developers`, `publishers`: pembuat dan penerbit game
- `release_year`: tahun rilis
- `short_description`: deskripsi singkat game

Simulasi data interaksi dibuat dengan 1000 user dan rating acak dari 1–5 terhadap masing-masing game.

- Pada dataset terdapat jumlah missing value per kolom:
	- appid                 0
	- title                 0
	- genres               16
	- developers           24
	- publishers           97
	- tags                 39
	- release_year          0
	- short_description    10
- Membersihkan nilai kosong pada kolom `genres`, `tags`, dan `short_description` dengan menggantinya menjadi string kosong (`''`).
- Menghapus baris yang tidak memiliki `appid` atau `title` (jika ada).
- Mengonversi `release_year` menjadi numerik untuk visualisasi dan analisis.
- Game dengan genre populer seperti Action, RPG, dan Indie mendominasi dataset.
- Rata-rata game memiliki 3 genre dan 4–5 tag.
- Beberapa genre/tags memiliki korelasi positif, misalnya 'Multiplayer' dengan 'Co-op' dan 'Online'.
- Content-based cocok untuk pengguna baru tanpa histori, sementara collaborative filtering efektif jika histori tersed

**Rubrik Tambahan:**
- Visualisasi distribusi genre, tag, dan tahun rilis ditampilkan pada tahap EDA.
- Korelasi antara jumlah genre/tag dan tahun rilis juga dianalisis.

## Data Preparation
- **Scraping Steam API** sebanyak 4000 game memakan waktu total **±4 jam 16 menit**, dilakukan bertahap dengan delay agar tidak diblokir server.
- Menghapus baris yang tidak memiliki `appid` atau `title` (jika ada).
- Membersihkan nilai kosong pada kolom `genres`, `tags`, dan `short_description` dengan menggantinya menjadi string kosong (`''`).
- Mengonversi `release_year` menjadi numerik untuk visualisasi dan analisis.
- Membuat kolom `combined` berisi gabungan genre + tag untuk content-based.
Membuat DataFrame ratings:
	- userID: Membuat ID acak antara 1–1000 sebagai representasi pengguna.
	- itemID: Menggunakan kolom appid dari DataFrame game sebagai ID item/game.
	- Meng-encode `userID` dan `itemID` menjadi indeks numerik.
- Membagi data interaksi menjadi train dan test set (80:20).
- Menyimpan mapping untuk inferensi.

**Rubrik Tambahan:**
- Proses encoding kategori dan pembuatan kombinasi teks genre-tag penting untuk memungkinkan model memahami preferensi pengguna.
- Berdasarkan 4000 dataset hasil scrapping, mayoritas game di dataset dirilis tahun 2015
- Berdasarkan 4000 dataset hasil scrapping, mayoritas game di dataset memiliki genre indie, Itu valid karena lebih banyak game buatan developer game independen di seluruh dunia daripada game buatan perusahaan yang dirilis tiap tahun nya.
- release_year tidak berkorelasi dengan num_genres maupun num_tags.
- num_genres hanya sedikit berkorelasi dengan num_tags (ditunjukkan oleh warna agak keunguan tapi tidak gelap total).
- Diagonal selalu bernilai 1.0 (kuning), karena fitur dikorelasikan dengan dirinya sendiri.
- Game modern (2010 ke atas) cenderung memiliki lebih banyak tag, mencerminkan metadata atau kategorisasi yang lebih kaya.
- Tidak ada hubungan linier kuat, tetapi pola visual menyarankan kemungkinan korelasi lemah antara jumlah genre dan tag.
- Distribusi waktu memperlihatkan bahwa data didominasi oleh game yang dirilis pada dekade terakhir.
- Fokus pada tren modern dalam desain dan metadata game.
- Cocok digunakan sebagai dasar eksplorasi fitur untuk rekomendasi berbasis konten atau analisis tren game modern.

## Modeling

1. **Collaborative Filtering**:  
   - Menggunakan Neural Network (ANN) dengan embedding untuk user dan item.  
   - Model dilatih dengan MSE loss, diuji dengan MAE dan metrik prediksi dalam ±1 poin.

2. **Content-based Filtering**:  
   - Implementasi di `Dasbor.py` menggunakan CountVectorizer dan cosine similarity.  
   - Mengambil input genre dan tag dari user baru.

### Modeling Parameters

#### Model 1: Collaborative Filtering (ANN dengan Keras)
- Embedding size: 32
- Hidden layers: shallow ([64]), medium ([128, 64]), deep ([256, 128, 64])
- Dropout: 0.3
- Optimizer: Adam (default parameter)
- Loss function: MSE (Mean Squared Error)
- Epochs: 200 (untuk semua model)
- Batch size: 50
- Validation split: 20%
- Compile args default: tidak ada custom learning rate
**Rubrik Tambahan:**
- Kelebihan CF: personalized, akurat untuk user aktif.
- Kelebihan CB: bisa digunakan oleh user baru (cold start).

### Inferensi Content Based Filtering
- Bagian ini menampilkan Top-N rekomendasi berdasarkan input genre dan tag yang dimasukkan secara manual oleh user. 
- Rekomendasi dihitung berdasarkan kemiripan cosine antara kombinasi `genres + tags` game dengan preferensi user.
Berdasarkan input:
input_genres = ['Action', 'Adventure']
input_tags = ['Multiplayer', 'Co-op']
sistem memprediksi 10 game untuk direkomendasikan adalah:
	- A Valley Without Wind 2
	- No More Room in Hell 2
	- Killing Floor - Toy Master
	- Planet Centauri
	- Gremlin Invasion: Survivor
	- AdventureQuest 3D
	- DC Universe™ Online
	- Zombasite
	- Sub Rosa
	- Worlds Adrift


### Inferensi ID User
- Menggunakan model ANN terbaik, sistem ini memprediksi semua game yang belum pernah dimainkan oleh user tertentu, kemudian menampilkan 10 game dengan prediksi rating tertinggi sebagai rekomendasi. 
Daftar rekomendasi di atas dihasilkan langsung dari notebook, dan urutan rekomendasi disusun berdasarkan skor prediksi tertinggi yang diberikan oleh model. Game yang direkomendasikan adalah:
	- Sherlock Holmes: The Mystery of the Persian Carpet
	- Majesty Gold HD
	- Silo 2
	- Journal
	- Panzer Elite Action Gold Edition
	- Darkness Within 1: In Pursuit of Loath Nolder
	- CRYPTARK
	- How to Take Off Your Mask
	- Life in Bunker
	- Space Pilgrim Episode I: Alpha Centauri

## Evaluation

Model ANN dibandingkan dalam 3 konfigurasi (`shallow`, `medium`, `deep`). Metrik evaluasi:
- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **Persentase prediksi rating dalam ±1 poin**


**Rubrik Tambahan:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- \( y_i \): nilai sebenarnya (ground truth)
- \( \hat{y}_i \): nilai prediksi dari model
- \( n \): jumlah total sampel
MSE menghitung rata-rata kuadrat selisih antara nilai aktual dan prediksi. Karena selisih dikuadratkan, MSE sangat sensitif terhadap kesalahan besar (outlier). Semakin kecil MSE, semakin baik prediksi model.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$
- Sama seperti MSE, tetapi menghitung rata-rata selisih mutlak tanpa dikuadratkan.
- MAE tidak sepeka MSE terhadap outlier.

## Evaluation Result
| Model   | MSE                | % Prediksi dalam ±1 poin |
|---------|--------------------|--------------------------|
| Shallow | 2.1556084156036377 |          42.375%         |
| Medium  | 2.4154422283172607 |          42.375%         |
| Deep    | 2.7552337646484375 |          40.75%          |

Model **shallow ANN** menghasilkan hasil terbaik secara MSE (terkecil) dan akurasi prediksi dalam ±1 poin (tertinggi, sama dengan medium), sehingga dipilih sebagai model terbaik dan disimpan ke `best_model_keras.h5`.
- Training setiap model: ±10–12 detik per model pada CPU (Intel i7 8th Gen)

Evaluasi CBF dengan Precision@K memiliki hasil 0.0007, yang berarti dari seluruh pengguna, hanya sekitar 0.07% dari Top-5 rekomendasi yang cocok dengan item yang pernah mereka beri rating tinggi.

