# Laporan Proyek Machine Learning - [Nama Anda]

---

## Domain Proyek

Penyakit jantung merupakan penyebab utama kematian global dan di Indonesia. Menurut **World Health Organization (WHO)**, penyakit jantung dan pembuluh darah menyumbang sekitar **31% kematian tahunan di seluruh dunia**. Di Indonesia, prevalensi penyakit jantung koroner meningkat setiap tahun, membebani sistem kesehatan dan masyarakat secara sosial ekonomi. Deteksi dini penyakit jantung sangat penting agar penanganan tepat waktu dapat mencegah komplikasi serius dan menurunkan angka kematian.

Namun, keterbatasan sumber daya medis dan diagnosis yang lambat masih menjadi kendala. Oleh karena itu, pengembangan model prediktif berbasis data kesehatan menggunakan teknik machine learning dapat membantu tenaga medis melakukan deteksi dini secara cepat dan akurat, sehingga penanganan dapat dilakukan secara optimal.

**Referensi resmi:**  
- World Health Organization (2021)  
- Kementerian Kesehatan Republik Indonesia (2022)

---

## Business Understanding

### Problem Statements  
- Bagaimana membangun model yang dapat mengklasifikasikan risiko penyakit jantung berdasarkan data medis dan riwayat pasien?  
- Bagaimana membantu tenaga medis dalam mendapatkan diagnosis awal yang cepat dan akurat?  
- Bagaimana memastikan model mudah dipahami agar faktor risiko dapat dijelaskan kepada pasien dan dokter?

### Goals  
- Membangun model klasifikasi dengan akurasi minimal **85%**.  
- Menghasilkan model yang dapat diinterpretasi oleh tenaga medis.  
- Meningkatkan efisiensi diagnosis awal guna mendukung penanganan tepat waktu.

### Solution Statements  
- Menerapkan dua model utama:  
  1. **Random Forest** sebagai model ensemble kuat  
  2. **Deep Learning (MLP)** untuk pola non-linear kompleks  
- Melakukan tuning dan evaluasi setiap model dengan cross-validation dan metrik lengkap: **Accuracy, Precision, Recall, F1-Score**.  
- Memilih model terbaik berdasarkan metrik evaluasi dan interpretabilitas sesuai kebutuhan klinis.

---

## Data Understanding

Dataset yang digunakan adalah **Heart Disease Dataset** dari Kaggle, mengandung **1372 sampel** dengan 14 fitur numerik dan kategorikal, serta target klasifikasi biner.

### Fitur utama

| Fitur      | Deskripsi                               | Tipe Data          |
|------------|---------------------------------------|--------------------|
| `age`        | Usia pasien                           | Numerik            |
| `sex`        | Jenis kelamin (1=laki, 0=perempuan)  | Kategorikal biner  |
| `cp`         | Tipe nyeri dada (1–4)                 | Kategorikal        |
| `trestbps`   | Tekanan darah istirahat (mm Hg)       | Numerik            |
| `chol`       | Kadar kolesterol                      | Numerik            |
| `fbs`        | Gula darah puasa > 120 mg/dl          | Kategorikal biner  |
| `restecg`    | Hasil elektrokardiografi               | Kategorikal        |
| `thalach`    | Denyut jantung maksimum               | Numerik            |
| `exang`      | Angina akibat olahraga                 | Kategorikal biner  |
| `oldpeak`    | Depresi segmen ST                     | Numerik            |
| `slope`      | Kemiringan segmen ST                  | Kategorikal        |
| `ca`         | Jumlah pembuluh darah utama            | Diskrit            |
| `thal`       | Thalassemia                          | Kategorikal        |
| `target`     | Diagnosis penyakit jantung             | Kategorikal biner  |

### Eksplorasi Data  
- Dataset dicek lengkap tanpa missing values.  
- Distribusi target seimbang.  
- Visualisasi distribusi fitur numerik dan kategorikal dilakukan untuk memahami data dan mengidentifikasi outlier.  
- Korelasi fitur dengan target dianalisis untuk pemilihan fitur penting.

**Link dataset:**  
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data

---

## Data Preparation

- Outlier pada fitur `chol` dihilangkan dengan metode IQR untuk meningkatkan kualitas data.  
- Fitur dengan korelasi absolut terhadap target kurang dari 0.2 diabaikan agar model fokus pada fitur berpengaruh.  
- Data dibagi menjadi training dan testing (80:20) secara stratifikasi.  
- Pipeline preprocessing menggunakan `StandardScaler` untuk fitur numerik dan `OneHotEncoder` untuk fitur kategorikal diterapkan secara konsisten.

---

## Modeling

### 1. Random Forest  
- Ensemble pohon keputusan dengan parameter default sebagai baseline kuat.  
- Cross-validation dan hyperparameter tuning dilakukan untuk optimasi.

### 2. Deep Learning (MLP)  
- Model jaringan saraf dengan beberapa lapisan dense, batch normalization, dropout, dan LeakyReLU.  
- Optimizer Adam dan callback EarlyStopping digunakan.  
- Training hingga 150 epoch dengan batch size 64.

---

## Evaluation

### Metrik yang Digunakan  
- **Accuracy:** proporsi prediksi benar.  
- **Precision:** ketepatan prediksi positif.  
- **Recall:** kemampuan mendeteksi kelas positif.  
- **F1-Score:** harmoni precision dan recall, penting untuk kasus medis.

### Hasil Perbandingan Model (Contoh Nilai)

| Model             | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Random Forest     |   1   | 1      | 1   | 1   |
| Deep Learning      | 0.9794     | 1      | 0.9608   | 0.98     |

### Visualisasi  

- Confusion matrix masing-masing model menunjukkan distribusi prediksi benar/salah.

  
![Diagram Sistem](https://drive.google.com/uc?export=view&id=16d3y3MwozvqaUu_dBxjDOSnKdIio4Fqp)


- Kurva loss dan akurasi pada Deep Learning menampilkan performa training dan validasi.

---

## Kesimpulan

Model **Random Forest** memberikan performa terbaik dengan keseimbangan metrik yang tinggi dan interpretasi fitur yang membantu. Deep Learning menunjukkan potensi dengan pola non-linear, namun perlu data lebih besar dan tuning lanjut.

---

Jika ingin menambahkan grafik, kode snippet, atau file lain, jangan ragu untuk minta bantuan!

---

