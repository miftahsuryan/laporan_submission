# Laporan Proyek Machine Learning - MIFTAHULLAH SURYA NUGRAHA

---

## Daftar Isi

- [Overview Proyek](#overview-proyek)  
- [Business Understanding](#business-understanding)  
  - [Problem Statements](#problem-statements)  
  - [Tujuan](#tujuan)  
  - [Solution Statements](#solution-statements)  
- [Data Understanding](#data-understanding)  
- [Data Preparation](#data-preparation)  
- [Modeling](#modeling)  
- [Evaluation](#evaluation)  
- [Kesimpulan](#kesimpulan)  

---

## Overview Proyek

Penyakit jantung merupakan penyebab utama kematian global, termasuk di Indonesia. Menurut **World Health Organization (WHO)**, penyakit jantung dan pembuluh darah menyumbang sekitar **31%** dari total kematian tahunan di seluruh dunia (World Health Organization, 2021). Di Indonesia, prevalensi penyakit jantung koroner terus meningkat setiap tahunnya, yang memberikan dampak signifikan terhadap sistem kesehatan nasional serta aspek sosial dan ekonomi masyarakat (Kementerian Kesehatan Republik Indonesia, 2022).  

Deteksi dini penyakit jantung sangat krusial agar penanganan yang cepat dan tepat dapat mencegah komplikasi serius serta menurunkan angka kematian.

Namun, keterbatasan sumber daya medis dan lambatnya proses diagnosis masih menjadi kendala utama dalam penanganan penyakit ini. Oleh karena itu, pengembangan model prediktif berbasis data kesehatan dengan menggunakan teknik *machine learning* berpotensi membantu tenaga medis dalam melakukan deteksi dini secara cepat dan akurat, sehingga intervensi yang optimal dapat dilakukan (Mathur et al., 2020).

---

### Referensi

- World Health Organization. (2021). *Cardiovascular diseases (CVDs)*. Retrieved from [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))  
- Kementerian Kesehatan Republik Indonesia. (2022). *Profil Kesehatan Indonesia Tahun 2021*. Jakarta: Kementerian Kesehatan RI.  
- Mathur, P., Srivastava, S., Xu, X., & Mehta, J. L. (2020). Artificial intelligence, machine learning, and cardiovascular disease. *Journal of the American College of Cardiology*, 75(23), 2879–2891. [https://doi.org/10.1016/j.jacc.2020.04.010](https://doi.org/10.1016/j.jacc.2020.04.010) Retrieved from [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7485162/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7485162/)

---

## Business Understanding

### Problem Statements  
- Bagaimana membangun model yang dapat mengklasifikasikan risiko penyakit jantung berdasarkan data medis dan riwayat pasien?  
- Bagaimana membantu tenaga medis dalam mendapatkan diagnosis awal yang cepat dan akurat?  

### Tujuan  
- Membangun model klasifikasi dengan akurasi minimal **90%**.  
- Menghasilkan model yang dapat diinterpretasi oleh tenaga medis.  

### Solution Statements  
- Menerapkan dua model utama:  
  1. **Random Forest** sebagai model ensemble kuat  
  2. **Deep Learning (MLP)** untuk pola non-linear kompleks  
- Melakukan evaluasi setiap model dengan metrik lengkap: *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.  
- Memilih model terbaik berdasarkan metrik evaluasi.

---

## Data Understanding

**Link dataset:**  
[Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)  

Dataset yang digunakan adalah **Heart Disease Dataset** dari Kaggle, mengandung **1025 sampel** dengan 14 fitur numerik dan kategorikal, serta target klasifikasi biner.

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

- Dataset telah diperiksa dan **tidak ditemukan nilai yang hilang (missing values)**, memastikan kualitas data yang baik untuk analisis lebih lanjut.

  ![Pemeriksaan Missing Values](https://github.com/user-attachments/assets/f0bac4ff-57da-46d2-bc7c-89e342b56c54 "Distribusi Missing Values")

- Distribusi kelas target relatif **seimbang**, sehingga model tidak bias pada salah satu kelas.

  ![Distribusi Target](https://github.com/user-attachments/assets/6f1bc2fb-d846-4ab3-ad4f-5523d54e42cf "Distribusi Kelas Target")

- Visualisasi distribusi fitur numerik dan kategorikal dilakukan guna memahami karakteristik data serta mengidentifikasi adanya outlier.

  ![Distribusi Fitur Numerik](https://github.com/user-attachments/assets/8365567b-c155-4872-a137-404a60ca737b "Distribusi Fitur Numerik")

  ![Distribusi Fitur Kategorikal](https://github.com/user-attachments/assets/45b3e1ae-762f-42ec-98f3-a0ba8b34f4ad "Distribusi Fitur Kategorikal")

- Analisis korelasi antara fitur-fitur dengan target dilakukan untuk memilih fitur-fitur yang memiliki pengaruh signifikan terhadap prediksi.

  ![Korelasi Fitur dengan Target](https://github.com/user-attachments/assets/ec69c4c5-afcd-42ff-86d6-7ef6a75fd157 "Korelasi Fitur dengan Target")

---

## Data Preparation

- Outlier pada fitur **`chol`** (kolesterol) dihapus menggunakan metode *Interquartile Range* (IQR).

  #### Deteksi Outlier dengan Metode IQR

  Interquartile Range (IQR) digunakan untuk mengukur variabilitas dengan membagi dataset menjadi empat bagian yang sama besar (kuartil). Data diurutkan secara menaik, kemudian dibagi menjadi empat bagian berdasarkan nilai kuartil:

  - **Q1 (kuartil pertama):** persentil ke-25, yaitu median dari data bagian bawah.
  - **Q2 (kuartil kedua / median):** persentil ke-50, nilai tengah dataset.
  - **Q3 (kuartil ketiga):** persentil ke-75, median dari data bagian atas.

  Untuk dataset dengan jumlah data \(2n\) atau \(2n + 1\), perhitungan kuartil dilakukan sebagai berikut:
  - Q2 adalah median seluruh dataset.
  - Q1 adalah median dari \(n\) data terkecil.
  - Q3 adalah median dari \(n\) data terbesar.

  Outlier diidentifikasi sebagai nilai data yang berada di luar batas:
  \[
  \text{Lower Bound} = Q1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q3 + 1.5 \times IQR
  \]
  dimana \(IQR = Q3 - Q1\).

  Data di luar batas ini dianggap outlier dan dikeluarkan untuk meningkatkan kualitas dan konsistensi data sebelum pemodelan.

- Outlier juga dihapus pada fitur-fitur lain seperti **`trestbps`** (tekanan darah saat istirahat) menggunakan metode IQR yang sama, serta penghapusan nilai minimum pada fitur **`thalach`** (denyut jantung maksimum) dan nilai maksimum pada fitur **`oldpeak`** (depresi segmen ST) untuk menjaga konsistensi data.

- Fitur dengan nilai korelasi absolut terhadap variabel target kurang dari 0.2 diabaikan agar model hanya fokus pada fitur yang berpengaruh signifikan terhadap prediksi.

- Data kemudian dibagi menjadi data training dan testing dengan perbandingan 80:20 secara stratifikasi berdasarkan label target agar proporsi kelas tetap seimbang.

- Pipeline preprocessing disusun menggunakan `StandardScaler` untuk normalisasi fitur numerik dan `OneHotEncoder` untuk konversi fitur kategorikal menjadi format numerik yang sesuai. Pipeline ini diterapkan secara konsisten pada data training dan testing untuk menjaga keseragaman proses.

---

## Modeling

### 1. Random Forest  
- Ensemble pohon keputusan dengan parameter default sebagai baseline yang kuat.  
- Cross-validation dengan *Stratified K-Fold* dan metrik F1-score digunakan untuk evaluasi.  
- Model dilatih pada data training dan diuji pada data testing.

### 2. Deep Learning (MLP)  
- Model jaringan saraf multilayer perceptron (MLP) dengan beberapa lapisan dense, batch normalization, dropout, dan LeakyReLU sebagai fungsi aktivasi.  
- Optimizer Adam dengan learning rate 0.0005 digunakan.  
- Callback EarlyStopping dan ReduceLROnPlateau dipasang untuk menghindari overfitting dan mengoptimalkan training.  
- Model dilatih hingga maksimal 150 epoch dengan batch size 64.

---

## Evaluation

### Metrik yang Digunakan  
- **Accuracy:** proporsi prediksi yang benar dari keseluruhan data.  
- **Precision:** ketepatan prediksi kelas positif.  
- **Recall:** kemampuan model mendeteksi semua kasus positif.  
- **F1-Score:** harmonisasi antara precision dan recall, sangat penting dalam konteks medis.

### Hasil Perbandingan Model

| Model          | Accuracy | Precision | Recall  | F1-Score |
|----------------|----------|-----------|---------|----------|
| Random Forest  | 1.0000   | 1.0000    | 1.0000  | 1.0000   |
| Deep Learning  | 0.9794   | 1.0000    | 0.9608  | 0.9798   |

### Visualisasi  

- Confusion matrix masing-masing model menunjukkan distribusi prediksi benar dan salah.

  ![Confusion Matrix RF](https://drive.google.com/uc?export=view&id=16d3y3MwozvqaUu_dBxjDOSnKdIio4Fqp "Confusion Matrix Random Forest")

- Kurva loss dan akurasi pada model Deep Learning menggambarkan performa training dan validasi.

---

## Kesimpulan

Model **Random Forest** memberikan performa terbaik dengan metrik evaluasi yang sangat baik dan interpretasi fitur yang membantu analisis medis. Model **Deep Learning** menunjukkan potensi dalam menangkap pola non-linear kompleks, namun membutuhkan data yang lebih besar dan tuning lebih lanjut untuk hasil optimal.

---

_Catatan:_  
Laporan ini disusun mengikuti kriteria dan rubrik penilaian resmi untuk proyek machine learning yang berbasis klasifikasi penyakit jantung.

