# Laporan Proyek Machine Learning - MIFTAHULLAH SURYA NUGRAHA

---

## Daftar Isi

- [Overview Proyek](#overview-proyek)  
- [Business Understanding](#business-understanding)  
  - [Problem Statements](#problem-statements)  
  - [Tujuan](#tujuan)  
  - [Solution Statements](#solution-statements)  
- [Data Understanding](#data-understanding)  
  - [Fitur Utama](#fitur-utama)  
  - [Statistik Deskriptif](#statistik-deskriptif)  
  - [Missing Values](#missing-values)  
  - [Distribusi Target](#distribusi-target)  
  - [Visualisasi Fitur](#visualisasi-fitur)  
  - [Deteksi Outlier](#deteksi-outlier)  
  - [Korelasi](#korelasi)  
  - [Analisis Kategorikal vs Target](#analisis-kategorikal-vs-target)  
- [Data Preparation](#data-preparation)  
- [Modeling](#modeling)  
- [Evaluation](#evaluation)  
- [Inference (Pengujian Model pada Data Baru)](#inference-pengujian-model-pada-data-baru)  
---

## Overview Proyek

Penyakit jantung merupakan penyebab utama kematian global, termasuk di Indonesia. Menurut **World Health Organization (WHO)**, penyakit jantung dan pembuluh darah menyumbang sekitar **31%** dari total kematian tahunan di seluruh dunia (World Health Organization, 2021). Di Indonesia, prevalensi penyakit jantung koroner terus meningkat setiap tahunnya, memberikan dampak signifikan terhadap sistem kesehatan nasional serta aspek sosial dan ekonomi masyarakat (Kementerian Kesehatan Republik Indonesia, 2022).  

Deteksi dini penyakit jantung sangat krusial agar penanganan yang cepat dan tepat dapat mencegah komplikasi serius serta menurunkan angka kematian.

Namun, keterbatasan sumber daya medis dan lambatnya proses diagnosis masih menjadi kendala utama dalam penanganan penyakit ini. Oleh karena itu, pengembangan model prediktif berbasis data kesehatan dengan teknik *machine learning* berpotensi membantu tenaga medis dalam melakukan deteksi dini secara cepat dan akurat, sehingga intervensi yang optimal dapat dilakukan (Mathur et al., 2020).

### Referensi:

- World Health Organization. (2021). *Cardiovascular diseases (CVDs)*. Retrieved from [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))  
- Kementerian Kesehatan Republik Indonesia. (2022). *Profil Kesehatan Indonesia Tahun 2021*. Jakarta: Kementerian Kesehatan RI.  
- Mathur, P., Srivastava, S., Xu, X., & Mehta, J. L. (2020). Artificial intelligence, machine learning, and cardiovascular disease. *Journal of the American College of Cardiology*, 75(23), 2879–2891. [https://doi.org/10.1016/j.jacc.2020.04.010](https://doi.org/10.1016/j.jacc.2020.04.010)

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
  1. **Random Forest** sebagai model ensemble yang kuat  
  2. **Deep Learning (MLP)** untuk mendeteksi pola non-linear kompleks  
- Melakukan evaluasi setiap model dengan metrik lengkap: *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.  
- Memilih model terbaik berdasarkan metrik evaluasi.

---

## Data Understanding

**Link dataset:**  
[Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)  

Dataset berisi **1025 sampel** dengan 14 fitur numerik dan kategorikal, serta target klasifikasi biner.

### Fitur Utama

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

### Statistik Deskriptif  
Statistik fitur numerik memberikan gambaran distribusi dan sebaran data tiap variabel utama.

### Missing Values  
Tidak ditemukan missing values.

### Distribusi Target  
Visualisasi distribusi target menunjukkan keseimbangan kelas.

![Distribusi Target](images/Distribusi-Target.png)

### Visualisasi Fitur  
Histogram dan KDE untuk fitur numerik, serta countplot untuk kategorikal, membantu memahami distribusi dan variasi data.

![Distribusi Data Numerik](images/Distribusi-Numerik.png)

![Distribusi Data Kategorikal](images/Distribusi-Kategorikal.png)

### Deteksi Outlier  
Boxplot mengindikasikan keberadaan outlier yang perlu ditangani.

![Deteksi Outlier](images/Deteksi-Outlier.png)

### Korelasi  
Matriks korelasi membantu memilih fitur yang paling relevan untuk prediksi.

![Matriks Korelasi](images/Matrix-Korelasi.png)

### Analisis Kategorikal vs Target  
Countplot dengan hue target menunjukkan hubungan kategori dan kelas target.

![Distribusi Kategorikal Target](images/Kategorikal-Target.png)

---

## Data Preparation

- Outlier pada fitur **`chol`** (kolesterol) dihapus menggunakan metode *Interquartile Range* (IQR).

  #### Deteksi Outlier dengan Metode IQR

  Interquartile Range (IQR) digunakan untuk mengukur variabilitas dengan membagi dataset menjadi empat bagian yang sama besar (kuartil). Data diurutkan secara menaik, kemudian dibagi menjadi empat bagian berdasarkan nilai kuartil:

  - **Q1 (kuartil pertama):** persentil ke-25, yaitu median dari data bagian bawah.
  - **Q2 (kuartil kedua / median):** persentil ke-50, nilai tengah dataset.
  - **Q3 (kuartil ketiga):** persentil ke-75, median dari data bagian atas.

  Outlier diidentifikasi sebagai nilai data di luar batas:
  \[
  \text{Lower Bound} = Q1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q3 + 1.5 \times IQR
  \]
  dengan \(IQR = Q3 - Q1\).

- Penghapusan outlier juga dilakukan pada fitur **`trestbps`**, serta penghapusan nilai minimum pada fitur **`thalach`** dan nilai maksimum pada fitur **`oldpeak`** untuk menjaga konsistensi data.

- Fitur dengan nilai korelasi absolut terhadap target kurang dari 0.2 diabaikan agar model fokus pada fitur berpengaruh signifikan.

- Data dibagi menjadi data training dan testing dengan rasio 80:20 menggunakan stratifikasi berdasarkan label target untuk menjaga keseimbangan kelas.

- Pipeline preprocessing dibangun menggunakan `StandardScaler` untuk normalisasi fitur numerik dan `OneHotEncoder` untuk encoding fitur kategorikal, diterapkan konsisten pada data training dan testing.

---

## Modeling

### 1. Random Forest  
- Random Forest merupakan metode ensemble pohon keputusan yang digunakan dengan parameter default sebagai baseline kuat.  
- Evaluasi menggunakan *Stratified K-Fold Cross-Validation* dengan metrik F1-score untuk menjaga keseimbangan kelas.  
- Model dilatih pada data training dan diuji pada data testing.  
  
- **Kelebihan:**  
  - Mudah diinterpretasikan karena struktur pohon keputusan jelas.  
  - Tahan terhadap overfitting, terutama pada data dengan banyak fitur dan noise.  
  - Efektif menangani data numerik dan kategorikal tanpa preprocessing rumit.  

- **Kekurangan:**  
  - Waktu pelatihan dan prediksi dapat meningkat jika jumlah pohon besar.  
  - Kurang optimal dalam menangkap pola non-linear sangat kompleks dibanding deep learning.  

### 2. Deep Learning (Multilayer Perceptron - MLP)  
- Model MLP dibangun dengan beberapa lapisan *dense*, *batch normalization*, *dropout*, dan fungsi aktivasi LeakyReLU.  
- Optimizer Adam dengan learning rate 0.0005 digunakan.  
- Callback *EarlyStopping* dan *ReduceLROnPlateau* diterapkan untuk mencegah overfitting dan mengoptimalkan training.  
- Model dilatih hingga maksimal 150 epoch dengan batch size 64.  
  
- **Kelebihan:**  
  - Mampu menangkap pola non-linear kompleks dan interaksi fitur sulit terdeteksi model lain.  
  - Fleksibel dan dapat dioptimalkan dengan teknik regularisasi untuk meningkatkan generalisasi.  

- **Kekurangan:**  
  - Membutuhkan waktu pelatihan lebih lama dan data lebih banyak agar optimal.  
  - Rentan overfitting tanpa teknik regularisasi dan validasi tepat.  
  - Interpretabilitas rendah dibanding Random Forest.  

---

## Evaluation

Dalam evaluasi performa algoritma *machine learning*, khususnya untuk klasifikasi, beberapa metrik penting yang umum digunakan adalah:

### 1. Accuracy  
Accuracy mengukur proporsi prediksi yang benar terhadap seluruh data yang diuji. Rumusnya adalah:

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}
$$

dimana:  
- *TP* = True Positive  
- *TN* = True Negative  
- *FP* = False Positive  
- *FN* = False Negative  

### 2. Precision  
Precision mengukur seberapa akurat prediksi positif yang dihasilkan, yaitu proporsi prediksi positif yang benar-benar positif:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### 3. Recall (Sensitivitas)  
Recall mengukur kemampuan model dalam menemukan semua kasus positif yang sebenarnya:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### 4. Specificity  
Specificity mengukur kemampuan model dalam mengidentifikasi kasus negatif yang benar:

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

### 5. F1-Score  
F1-Score adalah rata-rata harmonis antara Precision dan Recall yang memberikan ukuran keseimbangan keduanya:

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Hasil Perbandingan Model

| Model          | Accuracy | Precision | Recall  | F1-Score |
|----------------|----------|-----------|---------|----------|
| Random Forest  | 1.0000   | 1.0000    | 1.0000  | 1.0000   |
| Deep Learning  | 0.9794   | 1.0000    | 0.9608  | 0.9798   |

### Visualisasi  

Confusion matrix adalah matriks yang menunjukkan performa model klasifikasi dengan cara memvisualisasikan jumlah prediksi yang benar dan salah dalam beberapa kategori:

|                  | Prediksi Positif | Prediksi Negatif |
|------------------|------------------|------------------|
| **Data Positif**  | True Positive (TP) | False Negative (FN) |
| **Data Negatif**  | False Positive (FP) | True Negative (TN) |

- **True Positive (TP):** Prediksi model benar bahwa sampel termasuk kelas positif.  
- **False Positive (FP):** Prediksi model salah bahwa sampel positif padahal negatif (kesalahan tipe I).  
- **False Negative (FN):** Prediksi model salah bahwa sampel negatif padahal positif (kesalahan tipe II).  
- **True Negative (TN):** Prediksi model benar bahwa sampel termasuk kelas negatif.

Confusion matrix membantu memahami jenis kesalahan model yang terjadi dan memberikan gambaran yang lebih rinci dibanding metrik akurasi tunggal.

Learning curve adalah grafik yang menggambarkan performa model selama proses pelatihan terhadap jumlah epoch (iterasi) tertentu, biasanya dalam bentuk:

- **Loss:** Menunjukkan seberapa besar kesalahan model. Loss yang menurun menunjukkan model belajar memperbaiki prediksi.  
- **Accuracy:** Menunjukkan persentase prediksi yang benar dari total data, biasanya meningkat seiring pelatihan.

Learning curve sering dipakai untuk:

- Mendeteksi **overfitting**, yaitu ketika performa model pada data training sangat baik tetapi buruk pada data validasi. Ditandai dengan loss training rendah namun loss validasi mulai naik.  
- Mendeteksi **underfitting**, ketika model tidak belajar dengan baik dan performa tetap rendah di training maupun validasi.

---

- Confusion matrixdari hasil train model dan validasi:

  ![Confusion Matrix Random Forest](images/Confusion-Matrix-RF.png)

  ![Confusion Matrix Deep Learning](images/Confusion-Matrix-DL.png)
  
- Learning curve atau kurva loss dan akurasi model Deep Learning menggambarkan performa training dan validasi.

  ![Loss dan Accuracy Deep Learning](images/Loss-Accuracy-DL.png)

---

## Inference (Pengujian Model pada Data Baru)

Setelah pelatihan dan evaluasi, model diuji pada data baru untuk mengukur kemampuan generalisasi.

### Prosedur Inference

1. Data baru diproses menggunakan pipeline preprocessing yang sama.  
2. Model Random Forest dan Deep Learning digunakan untuk memprediksi target.  
3. Evaluasi menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.

### Hasil Inference

| Model          | Accuracy | Precision | Recall  | F1-Score |
|----------------|----------|-----------|---------|----------|
| Random Forest  | 0.9604   | 0.9487    | 0.9755  | 0.9619   |
| Deep Learning  | 0.9237   | 0.8993    | 0.9586  | 0.9280   |

![Inference Random Forest](images/Inference-RF.png)

![Inference Deep Learning](images/Inference-DL.png)

Model Random Forest unggul dalam prediksi pada data baru, sedangkan Deep Learning juga memberikan performa memadai.
