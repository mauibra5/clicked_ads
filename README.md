# Latar Belakang & Objektif
 - Sebuah perusahaan ingin mengetahui efektifitas sebuah iklan yang mereka tayangkan. 
 - Hal ini penting bagi perusahaan yang bergerak di bidang consultant digital marketing agar dapat mengetahui seberapa besar ketercapainnya iklan yang dipasarkan sehingga dapat menarik customers untuk melihat iklan.

# Batasan Masalah
 - Dalam tujuan mengurangi biaya pengiklanan dan meningkatkan conversion rate perusahaan memutuskan untuk membuat model machine learning yang dapat membedakan mana customer yang akan mengklik iklan & mana yang tidak, sehingga iklan dapat ditargetkan ke customer yang akan mengklik iklan. 
 
# Data dan Asumsi
Berikut penjelasan feature-feature yang ada di dalam dataset ini.
 - 'Daily Time Spent on Site': waktu yang dihabiskan customer di website (dalam menit)
 -   'Age': umur customer (dalam tahun)
 -   'Area Income':  pendapatan rata-rata wilayah geografis konsumen
 -   'Daily Internet Usage': rata-rata berapa menit per hari penggunaan internet
 -   'Male': apakah customer pria atau wanita (1 jika pria, 0 jika wanita)
 -   'Timestamp': waktu dimana customer mengklik/menutup iklan
 -   'Clicked on Ad': 1 jika mengklik iklan, 0 jika tidak
 -   'City': kota tempat tinggal customer
 -   'Province': provinsi tempat tinggal customer
 -   'Category': kategori product yang dikunjungi
 
# Analisis Data
## Data Distribution
 -    Saya melakukan statistical analysis untuk data numerik dengan df.describe() untuk mendapatkan keterangan statistik berupa count, min, max, mean, percentiles, dan standard deviation. Sedangkan untuk data kategorikal saya menggunakan df.select_dtypes(include='object').describe() dengan output keterangan count, unique, top, freq.
    
 -   Untuk melihat bentuk persebaran data masing-masing kolom (univariate analysis) saya menggunakan metode visualisasi histogram untuk kolom numerik, sedangkan untuk kolom kategorikal saya menggunakan bar plot.
    
 -   Untuk bivariate analysis saya menggunakan box plot, untuk mencari hubungan antar 3 feature: ‘Age’, 'Daily Internet Usage', 'Daily Time Spent on Site'.
 
 -   Untuk multivariate analysis saya menggunakan correlation heatmap untuk memudahkan melihat seberapa besar korelasi antar feature, dan pairplot untuk melihat hubungan persebaran data antar feature.
## Data Preprocessing
 -    Melihat jumlah null value pada masing-masing kolom dengan df.isna().sum() lalu saya memutuskan melakukan imputasi median dengan pertimbangan agar tidak mengubah sebaran data.
    
 -   Memeriksa apakah ada row yang duplicated dengan df[df.duplicated()] yang hasilnya tidak ada, maka tidak ada pemrosesan lebih lanjut untuk duplicated value.
    
 -   Mengekstrak value tahun, bulan, minggu, dan hari dari kolom ‘Timestamp’ dengan menggunakan Series.dt. Setelah mengekstrak kolom ‘Timestamp’ menjadi 4 kolom baru, lalu saya menghapus kolom originalnya. Setelah diperiksa lebih lanjut, kolom ‘year’ hanya berisi 1 unique value, maka saya memutuskan untuk menghapus kolom ‘year’ karena kolom dengan 1 unique value tidak memiliki pengaruh yang signifikan.
    
 -   Melakukan split data, pertama-tama membagi menjadi X(kolom berisi features) dan y(kolom target), kemudian membaginya lagi untuk masing-masing train dan test dengan persentase 80:20.
    
 -   Melakukan feature encoding untuk kolom kategorikal menggunakan fitur pandas get_dummies() sehingga masing-masing value dari kolom kategorikal menjadi kolom-kolom baru yang berisi value 1 dan 0.
 
# Simpulan
## Feature yang digunakan
 - Setelah dilakukan feature encoding, feature-feature dalam dataset menjadi berjumlah 65, feature-feature tersebut antara lain: 'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Clicked on Ad', 'month', 'week', 'day',
       'Male_Laki-Laki', 'Male_Perempuan', 'city_Balikpapan',
       'city_Bandar Lampung', 'city_Bandung', 'city_Banjarmasin', 'city_Batam',
       'city_Bekasi', 'city_Bogor', 'city_Cimahi', 'city_Denpasar',
       'city_Depok', 'city_Jakarta Barat', 'city_Jakarta Pusat',
       'city_Jakarta Selatan', 'city_Jakarta Timur', 'city_Jakarta Utara',
       'city_Makassar', 'city_Malang', 'city_Medan', 'city_Padang',
       'city_Pakanbaru', 'city_Palembang', 'city_Pontianak', 'city_Samarinda',
       'city_Semarang', 'city_Serang', 'city_Surabaya', 'city_Surakarta',
       'city_Tangerang', 'city_Tangerang Selatan', 'city_Tasikmalaya',
       'province_Bali', 'province_Banten',
       'province_Daerah Khusus Ibukota Jakarta', 'province_Jawa Barat',
       'province_Jawa Tengah', 'province_Jawa Timur',
       'province_Kalimantan Barat', 'province_Kalimantan Selatan',
       'province_Kalimantan Timur', 'province_Kepulauan Riau',
       'province_Lampung', 'province_Riau', 'province_Sulawesi Selatan',
       'province_Sumatra Barat', 'province_Sumatra Selatan',
       'province_Sumatra Utara', 'category_Bank', 'category_Electronic',
       'category_Fashion', 'category_Finance', 'category_Food',
       'category_Furniture', 'category_Health', 'category_House',
       'category_Otomotif', 'category_Travel'
  ## Performa model
  ![Confusion Matrix](https://github.com/mauibra5/clicked_ads/blob/main/confusion_matrix.png?raw=true)
 ![Feature Importance](https://github.com/mauibra5/clicked_ads/blob/main/feature_importance.png?raw=true)
 -   Dilihat dari hasil classification_report dan confusion matrix, perubahan tidak terlalu ekstrem sebelum dan sesudah dilakukannya normalisasi untuk algoritma DecisionTreeClassifier dan RandomForestClassifier, lain halnya dengan algoritma KNeighborsClassifier yang lebih sensitif terhadap range data, oleh karena itu untuk algoritma KNeighborsClassifier dibutuhkan proses normalisasi terlebih dahulu, dan hasilnya dapat dilihat ada perbaikan skor antara hasil KNeighborsClassifier sebelum dengan KNeighborsClassifier setelah dilakukan normalisasi.
    
-   Karena permasalahan bisnis di sini adalah menentukan mana customer yang akan mengklik iklan, maka kita akan menggunakan model dengan false positive paling rendah, yang artinya kita ingin seakurat mungkin menargetkan iklan ke customer-customer yang akan mengklik, sehingga biaya marketing tidak terbuang sia-sia dengan mengiklankan ke customer yang tidak berpotensi (false positive), dari confusion matrix dapat dilihat bahwa DecisionTreeClassifier setelah data dinormalisasi adalah algoritma dengan false positive yang paling rendah.
    
-   Berdasarkan hasil dari feature importance, 2 feature yang sangat signifikan untuk dipertimbangkan dalam menargetkan iklan adalah ‘Daily Internet Usage’ dan ‘Daily Time Spent on Site’.
  
# Saran
 -    Hasil EDA menunjukkan kebanyakan customer yang mengklik iklan terletak pada point rendah pada variabel ‘Daily Internet Usage’ dan ‘Daily Time Spent on Site’, yang artinya semakin sedikit penggunaan internet & waktu yang dihabiskan customer di website maka semakin besar potensi customer untuk mengklik iklan. Maka kita dapat fokus untuk memasang iklan pada website dengan konten-konten yang berbentuk short-form seperti artikel pendek atau video Tiktok/Youtube Shorts.
    
-   Melakukan evaluasi materi iklan, karena berdasarkan hasil EDA, kebanyakan customer yang mengklik iklan adalah customer dengan rentang umur 40-70, sedangkan customer dengan rentang umur di bawah 40 cenderung tidak mengklik iklan. Dapat diartikan bahwa materi iklan kurang ‘relatable’ untuk kelompok umur di bawah 40, maka direkomendasikan untuk membuat materi iklan yang lebih universal dan dapat meng-engage semua kelompok umur.
    
-   Pada kelompok pendapatan juga dapat dilakukan yang sama, hasil EDA menunjukkan customer dengan ‘Area Income’ yang tinggi cenderung tidak mengklik iklan, ini artinya harus dilakukan evaluasi lebih lanjut untuk materi iklan dimana nantinya materi iklan juga dapat menarik kelompok customer dengan ‘Area Income’ tinggi untuk mengklik iklan.