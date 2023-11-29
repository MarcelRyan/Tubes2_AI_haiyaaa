# Tubes2_AI_haiyaaa
Implementasi Algoritma Pembelajaran Mesin pada Pemrosesan Data: K-Nearest Neighbor & Naive-Bayes

## Daftar Isi
1. [Deskripsi Singkat](#deskripsi-singkat)
2. [Algoritma](#algoritma)
3. [Credits](#credits)

## Deskripsi Singkat
Data terdiri dari beberapa kolom berikut:
1. _battery_power_: Total energi baterai dalam satu waktu diukur dalam mAh
2. _blue_: Memiliki bluetooth atau tidak
3. _clock_speed_: Kecepatan mikroprosesor menjalankan instruksi
4. _dual_sim_: Memiliki dukungan dual sim atau tidak
5. _fc_: Resolusi kamera depan dalam megapiksel
6. _four_g_: Memiliki 4G atau tidak
7. _int_memory_: Memori internal dalam gigabyte
8. _m_dep_: Ketebalan ponsel dalam cm
9. _mobile_wt_: Berat ponsel
10. _n_cores_: Jumlah core prosesor
11. _pc_: Resolusi kamera utama dalam megapiksel
12. _px_height_: Tinggi resolusi piksel
13. _px_width_: Lebar resolusi piksel
14. _ram_: Ukuran RAM dalam megabyte
15. _sc_h_: Tinggi layar ponsel dalam cm
16. _sc_w_: Lebar layar ponsel dalam cm
17. _talk_time_: Waktu telepon maksimum dalam satu kali pengisian baterai
18. _three_g_: Memiliki 3G atau tidak
19. _touch_screen_: Memiliki layar sentuh atau tidak
20. _wifi_: Memiliki wifi atau tidak
21. _price_range_ (target): Rentang harga dengan nilai 0 (biaya rendah), 1 (biaya sedang), 2 (biaya tinggi) atau 3 (biaya sangat tinggi).

## Algoritma
### K-Nearest Neighbor
K-Nearest Neighbors (KNN) merupakan salah satu algoritma _supervised learning_  yang bersifat non-parametrik dan _lazy learning_.

Algoritma KNN mengukur jarak nilai data terhadap seluruh instansi data yang ada di data latih yang digunakan. Adapun jarak ditentukan dengan _Euclidean Distance_. Kemudian, KNN akan memilih sebanyak _k_ data yang memiliki jarak terdekat dan sejumlah _k_-data itu akan digunakan untuk mencari mayoritas nilai target yang diinginkan. Dalam kasus data yang telah ditentukan, dari sejumlah _k_ data akan diambil data mayoritas dengan nilai _price_range_ yang paling banyak. Data mayoritas tersebut yang akan dijadikan kesimpulan prediksi _price_range_ data tersebut.


### Naive-Bayes
Naive-Bayes adalah salah satu algoritma klasifikasi yang memanfaatkan teorema Bayes dengan asumsi bahwa setiap atribut yang digunakan untuk memprediksi tidak ada hubungan satu dengan yang lainnya. Untuk atribut numerik dan non-numerik, algoritma ini memiliki perhitungan yang berbeda.

Untuk melakukan prediksi menggunakan atribut non-numerik, digunakan rumus yaitu _P(A|B)_ dari teorema Bayes dengan A adalah klasifikasi target dan B adalah nilai dari atribut. Berbeda dengan atribut numerik, perhitungan perlu dilakukan dengan menggunakan rumus distribusi normal. Nilai prediksi yang diperoleh dari perhitungan tersebut kemudian dipasangkan dengan klasifikasi target yang sudah ada sehingga diketahui hasil prediksi berdasarkan nilai yang diperoleh dari kalkulasi tersebut.

## Credits
| NIM | Nama |
|-----|------|
|13521057|Hosea Nathanael Abetnego|
|13521059|Arleen Chrysantha Gunardi|
|13521127|Marcel Ryan Antony|
|13521145|Kenneth Dave Bahana|
