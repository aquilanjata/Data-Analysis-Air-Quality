# Data-Analysis-Air-Quality

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis kualitas udara berdasarkan data historis dari 12 stasiun pemantauan di Beijing (periode 1 Maret 2013 – 28 Februari 2017). Analisis berfokus pada identifikasi faktor meteorologi dan polutan yang memicu lonjakan PM2.5 (>75 μg/m³) selama musim dingin maupun non-musim dingin, serta mengevaluasi tren perubahan konsentrasi polutan dari tahun ke tahun.

## Struktur Direktori
```text
├───dashboard
│   ├───main_data.csv
│   └───dashboard.py
├───data
│   └───PRSA_Data_*.csv
├───notebook.ipynb
├───README.md
└───requirements.txt
```

## Setup Environment - Anaconda
```bash
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```bash
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Menjalankan Aplikasi Streamlit
```bash
streamlit run dashboard/dashboard.py
```
