# ======================================================================================
# APP.PY
# APLIKASI ANALISIS SENTIMEN ULASAN GOPAY
# TF-IDF + SMOTE + LOGISTIC REGRESSION
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ======================================================================================
# 1. KONFIGURASI HALAMAN
# ======================================================================================

st.set_page_config(
    page_title="Analisis Sentimen GoPay",
    page_icon=None,
    layout="wide"
)


# ======================================================================================
# 2. SESSION STATE
# ======================================================================================

if "mulai" not in st.session_state:
    st.session_state.mulai = False


# ======================================================================================
# 3. HALAMAN SELAMAT DATANG
# ======================================================================================

if not st.session_state.mulai:

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.title("Selamat Datang")

    st.subheader(
        "Aplikasi Analisis Sentimen Ulasan GoPay"
    )

    st.write(
        "Aplikasi ini digunakan untuk menganalisis sentimen "
        "ulasan pengguna GoPay menjadi sentimen positif atau negatif."
    )

    st.write("")

    st.info(
        "Sistem menggunakan metode TF-IDF, SMOTE, dan "
        "Logistic Regression untuk melakukan klasifikasi sentimen."
    )

    st.write("")
    st.write("")

    st.markdown(
        "### Silakan klik tombol di bawah untuk mulai melakukan analisis."
    )

    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        if st.button(
            "Mulai Analisis",
            type="primary",
            use_container_width=True
        ):

            st.session_state.mulai = True

            st.rerun()

    st.stop()


# ======================================================================================
# 4. TOMBOL KEMBALI KE HALAMAN UTAMA
# ======================================================================================

col1, col2, col3 = st.columns([1, 1, 5])

with col1:

    if st.button(
        "Halaman Utama",
        use_container_width=True
    ):

        st.session_state.mulai = False

        st.rerun()


# ======================================================================================
# 5. LOAD MODEL DAN VECTORIZER
# ======================================================================================

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


try:

    model, vectorizer = load_model()

except Exception as e:

    st.error(
        "Model atau vectorizer tidak dapat dimuat."
    )

    st.error(
        f"Detail error: {e}"
    )

    st.stop()


# ======================================================================================
# 6. LOAD STEMMER SASTRAWI
# ======================================================================================

@st.cache_resource
def load_stemmer():

    factory = StemmerFactory()

    return factory.create_stemmer()


stemmer = load_stemmer()


# ======================================================================================
# 7. PREPROCESSING
# Cleaning → Case Folding → Normalisasi → Tokenizing
# → Stopword Removal → Stemming
# ======================================================================================

def preprocessing(teks):

    teks = str(teks)


    # ==========================================================================
    # CLEANING
    # ==========================================================================

    # Menghapus URL
    teks = re.sub(
        r"http\S+|www\S+",
        " ",
        teks
    )

    # Menghapus mention
    teks = re.sub(
        r"@\w+",
        " ",
        teks
    )

    # Menghapus tanda # tetapi mempertahankan kata
    teks = re.sub(
        r"#(\w+)",
        r"\1",
        teks
    )

    # Menghapus tag HTML
    teks = re.sub(
        r"<.*?>",
        " ",
        teks
    )

    # Menghapus angka dan karakter selain huruf
    teks = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        teks
    )

    # Menghapus spasi berlebih
    teks = re.sub(
        r"\s+",
        " ",
        teks
    ).strip()


    # ==========================================================================
    # CASE FOLDING
    # ==========================================================================

    teks = teks.lower()


    # ==========================================================================
    # NORMALISASI
    # ==========================================================================

    kamus = {

        "gak": "tidak",
        "ga": "tidak",
        "gk": "tidak",
        "nggak": "tidak",
        "ngga": "tidak",

        "blm": "belum",
        "blom": "belum",

        "udh": "sudah",
        "udah": "sudah",

        "bgt": "banget",
        "bngt": "banget",

        "yg": "yang",
        "yng": "yang",

        "utk": "untuk",

        "krn": "karena",
        "karna": "karena",

        "tp": "tapi",
        "tpi": "tapi",

        "jd": "jadi",
        "jdi": "jadi",

        "dr": "dari",
        "dri": "dari",

        "sy": "saya",
        "sya": "saya",

        "gmn": "bagaimana",
        "gmna": "bagaimana",

        "app": "aplikasi",

        "eror": "error",
        "erorr": "error",

        "lemot": "lambat",
        "lelet": "lambat",

        "mantab": "mantap",
        "mantul": "mantap",

        "bgus": "bagus",
        "baguss": "bagus",

        "gopayy": "gopay",

        "dapet": "dapat",

        "klo": "kalau",

        "aja": "saja",

        "msh": "masih",

        "skrg": "sekarang",
        "skrng": "sekarang",

        "trs": "terus",
        "trus": "terus",

        "jgn": "jangan",

        "bsa": "bisa",

        "ngak": "tidak",
        "ngk": "tidak",

        "pke": "pakai",
        "pake": "pakai",

        "makasih": "terima kasih",
        "mksh": "terima kasih",

        "thx": "terima kasih",
        "thanks": "terima kasih"
    }


    kata_normalisasi = []

    for kata in teks.split():

        kata_baru = kamus.get(
            kata,
            kata
        )

        kata_normalisasi.extend(
            kata_baru.split()
        )


    teks = " ".join(
        kata_normalisasi
    )


    # ==========================================================================
    # TOKENIZING
    # ==========================================================================

    tokens = teks.split()


    # ==========================================================================
    # STOPWORD REMOVAL
    # ==========================================================================

    stopword_tambahan = {

        "nya",
        "nih",
        "sih",
        "dong",
        "deh",
        "lah",
        "kah",
        "pun",
        "aja",
        "saja",
        "ya",
        "yah",
        "yg",
        "yang",
        "dan",
        "di",
        "ke",
        "dari"
    }


    # Kata yang tetap dipertahankan
    # karena memiliki pengaruh terhadap sentimen

    kata_dipertahankan = {

        "tidak",
        "belum",
        "bukan",
        "jangan",
        "sulit",
        "rumit",
        "lambat",
        "gagal",
        "error",
        "bagus",
        "baik",
        "mantap",
        "mudah",
        "cepat",
        "aman",
        "puas",
        "kecewa"
    }


    hasil_stopword = []

    for kata in tokens:

        if (
            kata not in stopword_tambahan
            or kata in kata_dipertahankan
        ):

            hasil_stopword.append(kata)


    teks = " ".join(
        hasil_stopword
    )


    # ==========================================================================
    # STEMMING
    # ==========================================================================

    teks = stemmer.stem(
        teks
    )


    # ==========================================================================
    # FINAL CLEANING
    # ==========================================================================

    teks = re.sub(
        r"\s+",
        " ",
        teks
    ).strip()


    return teks


# ======================================================================================
# 8. FUNGSI PREDIKSI
# ======================================================================================

def prediksi_sentimen(teks):

    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------

    teks_bersih = preprocessing(
        teks
    )


    # --------------------------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------------------------

    X = vectorizer.transform(
        [teks_bersih]
    )


    # --------------------------------------------------------------------------
    # PREDIKSI
    # --------------------------------------------------------------------------

    prediksi = model.predict(
        X
    )[0]


    # --------------------------------------------------------------------------
    # PROBABILITAS
    # --------------------------------------------------------------------------

    probabilitas = model.predict_proba(
        X
    )[0]


    # --------------------------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------------------------

    confidence = (
        np.max(
            probabilitas
        ) * 100
    )


    # --------------------------------------------------------------------------
    # LABEL SENTIMEN
    # --------------------------------------------------------------------------

    if prediksi == 1:

        sentimen = "Positif"

    else:

        sentimen = "Negatif"


    return (
        sentimen,
        confidence,
        teks_bersih
    )


# ======================================================================================
# 9. HEADER HALAMAN UTAMA
# ======================================================================================

st.title(
    "Sentimen Review Analyzer"
)

st.subheader(
    "Aplikasi Analisis Sentimen Ulasan GoPay"
)

st.write(
    "Masukkan ulasan untuk mengetahui hasil klasifikasi "
    "sentimen secara otomatis."
)

st.divider()


# ======================================================================================
# 10. ANALISIS SATU ULASAN
# ======================================================================================

st.header(
    "Analisis Ulasan"
)

ulasan = st.text_area(
    "Masukkan ulasan",
    placeholder=(
        "Contoh: Aplikasi lambat dan sulit digunakan untuk transaksi"
    ),
    height=150
)


if st.button(
    "Analisis Sentimen",
    type="primary",
    use_container_width=True
):

    if not ulasan.strip():

        st.warning(
            "Silakan masukkan ulasan terlebih dahulu."
        )

    else:

        (
            sentimen,
            confidence,
            teks_bersih
        ) = prediksi_sentimen(
            ulasan
        )


        st.divider()

        st.subheader(
            "Hasil Analisis"
        )


        # ------------------------------------------------------------------
        # HASIL SENTIMEN
        # ------------------------------------------------------------------

        if sentimen == "Positif":

            st.success(
                "Sentimen Positif"
            )

        else:

            st.error(
                "Sentimen Negatif"
            )


        # ------------------------------------------------------------------
        # HASIL PREDIKSI DAN CONFIDENCE
        # ------------------------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "Hasil Prediksi"
            )

            st.markdown(
                f"### {sentimen}"
            )


        with col2:

            st.write(
                "Confidence"
            )

            st.markdown(
                f"### {confidence:.2f}%"
            )


        # ------------------------------------------------------------------
        # HASIL PREPROCESSING
        # ------------------------------------------------------------------

        with st.expander(
            "Lihat Hasil Preprocessing"
        ):

            st.write(
                teks_bersih
            )


# ======================================================================================
# 11. ANALISIS BANYAK ULASAN
# ======================================================================================

st.divider()

st.header(
    "Analisis Banyak Ulasan"
)

st.write(
    "Upload file CSV untuk melakukan klasifikasi "
    "terhadap beberapa ulasan sekaligus."
)

st.info(
    "File CSV harus memiliki kolom bernama 'content'."
)


uploaded_file = st.file_uploader(
    "Pilih file CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            "File CSV tidak dapat dibaca."
        )

        st.error(
            f"Detail error: {e}"
        )

        st.stop()


    # ------------------------------------------------------------------
    # CEK KOLOM
    # ------------------------------------------------------------------

    if "content" not in df.columns:

        st.error(
            "Kolom 'content' tidak ditemukan pada file CSV."
        )

        st.write(
            "Kolom yang tersedia:"
        )

        st.write(
            list(df.columns)
        )

        st.stop()


    st.success(
        f"{len(df)} ulasan berhasil dimuat."
    )


    # ------------------------------------------------------------------
    # PREVIEW DATA
    # ------------------------------------------------------------------

    with st.expander(
        "Lihat Data"
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


    # ------------------------------------------------------------------
    # TOMBOL ANALISIS SEMUA
    # ------------------------------------------------------------------

    if st.button(
        "Analisis Semua Ulasan",
        type="primary",
        use_container_width=True
    ):

        hasil_preprocessing = []

        hasil_prediksi = []

        hasil_confidence = []


        # --------------------------------------------------------------
        # PROSES SETIAP ULASAN
        # --------------------------------------------------------------

        with st.spinner(
            "Sedang menganalisis ulasan..."
        ):

            for teks in df["content"].fillna(""):

                (
                    sentimen,
                    confidence,
                    teks_bersih
                ) = prediksi_sentimen(
                    teks
                )


                hasil_preprocessing.append(
                    teks_bersih
                )

                hasil_prediksi.append(
                    sentimen
                )

                hasil_confidence.append(
                    round(
                        confidence,
                        2
                    )
                )


        # --------------------------------------------------------------
        # TAMBAHKAN HASIL KE DATAFRAME
        # --------------------------------------------------------------

        df["Hasil Preprocessing"] = (
            hasil_preprocessing
        )

        df["Hasil Prediksi"] = (
            hasil_prediksi
        )

        df["Confidence (%)"] = (
            hasil_confidence
        )


        # --------------------------------------------------------------
        # TAMPILKAN HASIL
        # --------------------------------------------------------------

        st.subheader(
            "Hasil Analisis"
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )


        # --------------------------------------------------------------
        # RINGKASAN
        # --------------------------------------------------------------

        jumlah_positif = int(
            (
                df["Hasil Prediksi"]
                == "Positif"
            ).sum()
        )


        jumlah_negatif = int(
            (
                df["Hasil Prediksi"]
                == "Negatif"
            ).sum()
        )


        st.subheader(
            "Ringkasan Hasil"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Ulasan",
                len(df)
            )


        with col2:

            st.metric(
                "Positif",
                jumlah_positif
            )


        with col3:

            st.metric(
                "Negatif",
                jumlah_negatif
            )


        # --------------------------------------------------------------
        # DOWNLOAD HASIL
        # --------------------------------------------------------------

        csv_hasil = df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        )


        st.download_button(
            "Download Hasil Analisis",
            data=csv_hasil,
            file_name="hasil_prediksi_sentimen.csv",
            mime="text/csv",
            use_container_width=True
        )


# ======================================================================================
# 12. FOOTER
# ======================================================================================

st.divider()

st.caption(
    "Sentimen Review Analyzer | "
    "Analisis Sentimen Ulasan GoPay | "
    "TF-IDF + SMOTE + Logistic Regression | "
    "2026"
)
