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
    page_title="Sentimen Review Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================================
# 2. SESSION STATE
# ======================================================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

if "halaman" not in st.session_state:
    st.session_state.halaman = "Login"


# ======================================================================================
# 3. DATA USER
# ======================================================================================

if "users" not in st.session_state:

    st.session_state.users = {

        "admin": {
            "password": "admin123",
            "role": "Administrator"
        },

        "user": {
            "password": "user123",
            "role": "User"
        }

    }


# ======================================================================================
# 4. CSS / TAMPILAN
# ======================================================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap'
    );


    /* ==========================================================================
       FONT UTAMA
       ========================================================================== */

    html,
    body,
    [class*="css"],
    .stApp,
    button,
    input,
    textarea,
    select {

        font-family: 'Poppins', sans-serif !important;

    }


    /* ==========================================================================
       BACKGROUND
       ========================================================================== */

    .stApp {

        background-color: #f7f8fc;

    }


    /* ==========================================================================
       SIDEBAR
       ========================================================================== */

    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #111827 0%,
            #1f2937 100%
        );

    }


    section[data-testid="stSidebar"] * {

        color: white !important;

    }


    section[data-testid="stSidebar"] .stButton button {

        background-color: transparent;

        color: white !important;

        border: 1px solid rgba(255,255,255,0.16);

        border-radius: 10px;

        font-weight: 500;

        min-height: 42px;

    }


    section[data-testid="stSidebar"] .stButton button:hover {

        background-color: rgba(255,255,255,0.10);

        border-color: rgba(255,255,255,0.30);

    }


    /* ==========================================================================
       BUTTON
       ========================================================================== */

    .stButton button {

        border-radius: 10px;

        font-family: 'Poppins', sans-serif !important;

        font-weight: 600;

    }


    /* ==========================================================================
       INPUT
       ========================================================================== */

    .stTextInput input,
    .stTextArea textarea {

        border-radius: 10px;

        font-family: 'Poppins', sans-serif !important;

    }


    /* ==========================================================================
       CONTAINER / CARD
       ========================================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        border-radius: 16px;

        border: 1px solid #e1e5eb;

        background-color: white;

    }


    /* ==========================================================================
       CAPTION UMUM
       ========================================================================== */

    .stCaption {

        font-family: 'Poppins', sans-serif !important;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================================================
# 5. HALAMAN LOGIN + SELAMAT DATANG
# ======================================================================================

def halaman_login():

    st.write("")
    st.write("")
    st.write("")


    kolom_kiri, kolom_tengah, kolom_kanan = st.columns(
        [1, 2, 1]
    )


    with kolom_tengah:

        # ----------------------------------------------------------------------
        # SELAMAT DATANG
        # ----------------------------------------------------------------------

        st.title(
            "Selamat Datang di Aplikasi"
        )


        st.subheader(
            "Sentimen Review Analyzer"
        )


        st.write(
            "Silakan login untuk mengakses aplikasi "
            "analisis sentimen GoPay."
        )


        st.write("")


        # ----------------------------------------------------------------------
        # USERNAME
        # ----------------------------------------------------------------------

        username = st.text_input(
            "Username",
            placeholder="Masukkan username",
            key="login_username"
        )


        # ----------------------------------------------------------------------
        # PASSWORD
        # ----------------------------------------------------------------------

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Masukkan password",
            key="login_password"
        )


        st.write("")


        # ----------------------------------------------------------------------
        # LOGIN
        # ----------------------------------------------------------------------

        if st.button(
            "Login",
            type="primary",
            use_container_width=True,
            key="login_button"
        ):

            if (
                username in st.session_state.users
                and
                st.session_state.users[username]["password"] == password
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                st.session_state.role = (
                    st.session_state.users[username]["role"]
                )

                st.session_state.halaman = "Dashboard"

                st.rerun()

            else:

                st.error(
                    "Username atau password salah."
                )


        st.write("")


        st.caption(
            "Analisis Sentimen Ulasan GoPay • 2026"
        )


# ======================================================================================
# 6. CEK LOGIN
# ======================================================================================

if not st.session_state.logged_in:

    halaman_login()

    st.stop()


# ======================================================================================
# 7. SIDEBAR
# ======================================================================================

with st.sidebar:

    st.title(
        "Sentimen Review"
    )


    st.caption(
        "Analyzer GoPay"
    )


    st.write("")


    # --------------------------------------------------------------------------
    # USER
    # --------------------------------------------------------------------------

    st.write(
        f"**{st.session_state.username}**"
    )


    st.caption(
        st.session_state.role
    )


    st.divider()


    # --------------------------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------------------------

    if st.button(
        "Dashboard",
        use_container_width=True,
        key="sidebar_dashboard"
    ):

        st.session_state.halaman = "Dashboard"

        st.rerun()


    # --------------------------------------------------------------------------
    # ANALISIS ULASAN
    # --------------------------------------------------------------------------

    if st.button(
        "Analisis Ulasan",
        use_container_width=True,
        key="sidebar_analisis"
    ):

        st.session_state.halaman = "Analisis Ulasan"

        st.rerun()


    # --------------------------------------------------------------------------
    # MANAGE USER
    # KHUSUS ADMINISTRATOR
    # --------------------------------------------------------------------------

    if st.session_state.role == "Administrator":

        if st.button(
            "Manage User",
            use_container_width=True,
            key="sidebar_manage_user"
        ):

            st.session_state.halaman = "Manage User"

            st.rerun()


    # --------------------------------------------------------------------------
    # TENTANG APLIKASI
    # --------------------------------------------------------------------------

    if st.button(
        "Tentang Aplikasi",
        use_container_width=True,
        key="sidebar_tentang"
    ):

        st.session_state.halaman = "Tentang Aplikasi"

        st.rerun()


    st.divider()


    # --------------------------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------------------------

    if st.button(
        "Logout",
        use_container_width=True,
        key="sidebar_logout"
    ):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.role = ""

        st.session_state.halaman = "Login"

        st.rerun()


# ======================================================================================
# 8. LOAD MODEL DAN VECTORIZER
# ======================================================================================

@st.cache_resource
def load_model():

    with open(
        "model.pkl",
        "rb"
    ) as file:

        model = pickle.load(file)


    with open(
        "vectorizer.pkl",
        "rb"
    ) as file:

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
# 9. LOAD STEMMER SASTRAWI
# ======================================================================================

@st.cache_resource
def load_stemmer():

    factory = StemmerFactory()

    return factory.create_stemmer()


stemmer = load_stemmer()


# ======================================================================================
# 10. PREPROCESSING
# ======================================================================================

def preprocessing(teks):

    teks = str(teks)


    # ==========================================================================
    # CLEANING
    # ==========================================================================

    teks = re.sub(
        r"http\S+|www\S+",
        " ",
        teks
    )


    teks = re.sub(
        r"@\w+",
        " ",
        teks
    )


    teks = re.sub(
        r"#(\w+)",
        r"\1",
        teks
    )


    teks = re.sub(
        r"<.*?>",
        " ",
        teks
    )


    teks = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        teks
    )


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
    # STOPWORD
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

            hasil_stopword.append(
                kata
            )


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
# 11. FUNGSI PREDIKSI
# ======================================================================================

def prediksi_sentimen(teks):

    teks_bersih = preprocessing(
        teks
    )


    X = vectorizer.transform(
        [teks_bersih]
    )


    prediksi = model.predict(
        X
    )[0]


    probabilitas = model.predict_proba(
        X
    )[0]


    confidence = (
        np.max(
            probabilitas
        ) * 100
    )


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
# 12. DASHBOARD
# ======================================================================================

def dashboard():

    # --------------------------------------------------------------------------
    # JUDUL
    # --------------------------------------------------------------------------

    st.title(
        "Dashboard"
    )


    st.write(
        "Selamat datang di aplikasi Sentimen Review Analyzer."
    )


    st.write("")


    # ==========================================================================
    # KARTU INFORMASI
    # ==========================================================================

    # Semua kartu menggunakan lebar proporsional.
    # Kolom Oversampling dibuat cukup lebar agar label tidak terpotong.
    # Semua kartu memiliki tinggi yang sama.

    col1, col2, col3, col4 = st.columns(
        [1.05, 1.40, 1.45, 0.95],
        gap="medium"
    )


    # --------------------------------------------------------------------------
    # METODE
    # --------------------------------------------------------------------------

    with col1:

        with st.container(
            border=True,
            height=145
        ):

            st.markdown(
                """
                <div style="
                    font-size:13px;
                    color:#718096;
                    white-space:nowrap;
                    margin-bottom:18px;
                ">
                    Metode
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    font-size:21px;
                    font-weight:600;
                    line-height:1.25;
                    white-space:nowrap;
                    color:#182033;
                ">
                    TF-IDF
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------------------------
    # OVERSAMPLING
    # --------------------------------------------------------------------------

    with col2:

        with st.container(
            border=True,
            height=145
        ):

            st.markdown(
                """
                <div style="
                    font-size:13px;
                    color:#718096;
                    white-space:nowrap;
                    margin-bottom:18px;
                ">
                    Oversampling
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    font-size:21px;
                    font-weight:600;
                    line-height:1.25;
                    white-space:nowrap;
                    color:#182033;
                ">
                    SMOTE
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------------------------

    with col3:

        with st.container(
            border=True,
            height=145
        ):

            st.markdown(
                """
                <div style="
                    font-size:13px;
                    color:#718096;
                    white-space:nowrap;
                    margin-bottom:12px;
                ">
                    Classifier
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    font-size:20px;
                    font-weight:600;
                    line-height:1.35;
                    color:#182033;
                ">
                    Logistic<br>
                    Regression
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------------------------
    # TAHUN
    # --------------------------------------------------------------------------

    with col4:

        with st.container(
            border=True,
            height=145
        ):

            st.markdown(
                """
                <div style="
                    font-size:13px;
                    color:#718096;
                    white-space:nowrap;
                    margin-bottom:18px;
                ">
                    Tahun
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                """
                <div style="
                    font-size:21px;
                    font-weight:600;
                    line-height:1.25;
                    white-space:nowrap;
                    color:#182033;
                ">
                    2026
                </div>
                """,
                unsafe_allow_html=True
            )


    st.write("")


    # ==========================================================================
    # FITUR APLIKASI
    # ==========================================================================

    st.subheader(
        "Fitur Aplikasi"
    )


    st.write("")


    # --------------------------------------------------------------------------
    # ANALISIS ULASAN
    # --------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            "Analisis Ulasan"
        )


        st.write(
            "Masukkan satu ulasan GoPay untuk mendapatkan "
            "hasil klasifikasi sentimen dan nilai confidence."
        )


        st.write("")


        if st.button(
            "Buka Analisis Ulasan",
            use_container_width=True,
            key="dashboard_analisis_satu"
        ):

            st.session_state.halaman = "Analisis Ulasan"

            st.rerun()


# ======================================================================================
# 13. ANALISIS SATU ULASAN
# ======================================================================================

def analisis_satu_ulasan():

    st.title(
        "Analisis Ulasan"
    )


    st.write(
        "Masukkan ulasan pengguna GoPay untuk mengetahui "
        "hasil klasifikasi sentimen."
    )


    st.divider()


    ulasan = st.text_area(

        "Masukkan ulasan",

        placeholder=(
            "Contoh: Aplikasi GoPay sangat mudah digunakan "
            "dan transaksi berjalan cepat."
        ),

        height=150,

        key="input_ulasan"

    )


    st.write("")


    if st.button(
        "Analisis Sentimen",
        type="primary",
        use_container_width=True,
        key="button_analisis"
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


            st.write("")


            if sentimen == "Positif":

                st.success(
                    "Sentimen Positif"
                )

            else:

                st.error(
                    "Sentimen Negatif"
                )


            st.write("")


            col1, col2 = st.columns(2)


            with col1:

                st.caption(
                    "Hasil Prediksi"
                )

                st.subheader(
                    sentimen
                )


            with col2:

                st.caption(
                    "Confidence"
                )

                st.subheader(
                    f"{confidence:.2f}%"
                )


            st.write("")


            with st.expander(
                "Lihat Hasil Preprocessing"
            ):

                st.write(
                    teks_bersih
                )


# ======================================================================================
# 14. MANAGE USER
# ======================================================================================

def manage_user():

    st.title(
        "Manage User"
    )


    st.write(
        "Kelola pengguna yang dapat mengakses aplikasi."
    )


    st.divider()


    # ==========================================================================
    # DAFTAR USER
    # ==========================================================================

    st.subheader(
        "Daftar User"
    )


    data_user = []


    for username, data in st.session_state.users.items():

        data_user.append(
            {
                "Username": username,
                "Role": data["role"]
            }
        )


    df_user = pd.DataFrame(
        data_user
    )


    st.dataframe(
        df_user,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # ==========================================================================
    # TAMBAH USER
    # ==========================================================================

    st.subheader(
        "Tambah User"
    )


    col1, col2 = st.columns(2)


    with col1:

        username_baru = st.text_input(
            "Username",
            key="username_baru"
        )


    with col2:

        password_baru = st.text_input(
            "Password",
            type="password",
            key="password_baru"
        )


    role_baru = st.selectbox(
        "Role",
        [
            "User",
            "Administrator"
        ],
        key="role_baru"
    )


    st.write("")


    if st.button(
        "Tambah User",
        type="primary",
        use_container_width=True,
        key="button_tambah_user"
    ):

        if not username_baru.strip():

            st.warning(
                "Username harus diisi."
            )

        elif not password_baru.strip():

            st.warning(
                "Password harus diisi."
            )

        elif username_baru in st.session_state.users:

            st.error(
                "Username sudah digunakan."
            )

        else:

            st.session_state.users[
                username_baru
            ] = {

                "password": password_baru,

                "role": role_baru

            }


            st.success(
                f"User '{username_baru}' berhasil ditambahkan."
            )


            st.rerun()


    st.divider()


    # ==========================================================================
    # HAPUS USER
    # ==========================================================================

    st.subheader(
        "Hapus User"
    )


    daftar_user = [

        username

        for username in st.session_state.users

        if username != st.session_state.username

    ]


    if len(daftar_user) > 0:

        user_hapus = st.selectbox(
            "Pilih user yang akan dihapus",
            daftar_user,
            key="user_hapus"
        )


        st.write("")


        if st.button(
            "Hapus User",
            use_container_width=True,
            key="button_hapus_user"
        ):

            del st.session_state.users[
                user_hapus
            ]


            st.success(
                f"User '{user_hapus}' berhasil dihapus."
            )


            st.rerun()

    else:

        st.info(
            "Tidak ada user lain yang dapat dihapus."
        )


# ======================================================================================
# 15. TENTANG APLIKASI
# ======================================================================================

def tentang_aplikasi():

    st.title(
        "Tentang Aplikasi"
    )


    st.write(
        "Informasi mengenai aplikasi Sentimen Review Analyzer."
    )


    st.divider()


    with st.container(
        border=True
    ):

        st.subheader(
            "Sentimen Review Analyzer"
        )


        st.write(
            "Aplikasi ini digunakan untuk menganalisis "
            "sentimen ulasan pengguna GoPay menjadi "
            "sentimen positif atau negatif dan confidence."
        )


        st.write("")


        st.write(
            "Sistem menggunakan preprocessing teks, "
            "TF-IDF, SMOTE, dan Logistic Regression "
            "untuk melakukan klasifikasi sentimen."
        )


        st.write("")


        st.write(
            "Pengguna dapat memasukkan satu ulasan "
            "untuk memperoleh hasil prediksi sentimen "
            "beserta nilai confidence."
        )


# ======================================================================================
# 16. ROUTING
# ======================================================================================

if st.session_state.halaman == "Dashboard":

    dashboard()


elif st.session_state.halaman == "Analisis Ulasan":

    analisis_satu_ulasan()


elif st.session_state.halaman == "Manage User":

    if st.session_state.role == "Administrator":

        manage_user()

    else:

        st.error(
            "Anda tidak memiliki akses ke halaman Manage User."
        )


elif st.session_state.halaman == "Tentang Aplikasi":

    tentang_aplikasi()


# ======================================================================================
# 17. FOOTER
# ======================================================================================

st.divider()


st.caption(
    "Sentimen Review Analyzer | "
    "Analisis Sentimen Ulasan GoPay | "
    "TF-IDF + SMOTE + Logistic Regression | "
    "2026"
)
