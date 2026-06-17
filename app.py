import streamlit as st
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import matplotlib.pyplot as plt


# =========================================================
# H-RESP: Heave Response Program in Random Waves
# Dibuat oleh: Muhammad Iqbal, S.T., M.T., Ph.D
# Teknik Perkapalan UNDIP
# Tanggal: 3 Juni 2025
# =========================================================


st.set_page_config(
    page_title="H-RESP",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 H-RESP")
st.subheader("Heave Response Program in Random Waves")

st.markdown("""
**Dibuat oleh:** Muhammad Iqbal, S.T., M.T., Ph.D  
**Departemen:** Teknik Perkapalan UNDIP  
**Tanggal Aplikasi Web Dibuat:** 18 Juni 2026
""")

st.divider()


def read_rao_file(uploaded_file, separator_option):
    """
    Membaca file RAO dari CSV/TXT.
    Kolom pertama: wave ratio
    Kolom kedua: Heave RAO
    """

    if separator_option == "Tab":
        sep = "\t"
    elif separator_option == "Titik koma (;)" :
        sep = ";"
    elif separator_option == "Koma (,)":
        sep = ","
    elif separator_option == "Spasi":
        sep = r"\s+"
    else:
        sep = None

    df = pd.read_csv(uploaded_file, sep=sep, engine="python")
    return df


def calculate_hresp(v, L, Hs, Tp, gama, heading_deg, df):
    """
    Fungsi utama perhitungan H-RESP.
    """

    heading = np.deg2rad(heading_deg)

    wave_ratio = df.iloc[:, 0].to_numpy(dtype=float)
    heave_rao = df.iloc[:, 1].to_numpy(dtype=float)

    # Menghapus data NaN jika ada
    valid = np.isfinite(wave_ratio) & np.isfinite(heave_rao)
    wave_ratio = wave_ratio[valid]
    heave_rao = heave_rao[valid]

    if len(wave_ratio) < 4:
        raise ValueError("Data RAO terlalu sedikit. Minimal diperlukan sekitar 4 titik data.")

    # ==== PROSES GEOMETRI DAN GELOMBANG ====
    wave_length = wave_ratio * L
    k = 2 * np.pi / wave_length
    w = np.sqrt(k * 9.81)

    w_e = w * (1 - (w * v * np.cos(heading)) / 9.81)

    # Hindari encounter frequency tidak valid
    valid_we = np.isfinite(w_e) & (w_e > 0)
    w = w[valid_we]
    w_e = w_e[valid_we]
    heave_rao = heave_rao[valid_we]

    if len(w_e) < 4:
        raise ValueError(
            "Data encounter frequency tidak cukup atau bernilai tidak valid. "
            "Coba cek heading, kecepatan kapal, atau data RAO."
        )

    # CubicSpline butuh x berurutan naik
    sort_idx = np.argsort(w_e)
    w_e = w_e[sort_idx]
    w = w[sort_idx]
    heave_rao = heave_rao[sort_idx]

    no_points = len(heave_rao)*2
    xx = np.linspace(np.min(w_e), np.max(w_e), no_points)

    # ==== INTERPOLASI RAO ====
    rao_spline = interp.CubicSpline(w_e, heave_rao)
    RAO = rao_spline(xx)

    # ==== SPEKTRUM GELOMBANG JONSWAP ====
    ww = np.linspace(np.min(w), np.max(w), no_points)
    we = ww * (1 - (ww * v * np.cos(heading)) / 9.81)

    fp = 2 * np.pi / Tp

    fac1 = (320 * Hs**2) / Tp**4
    sigma = np.where(ww <= fp, 0.07, 0.09)
    Aa = np.exp(-((ww / fp - 1) / (sigma * np.sqrt(2)))**2)
    fac2 = ww**-5
    fac31 = np.exp(-5 / 4 * (ww / fp)**-4)
    fac4 = gama**Aa

    S = fac1 * fac2 * fac31 * fac4

    denominator = 1 - (2 * ww * v / 9.81) * np.cos(heading)

    # Hindari pembagian dengan nol atau nilai sangat kecil
    valid_spec = np.isfinite(we) & np.isfinite(S) & np.isfinite(denominator) & (np.abs(denominator) > 1e-8)

    we = we[valid_spec]
    S_we = S[valid_spec] / denominator[valid_spec]

    if len(we) < 4:
        raise ValueError(
            "Data spektrum encounter tidak cukup untuk interpolasi. "
            "Coba cek parameter input."
        )

    # Sort untuk CubicSpline
    sort_spec = np.argsort(we)
    we = we[sort_spec]
    S_we = S_we[sort_spec]

    # Jika ada duplikasi nilai we, ambil nilai unik
    we_unique, unique_idx = np.unique(we, return_index=True)
    S_we_unique = S_we[unique_idx]

    if len(we_unique) < 4:
        raise ValueError(
            "Nilai encounter frequency terlalu banyak yang sama/duplikat. "
            "Interpolasi tidak dapat dilakukan."
        )

    s_we_spline = interp.CubicSpline(we_unique, S_we_unique)
    S_we_inter = s_we_spline(xx)

    # Nilai spektrum negatif akibat interpolasi kecil bisa dipotong ke nol
    S_we_inter = np.maximum(S_we_inter, 0)

    # ==== SPEKTRUM RESPON ====
    S_res = RAO**2 * S_we_inter
    velocity = xx**2 * RAO**2 * S_we_inter
    acceleration = xx**4 * S_res

    # ==== MOMEN DAN RMS ====
    m_0 = np.trapezoid(S_res, xx)
    m_2 = np.trapezoid(velocity, xx)
    m_4 = np.trapezoid(acceleration, xx)

    RMS_m0 = np.sqrt(m_0)
    RMS_m2 = np.sqrt(m_2)
    RMS_m4 = np.sqrt(m_4)

    results = {
        "m0": m_0,
        "m2": m_2,
        "m4": m_4,
        "RMS Heave": RMS_m0,
        "RMS Velocity": RMS_m2,
        "RMS Acceleration": RMS_m4,
    }

    plot_data = pd.DataFrame({
        "Encounter Frequency (rad/s)": xx,
        "RAO": RAO,
        "Wave Spectrum Encounter": S_we_inter,
        "Response Spectrum": S_res,
        "Velocity Spectrum": velocity,
        "Acceleration Spectrum": acceleration,
    })

    return results, plot_data


# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.header("Input Parameters")

v = st.sidebar.number_input(
    "Kecepatan kapal, v (m/s)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

L = st.sidebar.number_input(
    "Panjang kapal, L (m)",
    min_value=0.1,
    value=100.0,
    step=1.0
)

Hs = st.sidebar.number_input(
    "Tinggi gelombang signifikan, Hs (m)",
    min_value=0.0,
    value=2.0,
    step=0.1
)

Tp = st.sidebar.number_input(
    "Peak Period, Tp (s)",
    min_value=0.1,
    value=8.0,
    step=0.1
)

gama = st.sidebar.number_input(
    "Gamma JONSWAP",
    min_value=0.1,
    value=3.3,
    step=0.1
)

heading_deg = st.sidebar.number_input(
    "Heading / arah datang gelombang (derajat)",
    min_value=0.0,
    max_value=360.0,
    value=180.0,
    step=1.0
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload file RAO CSV/TXT",
    type=["csv", "txt"]
)

separator_option = st.sidebar.selectbox(
    "Pilih pemisah data",
    ["Tab", "Titik koma (;)", "Koma (,)", "Spasi"]
)

calculate_button = st.sidebar.button("Hitung H-RESP")


# =========================================================
# MAIN PAGE
# =========================================================

st.header("Panduan Format File RAO")

st.write("""
File RAO harus berisi minimal dua kolom:

1. **Kolom 1:** wave ratio  
2. **Kolom 2:** Heave RAO
""")

st.code(
"""wave_ratio    Heave_RAO
0.5           0.12
0.6           0.18
0.7           0.31
0.8           0.45""",
    language="text"
)

if uploaded_file is not None:
    try:
        df_preview = read_rao_file(uploaded_file, separator_option)

        st.subheader("Preview Data RAO")
        st.dataframe(df_preview.head())

        # Reset pointer file agar bisa dibaca ulang saat hitung
        uploaded_file.seek(0)

    except Exception as e:
        st.error(f"File tidak dapat dibaca. Detail error: {e}")


if calculate_button:
    if uploaded_file is None:
        st.warning("Silakan upload file RAO terlebih dahulu.")
    else:
        try:
            uploaded_file.seek(0)
            df = read_rao_file(uploaded_file, separator_option)

            results, plot_data = calculate_hresp(
                v=v,
                L=L,
                Hs=Hs,
                Tp=Tp,
                gama=gama,
                heading_deg=heading_deg,
                df=df
            )

            st.success("Perhitungan berhasil dilakukan.")

            st.header("Hasil Perhitungan")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("m0 - Energi Heave", f"{results['m0']:.4f}")
                st.metric("RMS Heave (m)", f"{results['RMS Heave']:.4f}")

            with col2:
                st.metric("m2 - Energi Kecepatan", f"{results['m2']:.4f}")
                st.metric("RMS Velocity (m/s)", f"{results['RMS Velocity']:.4f}")

            with col3:
                st.metric("m4 - Energi Percepatan", f"{results['m4']:.4f}")
                st.metric("RMS Acceleration (m/s²)", f"{results['RMS Acceleration']:.4f}")

            st.divider()

            st.header("Grafik Hasil")

            fig1, ax1 = plt.subplots()
            ax1.plot(
                plot_data["Encounter Frequency (rad/s)"],
                plot_data["RAO"]
            )
            ax1.set_xlabel("Encounter Frequency (rad/s)")
            ax1.set_ylabel("RAO")
            ax1.set_title("Interpolated Heave RAO")
            ax1.grid(True)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(
                plot_data["Encounter Frequency (rad/s)"],
                plot_data["Wave Spectrum Encounter"],
                label="Wave Spectrum Encounter"
            )
            ax2.plot(
                plot_data["Encounter Frequency (rad/s)"],
                plot_data["Response Spectrum"],
                label="Response Spectrum"
            )
            ax2.set_xlabel("Encounter Frequency (rad/s)")
            ax2.set_ylabel("Spectral Density")
            ax2.set_title("Wave Spectrum and Response Spectrum")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots()
            ax3.plot(
                plot_data["Encounter Frequency (rad/s)"],
                plot_data["Velocity Spectrum"],
                label="Velocity Spectrum"
            )
            ax3.plot(
                plot_data["Encounter Frequency (rad/s)"],
                plot_data["Acceleration Spectrum"],
                label="Acceleration Spectrum"
            )
            ax3.set_xlabel("Encounter Frequency (rad/s)")
            ax3.set_ylabel("Spectral Density")
            ax3.set_title("Velocity and Acceleration Spectrum")
            ax3.legend()
            ax3.grid(True)
            st.pyplot(fig3)

            st.header("Tabel Data Hasil")
            st.dataframe(plot_data)

            csv_output = plot_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Data Hasil sebagai CSV",
                data=csv_output,
                file_name="HRESP_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Terjadi error saat perhitungan.")
            st.exception(e)
