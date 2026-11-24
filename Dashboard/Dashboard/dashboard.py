import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Beijing Air Quality Dashboard (2013–2017)",
    page_icon="🌫️",
    layout="wide",
)


@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "main_data.csv"),
        "dashboard/main_data.csv",
        "main_data.csv",
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break

    if df is None:
        st.error(
            "File 'main_data.csv' tidak ditemukan di dalam folder 'dashboard/'. "
            "Pastikan 'main_data.csv' berada satu folder dengan 'dashboard.py'."
        )
        st.stop()

    # 1. DECODE station & year FIRST before creating derived columns
    if pd.api.types.is_numeric_dtype(df["station"]):
        inv_station_map = {
            0: "Aotizhongxin",
            1: "Changping",
            2: "Dingling",
            3: "Dongsi",
            4: "Guanyuan",
            5: "Gucheng",
            6: "Huairou",
            7: "Nongzhanguan",
            8: "Shunyi",
            9: "Tiantan",
            10: "Wanliu",
            11: "Wanshouxigong",
        }
        df["station"] = df["station"].map(inv_station_map)

    if df["year"].max() < 100:
        df["year"] = df["year"] + 2000

    # 2. CREATE derived columns after station & year are decoded
    if "Musim" not in df.columns:
        df["Musim"] = np.where(
            df["month"].isin([12, 1, 2]),
            "Musim Salju (Des–Feb)",
            "Musim Lainnya (Mar–Nov)",
        )
    if "High_PM25" not in df.columns:
        df["High_PM25"] = (df["PM2.5"] > 75).astype(int)

    if "Kategori_PM25" not in df.columns:
        bins_pm25 = [0, 35, 75, 115, 150, float("inf")]
        labels_pm25 = [
            "Baik (0–35)",
            "Sedang (35–75)",
            "Tidak Sehat Sensitif (75–115)",
            "Tidak Sehat (115–150)",
            "Sangat Tidak Sehat (>150)",
        ]
        df["Kategori_PM25"] = pd.cut(
            df["PM2.5"], bins=bins_pm25, labels=labels_pm25, include_lowest=True
        )

    if "Zona_Wilayah" not in df.columns:
        zona_stasiun = {
            "Dongsi": "Perkotaan Padat",
            "Guanyuan": "Perkotaan Padat",
            "Wanshouxigong": "Perkotaan Padat",
            "Nongzhanguan": "Perkotaan Padat",
            "Tiantan": "Perkotaan Padat",
            "Wanliu": "Perkotaan Padat",
            "Aotizhongxin": "Perkotaan Padat",
            "Gucheng": "Suburban / Penyangga",
            "Shunyi": "Suburban / Penyangga",
            "Changping": "Suburban / Penyangga",
            "Huairou": "Pedesaan / Utara",
            "Dingling": "Pedesaan / Utara",
        }
        df["Zona_Wilayah"] = df["station"].map(zona_stasiun)

    return df


df = load_data()

# ==========================================
# SIDEBAR FILTER (HANYA STASIUN & TAHUN)
# ==========================================
st.sidebar.title("Filter Eksplorasi")
st.sidebar.markdown("Pilih **Stasiun** dan **Tahun** untuk memfilter seluruh grafik.")

station_options = ["Semua Stasiun (12 Stasiun)"] + sorted(df["station"].unique().tolist())
selected_station = st.sidebar.selectbox("Pilih Stasiun Pemantauan", station_options)

year_options = ["Semua Tahun (2013–2017)"] + sorted(df["year"].unique().tolist())
selected_year = st.sidebar.selectbox("Pilih Tahun Observasi", year_options)

filtered_df = df.copy()
if selected_station != "Semua Stasiun (12 Stasiun)":
    filtered_df = filtered_df[filtered_df["station"] == selected_station]
if selected_year != "Semua Tahun (2013–2017)":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Cakupan Data Aktif:**\n\n"
    f"• Stasiun: **{selected_station}**\n\n"
    f"• Tahun: **{selected_year}**\n\n"
    f"• Total Rekaman: **{len(filtered_df):,} jam**"
)

# ==========================================
# HEADER & METRIK UTAMA (KPI)
# ==========================================
st.title("🌫️ Dashboard Analisis Kualitas Udara Beijing (2013–2017)")
st.markdown(
    f"Menampilkan hasil analisis eksploratori untuk **{selected_station}** pada periode **{selected_year}**."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rata-rata PM2.5", f"{filtered_df['PM2.5'].mean():.1f} μg/m³")
with col2:
    pct_spike = filtered_df["High_PM25"].mean() * 100
    st.metric("Jam Buruk (PM2.5 > 75)", f"{pct_spike:.1f}%")
with col3:
    st.metric("Rata-rata SO2", f"{filtered_df['SO2'].mean():.1f} μg/m³")
with col4:
    st.metric("Rata-rata NO2", f"{filtered_df['NO2'].mean():.1f} μg/m³")

st.markdown("---")

# ==========================================
# TABS ANALISIS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "1. Pemicu Lonjakan PM2.5 (Salju vs. Non-Salju)",
    "2. Tren Tahunan Polutan & Cuaca",
    "3. Clustering & Peta Geospatial",
])

with tab1:
    st.subheader("Pemicu Lonjakan PM2.5 (> 75 μg/m³): Musim Salju vs. Musim Lainnya")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**A. Kekuatan Korelasi (Spearman) Terhadap Lonjakan PM2.5 > 75**")
        features = ["PM10", "CO", "NO2", "SO2", "O3", "DEWP", "TEMP", "PRES", "RAIN", "WSPM"]
        corr_dict = {}
        for m in ["Musim Salju (Des–Feb)", "Musim Lainnya (Mar–Nov)"]:
            sub_m = filtered_df[filtered_df["Musim"] == m]
            if len(sub_m) > 10 and sub_m["High_PM25"].nunique() > 1:
                corr_dict[m] = sub_m[features + ["High_PM25"]].corr(method="spearman")["High_PM25"].drop("High_PM25")
            else:
                corr_dict[m] = pd.Series(np.nan, index=features)

        hm_df = pd.DataFrame(corr_dict)
        fig1, ax1 = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(hm_df, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=-0.7, vmax=0.85, linewidths=0.5, ax=ax1)
        ax1.set_title("Korelasi Spearman dengan Lonjakan PM2.5 (>75 μg/m³)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with c2:
        st.markdown("**B. Efek Dispersi Kecepatan Angin (WSPM) Terhadap Lonjakan PM2.5**")
        temp_wspm = filtered_df.copy()
        temp_wspm["WSPM_bin"] = pd.cut(
            temp_wspm["WSPM"],
            bins=[0, 1, 2, 3, 4, 6, 15],
            labels=["0–1", "1–2", "2–3", "3–4", "4–6", ">6"],
            include_lowest=True,
        )
        wspm_curve = temp_wspm.groupby(["WSPM_bin", "Musim"], observed=False)["High_PM25"].mean().unstack() * 100

        fig2, ax2 = plt.subplots(figsize=(6.5, 5))
        if "Musim Salju (Des–Feb)" in wspm_curve.columns:
            ax2.plot(
                wspm_curve.index.astype(str),
                wspm_curve["Musim Salju (Des–Feb)"],
                marker="o",
                linewidth=2.3,
                color="#d95f02",
                label="Musim Salju (Des–Feb)",
            )
        if "Musim Lainnya (Mar–Nov)" in wspm_curve.columns:
            ax2.plot(
                wspm_curve.index.astype(str),
                wspm_curve["Musim Lainnya (Mar–Nov)"],
                marker="s",
                linewidth=2.3,
                color="#1b9e77",
                label="Musim Lainnya (Mar–Nov)",
            )
        ax2.set_xlabel("Kecepatan Angin / WSPM (m/s)")
        ax2.set_ylabel("Persentase Jam PM2.5 > 75 μg/m³ (%)")
        ax2.set_title("Penurunan Risiko Lonjakan Saat Angin Menguat", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("**C. Pergeseran Titik Embun / Kelembapan (DEWP) Saat Normal vs. Lonjakan**")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    plot_dewp = filtered_df.copy()
    plot_dewp["Status"] = np.where(plot_dewp["High_PM25"] == 1, "Lonjakan (>75 μg/m³)", "Normal (≤75 μg/m³)")
    sns.boxplot(
        data=plot_dewp,
        x="Musim",
        y="DEWP",
        hue="Status",
        palette={"Normal (≤75 μg/m³)": "#1b9e77", "Lonjakan (>75 μg/m³)": "#d95f02"},
        showfliers=False,
        ax=ax3,
    )
    ax3.set_xlabel("")
    ax3.set_ylabel("Dew Point / DEWP (°C)")
    ax3.set_title("Distribusi Dew Point (DEWP) Pada Kondisi Udara Normal vs. Lonjakan", fontsize=10, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with tab2:
    st.subheader("Perbandingan Tren Tahunan Polutan & Kondisi Meteorologi")
    st.caption(
        "Catatan: Jika Anda memilih satu tahun tertentu di sidebar, grafik tren di bawah otomatis menampilkan "
        "seluruh rentang 2013–2017 untuk stasiun terpilih agar perbandingan antar-tahun tetap terlihat."
    )

    trend_base = df if selected_station == "Semua Stasiun (12 Stasiun)" else df[df["station"] == selected_station]
    pollutants = ["SO2", "NO2", "CO", "O3", "PM2.5", "PM10"]
    weather = ["WSPM", "DEWP", "TEMP", "RAIN"]

    st.markdown("**A. Rata-rata Konsentrasi 6 Polutan per Tahun (2013–2017)**")
    fig4, axes4 = plt.subplots(2, 3, figsize=(14, 7))
    axes4 = axes4.flatten()
    yearly_pol = trend_base.groupby("year")[pollutants].mean()

    for i, pol in enumerate(pollutants):
        axes4[i].plot(yearly_pol.index, yearly_pol[pol], marker="o", linewidth=2.4, color="#d95f02")
        axes4[i].set_title(f"Tren Tahunan {pol}", fontsize=10, fontweight="bold")
        axes4[i].set_xticks([2013, 2014, 2015, 2016, 2017])
        axes4[i].set_ylabel("Konsentrasi")
        axes4[i].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("**B. Rata-rata Faktor Meteorologi per Tahun (Pembanding Kondisi Cuaca)**")
    fig5, axes5 = plt.subplots(1, 4, figsize=(14, 3.6))
    yearly_wx = trend_base.groupby("year")[weather].mean()

    for i, wx in enumerate(weather):
        axes5[i].plot(yearly_wx.index, yearly_wx[wx], marker="s", linewidth=2.3, color="#1b9e77")
        axes5[i].set_title(f"Rata-rata {wx}", fontsize=10, fontweight="bold")
        axes5[i].set_xticks([2013, 2014, 2015, 2016, 2017])
        axes5[i].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

with tab3:
    st.subheader("Analisis Lanjutan: Binning Kualitas Udara, Manual Grouping & Peta Geospatial")
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("**A. Distribusi Kategori PM2.5 per Zona Wilayah & Musim**")
        ct_cluster = pd.crosstab(
            [filtered_df["Zona_Wilayah"], filtered_df["Musim"]],
            filtered_df["Kategori_PM25"],
            normalize="index",
        ) * 100

        fig6, ax6 = plt.subplots(figsize=(7.5, 5.2))
        ct_cluster.plot(
            kind="bar",
            stacked=True,
            color=["#1b9e77", "#a6d96a", "#fdae61", "#f46d43", "#d7191c"],
            edgecolor="white",
            ax=ax6,
        )
        ax6.set_xlabel("")
        ax6.set_ylabel("Persentase Jam (%)")
        ax6.set_title("Proporsi Kategori Kualitas Udara (Binning & Manual Grouping)", fontsize=10, fontweight="bold")
        ax6.tick_params(axis="x", rotation=25)
        ax6.legend(title="Kategori PM2.5", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    with col_right:
        st.markdown("**B. Peta Sebaran Spasial Stasiun Pemantauan Beijing**")
        koordinat_stasiun = {
            "Aotizhongxin": (39.982, 116.397),
            "Changping": (40.217, 116.230),
            "Dingling": (40.292, 116.220),
            "Dongsi": (39.929, 116.417),
            "Guanyuan": (39.929, 116.339),
            "Gucheng": (39.914, 116.184),
            "Huairou": (40.328, 116.628),
            "Nongzhanguan": (39.937, 116.461),
            "Shunyi": (40.127, 116.655),
            "Tiantan": (39.886, 116.407),
            "Wanliu": (39.987, 116.287),
            "Wanshouxigong": (39.878, 116.352),
        }

        map_source = df if selected_year == "Semua Tahun (2013–2017)" else df[df["year"] == selected_year]
        geo_summary = (
            map_source.groupby(["station", "Zona_Wilayah"])
            .agg(
                mean_pm25=("PM2.5", "mean"),
                pct_buruk=("High_PM25", lambda x: x.mean() * 100),
                mean_no2=("NO2", "mean"),
                mean_so2=("SO2", "mean"),
            )
            .reset_index()
        )

        peta = folium.Map(
            location=[40.05, 116.40],
            zoom_start=10,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
        )

        for _, row in geo_summary.iterrows():
            st_name = row["station"]
            lat, lon = koordinat_stasiun[st_name]
            is_selected = (selected_station == "Semua Stasiun (12 Stasiun)") or (st_name == selected_station)

            if row["mean_pm25"] >= 83:
                warna = "#d7191c"
            elif row["mean_pm25"] >= 78:
                warna = "#fdae61"
            else:
                warna = "#1b9e77"

            popup_html = f"""
            <b>Stasiun: {st_name}</b><br>
            Zona: {row['Zona_Wilayah']}<br>
            Rata-rata PM2.5: <b>{row['mean_pm25']:.1f} μg/m³</b><br>
            Jam Buruk (>75): <b>{row['pct_buruk']:.1f}%</b><br>
            Rata-rata NO2: {row['mean_no2']:.1f} μg/m³<br>
            Rata-rata SO2: {row['mean_so2']:.1f} μg/m³
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=(row["mean_pm25"] / 5.2) if is_selected else 6,
                color=warna,
                fill=True,
                fill_color=warna,
                fill_opacity=0.85 if is_selected else 0.25,
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"{st_name}: {row['mean_pm25']:.1f} μg/m³",
            ).add_to(peta)

        st_folium(peta, width=600, height=430, returned_objects=[])