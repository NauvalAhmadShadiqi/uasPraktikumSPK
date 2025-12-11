import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.title("Sistem Pendukung Keputusan – Clustering Lagu Spotify")
st.write("Metode: K-Means Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset Spotify (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Preview Dataset")
    st.dataframe(df.head())

    # Pilih fitur
    st.subheader("🔧 Pilih Fitur Numerik untuk Clustering")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    selected_features = st.multiselect("Pilih fitur:", numeric_cols, 
                                       default=["danceability", "energy", "tempo", "valence"])

    if len(selected_features) >= 2:
        X = df[selected_features]

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Tentukan jumlah cluster
        n_clusters = st.slider("Jumlah Cluster", 2, 10, 3)

        # Training K-Means
        km = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = km.fit_predict(X_scaled)

        st.subheader("📊 Hasil Clustering")
        st.dataframe(df[["cluster"] + selected_features].head())

        # Visualisasi scatter plot (2 fitur saja)
        st.subheader("📈 Visualisasi Cluster")
        if len(selected_features) >= 2:
            fig = px.scatter(df, x=selected_features[0], y=selected_features[1],
                             color="cluster", title="Scatter Plot Cluster")
            st.plotly_chart(fig)

        # Centroid
        st.subheader("📌 Centroid Cluster (dalam skala standar)")
        centroid_df = pd.DataFrame(km.cluster_centers_, columns=selected_features)
        st.dataframe(centroid_df)

        st.subheader("🔍 Prediksi Cluster Lagu Baru")
        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.number_input(f"Masukkan nilai {feature}", 
                                                  value=float(X[feature].mean()))
        
        if st.button("Prediksi"):
            input_scaled = scaler.transform([list(input_data.values())])
            pred_cluster = km.predict(input_scaled)[0]
            st.success(f"Lagu baru masuk ke Cluster **{pred_cluster}**")
    else:
        st.warning("Pilih minimal 2 fitur untuk clustering.")
