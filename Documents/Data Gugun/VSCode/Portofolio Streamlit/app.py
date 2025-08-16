#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

#Config
st.set_page_config(page_title="Customer Clustering Dashboard", layout="wide")
sns.set_style("whitegrid")

#Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "EDA", "Preprocessing", "Clustering", "Evaluation", "Cluster Insights"])

#Upload data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if page == "Upload Data":
    st.title("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("File berhasil terupload!")
        st.dataframe(df.head())
    else:
        st.warning("Please upload a CSV file.")

#EDA
if page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    if "df" in st.session_state:
        df = st.session_state.df

        st.subheader("Dataset Overview")
        st.write(df.shape)
        st.dataframe(df.head())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Numerical Distribution")
        num_cols = df.select_dtypes(include=np.number).columns
        col_choice = st.selectbox("Choose column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col_choice], kde=True, ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Please upload data first.")

#Preprocessing
if page == "Preprocessing":
    st.title("⚙ Data Preprocessing")
    if "df" in st.session_state:
        df = st.session_state.df.copy()

        # Example preprocessing: fill NA, scale, select features
        df['AGE'] = df['AGE'].fillna(df['AGE'].median())
        num_cols = df.select_dtypes(include=np.number).columns
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

        st.session_state.df_scaled = df_scaled
        st.success("Data scaled successfully!")
        st.dataframe(df_scaled.head())
    else:
        st.warning("Please upload data first.")

#Model Clustering
if page == "Clustering":
    st.title("📊 K-Means Clustering")
    if "df_scaled" in st.session_state:
        df_scaled = st.session_state.df_scaled
        k = st.slider("Select number of clusters (K)", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)

        df_scaled["Cluster"] = labels
        st.session_state.clustered_df = df_scaled
        st.success(f"Clustering done with K={k}")

        # PCA Visualization
        pca = PCA(n_components=2)
        components = pca.fit_transform(df_scaled.drop(columns="Cluster"))
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["Cluster"] = labels

        fig, ax = plt.subplots()
        sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="viridis", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Please preprocess data first.")

#Evaluation
if page == "Evaluation":
    st.title("📈 Model Evaluation")
    if "clustered_df" in st.session_state:
        df_scaled = st.session_state.clustered_df
        labels = df_scaled["Cluster"]

        sil_score = silhouette_score(df_scaled.drop(columns="Cluster"), labels)
        ch_score = calinski_harabasz_score(df_scaled.drop(columns="Cluster"), labels)

        st.metric("Silhouette Score", f"{sil_score:.4f}")
        st.metric("Calinski-Harabasz Score", f"{ch_score:.4f}")
    else:
        st.warning("Please run clustering first.")

#Model Insights
if page == "Cluster Insights":
    st.title("💡 Cluster Insights")
    if "clustered_df" in st.session_state:
        df_scaled = st.session_state.clustered_df
        st.subheader("Cluster Summary")
        st.write(df_scaled.groupby("Cluster").mean())

        st.subheader("Boxplot per Feature")
        feature = st.selectbox("Choose feature", df_scaled.columns[:-1])
        fig, ax = plt.subplots()
        sns.boxplot(x="Cluster", y=feature, data=df_scaled, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please run clustering first.")
