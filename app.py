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

# CLUSTERING
if page == "Clustering":
    st.title("📊 LRFMC K-Means Clustering")
    if "df" in st.session_state:
        df = st.session_state.df.copy()

        #Date conversion
        date_columns = ['FFP_DATE', 'FIRST_FLIGHT_DATE', 'LOAD_TIME', 'LAST_FLIGHT_DATE']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        #Fill missing categorical
        cat_cols = ['GENDER', 'WORK_CITY', 'WORK_PROVINCE', 'WORK_COUNTRY']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').str.lower()

        #Fill AGE missing
        if 'AGE' in df.columns:
            df['AGE'] = df['AGE'].fillna(df['AGE'].median())

        #Drop rows with missing SUM_YR_1 or SUM_YR_2
        for col in ['SUM_YR_1', 'SUM_YR_2']:
            if col in df.columns:
                df = df[df[col].notnull()]

        #Feature engineering L, R, F, M, C
        df['LENGTH'] = (df['LOAD_TIME'] - df['FFP_DATE']).dt.days
        lrfmc_features = ['LENGTH', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']
        df_lrfmc = df[lrfmc_features].copy()

        #Outlier capping
        cols_for_outlier_capping = ['LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']
        for col in cols_for_outlier_capping:
            high_cut = df_lrfmc[col].quantile(0.99)
            low_cut = df_lrfmc[col].quantile(0.01)
            df_lrfmc[col] = np.where(df_lrfmc[col] > high_cut, high_cut, df_lrfmc[col])
            df_lrfmc[col] = np.where(df_lrfmc[col] < low_cut, low_cut, df_lrfmc[col])

        #Scaling
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_lrfmc), columns=lrfmc_features)

        #K-Means
        k = st.slider("Select number of clusters (K)", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        df_scaled['Cluster'] = labels
        st.session_state.clustered_df = df_scaled

        # PCA Visualization
        pca = PCA(n_components=2)
        components = pca.fit_transform(df_scaled.drop(columns="Cluster"))
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["Cluster"] = labels

        st.subheader("PCA Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="viridis", ax=ax)
        st.pyplot(fig)

        st.subheader("Cluster Size")
        st.write(df_scaled["Cluster"].value_counts())
    else:
        st.warning("Please upload data first.")

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
