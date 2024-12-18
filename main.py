import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Streamlit App Title
st.title("EDA and Feature Engineering App")

# File Upload Section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the Dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Exploratory Data Analysis Section
    st.sidebar.header("EDA Options")
    if st.sidebar.checkbox("Show Summary Statistics"):
        st.write("### Summary Statistics")
        st.write(df.describe())

    if st.sidebar.checkbox("Show Missing Values"):
        st.write("### Missing Values")
        st.write(df.isnull().sum())

    if st.sidebar.checkbox("Visualizations"):
        st.write("### Visualizations")
        plot_type = st.selectbox("Choose a Plot Type", ["Histogram", "Correlation Matrix"])
        
        if plot_type == "Histogram":
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                column = st.selectbox("Choose Column", numeric_cols)
                plt.figure(figsize=(10, 5))
                sns.histplot(df[column], kde=True)
                st.pyplot(plt)
                plt.clf()
            else:
                st.warning("No numeric columns available for histograms.")

        elif plot_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                corr = numeric_df.corr()
                plt.figure(figsize=(10, 5))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
                st.pyplot(plt)
                plt.clf()
            else:
                st.warning("No numeric columns available for correlation matrix.")

    # Outlier Detection Section
    st.sidebar.header("Outlier Detection")
    if st.sidebar.checkbox("Perform Outlier Detection"):
        st.write("### Outlier Detection")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            outlier_column = st.selectbox("Choose a Numeric Column", numeric_cols)
            contamination = st.slider("Set Contamination Level", 0.01, 0.5, 0.1)
            
            # Check for missing or invalid values
            if df[outlier_column].isnull().sum() > 0:
                st.warning(f"The selected column '{outlier_column}' contains missing values. These will be filled with the median.")
                df[outlier_column] = df[outlier_column].fillna(df[outlier_column].median())
            
            if not np.isfinite(df[outlier_column]).all():
                st.warning(f"The selected column '{outlier_column}' contains non-finite values. These will be replaced with the median.")
                df[outlier_column] = df[outlier_column].replace([np.inf, -np.inf], df[outlier_column].median())
            
            # Using Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            df['Outlier'] = iso_forest.fit_predict(df[[outlier_column]])
            df['Outlier'] = df['Outlier'].map({1: "Normal", -1: "Outlier"})
            
            st.write(df[['Outlier', outlier_column]])
            st.write("### Outlier Visualization")
            sns.scatterplot(x=df.index, y=df[outlier_column], hue=df['Outlier'])
            st.pyplot(plt)
            plt.clf()
        else:
            st.warning("No numeric columns available for outlier detection.")

    # Feature Engineering Section
    st.sidebar.header("Feature Engineering")
    if st.sidebar.checkbox("Scale Features"):
        st.write("### Feature Scaling")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            st.write(df.head())
        else:
            st.warning("No numeric columns available for scaling.")

    if st.sidebar.checkbox("Encode Categorical Features"):
        st.write("### Categorical Encoding")
        cat_cols = df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
            df = pd.concat([df, encoded], axis=1).drop(cat_cols, axis=1)
            st.write(df.head())
        else:
            st.warning("No categorical columns available for encoding.")

    # Export Processed Data
    st.sidebar.header("Download Processed Data")
    if st.sidebar.button("Download"):
        st.write("### Download Link")
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="processed_data.csv",
            mime="text/csv"
        )
