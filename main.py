import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

# Streamlit App Title
st.title("EDA App")

# File Upload Section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def handle_missing_values(df, column, strategy):
    """
    Handle missing values in the selected column using the specified strategy.
    """
    if strategy == "Mean":
        df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == "Median":
        df[column].fillna(df[column].median(), inplace=True)
    elif strategy == "Mode":
        df[column].fillna(df[column].mode().iloc[0], inplace=True)
    elif strategy == "KNN":
        imputer = KNNImputer()
        df[[column]] = imputer.fit_transform(df[[column]])
    return df

def remove_outliers(df, columns):
    """
    Remove outliers from the selected columns based on the Isolation Forest model.
    """
    for column in columns:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['Outlier'] = iso_forest.fit_predict(df[[column]])
        df = df[df['Outlier'] == 1]  # Keep only normal data points
        df = df.drop(columns=['Outlier'])  # Drop the 'Outlier' column
    return df

if uploaded_file:
    # Load the Dataset (store it in session state)
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

    df = st.session_state.df
    original_df = df.copy()  # Keep the original dataframe to show comparison later
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Exploratory Data Analysis Section
    st.sidebar.header("EDA Options")
    if st.sidebar.checkbox("Show Summary Statistics"):
        st.write("### Summary Statistics")
        st.table(df.describe())
        st.write("#### Additional Descriptive Statistics")
        numeric_df = df.select_dtypes(include=np.number)  # Filter numeric columns
        st.write("Skewness:")
        st.table(numeric_df.skew())
        st.write("Kurtosis:")
        st.table(numeric_df.kurt())

    if st.sidebar.checkbox("Show Missing Values"):
        st.write("### Missing Values")
        st.write(df.isnull().sum())
        missing_columns = df.columns[df.isnull().any()].tolist()

        if missing_columns:
            column = st.selectbox("Select Column to Handle Missing Values", missing_columns, key="missing_column")
            strategy = st.selectbox("Choose Missing Value Handling Strategy", ["None", "Mean", "Median", "Mode", "KNN"], key="missing_strategy")

            if st.button("Handle Missing Values"):
                df = handle_missing_values(df, column, strategy)
                st.session_state.df = df  # Store the updated dataframe in session state
                st.success(f"Missing values in column '{column}' have been handled using the '{strategy}' strategy.")
                st.write("### Updated Dataset")
                st.dataframe(df.head())
                st.write("### Missing Values After Handling")
                st.write(df.isnull().sum())
        else:
            st.write("No columns with missing values.")

    if st.sidebar.checkbox("Visualizations"):
        st.write("### Visualizations")
        plot_type = st.selectbox("Choose a Plot Type", ["Histogram", "Correlation Matrix", "Boxplot", "Density Plot", "Scatterplot", "Pairplot", "Violin Plot", "Bar Plot"], key="plot_type")

        numeric_cols = df.select_dtypes(include=np.number).columns
        if plot_type == "Histogram":
            column = st.selectbox("Choose Column", numeric_cols, key="hist_column")
            plt.figure(figsize=(10, 5))
            sns.histplot(df[column], kde=True)
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include=np.number)
            corr = numeric_df.corr()
            plt.figure(figsize=(10, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Boxplot":
            column = st.selectbox("Choose Numeric Column", numeric_cols, key="boxplot_column")
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[column])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Density Plot":
            column = st.selectbox("Choose Numeric Column", numeric_cols, key="density_column")
            plt.figure(figsize=(10, 5))
            sns.kdeplot(df[column], shade=True)
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Scatterplot":
            x_col = st.selectbox("Choose X Column", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Choose Y Column", numeric_cols, key="scatter_y")
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=df[x_col], y=df[y_col])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Pairplot":
            sns.pairplot(df[numeric_cols])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Violin Plot":
            column = st.selectbox("Choose Numeric Column", numeric_cols, key="violin_column")
            plt.figure(figsize=(10, 5))
            sns.violinplot(x=df[column])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Bar Plot":
            column = st.selectbox("Choose Categorical Column", df.select_dtypes(include="object").columns, key="bar_column")
            plt.figure(figsize=(10, 5))
            sns.countplot(x=df[column])
            st.pyplot(plt)
            plt.clf()

    # Outlier Detection Section
    st.sidebar.header("Outlier Detection & Removal")
    if st.sidebar.checkbox("Perform Outlier Detection"):
        st.write("### Outlier Detection")
        outlier_columns = st.multiselect("Choose Numeric Columns", df.select_dtypes(include=np.number).columns, key="outlier_columns")
        contamination = st.slider("Set Contamination Level", 0.01, 0.5, 0.1, key="contamination")

        if outlier_columns:
            # Handle missing or invalid values in the selected columns
            for column in outlier_columns:
                if df[column].isnull().sum() > 0:
                    st.warning(f"The selected column '{column}' contains missing values. These will be filled with the median.")
                    df[column] = df[column].fillna(df[column].median())

                if not np.isfinite(df[column]).all():
                    st.warning(f"The selected column '{column}' contains non-finite values. These will be replaced with the median.")
                    df[column] = df[column].replace([np.inf, -np.inf], df[column].median())

            # Using Isolation Forest for outlier detection on multiple columns
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            df['Outlier'] = iso_forest.fit_predict(df[outlier_columns])
            df['Outlier'] = df['Outlier'].map({1: "Normal", -1: "Outlier"})

            st.write(df[outlier_columns + ['Outlier']].head())
            st.write("### Outlier Visualization")
            sns.pairplot(df, hue="Outlier", vars=outlier_columns)
            st.pyplot(plt)
            plt.clf()

            # Button to remove outliers from selected columns
            if st.sidebar.button("Remove Outliers"):
                df = remove_outliers(df, outlier_columns)
                st.session_state.df = df  # Store the updated dataframe in session state
                st.success(f"Outliers in columns '{', '.join(outlier_columns)}' have been removed.")
                st.write("### Updated Dataset (Outliers Removed)")
                st.dataframe(df.head())

                # Visualizing the comparison of data before and after removing outliers
                st.write("### Comparison Before and After Removing Outliers")

                # Comparison of each selected column before and after removal
                for column in outlier_columns:
                    plt.figure(figsize=(10, 5))
                    sns.scatterplot(x=original_df.index, y=original_df[column], color='blue', label=f"Before Removal - {column}")
                    sns.scatterplot(x=df.index, y=df[column], color='red', label=f"After Removal - {column}")
                    plt.legend()
                    st.pyplot(plt)
                    plt.clf()

    # Visualization After Modifications Section
    if st.sidebar.checkbox("Visualize Updated Data"):
        st.write("### Visualize Updated Dataset")
        plot_type = st.selectbox("Choose a Plot Type", ["Histogram", "Correlation Matrix", "Boxplot", "Density Plot", "Scatterplot", "Pairplot", "Violin Plot", "Bar Plot"], key="visualize_plot_type")

        numeric_cols = df.select_dtypes(include=np.number).columns
        if plot_type == "Histogram":
            column = st.selectbox("Choose Column", numeric_cols, key="visualize_hist_column")
            plt.figure(figsize=(10, 5))
            sns.histplot(df[column], kde=True)
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include=np.number)
            corr = numeric_df.corr()
            plt.figure(figsize=(10, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Boxplot":
            column = st.selectbox("Choose Numeric Column", numeric_cols, key="visualize_boxplot_column")
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[column])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Density Plot":
            column = st.selectbox("Choose Numeric Column", numeric_cols, key="visualize_density_column")
            plt.figure(figsize=(10, 5))
            sns.kdeplot(df[column], shade=True)
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Scatterplot":
            x_col = st.selectbox("Choose X Column", numeric_cols, key="visualize_scatter_x")
            y_col = st.selectbox("Choose Y Column", numeric_cols, key="visualize_scatter_y")
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=df[x_col], y=df[y_col])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Pairplot":
            sns.pairplot(df[numeric_cols])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Violin Plot":
            column = st.selectbox("Choose Numeric Column", numeric_cols, key="visualize_violin_column")
            plt.figure(figsize=(10, 5))
            sns.violinplot(x=df[column])
            st.pyplot(plt)
            plt.clf()

        elif plot_type == "Bar Plot":
            column = st.selectbox("Choose Categorical Column", df.select_dtypes(include="object").columns, key="visualize_bar_column")
            plt.figure(figsize=(10, 5))
            sns.countplot(x=df[column])
            st.pyplot(plt)
            plt.clf()

    st.sidebar.header("Download Updated Data")
    if st.sidebar.button("Download Updated Data", key="download_data"):
        st.write("### Download Updated Data")
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="updated_data.csv",
            mime="text/csv"
        )