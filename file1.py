import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create plots directory if not exists
os.makedirs("plots", exist_ok=True)

# Set plot style
sns.set(style="whitegrid")
plt.style.use("seaborn-v0_8-muted")

# Data Cleaning
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def run_data_cleaning(df):
    numerical_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    print("\n Data Cleaning ")

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    duplicates = df[df.duplicated()]
    print(f"\n Number of Duplicate Rows: {len(duplicates)}")
    df = df.drop_duplicates()
    for col in numerical_cols:
        original_len = len(df)
        df = remove_outliers_iqr(df, col)
        print(f"Removed outliers from '{col}': {original_len - len(df)} rows removed")
    for col in categorical_cols:
        df[col] = df[col].str.strip().str.lower()
    return df

# EDA Functions
def summary_statistics(df, numerical_cols):
    print("\n Summary Statistics (Numerical):")
    print(df[numerical_cols].describe())
    for col in numerical_cols:
        print(f"\n {col}:")
        print(f"  Median: {df[col].median()}")
        print(f"  Mode: {df[col].mode()[0]}")
        print(f"  Variance: {df[col].var():.2f}")
        print(f"  Skewness: {df[col].skew():.2f}")

def frequency_distributions(df, categorical_cols):
    print("\n Frequency Distributions (Categorical):")
    for col in categorical_cols:
        print(f"\n {col}:")
        print(df[col].value_counts())

def plot_histograms(df, numerical_cols):
    df[numerical_cols].hist(bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
    plt.suptitle(" Histograms of Numerical Features", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/histograms.png")
    plt.close()

def plot_boxplots(df, numerical_cols):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[numerical_cols])
    plt.title(" Boxplots of Numerical Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/boxplots.png")
    plt.close()

def correlation_analysis(df, numerical_cols):
    corr_matrix = df[numerical_cols].corr()
    print("\n Correlation Matrix:")
    print(corr_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(" Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

def scatter_plots(df, numerical_cols):
    sns.pairplot(df[numerical_cols])
    plt.suptitle(" Scatter Plots (Numerical Variables)", y=1.02)
    plt.savefig("plots/scatter_plots.png")
    plt.close()

def box_violin_by_category(df, numerical_cols, categorical_cols):
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[cat_col], y=df[num_col], palette="pastel")
            plt.title(f"Boxplot of {num_col} by {cat_col}")
            plt.tight_layout()
            plt.savefig(f"plots/boxplot_{num_col}_by_{cat_col}.png")
            plt.close()

            plt.figure(figsize=(8, 4))
            sns.violinplot(x=cat_col, y=num_col, data=df, palette="muted")
            plt.title(f"Violin Plot of {num_col} by {cat_col}")
            plt.tight_layout()
            plt.savefig(f"plots/violin_{num_col}_by_{cat_col}.png")
            plt.close()

def multivariate_pairplot(df, hue_col='smoker'):
    if hue_col in df.columns:
        sns.pairplot(df, hue=hue_col, diag_kind='kde', palette="Set2")
        plt.suptitle(f"Pairplot Colored by '{hue_col}'", y=1.02)
        plt.savefig(f"plots/multivariate_pairplot_by_{hue_col}.png")
        plt.close()

def grouped_barplot(df, group_col1='sex', group_col2='smoker', target='charges'):
    if all(col in df.columns for col in [group_col1, group_col2, target]):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=group_col1, y=target, hue=group_col2, data=df, palette='Set2')
        plt.title(f"Mean {target.title()} by {group_col1.title()} and {group_col2.title()}")
        plt.tight_layout()
        plt.savefig("plots/grouped_barplot.png")
        plt.close()

def heatmap_grouped_values(df, index_col='sex', col_col='region', value_col='charges'):
    if all(col in df.columns for col in [index_col, col_col, value_col]):
        pivot_table = df.pivot_table(index=index_col, columns=col_col, values=value_col, aggfunc='mean')
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".0f")
        plt.title(f"Average {value_col.title()} by {index_col.title()} and {col_col.title()}")
        plt.tight_layout()
        plt.savefig("plots/grouped_heatmap.png")
        plt.close()

# Run EDA
def run_eda(df):
    numerical_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    print(f"\n Numerical Features: {list(numerical_cols)}")
    print(f" Categorical Features: {list(categorical_cols)}")

    summary_statistics(df, numerical_cols)
    frequency_distributions(df, categorical_cols)
    plot_histograms(df, numerical_cols)
    plot_boxplots(df, numerical_cols)
    correlation_analysis(df, numerical_cols)
    scatter_plots(df, numerical_cols)
    box_violin_by_category(df, numerical_cols, categorical_cols)
    multivariate_pairplot(df, hue_col='smoker')
    grouped_barplot(df, group_col1='sex', group_col2='smoker', target='charges')
    heatmap_grouped_values(df, index_col='sex', col_col='region', value_col='charges')


def main():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    print("first 5 rows: \n")
    print(df.head())
    print("info: \n")
    print(df.info())
    print("statictics: \n")
    print(df.describe())
    print("rows, cols: \n")
    print(df.shape)
    print("null values count: \n")
    print(df.isnull().sum())

    df_cleaned = run_data_cleaning(df)
    run_eda(df_cleaned)

if __name__ == "__main__":
    main()
