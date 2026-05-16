# ========================= EDA.py =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== LOAD DATA ==================
df = pd.read_csv("covertype.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.describe())


# ================== MISSING VALUES ==================
print("\nMissing Values:")
print(df.isnull().sum())


# ================== DUPLICATES ==================
print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ================== TARGET DISTRIBUTION ==================
plt.figure(figsize=(8, 5))

sns.countplot(x='Cover_Type', data=df)

plt.title("Cover Type Distribution")

plt.show()


# ================== HISTOGRAM ==================
plt.figure(figsize=(8, 5))

sns.histplot(df['Elevation'], kde=True)

plt.title("Elevation Distribution")

plt.show()


# ================== BOXPLOT ==================
plt.figure(figsize=(8, 5))

sns.boxplot(x=df['Slope'])

plt.title("Slope Boxplot")

plt.show()


# ================== CORRELATION HEATMAP ==================
plt.figure(figsize=(14, 10))

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ================== SCATTERPLOT ==================
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x='Elevation',
    y='Slope',
    hue='Cover_Type',
    data=df
)

plt.title("Elevation vs Slope")

plt.show()


# ================== PAIRPLOT ================== 
sample_df = df.sample(500) 

sns.pairplot( sample_df, hue='Cover_Type' ) 

plt.show()