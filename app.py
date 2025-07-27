#```python
# ecommerce_rfm_analysis.ipynb
# ============================
# 📊 E-commerce Customer Segmentation + Product Recommendation
# Author: Rahul Rai
# Date: July 2025

# 📦 1. Imports & Dataset Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')

# 🧹 2. Data Cleaning
# Drop missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)
# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
# Remove negative quantity/price
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 📊 3. EDA Visuals
## 3.1 Transaction Volume by Country
country_tx = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)
country_tx.head(10).plot(kind='bar', title='Transactions by Country')
plt.show()

## 3.2 Top-Selling Products
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='barh', title='Top Selling Products')
plt.show()

## 3.3 Purchase Trends Over Time
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Date'] = df['InvoiceDate'].dt.date
trend = df.groupby('Date')['Quantity'].sum()
trend.plot(title='Purchase Trends Over Time', figsize=(12,4))
plt.show()

## 3.4 Monetary Distribution per Transaction
monetary = df['Quantity'] * df['UnitPrice']
sns.histplot(monetary, bins=50)
plt.title('Monetary Value Distribution')
plt.show()

# 💡 4. RFM Feature Engineering
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSum': lambda x: (df.loc[x.index, 'Quantity'] * df.loc[x.index, 'UnitPrice']).sum()
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 📈 5. Clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Elbow Method
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Elbow Curve')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()

# Silhouette Score
for k in range(2, 6):
    score = silhouette_score(rfm_scaled, KMeans(n_clusters=k, random_state=42).fit_predict(rfm_scaled))
    print(f"Silhouette Score for k={k}: {score:.4f}")

# Final KMeans
k_final = 4
kmeans = KMeans(n_clusters=k_final, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Label clusters manually
def label_cluster(row):
    if row['Recency'] < rfm['Recency'].quantile(0.25) and row['Frequency'] > rfm['Frequency'].quantile(0.75):
        return "High-Value"
    elif row['Frequency'] > rfm['Frequency'].median():
        return "Regular"
    elif row['Recency'] > rfm['Recency'].quantile(0.75):
        return "At-Risk"
    else:
        return "Occasional"
rfm['Segment'] = rfm.apply(label_cluster, axis=1)

# Cluster Visual
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment')
plt.title('Customer Segments')
plt.show()

# Save model
import joblib
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# 🔁 6. Product Recommendation
basket = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
sim_matrix = pd.DataFrame(cosine_similarity(basket.T),
                          index=basket.columns,
                          columns=basket.columns)

# Recommend Function
def get_similar_products(product_id, top_n=5):
    if product_id in sim_matrix:
        return sim_matrix[product_id].sort_values(ascending=False)[1:top_n+1]
    else:
        return "Product not found"

# 🔵 Streamlit UI (Standalone Option)
# Save this section in a separate streamlit_app.py if needed
import streamlit as st

st.title("🛍️ Shopper Spectrum: Customer Segmentation & Product Recommender")

# Recommendation Module
st.subheader("🔁 Product Recommendation")
product_name = st.text_input("Enter Product StockCode:")
if st.button("Get Recommendations"):
    recs = get_similar_products(product_name)
    st.write("### 🔎 Top 5 Similar Products:")
    st.write(recs)

# Customer Segmentation Module
st.subheader("📊 Customer Segment Predictor")
r = st.number_input("Recency (days)", min_value=1)
f = st.number_input("Frequency (transactions)", min_value=1)
m = st.number_input("Monetary (total spend)", min_value=1.0)
if st.button("Predict Cluster"):
    X = scaler.transform([[r, f, m]])
    cluster = kmeans.predict(X)[0]
    segment = ['High-Value', 'Regular', 'Occasional', 'At-Risk'][cluster]
    st.success(f"Predicted Segment: {segment}")

