import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# --- Load data ---
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
df.dropna(subset=["CustomerID", "Description", "Quantity", "UnitPrice", "InvoiceDate"], inplace=True)

# --- Clean data ---
df["Description"] = df["Description"].astype(str).str.strip().str.upper()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalSum"] = df["Quantity"] * df["UnitPrice"]

# --- Create product matrix for recommender ---
product_matrix = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum"
).fillna(0)

# --- RFM Table for Customer Segmentation ---
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm_df = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalSum": "sum"
}).reset_index()

rfm_df.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
rfm_df.dropna(inplace=True)

# --- Load pre-trained KMeans model ---
kmeans = joblib.load("kmeans_model.joblib")

# --- Sidebar ---
st.sidebar.title("🛍️ Shopper Spectrum")
module = st.sidebar.radio("Select Module", ["1️⃣ Product Recommender", "2️⃣ Customer Segmentation"])

# --- Product Recommender ---
if module.startswith("1️"):
    st.title("🎯 Product Recommender")

    # Dropdown for safe selection
    product_list = sorted(product_matrix.columns.tolist())
    selected_product = st.selectbox("🔍 Select a Product", product_list)

    if st.button("🔄 Recommend Similar Products"):
        try:
            # Create model and fit
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
            model_knn.fit(product_matrix.T.values)

            # Find index of selected product
            product_idx = list(product_matrix.columns).index(selected_product)
            distances, indices = model_knn.kneighbors([product_matrix.T.values[product_idx]], n_neighbors=6)

            st.success("🛒 Recommended Products:")
            for idx in indices.flatten()[1:]:
                st.write(f"• {product_matrix.columns[idx].title()}")
        except ValueError:
            st.error("❌ Product not found. Please select a valid product from the list.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
# --- Customer Segmentation ---
elif module.startswith("2️"):
    st.title("👥 Customer Segmentation")
    customer_id = st.number_input("🔑 Enter Customer ID", min_value=1, step=1)

    if st.button("📊 Segment Customer"):
        customer_data = rfm_df[rfm_df["CustomerID"] == customer_id]

        if customer_data.empty:
            st.error("❌ Customer ID not found.")
        else:
            segment = kmeans.predict(customer_data[["Recency", "Frequency", "Monetary"]])
            st.success(f"🎯 Customer belongs to Segment {segment[0]}")
