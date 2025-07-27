import pandas as pd
import numpy as np

def main():
    # 📁 Load data
    try:
        df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1', parse_dates=["InvoiceDate"])
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print("❌ File 'online_retail.csv' not found.")
        return

    # 🧹 Data Cleaning
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalSum"] = df["Quantity"] * df["UnitPrice"]

    # 📆 Snapshot Date
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # 📊 RFM Calculation
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalSum": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm.to_csv("rfm_data.csv", index=False)
    print("✅ rfm_data.csv saved.")

    # 🔄 Product Matrix
    # Create product matrix
    # Clean product descriptions
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()

    # Create product matrix (CustomerID x Product)
    product_matrix = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum"
    ).fillna(0)

    # Optional: clean column names again if needed
    product_matrix.columns = product_matrix.columns.str.strip().str.upper()



    product_matrix.to_pickle("product_matrix.pkl")
    print("✅ product_matrix.pkl saved.")

if __name__ == "__main__":
    main()
