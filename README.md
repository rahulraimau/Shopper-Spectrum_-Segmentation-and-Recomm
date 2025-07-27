# Shopper-Spectrum_-Segmentation-and-Recomm
# 🛍️ Shopper Spectrum

A smart and interactive retail dashboard that blends **Product Recommendation** and **Customer Segmentation** into a single seamless Streamlit application. Analyze customer behavior, recommend products, and segment buyers — all in one place!

![App Preview](assets/preview.gif)

---

## 📌 Modules

### 1️⃣ Product Recommender
🔎 Enter any product name and get smart recommendations based on purchase patterns.

- Input: Product Name (from dataset)
- Output: Top 5 Similar Products
- Model: KNN (item-based collaborative filtering)

### 2️⃣ Customer Segmentation
📊 Upload a CSV file with RFM metrics and instantly discover customer segments:

- Segments: 🏷️ Loyal, 🧊 At Risk, 🔄 Potential, 🆕 New, 💸 Lost
- Model: KMeans Clustering on Recency, Frequency, and Monetary values

---

## 🔧 Technical Stack

| Tool/Library        | Purpose                                  |
|---------------------|------------------------------------------|
| `Python`            | Core programming language                |
| `Pandas`, `NumPy`   | Data manipulation and analysis           |
| `Scikit-learn`      | ML models: KMeans, KNN                   |
| `Joblib`            | Model saving/loading                     |
| `Streamlit`         | Web app framework                        |
| `Matplotlib`, `Seaborn` | Data visualization                  |
| `Plotly` (optional) | Interactive plots                        |

---

## 📁 Dataset Used

- `online_retail.csv` – Transactional data (UCI ML Repository)
- `rfm_data.csv` – Preprocessed data with Recency, Frequency, Monetary values

---

## 📂 Folder Structure

Shopper-Spectrum/
│
├── app.py # Main Streamlit app
├── recommender_utils.py # Product recommendation logic
├── rfm_utils.py # Customer segmentation logic
├── models/
│ ├── kmeans_model.joblib # Trained KMeans clustering model
│ └── product_matrix.pkl # Pivoted customer-product matrix
├── data/
│ ├── online_retail.csv # Original retail dataset
│ └── rfm_data.csv # Cleaned RFM dataset
├── assets/
│ ├── preview.gif # App usage animation
│ ├── recommender.png # Screenshot: Product Recommender
│ └── segmentation.png # Screenshot: Customer Segmentation
├── README.md
└── requirements.txt



## 📷 Screenshots

### 🏪 Product Recommendation

![Recommender Screenshot](assets/recommender.png)

### 👥 Customer Segmentation

![Segmentation Screenshot](assets/segmentation.png)

---

## 🛠️ Setup Instructions

1. **Clone the Repo**

git clone https://github.com/yourusername/Shopper-Spectrum.git
cd Shopper-Spectrum
Install Dependencies


pip install -r requirements.txt
Run the App


streamlit run app.py
🚀 Demo Video
📺 Click to Watch Full Demo

⭐ Features
✅ Real-time Product Recommendations
✅ Automated Customer Segmentation
✅ Interactive Visualizations
✅ Clean and User-friendly Streamlit Interface
✅ Easy to Extend with Your Own Data

🤖 Built With
Python 3.10+

Streamlit

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

Joblib

📬 Contact
Rahul Rai
📧 rahulraimau5@gmail.com
🔗 GitHub | LinkedIn


