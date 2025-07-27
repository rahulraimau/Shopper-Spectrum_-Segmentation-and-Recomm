# Shopper-Spectrum_-Segmentation-and-Recomm
# 🛍️ Shopper Spectrum

An intelligent Streamlit dashboard for **Product Recommendation** and **Customer Segmentation** based on retail transaction data. Built using **KMeans Clustering**, **KNN**, and RFM analysis.

![App Preview](assets/preview.gif)

---

## 📌 Modules

### 1️⃣ Product Recommender
🔎 Enter any product name and get smart recommendations based on other users' purchase behavior.

- Input: Product Name
- Output: Top 5 Similar Products

> 💡 Powered by KNN using item-based collaborative filtering.

---

### 2️⃣ Customer Segmentation
📊 Upload a CSV with RFM metrics (Recency, Frequency, Monetary), and discover which segment your customer belongs to:

- 🏷️ Loyal
- 🧊 At Risk
- 🔄 Potential
- 🆕 New
- 💸 Lost

> 💡 Powered by KMeans clustering on RFM data.

---

## 📁 Dataset Used

- `online_retail.csv` – Raw transactions (source: UCI ML Repo)
- `rfm_data.csv` – Cleaned Recency, Frequency, Monetary values for each CustomerID

---

## 🔧 Setup Instructions

### 🔗 Clone this repo
```bash
git clone https://github.com/yourusername/Shopper-Spectrum.git
cd Shopper-Spectrum
📦 Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
▶️ Run Streamlit App
bash
Copy
Edit
streamlit run app.py
📂 Folder Structure
bash
Copy
Edit
Shopper-Spectrum/
│
├── app.py                  # Main Streamlit app
├── recommender_utils.py    # KNN recommendation functions
├── rfm_utils.py            # RFM segmentation functions
├── models/
│   ├── kmeans_model.joblib # Trained KMeans model
│   └── product_matrix.pkl  # Pivoted item matrix (if using offline)
├── data/
│   ├── online_retail.csv   # Raw retail data
│   └── rfm_data.csv        # Cleaned RFM data
├── assets/
│   └── preview.gif         # App screenshot/gif
├── README.md
└── requirements.txt
📷 Screenshots
🏪 Product Recommendation

👥 Customer Segmentation

🚀 Demo Video
📺 Click to Watch Demo

🤖 Built With
Python

Pandas, NumPy, Scikit-learn

Streamlit

Joblib

KMeans, KNN

📬 Contact
Rahul Rai
📧 rahulraimau5@gmail.com
🔗 GitHub | LinkedIn

⭐️ Give it a Star
If you found this project helpful, feel free to star ⭐ the repo and share it with others!

yaml
Copy
Edit

---

### 📸 Assets You Should Add:
Save these into an `assets/` folder in your repo:

1. `preview.gif`: GIF showing both modules being used.
2. `recommender.png`: Screenshot of Product Recommendation module.
3. `segmentation.png`: Screenshot of Segmentation result.

If you want, I can generate placeholder images for you or help design your GIF with a demo video.

Would you like a `requirements.txt` and automatic `README` generator too?






You said:
add also technical tag


ChatGPT said:
Here is the updated and complete README.md for your 🛍️ Shopper Spectrum project, now including a 🔧 Technical Stack section to highlight tools, libraries, and models used:

markdown
Copy
Edit
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

yaml
Copy
Edit

---

## 📷 Screenshots

### 🏪 Product Recommendation

![Recommender Screenshot](assets/recommender.png)

### 👥 Customer Segmentation

![Segmentation Screenshot](assets/segmentation.png)

---

## 🛠️ Setup Instructions

1. **Clone the Repo**
```bash
git clone https://github.com/yourusername/Shopper-Spectrum.git
cd Shopper-Spectrum
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
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
