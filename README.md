# Nada-Projects

# Customer Segmentation using Unsupervised Machine Learning

## Overview

This project performs **Customer Segmentation** using an unsupervised machine learning approach. The goal is to group customers with similar purchasing behavior and demographic characteristics, enabling businesses to design targeted marketing campaigns and improve customer engagement.

The project includes:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Principal Component Analysis (PCA)
- Agglomerative Hierarchical Clustering
- Customer Cluster Visualization

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Nada-Projects.git
```

Navigate to the project directory:

```bash
cd Nada-Projects
```

Install all the required dependencies using the `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

---

## Dataset

**Dataset:** Marketing Campaign Dataset

The dataset contains customer demographic information, purchasing behavior, campaign responses, and spending across multiple product categories.

### Features

- Customer ID
- Year of Birth
- Education
- Marital Status
- Income
- Number of Children
- Product Spending
- Purchase Channels
- Campaign Acceptance
- Recency
- Customer Enrollment Date

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Yellowbrick

---

## Project Workflow

### 1. Import Required Libraries

The project uses Python libraries for:

- Data manipulation
- Data visualization
- Feature engineering
- Data preprocessing
- Dimensionality reduction
- Clustering

### 2. Load Dataset

The dataset is loaded using Pandas.

```python
pd.read_csv("marketing_campaign.csv", sep="\t")
```

### 3. Data Cleaning

The preprocessing stage includes:

- Removing missing values
- Converting customer enrollment dates into datetime format
- Creating customer tenure (`Customer_For`)
- Removing unnecessary columns
- Removing outliers

Outlier thresholds:

- Age < 90
- Income < 600000

### 4. Feature Engineering

The following features were created:

- **Age**
- **Customer_For**
- **total_spent**
- **living_with_partner**
- **Children**
- **Family_Size**
- **Is_Parent**

Education levels were grouped into:

- Under Graduate
- Graduate
- Post Graduate

### 5. Exploratory Data Analysis (EDA)

Visualizations include:

- Pairplots
- Scatterplots
- Countplots
- Swarm plots
- Boxen plots
- KDE Joint plots

### 6. Data Preprocessing

- Label Encoding
- StandardScaler

### 7. Principal Component Analysis (PCA)

The dataset is reduced to **3 principal components** to simplify clustering and visualization.

### 8. Cluster Selection

The optimal number of clusters is determined using the **Elbow Method** with **KElbowVisualizer**.

### 9. Customer Segmentation

Agglomerative Hierarchical Clustering is applied with:

```
n_clusters = 4
```

### 10. Cluster Visualization

The project visualizes:

- 3D PCA Cluster Plot
- Cluster Distribution
- Income vs Spending
- Spending Distribution
- Income Distribution
- Promotions Accepted
- Deal Purchases
- Customer Demographics

## Business Insights

The generated customer segments help identify:

- High-value customers
- Budget-conscious customers
- Frequent deal seekers
- Promotion-responsive customers
- Families vs. individual shoppers
- Loyal customers

These insights can support:

- Personalized marketing campaigns
- Customer retention
- Product recommendations
- Promotion optimization
- Better customer relationship management

---

## Future Improvements

- Compare Agglomerative Clustering with K-Means.
- Evaluate clusters using Silhouette Score and Davies–Bouldin Index.
- Build an interactive Streamlit dashboard.
- Deploy the clustering model.
- Automate segmentation for new customer data.

---

## Results

The project successfully segments customers into **four distinct groups** based on demographics, purchasing behavior, and campaign responses. These clusters provide actionable insights for targeted marketing and customer engagement.

---

## Acknowledgements

This project was inspired by the YouTube tutorial **[Watch Me Do a Full Data Analytics Project with Machine Learning](https://youtu.be/k_7Ise59GQY?si=44iWhq1oDjPZ5R6K)** by **Lore So What**.

The implementation was recreated independently as a learning exercise. I added detailed code comments, documentation, and minor customizations to the workflow to strengthen my understanding of customer segmentation using Principal Component Analysis (PCA) and Agglomerative Hierarchical Clustering.

---

## Author

**Nada Rajguru**

If you found this project helpful, consider giving it a ⭐ on GitHub.