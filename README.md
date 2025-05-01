# 🏡 House Price Prediction in the Browser with JavaScript

This project demonstrates how to build a simple machine learning model in the browser using **TensorFlow.js** to predict house prices from CSV data.

It covers loading and parsing CSV files, training a neural network regression model, and visualizing predictions interactively.

---

## 🚀 Features

- 📂 Upload CSV files directly in the browser  
- 🧹 Filter and clean data (`LotArea`, `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `SalePrice`)  
- 🤖 Train a neural network with TensorFlow.js  
- 📊 Visualize predictions vs. actual values using Chart.js  
- 💥 Compute training and testing Mean Absolute Error (MAE)

---

## 🔧 Tech Stack

- **TensorFlow.js** — machine learning in the browser  
- **PapaParse** — CSV parsing  
- **Chart.js** — data visualization  
- **JavaScript (vanilla)** — frontend logic

---

## 💡 Limitations

While JavaScript is powerful for small to medium data tasks and interactive visualizations, it’s not ideal for advanced analysis or large datasets compared to Python’s mature ecosystem (NumPy, pandas, scikit-learn, etc.).

---

## 📦 How to Run

1. Clone or download the repository.
2. Open `index.html` in your web browser.
3. Upload a CSV file with columns: `SalePrice, LotArea, OverallQual, GrLivArea, GarageCars, TotalBsmtSF`.
4. View metrics and predictions right in the browser!

---


