# NetflixStreamLens

> A Data-Driven Exploration of Netflix Movies & TV Shows using Python

---

## 📌 Objective

The objective of this project is to analyze content trends on Netflix and derive insights into:
- Genre popularity
- Content release patterns
- Regional distribution
- Age ratings
- Evolution of content type (Movies vs. TV Shows)

---

## 📊 Dataset

- 📁 **Source**: [Netflix Movies and TV Shows Dataset – Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- 🔢 **Size**: ~8800+ rows × 12 columns

---

## 🛠️ Tools & Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Seaborn, Matplotlib, Plotly
- Git & GitHub

---

## 🔍 EDA Process Overview

### ✅ Step 1: Environment Setup  
- Installed required libraries  
- Setup working notebook

### ✅ Step 2: Dataset Loading  
- Loaded CSV using `pandas`  
- Basic shape and null checks

### ✅ Step 3: Data Cleaning  
- Handled missing values (e.g., director, cast)  
- Parsed `date_added`, `duration`, `listed_in`  
- Standardized columns and removed duplicates

### ✅ Step 4: Univariate Analysis  
- Content type distribution  
- Top genres  
- Rating categories  
- Release year trends

### ✅ Step 5: Multivariate Analysis  
- Country-wise content breakdown  
- Heatmaps of monthly/yearly additions  
- Rating vs type comparisons  
- Director/cast frequency

### ✅ Step 6: Interactive Visualization  
- Built engaging visuals using Plotly:
  - Pie Charts
  - Bar Charts
  - Choropleth Maps
  - Treemaps

---

## 📈 Key Takeaways

- Netflix is increasingly pushing TV Shows over Movies post-2015.
- U.S. remains the dominant content contributor, but international content is growing.
- Binge-worthy content with short duration and mature themes is preferred.
- Great scope exists for regional and genre-specific recommendations.

---

## 💡 Possible Extensions

- Compare with Amazon Prime or Disney+ datasets
- Apply Machine Learning for content recommendation
- Perform Sentiment Analysis on show descriptions using NLP


