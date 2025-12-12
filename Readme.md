# 🌍 WanderLust AI | Smart Sustainable Tourism Guide

**WanderLust AI** is an intelligent tourism analytics and prediction platform designed to help stakeholders understand travel trends, sustainability impacts, and tourist satisfaction.

Built as a Final Year Project, this application leverages Machine Learning (Random Forest & Logistic Regression) to predict tourist sentiment and uses Streamlit to provide an interactive dashboard for exploring global heritage sites, travel costs, and environmental data.

---

## 🚀 Key Features

* **📊 Global Dashboard:** Real-time overview of UNESCO heritage sites, travel reviews, and average trip costs.
* **🤖 AI Sentiment Predictor:** A Machine Learning tool that predicts whether a tourist will have a "Positive" or "Negative" experience based on age, travel purpose, expense level, and crowd density.
* **🗺️ UNESCO Explorer:** Interactive map to filter and explore World Heritage Sites by country, region, and category.
* **🌱 Sustainability Insights:** Visual analytics correlating carbon emissions, travel modes, and traffic congestion.
* **⚙️ Model Evaluation:** Transparent view of the AI's performance, including Confusion Matrices and Cross-Validation scores comparing Random Forest vs. Logistic Regression.

---

## 📂 Project Structure

```text
Sustainable-Tourism-Dashboard/
│
├── data/                        # Place all CSV datasets here
│   ├── ecotourism_dataset.csv
│   ├── Destination Reviews (final).csv
│   ├── Sustainable_road_tourism_dataset.csv
│   ├── Travel details dataset.csv
│   └── ThrowbackDataThursday...UNESCO...csv
│
├── notebooks
|   |
|   └──AI-Planner.ipynb         #Cleans and encodes raw data &
|       models/                      
│           ├── best_model.pkl      # Stores trained models (.pkl files)
│           ├── encoders.pkl
│           └── model_metrics.pkl
│          
├── app.py                       # Script 3: Main Streamlit Dashboard
├── requirements.txt             # List of python dependencies
└── README.md                    # Project documentation