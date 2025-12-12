# 🌍 AI Powered Guide For Sustainable Tourism Prediction

This is an intelligent tourism analytics and prediction platform designed to help stakeholders, travelers, and agencies understand travel trends, sustainability impacts, and tourist satisfaction.

Built as a Final Year Project, this application leverages **Machine Learning** (Random Forest & Logistic Regression) to predict tourist sentiment and uses **Streamlit** to provide an interactive dashboard for exploring global heritage sites, travel costs, and environmental data.

---

## 🚀 Key Features

* **📊 Global Dashboard:** Get a real-time overview of tourism data, including UNESCO heritage sites counts, analysis of travel reviews, and average trip cost estimations.
* **🤖 AI Sentiment Predictor:** An intelligent Machine Learning tool that predicts whether a tourist will have a **"Positive"** or **"Negative"** experience based on input parameters such as:
    * Visitor Age
    * Travel Purpose
    * Expense Level
    * Crowd Density
* **🗺️ UNESCO Explorer:** An interactive geospatial map to filter and explore World Heritage Sites by country, region, and category (Cultural/Natural).
* **🌱 Sustainability Insights:** Visual analytics correlating carbon emissions with travel modes (Bus, Plane, Train) and traffic congestion levels.
* **⚙️ Model Evaluation:** A transparent view of the AI's performance, displaying Confusion Matrices and Cross-Validation scores to compare the accuracy of Random Forest vs. Logistic Regression models.

---

## 📂 Project Structure

```text
Sustainable-Tourism-Dashboard/
│
├── data/                        # Contains all raw CSV datasets
│   ├── ecotourism_dataset.csv
│   ├── Destination Reviews (final).csv
│   ├── Sustainable_road_tourism_dataset.csv
│   ├── Travel details dataset.csv
│   └── ThrowbackDataThursday 2019 Week 6 - UNESCO World Heritage Sites.hyper.csv
│
├── notebooks/                   # Jupyter Notebooks for training & logic
│   ├── AI-Planner.ipynb         # Main script: Cleans data, trains models, saves artifacts
│   │
│   └── models/                  # Directory where trained models are saved
│       ├── best_model.pkl       # The saved best performing model
│       ├── encoders.pkl         # Label encoders for categorical data
│       └── model_metrics.pkl    # Accuracy scores and confusion matrices
│
├── app.py                       # Main Streamlit Dashboard Application
├── requirements.txt             # List of Python dependencies
└── README.md                    # Project documentation
````

-----

## 🛠️ Installation & Setup

Follow these steps to set up the project locally.

### 1\. Clone the Repository

Download the project files to your local machine.

### 2\. Set up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Run the following command to install Streamlit, Pandas, Scikit-Learn, and Plotly:

```bash
pip install -r requirements.txt
```

-----

## 🏃‍♂️ How to Run

To run the application, you first need to ensure the models are trained, and then launch the dashboard.

### Step 1: Train the Models

Open and run the Jupyter Notebook located in `notebooks/AI-Planner.ipynb`.

  * This notebook processes the data in the `data/` folder.
  * It trains the Random Forest and Logistic Regression models.
  * It saves the `best_model.pkl`, `encoders.pkl`, and `model_metrics.pkl` into the `notebooks/models/` directory.

### Step 2: Launch the App

Return to the root directory and run the Streamlit app:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

-----

## 📊 Datasets Used

This project aggregates data from multiple sources to provide comprehensive insights:

1.  **Ecotourism Dataset:** Used for training the Sentiment Analysis AI.
2.  **UNESCO World Heritage Sites:** Geospatial data for the interactive map.
3.  **Sustainable Road Tourism:** Data on carbon emissions and transport modes.
4.  **Travel Details Dataset:** Information on travel costs, duration, and traveler demographics.
5.  **Destination Reviews:** User feedback and ratings for various locations.

-----

## ⚙️ Technologies

  * **Language:** Python 3.9+
  * **Web Framework:** Streamlit
  * **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression)
  * **Data Manipulation:** Pandas, NumPy
  * **Visualization:** Plotly, Matplotlib, Seaborn

-----

## 🤝 Contact

**Developer:** Sharoobini
*Final Year Project - AI Powered Guide for Sustainable Tourism Prediction*

```
```