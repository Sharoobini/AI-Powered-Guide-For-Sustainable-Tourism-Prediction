import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sustainable Tourism Dashboard",
    layout="wide"
)

# --- ROBUST DATA LOADER ---
@st.cache_data
def load_datasets():
    # Helper to find file in common locations
    def get_path(fname):
        possible = [
            Path(fname), 
            Path("data") / fname, 
            Path("../data") / fname, 
            Path("notebooks/data") / fname
        ]
        for p in possible:
            if p.exists(): return p
        return None

    data = {}
    files = {
        "unesco": "ThrowbackDataThursday 2019 Week 6 - UNESCO World Heritage Sites.hyper.csv",
        "details": "Travel details dataset.csv",
        "sustainable": "Sustainable_road_tourism_dataset.csv",
        "reviews": "Destination Reviews (final).csv"
    }
    
    for key, f in files.items():
        p = get_path(f)
        if p:
            try:
                # Smart Separator Detection
                sep = ','
                if "UNESCO" in f: sep = '\t'
                
                df = pd.read_csv(p, sep=sep)
                if df.shape[1] < 2: # Retry if failed
                    alt_sep = ',' if sep == '\t' else '\t'
                    try: df = pd.read_csv(p, sep=alt_sep)
                    except: pass
                
                df.columns = df.columns.str.strip()
                data[key] = df
            except:
                try: 
                    df = pd.read_csv(p) # Last resort
                    data[key] = df
                except: pass
    return data

@st.cache_resource
def load_model_resources():
    model_dir = None
    for p in [Path("models"), Path("../models"), Path("notebooks/models"), Path(".")]:
        if (p / "best_model.pkl").exists():
            model_dir = p
            break
            
    if not model_dir: return None, None, None
    
    model = joblib.load(model_dir / "best_model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")
    metrics = joblib.load(model_dir / "model_metrics.pkl")
    return model, encoders, metrics

# --- LOAD ---
datasets = load_datasets()
model, encoders, metrics = load_model_resources()

if not model:
    st.error("⚠️ Models not found. Please run '01_process_data.py' and '02_train_models.py' first.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("WanderLust AI")
nav = st.sidebar.radio("Navigation", [
    "Dashboard & Data Info", 
    "UNESCO Explorer", 
    "AI Sentiment Predictor", 
    "Sustainability Insights", 
    "Model Performance"
])

# ==========================================
# 1. DASHBOARD & DATA INFO
# ==========================================
if nav == "Dashboard & Data Info":
    st.title("Global Tourism Dashboard")
    
    # Metrics
    n_unesco = len(datasets.get('unesco', []))
    n_reviews = len(datasets.get('reviews', []))
    avg_cost = 0
    if 'details' in datasets and 'Accommodation cost' in datasets['details'].columns:
        cost = datasets['details']['Accommodation cost'].astype(str).str.replace(r'[$,a-zA-Z\s]', '', regex=True)
        avg_cost = pd.to_numeric(cost, errors='coerce').mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Heritage Sites", n_unesco)
    c2.metric("Travel Reviews", n_reviews)
    c3.metric("Avg Trip Cost", f"${avg_cost:,.0f}" if pd.notnull(avg_cost) else "N/A")

    st.markdown("---")
    
    # --- REQUIREMENT: Dataset Overview (Shape, Columns, Types) ---
    st.subheader("📂 Dataset Overview")
    
    dataset_choice = st.selectbox("Select Dataset to Inspect:", list(datasets.keys()))
    
    if dataset_choice:
        df_selected = datasets[dataset_choice]
        
        # 1. Shape & Columns
        st.write(f"**Shape:** {df_selected.shape[0]} Rows, {df_selected.shape[1]} Columns")
            
        # 2. Sample Data
        st.write("**Sample Data:**")
        st.dataframe(df_selected.head(10), use_container_width=True)

# ==========================================
# 2. UNESCO EXPLORER
# ==========================================
elif nav == "UNESCO Explorer":
    st.title("UNESCO Heritage Sites")
    if 'unesco' in datasets:
        df = datasets['unesco']
        
        # Filters (Interactive)
        c1, c2 = st.columns(2)
        cat_col = 'Category' if 'Category' in df.columns else df.columns[0]
        reg_col = 'Region' if 'Region' in df.columns else (df.columns[1] if len(df.columns)>1 else df.columns[0])

        cats = ["All"] + list(df[cat_col].dropna().unique())
        sel_cat = c1.selectbox("Filter by Category", cats)
        
        regs = ["All"] + list(df[reg_col].dropna().unique())
        sel_reg = c2.selectbox("Filter by Region", regs)
        
        if sel_cat != "All": df = df[df[cat_col] == sel_cat]
        if sel_reg != "All": df = df[df[reg_col] == sel_reg]
        
        # Map
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            st.map(df[['Latitude', 'Longitude']].dropna().rename(columns={'Latitude':'latitude', 'Longitude':'longitude'}))
        
        # Chart
        if 'Country' in df.columns:
            top_countries = df['Country'].value_counts().head(10).reset_index()
            top_countries.columns = ['Country', 'Count']
            fig = px.bar(top_countries, x='Count', y='Country', orientation='h', title="Top 10 Countries")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. AI PREDICTOR
# ==========================================
elif nav == "AI Sentiment Predictor":
    st.title("Tourist Sentiment AI")
    st.markdown("Predict user satisfaction based on trip details.")
    
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 90, 30)
            v_type = st.selectbox("Group", encoders['Visit_Type'].classes_)
            purp = st.selectbox("Purpose", encoders['Travel_Purpose'].classes_)
            crowd = st.selectbox("Crowd", encoders['Crowd_Level'].classes_)
        with c2:
            exp = st.selectbox("Expense", encoders['Expense_Level'].classes_)
            eco = st.slider("Eco Rating", 1.0, 5.0, 4.0)
            serv = st.slider("Service", 1.0, 5.0, 4.0)
            act = st.number_input("Activities", 0, 20, 3)
            
        if st.form_submit_button("Predict"):
            # Encode
            vec = [age, 
                   encoders['Visit_Type'].transform([v_type])[0],
                   encoders['Travel_Purpose'].transform([purp])[0],
                   eco, serv,
                   encoders['Crowd_Level'].transform([crowd])[0],
                   encoders['Expense_Level'].transform([exp])[0],
                   act]
            
            # Predict
            pred_idx = model.predict([vec])[0]
            prob = model.predict_proba([vec])[0]
            label = encoders['target'].inverse_transform([pred_idx])[0]
            
            st.success(f"Prediction: **{label}**")
            st.metric("Confidence", f"{max(prob):.1%}")
            
            # Probability Chart
            probs_df = pd.DataFrame({"Label": encoders['target'].classes_, "Probability": prob})
            fig = px.bar(probs_df, x="Label", y="Probability")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. SUSTAINABILITY
# ==========================================
elif nav == "Sustainability Insights":
    st.title("Sustainability Analytics")
    if 'sustainable' in datasets:
        df = datasets['sustainable']
        
        # 3 Charts Requirement Met Here
        fig1 = px.scatter(df, x="Carbon_Emissions", y="Sustainability_Score", color="Travel_Mode", title="Emissions vs Sustainability")
        st.plotly_chart(fig1, use_container_width=True)
        
        c1, c2 = st.columns(2)
        fig2 = px.box(df, x="Travel_Mode", y="Carbon_Emissions", title="Emissions by Mode")
        c1.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.pie(df, names="Traffic_Congestion_Level", title="Traffic Congestion")
        c2.plotly_chart(fig3, use_container_width=True)

# ==========================================
# 5. MODEL PERFORMANCE
# ==========================================
elif nav == "Model Performance":
    st.title("Model Evaluation")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Model", metrics['best_model'])
    c2.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
    
    # CV Metric Display
    cv_key = 'rf_cv' if metrics['best_model'] == 'Random Forest' else 'lr_cv'
    if cv_key in metrics:
        c3.metric("Cross-Val Score", f"{metrics[cv_key]:.2%}")
    
    st.divider()
    
    col_cm, col_comp = st.columns(2)
    
    # Confusion Matrix
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = np.array(metrics['confusion_matrix'])
        labels = encoders['target'].classes_
        fig_cm = ff.create_annotated_heatmap(z=cm, x=list(labels), y=list(labels), colorscale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    # Comparison Chart
    with col_comp:
        st.subheader("Model Comparison")
        comp_df = pd.DataFrame({
            "Model": ["Random Forest", "Logistic Regression"],
            "Accuracy": [metrics.get('rf_acc',0), metrics.get('lr_acc',0)]
        })
        fig_comp = px.bar(comp_df, x="Model", y="Accuracy", title="Accuracy Comparison")
        fig_comp.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_comp, use_container_width=True)