import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="🤖 Data Scientist Human-Like Agent", layout="wide")
st.title("🤖 Data Scientist Human-Like Agent (2050 Vision)")
st.caption("End-to-End AI-powered Data Science workflow using 8 Agentic AI Agents")

# -------------------------------
# DATA SOURCES
# -------------------------------
st.sidebar.header("📊 Choose Dataset")
dataset_choice = st.sidebar.radio(
    "Select dataset source:",
    ["Synthetic: Sales CSV", "Synthetic: IoT Sensors CSV", "Real-time: COVID-19", "Real-time: Earthquakes"]
)

# Synthetic datasets
if dataset_choice == "Synthetic: Sales CSV":
    data = pd.DataFrame({
        "customer_id": range(1, 21),
        "age": np.random.randint(18, 60, 20),
        "spending": np.random.randint(100, 1000, 20),
        "churn": np.random.randint(0, 2, 20)
    })

elif dataset_choice == "Synthetic: IoT Sensors CSV":
    data = pd.DataFrame({
        "device_id": range(1, 21),
        "temperature": np.random.uniform(15, 40, 20),
        "humidity": np.random.uniform(20, 90, 20),
        "failure": np.random.randint(0, 2, 20)
    })

elif dataset_choice == "Real-time: COVID-19":
    url = "https://disease.sh/v3/covid-19/all"
    try:
        covid = requests.get(url).json()
        data = pd.DataFrame([covid])
    except:
        data = pd.DataFrame()
        st.warning("⚠️ Could not fetch COVID-19 data.")

elif dataset_choice == "Real-time: Earthquakes":
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    try:
        eq = requests.get(url).json()
        data = pd.DataFrame([{
            "place": f["properties"]["place"],
            "mag": f["properties"]["mag"],
            "time": f["properties"]["time"]
        } for f in eq["features"]])
    except:
        data = pd.DataFrame()
        st.warning("⚠️ Could not fetch Earthquake data.")

st.write("### 📂 Preview of Data")
st.dataframe(data.head())

# -------------------------------
# AGENTS
# -------------------------------

def cleaning_agent(df):
    st.subheader("🧹 Data Collection & Cleaning Agent")
    st.write("Removing missing values & duplicates...")
    before = df.shape
    df = df.dropna().drop_duplicates()
    after = df.shape
    st.write(f"Shape before: {before}, after cleaning: {after}")
    return df

def eda_agent(df):
    st.subheader("🔍 Exploratory Data Analysis Agent")
    st.write(df.describe())
    st.bar_chart(df.select_dtypes(include=[np.number]).iloc[:, :2])

def feature_engineering_agent(df):
    st.subheader("⚙️ Feature Engineering Agent")
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[18,30,45,60], labels=["18-30","31-45","46-60"])
        st.write("Added age_group feature ✅")
    return df

def model_building_agent(df):
    st.subheader("🤖 Model Building Agent")
    if "churn" not in df.columns:
        st.warning("Model requires Sales CSV with churn column.")
        return None

    X = df[["age", "spending"]]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"RandomForest Accuracy: {acc:.2f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model saved as model.pkl ✅")

def experimentation_agent(df):
    st.subheader("⚗️ Experimentation Agent")
    if "churn" not in df.columns:
        st.warning("Experimentation requires Sales CSV with churn column.")
        return
    X = df[["age", "spending"]]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        results[name] = accuracy_score(y_test, m.predict(X_test))

    st.write("### Model Comparison")
    st.write(results)

def storytelling_agent(df):
    st.subheader("📖 Data Storytelling Agent")
    st.write("Generating human-friendly insights...")
    for col in df.select_dtypes(include=[np.number]).columns[:2]:
        st.line_chart(df[col])

def deployment_agent():
    st.subheader("🚀 Deployment Agent")
    st.info("In real life, this would deploy to AWS/GCP. Here, we just saved model.pkl ✅")

def scaling_agent(df):
    st.subheader("📈 Scaling Agent")
    df_big = pd.concat([df]*10, ignore_index=True)
    st.write(f"Scaled dataset: {df.shape} → {df_big.shape}")
    return df_big

# -------------------------------
# Limitations & Instructions
# -------------------------------
st.markdown("### ⚠️ Limitations you should know:")
st.info("""
- Some agents only work for specific datasets (e.g., **Model Building** requires `sales.csv`).  
- If you switch to COVID/earthquake data, only **EDA, Storytelling, Cleaning, and Scaling** will make sense.  
- This is a prototype — in real production, each agent would be separate modules, maybe even separate microservices with their own intelligence.  
""")

st.markdown("### 📘 How to Use the 'Data Scientist Human-Like Agent'")
st.markdown("""
1. Select dataset from sidebar (Synthetic CSVs or Real-time APIs).  
2. Start with Cleaning Agent to preprocess data.  
3. Use EDA Agent to explore.  
4. Add new features via Feature Engineering.  
5. Train models via Model Building (only Sales CSV).  
6. Compare models via Experimentation Agent.  
7. Generate insights via Storytelling Agent.  
8. Save & deploy model with Deployment Agent.  
9. Scale data with Scaling Agent to test performance.  
10. Repeat workflow for other datasets.  
11. COVID & Earthquake are best for EDA + Storytelling.  
12. Sales CSV best for full ML pipeline.  
13. IoT CSV best for anomaly analysis.  
14. Download model.pkl for external use.  
15. This prototype simulates a real Data Scientist.  
""")

# -------------------------------
# Run Agents in Sequence
# -------------------------------
if st.sidebar.button("🚀 Run Agents Pipeline"):
    data = cleaning_agent(data)
    eda_agent(data)
    data = feature_engineering_agent(data)
    model_building_agent(data)
    experimentation_agent(data)
    storytelling_agent(data)
    deployment_agent()
    data = scaling_agent(data)
