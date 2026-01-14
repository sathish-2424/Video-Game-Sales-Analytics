# =====================================================
# Multi-Platform Video Game Sales Analytics
# Company-Level Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Video Game Sales Analytics",
    page_icon="🎮",
    layout="wide"
)

# =====================================================
# UI STYLING
# =====================================================

st.markdown("""
<style>
.main { padding: 1.5rem; }
h1, h2 { color: #1f2937; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🎮 Video Game Sales Analytics")

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():

    def standardize(df, platform_name=None):
        cols = {c.lower(): c for c in df.columns}

        def find(keys):
            for k in cols:
                if any(word in k for word in keys):
                    return cols[k]
            return None

        df = df.rename(columns={
            find(["name", "game", "title"]): "name",
            find(["year"]): "year",
            find(["global_sales", "globalsales", "global"]): "global_sales",
            find(["genre"]): "genre",
            find(["publisher"]): "publisher",
            find(["platform"]): "platform",
        })

        if "platform" not in df.columns:
            df["platform"] = platform_name

        return df[[
            "name", "platform", "year",
            "genre", "publisher", "global_sales"
        ]]

    global_df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv", encoding="latin1")
    ps4_df = pd.read_csv("PS4_GamesSales.csv", encoding="latin1")
    xbox_df = pd.read_csv("XboxOne_GameSales.csv", encoding="latin1")

    global_df = standardize(global_df)
    ps4_df = standardize(ps4_df, "PS4")
    xbox_df = standardize(xbox_df, "Xbox One")

    df = pd.concat([global_df, ps4_df, xbox_df], ignore_index=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["age"] = 2021 - df["year"]

    df.dropna(subset=["global_sales"], inplace=True)

    for col in ["platform", "genre", "publisher"]:
        df[col] = df[col].fillna("Unknown")

    return df

df = load_data()

# =====================================================
# KPI METRICS
# =====================================================

st.subheader("📌 Key Business Metrics")

k1, k2, k3 = st.columns(3)
k1.metric("🎯 Total Games", f"{df.shape[0]:,}")
k2.metric("💰 Total Global Sales (M)", f"{df['global_sales'].sum():.2f}")
k3.metric("📊 Avg Sales per Game (M)", f"{df['global_sales'].mean():.2f}")

st.divider()

# =====================================================
# VISUAL ANALYTICS
# =====================================================

st.subheader("📊 Sales Insights")

platform_sales = df.groupby("platform")["global_sales"].sum().reset_index()
genre_sales = (
    df.groupby("genre")["global_sales"]
    .sum()
    .reset_index()
    .sort_values("global_sales", ascending=False)
)
year_sales = df.groupby("year")["global_sales"].sum().reset_index()

fig1 = px.bar(
    platform_sales,
    x="platform",
    y="global_sales",
    title="Platform-wise Global Sales",
    labels={"global_sales": "Sales (Million Units)"}
)

fig2 = px.bar(
    genre_sales,
    x="global_sales",
    y="genre",
    orientation="h",
    title="Genre-wise Global Sales"
)

fig3 = px.line(
    year_sales,
    x="year",
    y="global_sales",
    title="Global Sales Trend Over Time",
    markers=True
)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig1, use_container_width=True)
with c2:
    st.plotly_chart(fig2, use_container_width=True)

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# =====================================================
# MODEL TRAINING (INTERNAL)
# =====================================================

@st.cache_resource
def train_model(data):

    X = data[["platform", "genre", "publisher", "age"]]
    y = data["global_sales"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, ["age"]),
        ("cat", categorical_pipeline, ["platform", "genre", "publisher"])
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)
    return model

with st.spinner("Preparing predictive model..."):
    model = train_model(df)


# =====================================================
# PREDICTION SECTION
# =====================================================

st.subheader("🎯 Sales Prediction Simulator")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        u_platform = st.selectbox("Platform", sorted(df["platform"].unique()))
        u_genre = st.selectbox("Genre", sorted(df["genre"].unique()))

    with c2:
        u_publisher = st.selectbox("Publisher", sorted(df["publisher"].unique()))
        u_age = st.slider("Game Age (Years)", 0, 40, 5)

    submit = st.form_submit_button("🔮 Predict Sales")

if submit:
    input_df = pd.DataFrame([{
        "platform": u_platform,
        "genre": u_genre,
        "publisher": u_publisher,
        "age": u_age
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Global Sales: **{prediction:.2f} Million Units**")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
