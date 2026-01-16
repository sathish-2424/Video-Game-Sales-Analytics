# =====================================================
# Video Game Sales Analytics
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Video Game Sales Analytics",
    page_icon="🎮",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🎮 Video Game Sales Analytics")

# =====================================================
# DATA LOADING (NO CLEANING)
# =====================================================

@st.cache_data
def load_data():
    ps4_df = pd.read_csv("PS4.csv")
    xbox_df = pd.read_csv("XboxOne.csv")

    ps4_df["platform"] = "PS4"
    xbox_df["platform"] = "Xbox One"

    return pd.concat([ps4_df, xbox_df], ignore_index=True)


df = load_data()

# =====================================================
# KPI METRICS
# =====================================================

st.subheader("📌 Key Business Metrics")

k1, k2, k3 = st.columns(3)

k1.metric("🎯 Total Games", df.shape[0])
k2.metric("💰 Total Global Sales (M)", round(df["Global"].sum(), 2))
k3.metric("📊 Avg Sales per Game (M)", round(df["Global"].mean(), 2))

st.divider()

# =====================================================
# VISUAL ANALYTICS
# =====================================================

st.subheader("📊 Sales Insights")

# Platform-wise sales
platform_sales = df.groupby("platform")["Global"].sum().reset_index()

# Genre-wise sales
genre_sales = (
    df.groupby("Genre")["Global"]
    .sum()
    .reset_index()
    .sort_values("Global", ascending=False)
)

# Year-wise sales
year_sales = df.groupby("Year")["Global"].sum().reset_index()

# Charts
fig1 = px.bar(
    platform_sales,
    x="platform",
    y="Global",
    title="Platform-wise Global Sales",
    labels={"Global": "Sales (Million Units)"}
)

fig2 = px.bar(
    genre_sales,
    x="Global",
    y="Genre",
    orientation="h",
    title="Genre-wise Global Sales"
)

fig3 = px.line(
    year_sales,
    x="Year",
    y="Global",
    title="Year-wise Global Sales Trend",
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
# PS4 COUNTRY SALES ANALYSIS
# =====================================================

ps4_df = df[df["platform"] == "PS4"]

country_sales = pd.DataFrame({
    "Country": ["North America", "Europe", "Japan", "Rest of World"],
    "Sales": [
        ps4_df["North America"].sum(),
        ps4_df["Europe"].sum(),
        ps4_df["Japan"].sum(),
        ps4_df["Rest of World"].sum()
    ]
})

top_country = country_sales.loc[country_sales["Sales"].idxmax()]

st.subheader("🌍 PS4 Sales by Country")

fig_country = px.bar(
    country_sales,
    x="Country",
    y="Sales",
    title="🌎 PS4 Regional Sales Distribution",
    color="Sales",
    color_continuous_scale="Blues",
    text_auto=".2f"
)

fig_country.update_layout(
    xaxis_title="Region",
    yaxis_title="Sales (Million Units)",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_country, use_container_width=True)





# =====================================================
# MODEL TRAINING
# =====================================================

@st.cache_resource
def train_model(data):
    X = data[["platform", "Genre"]]
    y = data["Global"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["platform", "Genre"])
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)
    return model


with st.spinner("Training prediction model..."):
    model = train_model(df)

# =====================================================
# PREDICTION SECTION
# =====================================================

st.subheader("🎯 Sales Prediction Simulator")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        u_platform = st.selectbox(
            "Platform",
            sorted(df["platform"].unique())
        )

    with c2:
        u_genre = st.selectbox(
            "Genre",
            sorted(df["Genre"].unique())
        )

    submit = st.form_submit_button("🔮 Predict Sales")

if submit:
    input_df = pd.DataFrame({
        "platform": [u_platform],
        "Genre": [u_genre]
    })

    prediction = model.predict(input_df)[0]

    st.success(
        f"📈 Predicted Global Sales: **{prediction:.2f} Million Units**"
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
