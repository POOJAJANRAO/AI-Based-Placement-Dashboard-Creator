# streamlit_app.py — Power BI‑like College Placement Dashboard (with ML/DL)
# --------------------------------------------------------------
# How to run:
#   1) pip install streamlit pandas numpy plotly scikit-learn
#   2) streamlit run streamlit_app.py
# --------------------------------------------------------------
# Features
#   • Upload or auto-generate sample placement dataset
#   • Power BI–like KPIs, slicers (filters), and interactive charts (Plotly)
#   • Download filtered data
#   • ML: Logistic Regression / MLP (Neural Net) to predict placement probability
#   • "What‑if" panel to simulate candidate profile and get prediction
# --------------------------------------------------------------

import io
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="College Placement Dashboard",
    page_icon="🎓",
    layout="wide",
)

# -----------------------
# Helpers & cache
# -----------------------

@st.cache_data(show_spinner=False)
def generate_sample_data(n_students: int = 800, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    years = np.random.choice(range(2019, datetime.now().year + 1), size=n_students)
    depts = np.random.choice(
        ["CSE", "ECE", "ME", "CE", "EEE", "IT", "AIML"], size=n_students, p=[0.28,0.16,0.14,0.12,0.1,0.12,0.08]
    )
    genders = np.random.choice(["Female", "Male", "Other"], size=n_students, p=[0.42,0.56,0.02])
    cgpa = np.clip(np.random.normal(7.4, 1.1, size=n_students), 4.5, 10.0)

    # Skills bucket
    skills_pool = [
        "Python", "Java", "C++", "SQL", "Data Science", "ML", "DL", "NLP",
        "Computer Vision", "React", "Flutter", "DevOps", "Cloud", "Golang",
    ]
    skills_counts = np.random.choice([2,3,4,5], size=n_students, p=[0.2,0.45,0.25,0.10])
    skills = [", ".join(np.random.choice(skills_pool, size=k, replace=False)) for k in skills_counts]

    companies = np.random.choice([
        "TCS", "Infosys", "Wipro", "Accenture", "Google", "Amazon", "Microsoft",
        "Adobe", "Flipkart", "BYJU'S", "Reliance Jio", "PhonePe", "HCL"
    ], size=n_students, p=[0.13,0.11,0.1,0.1,0.03,0.04,0.03,0.03,0.05,0.03,0.09,0.08,0.18])

    roles = np.random.choice([
        "Software Engineer", "Data Analyst", "Data Scientist", "SDE Intern",
        "ML Engineer", "DevOps Engineer", "System Engineer", "Business Analyst"
    ], size=n_students)

    # Placement probability influenced by CGPA + AIML dept + skills
    base = (cgpa - 5.5) * 0.18
    aiml_boost = (depts == "AIML") * 0.2
    ds_skill = np.array(["Data Science" in s or "ML" in s or "DL" in s for s in skills]) * 0.15
    prob = 1 / (1 + np.exp(-(base + aiml_boost + ds_skill)))
    placed_flag = np.random.binomial(1, np.clip(prob, 0.05, 0.95))

    # CTC for placed only
    ctc = np.where(
        placed_flag == 1,
        np.round(np.random.lognormal(mean=2.0, sigma=0.35, size=n_students) * 1.8, 2),
        np.nan,
    )

    offer_dates = [
        datetime(int(y), int(np.random.choice(range(7, 13))), int(np.random.choice(range(1, 28))))
        for y in years
    ]

    df = pd.DataFrame({
        "StudentID": [f"S{1000+i}" for i in range(n_students)],
        "Year": years,
        "Department": depts,
        "Gender": genders,
        "CGPA": np.round(cgpa, 2),
        "Skills": skills,
        "Company": companies,
        "Role": roles,
        "OfferDate": offer_dates,
        "CTC_LPA": ctc,  # LPA
        "Status": np.where(placed_flag==1, "Placed", "Unplaced"),
    })

    # Make some unplaced rows have no company/role
    df.loc[df["Status"] == "Unplaced", ["Company", "Role", "OfferDate"]] = [np.nan, np.nan, pd.NaT]
    return df


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist (or raise helpful error)
    expected = [
        "StudentID", "Year", "Department", "Gender", "CGPA", "Skills",
        "Company", "Role", "OfferDate", "CTC_LPA", "Status"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nExpected columns: {expected}")

    # Dtypes
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["CGPA"] = pd.to_numeric(df["CGPA"], errors="coerce")
    if not np.issubdtype(df["OfferDate"].dtype, np.datetime64):
        df["OfferDate"] = pd.to_datetime(df["OfferDate"], errors="coerce")
    return df


# -----------------------
# Sidebar — data source
# -----------------------
st.sidebar.title("📁 Data Source")
uploaded = st.sidebar.file_uploader("Upload placement CSV (optional)", type=["csv"])

try:
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df = _coerce_columns(df_raw)
        st.sidebar.success("Dataset loaded ✔")
    else:
        df = generate_sample_data()
        st.sidebar.info("Using auto‑generated sample dataset")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

# -----------------------
# Sidebar — filters (slicers)
# -----------------------
st.sidebar.title("🔎 Filters")
years = sorted([int(y) for y in df["Year"].dropna().unique()])
sel_years = st.sidebar.multiselect("Year", years, default=years)

branches = sorted(df["Department"].dropna().unique())
sel_branches = st.sidebar.multiselect("Department", branches, default=branches)

companies = sorted([c for c in df["Company"].dropna().unique()])
sel_companies = st.sidebar.multiselect("Company (Placed)", companies)

genders = sorted(df["Gender"].dropna().unique())
sel_genders = st.sidebar.multiselect("Gender", genders, default=genders)

min_cgpa, max_cgpa = float(df["CGPA"].min()), float(df["CGPA"].max())
sel_cgpa = st.sidebar.slider("CGPA range", min_cgpa, max_cgpa, (min_cgpa, max_cgpa))

# Apply filters
f = df[
    df["Year"].isin(sel_years)
    & df["Department"].isin(sel_branches)
    & df["Gender"].isin(sel_genders)
    & df["CGPA"].between(sel_cgpa[0], sel_cgpa[1])
]
if sel_companies:
    f = f[(f["Company"].isin(sel_companies)) | (f["Status"] == "Unplaced")]

# -----------------------
# KPI cards (Power BI‑like)
# -----------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

total_students = len(f)
placed = int((f["Status"] == "Placed").sum())
placement_rate = (placed / total_students * 100) if total_students else 0
avg_ctc = f.loc[f["Status"]=="Placed", "CTC_LPA"].mean()
med_ctc = f.loc[f["Status"]=="Placed", "CTC_LPA"].median()
high_ctc = f.loc[f["Status"]=="Placed", "CTC_LPA"].max()

col1.metric("Total Students", f"{total_students}")
col2.metric("Placed", f"{placed}")
col3.metric("Placement Rate", f"{placement_rate:0.1f}%")
col4.metric("Avg CTC (LPA)", f"{(avg_ctc or 0):0.2f}")
col5.metric("Median CTC (LPA)", f"{(med_ctc or 0):0.2f}")
col6.metric("Highest CTC (LPA)", f"{(high_ctc or 0):0.2f}")

st.markdown("---")

# -----------------------
# Visuals
# -----------------------
vc1, vc2 = st.columns(2)

# Placement rate by department
by_dept = (
    f.groupby(["Department", "Status"]).size().reset_index(name="Count")
)
if not by_dept.empty:
    total_by_dept = by_dept.groupby("Department")["Count"].sum().reset_index(name="Total")
    placed_by_dept = by_dept[by_dept["Status"] == "Placed"]["Count"].groupby(by_dept["Department"]).sum()
    rate_df = total_by_dept.merge(placed_by_dept, left_on="Department", right_index=True, how="left").fillna(0)
    rate_df["PlacementRate%"] = (rate_df["Count"] / rate_df["Total"]) * 100
    fig = px.bar(rate_df.sort_values("PlacementRate%", ascending=False), x="Department", y="PlacementRate%",
                 title="Placement Rate by Department", text=rate_df["PlacementRate%"].round(1))
    fig.update_layout(yaxis_title="Rate (%)", xaxis_title="Department")
    vc1.plotly_chart(fig, use_container_width=True)

# Top companies by offers
top_comp = f[f["Status"]=="Placed"].groupby("Company").size().reset_index(name="Offers").sort_values("Offers", ascending=False).head(15)
if not top_comp.empty:
    fig2 = px.bar(top_comp, x="Offers", y="Company", orientation="h", title="Top Hiring Companies (Offers)")
    vc2.plotly_chart(fig2, use_container_width=True)

# Trend: Avg CTC & Placement Rate by Year
st.markdown("\n")
tr1, tr2 = st.columns(2)

trend = f.copy()
if not trend.empty:
    yearly = trend.groupby("Year").agg(
        total=("StudentID", "count"),
        placed=("Status", lambda s: (s=="Placed").sum()),
        avg_ctc=("CTC_LPA", "mean"),
    ).reset_index()
    yearly["placement_rate"] = yearly["placed"] / yearly["total"] * 100

    fig3 = px.line(yearly, x="Year", y="placement_rate", markers=True, title="Placement Rate by Year (%)")
    tr1.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(yearly, x="Year", y="avg_ctc", markers=True, title="Average CTC by Year (LPA)")
    tr2.plotly_chart(fig4, use_container_width=True)

# Box: CTC distribution by Department
box_df = f[f["Status"]=="Placed"]
if not box_df.empty:
    fig5 = px.box(box_df, x="Department", y="CTC_LPA", title="CTC Distribution by Department")
    st.plotly_chart(fig5, use_container_width=True)

# Detailed table + download
st.markdown("---")
st.subheader("📋 Detailed Records (Filtered)")
st.dataframe(f.sort_values(["Year", "Department", "StudentID"]))

csv = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv, file_name="placement_filtered.csv", mime="text/csv")

# -----------------------
# ML / DL: Placement Prediction (What‑if)
# -----------------------
st.markdown("---")
st.header("🧠 Placement Prediction — What‑if Analysis (ML / Neural Net)")

with st.expander("About the model"):
    st.write(
        """
        We train a quick model on the filtered dataset to predict whether a student gets placed.
        Features used: CGPA, Department, Gender, and count of key skills (ML/DS/DL/CV/NLP).
        Choose between Logistic Regression (fast & interpretable) or MLP Neural Net (non‑linear).
        """
    )

model_type = st.radio("Model Type", ["Logistic Regression", "MLP Neural Net"], horizontal=True)

# Prepare features
mldf = f.dropna(subset=["CGPA", "Department", "Gender", "Status"]).copy()
# Skill signal: count of AI/DS related skills
ai_keywords = ["ML", "DL", "NLP", "Computer Vision", "Data Science"]

def count_ai_skills(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return sum(1 for kw in ai_keywords if kw.lower() in s.lower())

mldf["AI_Skill_Count"] = mldf["Skills"].apply(count_ai_skills)

if len(mldf) >= 50 and mldf["Status"].nunique() == 2:
    X = mldf[["CGPA", "Department", "Gender", "AI_Skill_Count"]]
    y = (mldf["Status"] == "Placed").astype(int)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Department", "Gender"]),
        ("num", "passthrough", ["CGPA", "AI_Skill_Count"]),
    ])

    if model_type == "Logistic Regression":
        clf = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000))
        ])
    else:
        clf = Pipeline([
            ("pre", pre),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42))
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc*100:0.1f}%")
    m2.metric("F1‑score", f"{f1:0.2f}")
    m3.metric("Train size", f"{len(X_train)}")

    st.subheader("🔮 What‑if Prediction")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inp_dept = st.selectbox("Department", branches)
    with c2:
        inp_gender = st.selectbox("Gender", genders)
    with c3:
        inp_cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=float(np.nanmax([7.5, mldf["CGPA"].median()])))
    with c4:
        inp_ai_skills = st.slider("AI Skill Count (ML/DL/NLP/CV/DS)", 0, 5, 2)

    if st.button("Predict Placement Probability"):
        sample = pd.DataFrame({
            "Department": [inp_dept],
            "Gender": [inp_gender],
            "CGPA": [inp_cgpa],
            "AI_Skill_Count": [inp_ai_skills],
        })
        prob = float(clf.predict_proba(sample)[0, 1])
        st.success(f"Estimated placement probability: **{prob*100:0.1f}%**")
else:
    st.info("Not enough balanced data after filtering to train the predictor. Try removing some filters or use the full dataset.")

# -----------------------
# Theming tweak (subtle Power BI vibe)
# -----------------------
st.markdown(
    """
    <style>
    .stMetric {border: 1px solid #eee; padding: 8px; border-radius: 12px;}
    .css-zt5igj, .eczjsme2 {font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
