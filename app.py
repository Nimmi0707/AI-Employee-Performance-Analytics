
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Employee Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM DASHBOARD STYLE
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #f4f7fb;
    }

    .dashboard-header {
        background: linear-gradient(90deg, #173f70, #1f5f9d);
        padding: 18px 24px;
        border-radius: 10px;
        color: white;
        margin-bottom: 18px;
    }

    .dashboard-header h1 {
        color: white;
        font-size: 30px;
        margin: 0;
    }

    .dashboard-header p {
        color: #e8f1fb;
        margin: 6px 0 0 0;
        font-size: 14px;
    }

    .kpi {
        background: white;
        padding: 14px 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border: 1px solid #e5eaf0;
    }

    .kpi-title {
        color: #52657a;
        font-size: 14px;
        font-weight: 600;
    }

    .kpi-value {
        color: #173f70;
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    [data-testid="stSidebar"] {
        background-color: #eef3f8;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA + MODEL
# =========================================================
dataset_path = Path("employee_performance_dataset.csv")
model_path = Path("employee_performance_model.pkl")

if not dataset_path.exists():
    st.error("employee_performance_dataset.csv was not found.")
    st.stop()

if not model_path.exists():
    st.error("employee_performance_model.pkl was not found.")
    st.stop()

df = pd.read_csv(dataset_path)
model = joblib.load(model_path)

required_columns = [
    "Experience_Years",
    "Training_Hours",
    "Attendance_Percentage",
    "Projects_Completed",
    "Performance_Score"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Required columns are missing: {', '.join(missing)}")
    st.stop()

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.title("🎛️ Dashboard Filters")
st.sidebar.caption("Filter the analytics dashboard")

score_min = float(df["Performance_Score"].min())
score_max = float(df["Performance_Score"].max())

score_range = st.sidebar.slider(
    "Performance Score",
    min_value=score_min,
    max_value=score_max,
    value=(score_min, score_max)
)

exp_min = float(df["Experience_Years"].min())
exp_max = float(df["Experience_Years"].max())

experience_range = st.sidebar.slider(
    "Experience (Years)",
    min_value=exp_min,
    max_value=exp_max,
    value=(exp_min, exp_max)
)

training_min = float(df["Training_Hours"].min())
training_max = float(df["Training_Hours"].max())

training_range = st.sidebar.slider(
    "Training Hours",
    min_value=training_min,
    max_value=training_max,
    value=(training_min, training_max)
)

attendance_range = st.sidebar.slider(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=(
        max(0.0, float(df["Attendance_Percentage"].min())),
        min(100.0, float(df["Attendance_Percentage"].max()))
    )
)

projects_min = int(df["Projects_Completed"].min())
projects_max = int(df["Projects_Completed"].max())

projects_range = st.sidebar.slider(
    "Projects Completed",
    min_value=projects_min,
    max_value=projects_max,
    value=(projects_min, projects_max)
)

filtered = df[
    (df["Performance_Score"].between(score_range[0], score_range[1]))
    & (df["Experience_Years"].between(experience_range[0], experience_range[1]))
    & (df["Training_Hours"].between(training_range[0], training_range[1]))
    & (df["Attendance_Percentage"].between(attendance_range[0], attendance_range[1]))
    & (df["Projects_Completed"].between(projects_range[0], projects_range[1]))
].copy()

if filtered.empty:
    st.warning("No employees match the selected filters.")
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="dashboard-header">
    <h1>📊 EMPLOYEE PERFORMANCE ANALYSIS</h1>
    <p>
        Interactive Machine Learning dashboard for analysing employee performance,
        experience, training, attendance and project completion.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI CARDS
# =========================================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Total Employees</div>'
        f'<div class="kpi-value">{len(filtered):,}</div></div>',
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Avg Performance</div>'
        f'<div class="kpi-value">{filtered["Performance_Score"].mean():.2f}</div></div>',
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Avg Experience</div>'
        f'<div class="kpi-value">{filtered["Experience_Years"].mean():.1f}</div></div>',
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Avg Attendance</div>'
        f'<div class="kpi-value">{filtered["Attendance_Percentage"].mean():.1f}%</div></div>',
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        f'<div class="kpi"><div class="kpi-title">Avg Training Hours</div>'
        f'<div class="kpi-value">{filtered["Training_Hours"].mean():.1f}</div></div>',
        unsafe_allow_html=True
    )

st.write("")

# =========================================================
# PERFORMANCE CATEGORY
# =========================================================
def performance_category(score):
    if score < 60:
        return "Needs Improvement"
    elif score < 75:
        return "Average"
    elif score < 90:
        return "Good"
    return "Excellent"

filtered["Performance_Category"] = filtered["Performance_Score"].apply(
    performance_category
)

# =========================================================
# ROW 1 - DONUT + BAR + DONUT
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    category_counts = (
        filtered["Performance_Category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Employees")
    )

    fig = px.pie(
        category_counts,
        names="Category",
        values="Employees",
        hole=0.55,
        title="Employees by Performance Category"
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    exp_summary = (
        filtered.groupby("Experience_Years", as_index=False)["Performance_Score"]
        .mean()
    )

    fig = px.bar(
        exp_summary,
        x="Experience_Years",
        y="Performance_Score",
        title="Average Performance by Experience",
        labels={
            "Experience_Years": "Experience (Years)",
            "Performance_Score": "Average Performance"
        }
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c3:
    score_bins = pd.cut(
        filtered["Performance_Score"],
        bins=[-float("inf"), 60, 75, 90, float("inf")],
        labels=["< 60", "60–75", "75–90", "90+"]
    )

    score_counts = (
        score_bins.value_counts()
        .sort_index()
        .rename_axis("Score Range")
        .reset_index(name="Employees")
    )

    fig = px.pie(
        score_counts,
        names="Score Range",
        values="Employees",
        hole=0.55,
        title="Performance Score Distribution"
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ROW 2 - SCATTER CHARTS
# =========================================================
c4, c5 = st.columns(2)

with c4:
    fig = px.scatter(
        filtered,
        x="Training_Hours",
        y="Performance_Score",
        size="Projects_Completed",
        hover_data=[
            "Experience_Years",
            "Attendance_Percentage"
        ],
        trendline="ols",
        title="Training Hours vs Performance Score",
        labels={
            "Training_Hours": "Training Hours",
            "Performance_Score": "Performance Score"
        }
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

with c5:
    fig = px.scatter(
        filtered,
        x="Attendance_Percentage",
        y="Performance_Score",
        size="Projects_Completed",
        hover_data=[
            "Experience_Years",
            "Training_Hours"
        ],
        trendline="ols",
        title="Attendance vs Performance Score",
        labels={
            "Attendance_Percentage": "Attendance (%)",
            "Performance_Score": "Performance Score"
        }
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ROW 3 - PROJECTS + EXPERIENCE
# =========================================================
c6, c7 = st.columns(2)

with c6:
    project_summary = (
        filtered.groupby("Projects_Completed", as_index=False)["Performance_Score"]
        .mean()
    )

    fig = px.bar(
        project_summary,
        x="Projects_Completed",
        y="Performance_Score",
        title="Projects Completed vs Average Performance",
        labels={
            "Projects_Completed": "Projects Completed",
            "Performance_Score": "Average Performance"
        }
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with c7:
    fig = px.scatter(
        filtered,
        x="Experience_Years",
        y="Performance_Score",
        color="Performance_Category",
        title="Experience vs Performance",
        labels={
            "Experience_Years": "Experience (Years)",
            "Performance_Score": "Performance Score"
        }
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CORRELATION
# =========================================================
st.subheader("🔗 Feature Correlation")

corr_columns = [
    "Experience_Years",
    "Training_Hours",
    "Attendance_Percentage",
    "Projects_Completed",
    "Performance_Score"
]

corr = filtered[corr_columns].corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    title="Correlation Heatmap"
)
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# AI PREDICTION
# =========================================================
st.divider()
st.subheader("🤖 AI Employee Performance Prediction")
st.write("Enter employee details and click Predict Performance.")

p1, p2, p3, p4 = st.columns(4)

with p1:
    experience = st.number_input(
        "Experience (Years)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0
    )

with p2:
    training = st.number_input(
        "Training Hours",
        min_value=0.0,
        max_value=500.0,
        value=60.0,
        step=5.0
    )

with p3:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

with p4:
    projects = st.number_input(
        "Projects Completed",
        min_value=0,
        max_value=100,
        value=8,
        step=1
    )

if st.button("🚀 Predict Performance", use_container_width=True):
    input_data = pd.DataFrame({
        "Experience_Years": [experience],
        "Training_Hours": [training],
        "Attendance_Percentage": [attendance],
        "Projects_Completed": [projects]
    })

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Performance Score: {prediction:.2f}")

    fig = px.bar(
        x=["Predicted Score"],
        y=[prediction],
        range_y=[0, 100],
        text=[f"{prediction:.2f}"],
        title="AI Predicted Performance"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(input_data, use_container_width=True)

# =========================================================
# DATASET
# =========================================================
with st.expander("📂 View Filtered Dataset"):
    st.dataframe(filtered, use_container_width=True)
