"""
AI Employee Performance Analytics System
M.Tech CSE Academic Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Employee Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ── */
.stApp { background: linear-gradient(135deg, #F0F4FF 0%, #F8FAFC 50%, #EEF2FF 100%); }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #312E81 40%, #4338CA 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #E0E7FF !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(79,70,229,0.10);
    border-top: 4px solid;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(79,70,229,0.18); }
.kpi-value   { font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin-bottom: 6px; }
.kpi-label   { font-size: 0.88rem; color: #64748B; font-weight: 500; letter-spacing: 0.03em; text-transform: uppercase; }
.kpi-delta   { font-size: 0.8rem; font-weight: 600; margin-top: 6px; }
.kpi-icon    { font-size: 1.8rem; margin-bottom: 8px; }

/* ── Section headings ── */
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #1E1B4B;
    border-left: 4px solid #4F46E5; padding-left: 12px;
    margin: 20px 0 12px 0;
}

/* ── Chart card ── */
.chart-card {
    background: white; border-radius: 16px; padding: 20px;
    box-shadow: 0 2px 12px rgba(79,70,229,0.08);
    margin-bottom: 16px;
}

/* ── Page title banner ── */
.page-title {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    box-shadow: 0 6px 24px rgba(79,70,229,0.30);
}
.page-title h1 { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.page-title p  { font-size: 1rem; margin: 6px 0 0 0; opacity: 0.88; }

/* ── Employee detail card ── */
.emp-card {
    background: white; border-radius: 16px; padding: 24px;
    box-shadow: 0 2px 16px rgba(79,70,229,0.10); margin-bottom: 16px;
}

/* ── Badge ── */
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.80rem; font-weight: 700; letter-spacing: 0.04em;
}
.badge-high   { background: #DCFCE7; color: #166534; }
.badge-medium { background: #FEF9C3; color: #854D0E; }
.badge-low    { background: #FEE2E2; color: #991B1B; }

/* ── Prediction result card ── */
.pred-result {
    border-radius: 16px; padding: 28px; text-align: center;
    font-size: 2rem; font-weight: 800; margin: 12px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.pred-high   { background: linear-gradient(135deg,#DCFCE7,#BBF7D0); color:#166534; border: 2px solid #86EFAC; }
.pred-medium { background: linear-gradient(135deg,#FEF9C3,#FEF08A); color:#854D0E; border: 2px solid #FDE047; }
.pred-low    { background: linear-gradient(135deg,#FEE2E2,#FECACA); color:#991B1B; border: 2px solid #FCA5A5; }

/* ── Info table ── */
.info-row { display: flex; justify-content: space-between; padding: 8px 0;
            border-bottom: 1px solid #F1F5F9; }
.info-key { color: #64748B; font-size: 0.88rem; font-weight: 500; }
.info-val { color: #1E293B; font-size: 0.88rem; font-weight: 600; }

/* ── Metric strip ── */
.metric-strip {
    background: white; border-radius: 12px; padding: 14px 20px;
    display: flex; align-items: center; gap: 12px;
    box-shadow: 0 1px 6px rgba(79,70,229,0.07); margin-bottom: 10px;
}

/* ── About card ── */
.about-card {
    background: white; border-radius: 16px; padding: 28px 32px;
    box-shadow: 0 2px 12px rgba(79,70,229,0.08); margin-bottom: 20px;
}
.about-card h3 { color: #4F46E5; font-size: 1.1rem; font-weight: 700; margin-bottom: 10px; }

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] {
    background: white; border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 8px rgba(79,70,229,0.07);
}
.stButton > button {
    background: linear-gradient(135deg,#4F46E5,#7C3AED) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 28px !important;
    box-shadow: 0 3px 12px rgba(79,70,229,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important; border-color: #C7D2FE !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib theme ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PALETTE = {
    "High":   "#4F46E5",
    "Medium": "#F59E0B",
    "Low":    "#EF4444",
}
DEPT_COLORS = ["#4F46E5", "#7C3AED", "#06B6D4", "#10B981",
               "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6"]

# ─── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "employees.csv")
    df = pd.read_csv(path)
    return df

# ─── ML model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def build_model(df):
    features = ["age", "years_experience", "salary", "attendance_rate",
                "projects_completed", "training_hours", "peer_rating",
                "manager_rating", "customer_satisfaction", "overtime_hours"]
    target = "performance_label"

    le = LabelEncoder()
    y = le.fit_transform(df[target])
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)

    gb  = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)

    # Ensemble (soft voting via averaged probas)
    rf_proba = rf.predict_proba(X_test)
    gb_proba = gb.predict_proba(X_test)
    avg_proba = (rf_proba + gb_proba) / 2
    y_pred_ens = np.argmax(avg_proba, axis=1)

    acc = accuracy_score(y_test, y_pred_ens)
    f1  = f1_score(y_test, y_pred_ens, average="weighted")
    cm  = confusion_matrix(y_test, y_pred_ens)
    cr  = classification_report(y_test, y_pred_ens, target_names=le.classes_, output_dict=True)
    cv  = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")

    return dict(
        rf=rf, gb=gb, le=le, scaler=scaler, features=features,
        acc=acc, f1=f1, cm=cm, cr=cr, cv=cv,
        X_test=X_test, y_test=y_test, y_pred=y_pred_ens,
        feature_importances=rf.feature_importances_,
    )

# ─── Helper: render KPI card ─────────────────────────────────────────────────
def kpi_card(icon, label, value, color, delta=""):
    delta_html = f'<div class="kpi-delta" style="color:{color};">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:{color};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

# ─── Helper: section title ───────────────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px;">
            <div style="font-size:2.8rem;">📊</div>
            <div style="font-size:1.1rem;font-weight:800;color:#E0E7FF;letter-spacing:-0.3px;">
                AI Performance<br>Analytics
            </div>
            <div style="font-size:0.72rem;color:#A5B4FC;margin-top:4px;font-weight:500;">
                M.Tech CSE Academic Project
            </div>
        </div>
        <hr style="border-color:#4338CA;margin:10px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠  Dashboard", "👤  Employee Analytics",
             "🏢  Department Analytics", "🤖  ML Prediction",
             "ℹ️  About Project"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#4338CA;margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#A5B4FC;padding:0 4px;">
        <b style="color:#C7D2FE;">Quick Stats</b>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="color:#C7D2FE;font-size:0.82rem;padding:6px 4px;line-height:2;">
            👥 Total Employees: <b style="color:white;">{len(df)}</b><br>
            🏢 Departments: <b style="color:white;">{df['department'].nunique()}</b><br>
            🏅 Avg Score: <b style="color:white;">{df['performance_score'].mean():.1f}</b><br>
            ⭐ High Performers: <b style="color:white;">{(df['performance_label']=='High').sum()}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#4338CA;margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem;color:#818CF8;text-align:center;">
            Built with ❤️ using Streamlit + ML<br>
            © 2024 Academic Project
        </div>""", unsafe_allow_html=True)

    return page.split("  ", 1)[-1].strip()

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def page_dashboard(df, model):
    st.markdown("""
    <div class="page-title">
        <h1>📊 Executive Dashboard</h1>
        <p>Real-time overview of employee performance across the organisation</p>
    </div>""", unsafe_allow_html=True)

    # ── KPIs ──
    total = len(df)
    avg_score = df["performance_score"].mean()
    high_pct = (df["performance_label"] == "High").sum() / total * 100
    avg_attend = df["attendance_rate"].mean()
    avg_salary = df["salary"].mean()
    high_count = (df["performance_label"] == "High").sum()
    med_count  = (df["performance_label"] == "Medium").sum()
    low_count  = (df["performance_label"] == "Low").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("👥", "Total Employees", total, "#4F46E5")
    with c2: kpi_card("🏅", "Avg Performance Score", f"{avg_score:.1f}", "#10B981", "out of 100")
    with c3: kpi_card("⭐", "High Performers", f"{high_pct:.1f}%", "#7C3AED", f"{high_count} employees")
    with c4: kpi_card("🕐", "Avg Attendance Rate", f"{avg_attend:.1f}%", "#06B6D4")
    with c5: kpi_card("💰", "Avg Salary", f"₹{avg_salary/1000:.0f}K", "#F59E0B")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Pie + Bar side by side ──
    section("Performance Distribution & Department Overview")
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        label_counts = df["performance_label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        colors_pie = [PALETTE[l] for l in label_counts.index]
        wedges, texts, autotexts = ax.pie(
            label_counts.values, labels=label_counts.index,
            autopct="%1.1f%%", colors=colors_pie,
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for t in texts: t.set_fontsize(11); t.set_fontweight("600")
        for at in autotexts: at.set_fontsize(10); at.set_color("white"); at.set_fontweight("700")
        ax.set_title("Performance Label Distribution", fontsize=13, fontweight="bold",
                     color="#1E1B4B", pad=12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        dept_avg = (df.groupby("department")["performance_score"]
                      .mean().sort_values(ascending=True))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(dept_avg.index, dept_avg.values,
                       color=DEPT_COLORS[:len(dept_avg)], edgecolor="white",
                       linewidth=1.2, height=0.65)
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}", va="center", fontsize=9.5,
                    fontweight="600", color="#374151")
        ax.set_xlim(0, 105)
        ax.set_xlabel("Average Performance Score", fontsize=10, color="#64748B")
        ax.set_title("Avg Performance Score by Department", fontsize=13,
                     fontweight="bold", color="#1E1B4B", pad=10)
        ax.tick_params(axis="y", labelsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Stacked bar (label per dept) + Score distribution ──
    col3, col4 = st.columns([1.6, 1])

    with col3:
        section("Performance Labels per Department")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        pivot = (df.groupby(["department", "performance_label"])
                   .size().unstack(fill_value=0))
        for lbl in ["High", "Medium", "Low"]:
            if lbl not in pivot.columns:
                pivot[lbl] = 0
        pivot = pivot[["High", "Medium", "Low"]]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        bottom = np.zeros(len(pivot))
        for lbl, clr in PALETTE.items():
            ax.bar(pivot.index, pivot[lbl].values, label=lbl,
                   bottom=bottom, color=clr, edgecolor="white", linewidth=0.8)
            bottom += pivot[lbl].values
        ax.set_ylabel("Number of Employees", fontsize=10, color="#64748B")
        ax.tick_params(axis="x", rotation=20, labelsize=9.5)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        section("Score Distribution")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.4))
        ax.hist(df["performance_score"], bins=18, color="#4F46E5",
                edgecolor="white", linewidth=0.8, alpha=0.87)
        ax.axvline(avg_score, color="#EF4444", linewidth=2, linestyle="--",
                   label=f"Mean: {avg_score:.1f}")
        ax.set_xlabel("Performance Score", fontsize=10, color="#64748B")
        ax.set_ylabel("Count", fontsize=10, color="#64748B")
        ax.set_title("Score Histogram", fontsize=13, fontweight="bold",
                     color="#1E1B4B", pad=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Salary vs Score scatter + Top Performers Table ──
    col5, col6 = st.columns([1, 1])

    with col5:
        section("Salary vs. Performance Score")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for lbl, clr in PALETTE.items():
            sub = df[df["performance_label"] == lbl]
            ax.scatter(sub["salary"] / 1000, sub["performance_score"],
                       c=clr, label=lbl, alpha=0.75, s=55, edgecolors="white",
                       linewidths=0.5)
        z = np.polyfit(df["salary"] / 1000, df["performance_score"], 1)
        p = np.poly1d(z)
        xs = np.linspace(df["salary"].min() / 1000, df["salary"].max() / 1000, 200)
        ax.plot(xs, p(xs), "k--", linewidth=1.2, alpha=0.5, label="Trend")
        ax.set_xlabel("Salary (₹K)", fontsize=10, color="#64748B")
        ax.set_ylabel("Performance Score", fontsize=10, color="#64748B")
        ax.set_title("Salary vs. Performance", fontsize=13, fontweight="bold",
                     color="#1E1B4B", pad=10)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        section("🏆 Top 10 Performers")
        top10 = (df.sort_values("performance_score", ascending=False)
                   .head(10)[["name", "department", "position",
                               "performance_score", "performance_label"]]
                   .reset_index(drop=True))
        top10.index = top10.index + 1

        def color_label(val):
            clr = {"High": "#166534", "Medium": "#854D0E", "Low": "#991B1B"}.get(val, "black")
            bg  = {"High": "#DCFCE7", "Medium": "#FEF9C3", "Low": "#FEE2E2"}.get(val, "white")
            return f"background-color:{bg};color:{clr};font-weight:700;border-radius:8px;"

        styled = top10.style.applymap(color_label, subset=["performance_label"])
        st.dataframe(styled, use_container_width=True, height=340)

    # ── Row 4: Heatmap correlation ──
    section("Feature Correlation Heatmap")
    num_cols = ["age", "years_experience", "salary", "performance_score",
                "attendance_rate", "projects_completed", "training_hours",
                "peer_rating", "manager_rating", "overtime_hours"]
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8.5},
                cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Matrix — Key Performance Features",
                 fontsize=13, fontweight="bold", color="#1E1B4B", pad=12)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: EMPLOYEE ANALYTICS
# ═══════════════════════════════════════════════════════════════════
def page_employee(df):
    st.markdown("""
    <div class="page-title">
        <h1>👤 Employee Analytics</h1>
        <p>Search, filter, and explore individual employee performance profiles</p>
    </div>""", unsafe_allow_html=True)

    # ── Filters ──
    section("Search & Filter Employees")
    fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1.5, 1.5])
    with fc1:
        search = st.text_input("🔍 Search by Name or Employee ID", placeholder="e.g. Arjun or E001")
    with fc2:
        dept_opts = ["All"] + sorted(df["department"].unique().tolist())
        dept_f = st.selectbox("Department", dept_opts)
    with fc3:
        label_opts = ["All", "High", "Medium", "Low"]
        label_f = st.selectbox("Performance Label", label_opts)
    with fc4:
        gender_opts = ["All", "Male", "Female"]
        gender_f = st.selectbox("Gender", gender_opts)

    sc1, sc2 = st.columns(2)
    with sc1:
        score_range = st.slider("Performance Score Range", 0, 100, (0, 100))
    with sc2:
        exp_range = st.slider("Years of Experience", 0, 30, (0, 30))

    filtered = df.copy()
    if search:
        q = search.lower()
        filtered = filtered[
            filtered["name"].str.lower().str.contains(q) |
            filtered["employee_id"].str.lower().str.contains(q)
        ]
    if dept_f != "All":
        filtered = filtered[filtered["department"] == dept_f]
    if label_f != "All":
        filtered = filtered[filtered["performance_label"] == label_f]
    if gender_f != "All":
        filtered = filtered[filtered["gender"] == gender_f]
    filtered = filtered[
        (filtered["performance_score"] >= score_range[0]) &
        (filtered["performance_score"] <= score_range[1]) &
        (filtered["years_experience"] >= exp_range[0]) &
        (filtered["years_experience"] <= exp_range[1])
    ]

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Employees Found", len(filtered))
    rc2.metric("Avg Score", f"{filtered['performance_score'].mean():.1f}" if len(filtered) else "N/A")
    rc3.metric("High Performers", (filtered["performance_label"] == "High").sum() if len(filtered) else 0)

    # ── Table ──
    section(f"Employee Records  ({len(filtered)} results)")
    display_cols = ["employee_id", "name", "age", "gender", "department",
                    "position", "years_experience", "salary",
                    "performance_score", "attendance_rate", "performance_label"]
    if len(filtered) == 0:
        st.info("No employees match the current filters.")
        return

    def row_style(val):
        color_map = {"High": "#DCFCE7", "Medium": "#FEF9C3", "Low": "#FEE2E2"}
        return f"background-color:{color_map.get(val,'white')};font-weight:700;" if val in color_map else ""

    styled_df = filtered[display_cols].reset_index(drop=True)
    styled_df.index += 1
    st.dataframe(
        styled_df.style.applymap(row_style, subset=["performance_label"]),
        use_container_width=True, height=340,
    )

    # ── Employee Detail ──
    section("Employee Detail View")
    emp_names = filtered["name"].tolist()
    sel_emp = st.selectbox("Select an employee to view full profile", emp_names)
    if sel_emp:
        emp = filtered[filtered["name"] == sel_emp].iloc[0]
        lbl = emp["performance_label"]
        badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(lbl, "")

        dc1, dc2, dc3 = st.columns([1.6, 1, 1])
        with dc1:
            st.markdown(f"""
            <div class="emp-card">
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
                    <div style="width:64px;height:64px;border-radius:50%;
                        background:linear-gradient(135deg,#4F46E5,#7C3AED);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.6rem;color:white;font-weight:800;">
                        {emp['name'][0]}
                    </div>
                    <div>
                        <div style="font-size:1.25rem;font-weight:800;color:#1E1B4B;">
                            {emp['name']}
                        </div>
                        <div style="color:#64748B;font-size:0.9rem;">{emp['position']}</div>
                        <span class="badge {badge_cls}">{lbl} Performer</span>
                    </div>
                </div>
                <div class="info-row"><span class="info-key">Employee ID</span>
                    <span class="info-val">{emp['employee_id']}</span></div>
                <div class="info-row"><span class="info-key">Department</span>
                    <span class="info-val">{emp['department']}</span></div>
                <div class="info-row"><span class="info-key">Age</span>
                    <span class="info-val">{emp['age']} years</span></div>
                <div class="info-row"><span class="info-key">Gender</span>
                    <span class="info-val">{emp['gender']}</span></div>
                <div class="info-row"><span class="info-key">Experience</span>
                    <span class="info-val">{emp['years_experience']} years</span></div>
                <div class="info-row"><span class="info-key">Salary</span>
                    <span class="info-val">₹{emp['salary']:,}</span></div>
            </div>""", unsafe_allow_html=True)

        with dc2:
            st.markdown('<div class="emp-card" style="height:100%;">', unsafe_allow_html=True)
            metrics = [
                ("🏅", "Performance Score", f"{emp['performance_score']:.0f}/100"),
                ("🕐", "Attendance Rate",   f"{emp['attendance_rate']:.1f}%"),
                ("📁", "Projects Done",     str(int(emp['projects_completed']))),
                ("📚", "Training Hours",    f"{int(emp['training_hours'])} hrs"),
                ("⏰", "Overtime Hours",    f"{int(emp['overtime_hours'])} hrs"),
            ]
            rows = "".join([
                f'<div class="info-row"><span class="info-key">{ic} {k}</span>'
                f'<span class="info-val">{v}</span></div>'
                for ic, k, v in metrics])
            st.markdown(
                f'<div style="font-weight:700;color:#4F46E5;margin-bottom:12px;">📊 KPIs</div>{rows}',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with dc3:
            st.markdown('<div class="emp-card" style="height:100%;">', unsafe_allow_html=True)
            ratings = [
                ("⭐", "Peer Rating",     emp['peer_rating']),
                ("👔", "Manager Rating",  emp['manager_rating']),
                ("😊", "Customer Sat.",   emp['customer_satisfaction']),
            ]
            rows2 = "".join([
                f'<div class="info-row"><span class="info-key">{ic} {k}</span>'
                f'<span class="info-val">{"★" * int(round(v))} ({v:.1f})</span></div>'
                for ic, k, v in ratings])
            st.markdown(
                f'<div style="font-weight:700;color:#4F46E5;margin-bottom:12px;">🌟 Ratings</div>{rows2}',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Radar-style bar chart for employee ──
        section("Performance Radar (Normalised Scores)")
        rad_cols = ["performance_score", "attendance_rate", "peer_rating",
                    "manager_rating", "customer_satisfaction", "training_hours"]
        rad_labels = ["Perf Score", "Attendance", "Peer Rating",
                      "Mgr Rating", "Cust Sat", "Training Hrs"]

        # Normalize each feature to 0–100 scale
        maxes = {"performance_score": 100, "attendance_rate": 100,
                 "peer_rating": 5, "manager_rating": 5,
                 "customer_satisfaction": 5, "training_hours": 95}
        norm_vals = [emp[c] / maxes[c] * 100 for c in rad_cols]

        fig, ax = plt.subplots(figsize=(9, 2.8))
        colors_bar = ["#4F46E5" if v >= 80 else "#F59E0B" if v >= 60 else "#EF4444"
                      for v in norm_vals]
        bars = ax.barh(rad_labels, norm_vals, color=colors_bar,
                       edgecolor="white", linewidth=0.8, height=0.55)
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.0f}%", va="center",
                    fontsize=9.5, fontweight="600", color="#374151")
        ax.set_xlim(0, 112)
        ax.set_xlabel("Normalised Score (%)", fontsize=9, color="#64748B")
        patches = [
            mpatches.Patch(color="#4F46E5", label="Strong (≥80%)"),
            mpatches.Patch(color="#F59E0B", label="Average (60–79%)"),
            mpatches.Patch(color="#EF4444", label="Needs Improvement (<60%)"),
        ]
        ax.legend(handles=patches, fontsize=8.5, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: DEPARTMENT ANALYTICS
# ═══════════════════════════════════════════════════════════════════
def page_department(df):
    st.markdown("""
    <div class="page-title">
        <h1>🏢 Department Analytics</h1>
        <p>Deep-dive into department-level performance, salary and workforce composition</p>
    </div>""", unsafe_allow_html=True)

    departments = sorted(df["department"].unique().tolist())
    sel_dept = st.selectbox("Select Department", ["All Departments"] + departments)
    ddf = df if sel_dept == "All Departments" else df[df["department"] == sel_dept]

    # ── KPIs for selected dept ──
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("👥", "Employees", len(ddf), "#4F46E5")
    with k2: kpi_card("🏅", "Avg Perf Score", f"{ddf['performance_score'].mean():.1f}", "#10B981")
    with k3: kpi_card("💰", "Avg Salary", f"₹{ddf['salary'].mean()/1000:.0f}K", "#F59E0B")
    with k4: kpi_card("🕐", "Avg Attendance", f"{ddf['attendance_rate'].mean():.1f}%", "#06B6D4")

    st.markdown("<br>", unsafe_allow_html=True)

    if sel_dept == "All Departments":
        # ── Multi-dept comparison charts ──
        section("Department Comparison — Key Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            dept_metrics = df.groupby("department").agg({
                "performance_score": "mean",
                "salary": "mean",
                "attendance_rate": "mean",
                "projects_completed": "mean",
            }).round(1)
            fig, ax = plt.subplots(figsize=(6, 4.8))
            x = np.arange(len(dept_metrics))
            w = 0.4
            ax.bar(x, dept_metrics["performance_score"], w, label="Avg Perf Score",
                   color="#4F46E5", edgecolor="white")
            ax.bar(x + w, dept_metrics["attendance_rate"], w, label="Avg Attendance %",
                   color="#06B6D4", edgecolor="white")
            ax.set_xticks(x + w / 2)
            ax.set_xticklabels(dept_metrics.index, rotation=20, ha="right", fontsize=9)
            ax.legend(fontsize=9)
            ax.set_title("Performance Score vs. Attendance Rate", fontsize=12,
                         fontweight="bold", color="#1E1B4B", pad=10)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            dept_salary = df.groupby("department")["salary"].agg(["mean", "min", "max"]).reset_index()
            dept_salary.columns = ["department", "mean", "min", "max"]
            dept_salary = dept_salary.sort_values("mean", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4.8))
            ax.barh(dept_salary["department"], dept_salary["mean"] / 1000,
                    color=DEPT_COLORS[:len(dept_salary)], edgecolor="white", height=0.65)
            for i, row in dept_salary.iterrows():
                ax.text(row["mean"] / 1000 + 0.5,
                        list(dept_salary["department"]).index(row["department"]),
                        f"₹{row['mean']/1000:.0f}K", va="center",
                        fontsize=9, fontweight="600", color="#374151")
            ax.set_xlabel("Average Salary (₹K)", fontsize=10, color="#64748B")
            ax.set_title("Average Salary by Department", fontsize=12,
                         fontweight="bold", color="#1E1B4B", pad=10)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Heatmap of dept averages ──
        section("Department Feature Heatmap")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        heat_cols = ["performance_score", "attendance_rate", "projects_completed",
                     "training_hours", "peer_rating", "manager_rating", "overtime_hours"]
        heat_data = df.groupby("department")[heat_cols].mean().round(1)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Department-wise Average Feature Values",
                     fontsize=12, fontweight="bold", color="#1E1B4B", pad=12)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Single department deep-dive ──
        col1, col2 = st.columns(2)

        with col1:
            section(f"Gender Distribution — {sel_dept}")
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            gender_counts = ddf["gender"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(gender_counts.values, labels=gender_counts.index,
                   autopct="%1.1f%%", colors=["#4F46E5", "#EC4899"],
                   startangle=90,
                   wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2))
            ax.set_title(f"Gender Split in {sel_dept}", fontsize=12,
                         fontweight="bold", color="#1E1B4B")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            section(f"Performance Labels — {sel_dept}")
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            lbl_counts = ddf["performance_label"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_pie = [PALETTE.get(l, "#999") for l in lbl_counts.index]
            ax.pie(lbl_counts.values, labels=lbl_counts.index,
                   autopct="%1.1f%%", colors=colors_pie,
                   startangle=140,
                   wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2))
            ax.set_title(f"Performance Distribution in {sel_dept}", fontsize=12,
                         fontweight="bold", color="#1E1B4B")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Score by position ──
        section(f"Performance Score by Position — {sel_dept}")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        pos_avg = ddf.groupby("position")["performance_score"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(3, len(pos_avg) * 0.52)))
        ax.barh(pos_avg.index, pos_avg.values,
                color=DEPT_COLORS[:len(pos_avg)], edgecolor="white",
                linewidth=0.8, height=0.65)
        for i, v in enumerate(pos_avg.values):
            ax.text(v + 0.3, i, f"{v:.1f}", va="center",
                    fontsize=9.5, fontweight="600", color="#374151")
        ax.set_xlabel("Average Performance Score", fontsize=10, color="#64748B")
        ax.set_xlim(0, 108)
        ax.set_title(f"Performance Score by Position in {sel_dept}",
                     fontsize=12, fontweight="bold", color="#1E1B4B", pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Employee list for dept ──
        section(f"Employee Roster — {sel_dept}")
        st.dataframe(
            ddf[["employee_id", "name", "position", "years_experience",
                 "salary", "performance_score", "performance_label"]]
              .reset_index(drop=True),
            use_container_width=True, height=300,
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE 4: ML PREDICTION
# ═══════════════════════════════════════════════════════════════════
def page_ml(df, model):
    st.markdown("""
    <div class="page-title">
        <h1>🤖 ML Performance Prediction</h1>
        <p>Ensemble Random Forest + Gradient Boosting — predicts High / Medium / Low performance</p>
    </div>""", unsafe_allow_html=True)

    # ── Model metrics banner ──
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: kpi_card("🎯", "Model Accuracy", f"{model['acc']*100:.1f}%", "#4F46E5")
    with mc2: kpi_card("📐", "Weighted F1 Score", f"{model['f1']*100:.1f}%", "#10B981")
    with mc3: kpi_card("🔁", "5-Fold CV Accuracy", f"{model['cv'].mean()*100:.1f}%", "#7C3AED")
    with mc4: kpi_card("📊", "Algorithm", "Ensemble RF+GB", "#06B6D4")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction form ──
    section("🔮 Predict Employee Performance")
    st.markdown("""
    <div style="background:#EEF2FF;border-radius:12px;padding:14px 18px;margin-bottom:16px;
        border-left:4px solid #4F46E5;">
        <b style="color:#4F46E5;">How it works:</b> Enter employee attributes below and
        the ensemble ML model will predict their performance tier with confidence scores.
    </div>""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            age = st.number_input("Age", 20, 65, 32)
            salary = st.number_input("Annual Salary (₹)", 30000, 300000, 75000, step=5000)
            projects = st.number_input("Projects Completed", 0, 30, 10)
            peer_rating = st.slider("Peer Rating (1–5)", 1.0, 5.0, 4.0, 0.1)
        with p2:
            exp = st.number_input("Years of Experience", 0, 35, 8)
            attend = st.slider("Attendance Rate (%)", 70.0, 100.0, 93.0, 0.5)
            training = st.number_input("Training Hours", 0, 100, 40)
            mgr_rating = st.slider("Manager Rating (1–5)", 1.0, 5.0, 4.0, 0.1)
        with p3:
            overtime = st.number_input("Overtime Hours (monthly)", 0, 40, 8)
            cust_sat = st.slider("Customer Satisfaction (1–5)", 1.0, 5.0, 4.0, 0.1)
            dept = st.selectbox("Department", sorted(df["department"].unique()))
            gender = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("🚀  Predict Performance", use_container_width=True)

    if submitted:
        inp = np.array([[age, exp, salary, attend, projects,
                         training, peer_rating, mgr_rating, cust_sat, overtime]])
        inp_scaled = model["scaler"].transform(inp)

        rf_prob = model["rf"].predict_proba(inp_scaled)[0]
        gb_prob = model["gb"].predict_proba(inp_scaled)[0]
        avg_prob = (rf_prob + gb_prob) / 2
        pred_idx = np.argmax(avg_prob)
        pred_label = model["le"].inverse_transform([pred_idx])[0]
        confidence = avg_prob[pred_idx] * 100

        cls_map = {"High": "pred-high", "Medium": "pred-medium", "Low": "pred-low"}
        emoji_map = {"High": "🏆", "Medium": "✅", "Low": "⚠️"}

        st.markdown(f"""
        <div class="pred-result {cls_map[pred_label]}">
            {emoji_map[pred_label]}  Predicted Performance: {pred_label.upper()}
            <div style="font-size:1rem;font-weight:600;margin-top:8px;opacity:0.8;">
                Confidence: {confidence:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

        # Probability breakdown
        section("Prediction Confidence Breakdown")
        classes = model["le"].classes_
        fig, ax = plt.subplots(figsize=(6, 2.8))
        colors_p = [PALETTE[c] for c in classes]
        bars = ax.barh(classes, avg_prob * 100, color=colors_p,
                       edgecolor="white", height=0.5)
        for bar in bars:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center",
                    fontsize=10, fontweight="700")
        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)", fontsize=10, color="#64748B")
        ax.set_title("Class Probability from Ensemble Model",
                     fontsize=11, fontweight="bold", color="#1E1B4B")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Advice
        if pred_label == "High":
            st.success("✅ **High Performer Detected!** Consider fast-track promotion, mentorship, and leadership roles.")
        elif pred_label == "Medium":
            st.warning("🔄 **Average Performer.** Targeted coaching, additional training, and clear KPI-setting recommended.")
        else:
            st.error("⚠️ **Needs Improvement.** Recommend performance improvement plan (PIP), closer supervision, and skill development.")

    # ── Model Insights ──
    st.markdown("<hr style='margin:24px 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)
    section("📈 Model Insights & Diagnostics")

    mi_col1, mi_col2 = st.columns(2)

    with mi_col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        imp = model["feature_importances"]
        feat_names = [f.replace("_", " ").title() for f in model["features"]]
        idx = np.argsort(imp)
        ax.barh([feat_names[i] for i in idx], imp[idx],
                color=["#4F46E5" if imp[i] > np.median(imp) else "#A5B4FC" for i in idx],
                edgecolor="white")
        ax.set_xlabel("Importance Score", fontsize=10, color="#64748B")
        ax.set_title("Feature Importances (Random Forest)",
                     fontsize=12, fontweight="bold", color="#1E1B4B", pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with mi_col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        classes = model["le"].classes_
        cm = model["cm"]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.5, cbar=False,
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_xlabel("Predicted Label", fontsize=10, color="#64748B")
        ax.set_ylabel("True Label", fontsize=10, color="#64748B")
        ax.set_title("Confusion Matrix (Test Set)",
                     fontsize=12, fontweight="bold", color="#1E1B4B", pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-class metrics ──
    section("Per-Class Classification Report")
    cr = model["cr"]
    classes = model["le"].classes_
    report_rows = []
    for cls in classes:
        report_rows.append({
            "Class": cls,
            "Precision": f"{cr[cls]['precision']:.3f}",
            "Recall":    f"{cr[cls]['recall']:.3f}",
            "F1-Score":  f"{cr[cls]['f1-score']:.3f}",
            "Support":   int(cr[cls]["support"]),
        })
    st.dataframe(pd.DataFrame(report_rows).set_index("Class"),
                 use_container_width=True)

    # ── CV scores ──
    section("5-Fold Cross-Validation Accuracy")
    cv = model["cv"]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.bar(range(1, 6), cv * 100, color="#4F46E5", edgecolor="white", width=0.55)
    for i, v in enumerate(cv):
        ax.text(i + 1, v * 100 + 0.3, f"{v*100:.1f}%", ha="center",
                fontsize=10, fontweight="700", color="#1E1B4B")
    ax.axhline(cv.mean() * 100, color="#EF4444", linewidth=1.8, linestyle="--",
               label=f"Mean: {cv.mean()*100:.1f}%")
    ax.set_ylim(50, 105)
    ax.set_xlabel("Fold", fontsize=10, color="#64748B")
    ax.set_ylabel("Accuracy (%)", fontsize=10, color="#64748B")
    ax.set_title("Cross-Validation Fold Accuracy", fontsize=12,
                 fontweight="bold", color="#1E1B4B", pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# PAGE 5: ABOUT
# ═══════════════════════════════════════════════════════════════════
def page_about(df):
    st.markdown("""
    <div class="page-title">
        <h1>ℹ️ About This Project</h1>
        <p>M.Tech CSE Academic Project — AI-Powered Employee Performance Analytics System</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Project Overview</h3>
            <p style="color:#475569;line-height:1.8;">
                This system provides an end-to-end AI-powered analytics platform for HR teams and
                management to monitor, analyse, and predict employee performance. It combines
                exploratory data analysis with machine learning to deliver actionable insights
                at both individual and organisational levels.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🛠️ Technology Stack</h3>
            <table style="width:100%;border-collapse:collapse;">
                <tr style="background:#F8FAFC;">
                    <td style="padding:8px;font-weight:600;color:#4F46E5;width:40%;">Layer</td>
                    <td style="padding:8px;font-weight:600;color:#4F46E5;">Technology</td>
                </tr>
                <tr><td style="padding:8px;color:#64748B;">Frontend / UI</td>
                    <td style="padding:8px;">Streamlit 1.40</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:8px;color:#64748B;">Data Processing</td>
                    <td style="padding:8px;">Pandas, NumPy</td></tr>
                <tr><td style="padding:8px;color:#64748B;">Machine Learning</td>
                    <td style="padding:8px;">Scikit-learn (RF + GB Ensemble)</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:8px;color:#64748B;">Visualisation</td>
                    <td style="padding:8px;">Matplotlib, Seaborn, Plotly</td></tr>
                <tr><td style="padding:8px;color:#64748B;">Language</td>
                    <td style="padding:8px;">Python 3.13</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:8px;color:#64748B;">Dataset</td>
                    <td style="padding:8px;">150 synthetic employees (CSV)</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🤖 Machine Learning Approach</h3>
            <ul style="color:#475569;line-height:2;">
                <li><b>Algorithm:</b> Ensemble of Random Forest + Gradient Boosting</li>
                <li><b>Target:</b> 3-class classification — High / Medium / Low</li>
                <li><b>Features:</b> 10 numeric attributes</li>
                <li><b>Validation:</b> 75/25 stratified split + 5-fold CV</li>
                <li><b>Outputs:</b> Class label, confidence %, feature importances</li>
                <li><b>Scaler:</b> StandardScaler (zero-mean, unit-variance)</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <h3>✨ Key Features</h3>
            <ul style="color:#475569;line-height:2.2;">
                <li>📊 Executive KPI Dashboard</li>
                <li>🔍 Employee Search & Filter</li>
                <li>👤 Individual Performance Profiles</li>
                <li>🏢 Department-wise Analytics</li>
                <li>📈 Interactive Visualisation Charts</li>
                <li>🤖 ML Performance Prediction</li>
                <li>🎯 Confidence Score & Probability Breakdown</li>
                <li>📉 Confusion Matrix & CV Diagnostics</li>
                <li>💡 Actionable Improvement Suggestions</li>
            </ul>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>📋 Dataset Details</h3>
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:7px;color:#64748B;font-size:0.9rem;">Total Records</td>
                    <td style="padding:7px;font-weight:600;">150 Employees</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:7px;color:#64748B;font-size:0.9rem;">Departments</td>
                    <td style="padding:7px;font-weight:600;">8 Departments</td></tr>
                <tr><td style="padding:7px;color:#64748B;font-size:0.9rem;">Features</td>
                    <td style="padding:7px;font-weight:600;">16 Attributes</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:7px;color:#64748B;font-size:0.9rem;">Target Classes</td>
                    <td style="padding:7px;font-weight:600;">High / Medium / Low</td></tr>
                <tr><td style="padding:7px;color:#64748B;font-size:0.9rem;">Format</td>
                    <td style="padding:7px;font-weight:600;">CSV</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        # Dataset stats
        section("Live Dataset Summary")
        st.dataframe(df.describe().round(2), use_container_width=True, height=260)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    df = load_data()
    with st.spinner("⚙️  Training ML models — first load may take a moment…"):
        model = build_model(df)

    page = sidebar(df)

    if page == "Dashboard":
        page_dashboard(df, model)
    elif page == "Employee Analytics":
        page_employee(df)
    elif page == "Department Analytics":
        page_department(df)
    elif page == "ML Prediction":
        page_ml(df, model)
    elif page == "About Project":
        page_about(df)


if __name__ == "__main__":
    main()
