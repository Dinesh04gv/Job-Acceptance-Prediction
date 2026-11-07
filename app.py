import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from pathlib import Path
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #101827, #17243a);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101827, #253858);
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #64d8ff;
    margin-bottom: 5px;
}

.subtitle {
    color: #b8c7dc;
    font-size: 16px;
    margin-bottom: 25px;
}

.kpi-card {
    background: linear-gradient(135deg, #253858, #344e75);
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #49688f;
    text-align: center;
    min-height: 125px;
}

.kpi-label {
    color: #c5d5e8;
    font-size: 15px;
}

.kpi-value {
    color: #64d8ff;
    font-size: 29px;
    font-weight: bold;
    margin-top: 10px;
}

.section-title {
    color: #64d8ff;
    font-size: 23px;
    font-weight: bold;
    margin-top: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg, #168aad, #4361ee);
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #4361ee, #168aad);
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. PROJECT PATH
# --------------------------------------------------

BASE_FOLDER = Path(__file__).parent
MODEL_FOLDER = BASE_FOLDER / "saved_model"


# --------------------------------------------------
# 4. MYSQL DATABASE CONNECTION
# --------------------------------------------------

@st.cache_data(ttl=60)
def load_candidate_data():

    connection = None

    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )

        query = "SELECT * FROM candidates"

        data = pd.read_sql(query, connection)

        return data

    finally:
        if connection is not None and connection.is_connected():
            connection.close()


# --------------------------------------------------
# 5. LOAD MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model_artifacts():

    model = XGBClassifier()

    model.load_model(
        str(MODEL_FOLDER / "xgboost_model.json")
    )

    encoder = joblib.load(
        MODEL_FOLDER / "encoder.pkl"
    )

    scaler = joblib.load(
        MODEL_FOLDER / "scaler.pkl"
    )

    imputer = joblib.load(
        MODEL_FOLDER / "imputer.pkl"
    )

    numerical_features = joblib.load(
        MODEL_FOLDER / "numerical_features.pkl"
    )

    categorical_features = joblib.load(
        MODEL_FOLDER / "categorical_features.pkl"
    )

    feature_names = joblib.load(
        MODEL_FOLDER / "feature_names.pkl"
    )

    return (
        model,
        encoder,
        scaler,
        imputer,
        list(numerical_features),
        list(categorical_features),
        feature_names
    )


# --------------------------------------------------
# 6. LOAD DATABASE
# --------------------------------------------------

try:

    df = load_candidate_data()

except Exception as e:

    st.error("Unable to connect to the MySQL database.")

    st.write(
        "Please check your MySQL service and "
        "the .streamlit/secrets.toml configuration."
    )

    st.exception(e)

    st.stop()


if df.empty:
    st.warning("No candidate records were found in the database.")
    st.stop()


# --------------------------------------------------
# 7. COMMON DATA PREPARATION
# --------------------------------------------------

df["status"] = (
    df["status"]
    .astype(str)
    .str.strip()
    .str.title()
)

total_candidates = len(df)

placed_count = (
    df["status"] == "Placed"
).sum()

not_placed_count = (
    df["status"] == "Not Placed"
).sum()

placement_rate = (
    placed_count / total_candidates * 100
    if total_candidates else 0
)

average_technical_score = (
    df["technical_score"].mean()
)

average_skills_match = (
    df["skills_match_percentage"].mean()
)


# --------------------------------------------------
# 8. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown(
    "<h1 style='color:#64d8ff;'>💼 Job Analytics</h1>",
    unsafe_allow_html=True
)

st.sidebar.write("Job Acceptance Prediction System")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Candidate Analysis",
        "Placement Insights",
        "Candidate Explorer",
        "Job Acceptance Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    f"""
    **Database Status:** Connected

    **Total Candidates:** {total_candidates:,}
    """
)


# --------------------------------------------------
# 9. REUSABLE KPI CARD
# --------------------------------------------------

def show_kpi(label, value):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.markdown(
        '<div class="main-title">💼 Job Acceptance Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Recruitment analytics and candidate placement insights</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_kpi(
            "Total Candidates",
            f"{total_candidates:,}"
        )

    with col2:
        show_kpi(
            "Placed Candidates",
            f"{placed_count:,}"
        )

    with col3:
        show_kpi(
            "Not Placed",
            f"{not_placed_count:,}"
        )

    with col4:
        show_kpi(
            "Placement Rate",
            f"{placement_rate:.2f}%"
        )

    st.markdown(
        '<div class="section-title">📊 Candidate Placement Distribution</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        status_counts = df["status"].value_counts().reset_index()

        status_counts.columns = ["Status", "Candidates"]

        fig = px.pie(
            status_counts,
            names="Status",
            values="Candidates",
            hole=0.55,
            title="Placement Status Distribution",
            color="Status",
            color_discrete_map={
                "Placed": "#06d6a0",
                "Not Placed": "#ef476f"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, width="stretch")

    with col2:

        fig = px.bar(
            status_counts,
            x="Status",
            y="Candidates",
            color="Status",
            title="Placed vs Not Placed Candidates",
            color_discrete_map={
                "Placed": "#06d6a0",
                "Not Placed": "#ef476f"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title">📈 Candidate Performance</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        show_kpi(
            "Average Technical Score",
            f"{average_technical_score:.2f}"
        )

    with col2:

        show_kpi(
            "Average Skills Match",
            f"{average_skills_match:.2f}%"
        )

    st.markdown(
        '<div class="section-title">🏢 Company Tier Distribution</div>',
        unsafe_allow_html=True
    )

    tier_data = (
        df.groupby(["company_tier", "status"])
        .size()
        .reset_index(name="Candidates")
    )

    fig = px.bar(
        tier_data,
        x="company_tier",
        y="Candidates",
        color="status",
        barmode="group",
        title="Placement Status by Company Tier",
        color_discrete_map={
            "Placed": "#06d6a0",
            "Not Placed": "#ef476f"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# PAGE 2: CANDIDATE ANALYSIS
# --------------------------------------------------

elif page == "Candidate Analysis":

    st.markdown(
        '<div class="main-title">👥 Candidate Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore candidate demographics, education and skills</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="age_years",
            color="status",
            nbins=20,
            title="Candidate Age Distribution",
            color_discrete_map={
                "Placed": "#06d6a0",
                "Not Placed": "#ef476f"
            }
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    with col2:

        gender_data = (
            df.groupby(["gender", "status"])
            .size()
            .reset_index(name="Candidates")
        )

        fig = px.bar(
            gender_data,
            x="gender",
            y="Candidates",
            color="status",
            barmode="group",
            title="Gender Distribution by Placement Status"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:

        specialization_data = (
            df.groupby(["degree_specialization", "status"])
            .size()
            .reset_index(name="Candidates")
        )

        fig = px.bar(
            specialization_data,
            x="degree_specialization",
            y="Candidates",
            color="status",
            barmode="group",
            title="Degree Specialization vs Placement"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-35
        )

        st.plotly_chart(fig, width="stretch")

    with col2:

        fig = px.box(
            df,
            x="status",
            y="degree_percentage",
            color="status",
            title="Degree Percentage by Placement Status"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title">📚 Academic and Technical Scores</div>',
        unsafe_allow_html=True
    )

    score_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "technical_score",
        "aptitude_score",
        "communication_score"
    ]

    selected_score = st.selectbox(
        "Select a score",
        score_columns
    )

    fig = px.histogram(
        df,
        x=selected_score,
        color="status",
        marginal="box",
        title=f"{selected_score} Distribution"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# PAGE 3: PLACEMENT INSIGHTS
# --------------------------------------------------

elif page == "Placement Insights":

    st.markdown(
        '<div class="main-title">📈 Placement Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Understand the factors associated with candidate placement</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        internship_data = (
            df.groupby(["internship_experience", "status"])
            .size()
            .reset_index(name="Candidates")
        )

        fig = px.bar(
            internship_data,
            x="internship_experience",
            y="Candidates",
            color="status",
            barmode="group",
            title="Internship Experience vs Placement"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    with col2:

        role_data = (
            df.groupby(["job_role_match", "status"])
            .size()
            .reset_index(name="Candidates")
        )

        fig = px.bar(
            role_data,
            x="job_role_match",
            y="Candidates",
            color="status",
            barmode="group",
            title="Job Role Match vs Placement"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            x="status",
            y="technical_score",
            color="status",
            title="Technical Score vs Placement"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    with col2:

        fig = px.box(
            df,
            x="status",
            y="skills_match_percentage",
            color="status",
            title="Skills Match vs Placement"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title">💰 Expected Salary Analysis</div>',
        unsafe_allow_html=True
    )

    fig = px.box(
        df,
        x="company_tier",
        y="expected_ctc_lpa",
        color="status",
        title="Expected CTC by Company Tier"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, width="stretch")

    st.markdown(
        '<div class="section-title">📊 Average Performance by Status</div>',
        unsafe_allow_html=True
    )

    performance = (
        df.groupby("status")[
            [
                "technical_score",
                "aptitude_score",
                "communication_score",
                "skills_match_percentage"
            ]
        ]
        .mean()
        .reset_index()
    )

    st.dataframe(
        performance.round(2),
        hide_index=True,
        width="stretch"
    )


# --------------------------------------------------
# PAGE 4: CANDIDATE EXPLORER
# --------------------------------------------------

elif page == "Candidate Explorer":

    st.markdown(
        '<div class="main-title">🔍 Candidate Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Search, filter and explore candidate records</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        status_filter = st.selectbox(
            "Placement Status",
            ["All", "Placed", "Not Placed"]
        )

    with col2:

        tier_options = ["All"] + sorted(
            df["company_tier"].dropna().astype(str).unique().tolist()
        )

        tier_filter = st.selectbox(
            "Company Tier",
            tier_options
        )

    with col3:

        internship_options = ["All"] + sorted(
            df["internship_experience"].dropna().astype(str).unique().tolist()
        )

        internship_filter = st.selectbox(
            "Internship Experience",
            internship_options
        )

    filtered_df = df.copy()

    if status_filter != "All":

        filtered_df = filtered_df[
            filtered_df["status"] == status_filter
        ]

    if tier_filter != "All":

        filtered_df = filtered_df[
            filtered_df["company_tier"].astype(str) == tier_filter
        ]

    if internship_filter != "All":

        filtered_df = filtered_df[
            filtered_df["internship_experience"].astype(str)
            == internship_filter
        ]

    search_text = st.text_input(
        "Search by Candidate ID or Degree Specialization"
    )

    if search_text:

        search_mask = (
            filtered_df["candidate_id"].astype(str).str.contains(
                search_text, case=False, na=False
            )
            |
            filtered_df["degree_specialization"].astype(str).str.contains(
                search_text, case=False, na=False
            )
        )

        filtered_df = filtered_df[search_mask]

    st.write(
        f"**Matching candidates:** {len(filtered_df):,}"
    )

    display_columns = [
        "candidate_id",
        "age_years",
        "gender",
        "degree_specialization",
        "technical_score",
        "skills_match_percentage",
        "company_tier",
        "job_role_match",
        "expected_ctc_lpa",
        "status"
    ]

    st.dataframe(
        filtered_df[display_columns],
        hide_index=True,
        width="stretch"
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Candidates CSV",
        data=csv_data,
        file_name="filtered_candidates.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# PAGE 5: JOB ACCEPTANCE PREDICTION
# --------------------------------------------------

elif page == "Job Acceptance Prediction":

    st.markdown(
        '<div class="main-title">🤖 Job Acceptance Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Predict candidate placement status using the trained XGBoost model</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter candidate details below. The trained model will estimate "
        "the probability of the candidate being placed."
    )

    # Load model files

    try:

        (
            model,
            encoder,
            scaler,
            imputer,
            numerical_features,
            categorical_features,
            feature_names
        ) = load_model_artifacts()

    except Exception as e:

        st.error("Unable to load the saved model files.")

        st.write(
            "Check that all model files are present inside the saved_model folder."
        )

        st.exception(e)

        st.stop()

    # Numeric columns used for imputation

    impute_columns = [
        "ssc_percentage",
        "hsc_percentage",
        "notice_period_days",
        "employment_gap_months"
    ]

    st.markdown(
        '<div class="section-title">📊 Numerical Candidate Details</div>',
        unsafe_allow_html=True
    )

    input_values = {}

    with st.form("prediction_form"):

        numerical_columns = st.columns(3)

        for index, feature in enumerate(numerical_features):

            series = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).dropna()

            if series.empty:
                minimum = 0.0
                maximum = 100.0
                default = 0.0

            else:
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())

            if minimum == maximum:
                minimum -= 1.0
                maximum += 1.0

            with numerical_columns[index % 3]:

                input_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                    step=0.1,
                    key=f"num_{feature}"
                )

        st.markdown(
            '<div class="section-title">📝 Categorical Candidate Details</div>',
            unsafe_allow_html=True
        )

        categorical_columns = st.columns(2)

        for index, feature in enumerate(categorical_features):

            options = (
                df[feature]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            options = sorted(set(options))

            if "Unknown" not in options:
                options.append("Unknown")

            if not options:
                options = ["Unknown"]

            with categorical_columns[index % 2]:

                input_values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options,
                    key=f"cat_{feature}"
                )

        st.divider()

        submitted = st.form_submit_button(
            "🔮 Predict Job Acceptance",
            width="stretch"
        )

    # Prediction processing

    if submitted:

        try:

            input_df = pd.DataFrame([input_values])

            # Step 1: Impute missing numeric features

            input_df[impute_columns] = imputer.transform(
                input_df[impute_columns]
            )

            # Step 2: Scale numerical features

            numerical_input = scaler.transform(
                input_df[numerical_features]
            )

            # Step 3: Encode categorical features

            categorical_input = encoder.transform(
                input_df[categorical_features]
            )

            if hasattr(categorical_input, "toarray"):
                categorical_input = categorical_input.toarray()

            # Step 4: Combine numerical and categorical features

            final_input = np.hstack(
                [
                    numerical_input,
                    categorical_input
                ]
            )

            # Step 5: Check feature count

            if final_input.shape[1] != model.n_features_in_:

                st.error(
                    f"Feature mismatch: Model expects "
                    f"{model.n_features_in_} features, but received "
                    f"{final_input.shape[1]}."
                )

                st.stop()

            # Step 6: Predict

            prediction = model.predict(final_input)[0]

            probabilities = model.predict_proba(final_input)[0]

            positive_class_index = list(model.classes_).index(1)

            placed_probability = (
                probabilities[positive_class_index] * 100
            )

            not_placed_probability = 100 - placed_probability

            st.divider()

            st.markdown(
                '<div class="section-title">🎯 Prediction Result</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Placed Probability",
                    f"{placed_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Not Placed Probability",
                    f"{not_placed_probability:.2f}%"
                )

            if prediction == 1:

                st.success(
                    "🎉 Prediction: Candidate is likely to be PLACED."
                )

            else:

                st.warning(
                    "⚠️ Prediction: Candidate is likely to be NOT PLACED."
                )

            st.progress(
                int(round(placed_probability))
            )

            st.caption(
                "The probability is a model estimate, not a guarantee "
                "of an actual job offer or acceptance."
            )

            # Simple input-based suggestions

            st.markdown(
                '<div class="section-title">💡 Candidate Improvement Suggestions</div>',
                unsafe_allow_html=True
            )

            suggestions = []

            if input_values["technical_score"] < 60:
                suggestions.append(
                    "Improve technical knowledge through coding practice "
                    "and technical interview preparation."
                )

            if input_values["aptitude_score"] < 60:
                suggestions.append(
                    "Practice quantitative aptitude, logical reasoning "
                    "and problem-solving."
                )

            if input_values["communication_score"] < 60:
                suggestions.append(
                    "Improve communication skills through mock interviews "
                    "and group discussions."
                )

            if input_values["skills_match_percentage"] < 60:
                suggestions.append(
                    "Develop skills that better match the targeted job role."
                )

            if input_values["job_role_match"] == "Not Matched":
                suggestions.append(
                    "Apply for positions that match the candidate's "
                    "education and technical skills."
                )

            if suggestions:

                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

            else:

                st.success(
                    "The selected technical, aptitude, communication "
                    "and skills-match scores meet the basic suggestion thresholds."
                )

            st.caption(
                "These suggestions are simple input-based checks, "
                "not explanations generated by SHAP."
            )

        except Exception as e:

            st.error("An error occurred during prediction.")

            st.exception(e)


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="text-align:center; color:#aabbd0;">
        <small>Job Acceptance Prediction System</small><br>
        <small>Machine Learning | XGBoost | MySQL | Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)