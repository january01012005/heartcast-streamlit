import streamlit as st
import polars as pl
import pandas as pd
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import base64


# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION & THEME ---
st.set_page_config(
    page_title="HeartCast",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- HELPER: IMAGE BACKGROUND ---
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""


# Get the base64 string of the background image
img = get_img_as_base64("background.jpg")

# --- 2. CUSTOM CSS ---
page_bg_img = f"""
<style>

/* --- FIX ASK HEARTY BUTTON TO BOTTOM --- */
.ask-hearty-container {{
    position: absolute;
    bottom: 25px;
    left: 15px;
    right: 15px;
}}

/* --- CHAT POPUP FIX --- */
.chat-popup {{
    pointer-events: auto;
}}

/* GLOBAL FONT */
* {{
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}}

/* MAIN BACKGROUND — KEEP IMAGE */
.stApp {{
    background-image:
        linear-gradient(rgba(14,17,23,0.75), rgba(14,17,23,0.65)),
        url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* --- SIDEBAR (LIGHTER & FIXED) --- */
[data-testid="stSidebar"] {{
    background-color: rgba(245, 246, 248, 0.92);
    border-right: 1px solid rgba(0,0,0,0.08);
}}

/* Disable collapse button */
button[kind="header"] {{
    display: none;
}}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    color: #1F2937;
}}

/* LOGO AREA EMPHASIS */
.sidebar-logo {{
    margin-top: 10px;
    margin-bottom: 30px;
}}

/* --- HEADER TEXT --- */
h1, h2, h3 {{
    color: #E09900 !important;
    font-weight: 700;
    letter-spacing: -0.5px;
}}

.slogan {{
    color: #D1D5DB;
    font-size: 1.6rem;
    font-weight: 400;
    margin-top: -10px;
}}

.credits {{
    color: #9CA3AF;
    font-size: 0.9rem;
    margin-bottom: 25px;
}}

/* --- GLASSMORPHISM BASE CARD --- */
.css-card,
.metric-card-orange,
.metric-card-blue,
.metric-card-green,
.stat-card-small {{
    background-color: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-left: 5px solid #FF5722;
    border-radius: 14px;  /* rounded rectangle */
    padding: 16px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
}}

/* --- CARD HOVER: TURN SOLID --- */
.css-card:hover,
.metric-card-orange:hover,
.metric-card-blue:hover,
.metric-card-green:hover,
.stat-card-small:hover {{
    background-color: rgba(255, 255, 255, 0.95);
    color: #111827;
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}}

/* Ensure text switches to dark on hover */
.css-card:hover h1,
.css-card:hover h2,
.css-card:hover h3,
.css-card:hover p,
.css-card:hover span {{
    color: #111827 !important;
}}

/* --- ORANGE TEXT THEME FOR CARDS --- */
.css-card h1,
.css-card h2,
.css-card h3,
.css-card p,
.css-card span,
.metric-card-orange h3,
.metric-card-blue h3,
.metric-card-green h3,
.metric-card-orange p,
.metric-card-blue p,
.metric-card-green p {{
    color: #FF8A50 !important; /* soft modern orange */
}}

/* Chest pain indicator text */
.stat-card-small h4,
.stat-card-small span {{
    color: #FF8A50 !important;
}}

/* On hover → solid card, darker readable orange */
.css-card:hover h1,
.css-card:hover h2,
.css-card:hover h3,
.css-card:hover p,
.css-card:hover span,
.stat-card-small:hover h4,
.stat-card-small:hover span {{
    color: #D84315 !important;
}}

/* --- SIDEBAR NAVIGATION LABEL --- */
.sidebar-navigation-title {{
    color: #FF5722;
    font-weight: 600;
    letter-spacing: 0.12em;
}}

/* METRIC CARDS — SOFTER COLORS */
.metric-card-orange,
.metric-card-blue,
.metric-card-green {{
    border-radius: 16px;
    padding: 18px;
    box-shadow: none;
    border: 1px solid rgba(255,255,255,0.25);
}}

/* BUTTONS */
div.stButton > button {{
    background-color: #F9FAFB;
    color: #111827;
    border-radius: 999px;
    border: 1px solid #E5E7EB;
    font-weight: 500;
}}

div.stButton > button:hover {{
    background-color: #FF5722;
    color: white;
}}

/* TABS — CLEAN & LIGHT */
.stTabs [data-baseweb="tab"] {{
    background-color: rgba(255,255,255,0.15);
    border-radius: 10px;
    color: #E5E7EB;
}}

.stTabs [aria-selected="true"] {{
    background-color: white !important;
    color: #111827 !important;
}}

/* PROGRESS BAR */
.stProgress > div > div > div > div {{
    background-color: #FF5722;
}}

/* --- TOP NAV TABS: RIGHT ALIGNED & EQUAL WIDTH --- */
.stTabs [data-baseweb="tab-list"] {{
    display: flex;
    justify-content: flex-end;
    gap: 12px;
}}

.stTabs [data-baseweb="tab"] {{
    width: 200px;
    text-align: center;
    font-weight: 500;
}}

/* --- REMOVE STREAMLIT DEFAULT BORDERS & OUTLINES --- */
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
div[data-testid="stMetric"],
div[data-testid="stPlotlyChart"],
div[data-testid="stPyplot"],
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {{
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}}

/* Remove focus outlines */
*:focus {{
    outline: none !important;
    box-shadow: none !important;
}}

/* Prevent nested cards from creating double borders */
.css-card * {{
    border: none !important;
}}

/* --- INSIGHT TOOLTIP --- */
.insight-wrapper {{
    position: relative;
    display: inline-block;
    width: 100%;
}}

.insight-tooltip {{
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 110%;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255,255,255,0.95);
    color: #111827;
    padding: 14px 16px;
    border-radius: 12px;
    font-size: 0.85rem;
    width: 280px;
    text-align: left;
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
    z-index: 10;
}}

.insight-wrapper:hover .insight-tooltip {{
    visibility: visible;
    opacity: 1;
}}

</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)


# --- 3. HELPER FUNCTIONS ---

def draw_progress_bar(current_step):
    steps = ["Problem", "Data", "EDA", "Modeling", "Deployment"]
    step_mapping = {
        "Dashboard": 0,
        "Analytics (EDA)": 2,
        "Model & Clusters": 3,
        "Live Diagnosis": 4
    }
    current_index = step_mapping.get(current_step, 0)

    # Styled progress container
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="css-card" style="padding: 15px; border-top: 2px solid #FF5722;">',
                    unsafe_allow_html=True)
        st.caption("Project Lifecycle Status")
        cols = st.columns(len(steps))
        for i, (col, step_name) in enumerate(zip(cols, steps)):
            with col:
                if i < current_index:
                    st.markdown(f"<span style='color:#4CAF50; font-weight:bold'>✅ {step_name}</span>",
                                unsafe_allow_html=True)
                    st.progress(100)
                elif i == current_index:
                    st.markdown(f"<span style='color:#FF5722; font-weight:bold'>🔵 {step_name}</span>",
                                unsafe_allow_html=True)
                    st.progress(50)
                else:
                    st.markdown(f"<span style='color:gray'>{step_name}</span>", unsafe_allow_html=True)
                    st.progress(0)
        st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data
def load_and_clean_data():
    try:
        df = pl.read_csv("heart.csv")
        # Fix for Polars sum issue
        null_count = sum(df.null_count().row(0))

        # Mapping Binary columns
        df_clean = df.with_columns([
            pl.col("Sex").replace({"M": 1, "F": 0}).cast(pl.Int64),
            pl.col("ExerciseAngina").replace({"Y": 1, "N": 0}).cast(pl.Int64)
        ])

        categorical_cols = ["ChestPainType", "RestingECG", "ST_Slope"]
        df_dummies = df_clean.select(categorical_cols).to_dummies()
        df_final = pl.concat([df_clean.drop(categorical_cols), df_dummies], how="horizontal")

        return df, df_final
    except FileNotFoundError:
        return None, None


# --- 4. DATA LOADING ---

df_raw, df_model = load_and_clean_data()

if df_raw is not None:
    open_chat = False

    # --- SIDEBAR ---
    with st.sidebar:
        logo_base64 = get_img_as_base64("logo.png")

        # LOGO HOLDER (Fixed Visibility)
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:32px;">
            <img 
                src="data:image/png;base64,{logo_base64}"
                style="
                    width: 350px;
                    max-width: 100%;
                    margin-bottom: 18px;
                "
            />
            <div class="sidebar-navigation-title">
                NAVIGATION
            </div>
        </div>
        """, unsafe_allow_html=True)


        # NAVIGATION RADIO (Only affects Home Tab)
        selected_dashboard_tab = st.radio("DASHBOARD MENU",
                                          ["Dashboard", "Analytics (EDA)", "Model & Clusters", "Live Diagnosis"],
                                          label_visibility="collapsed"
                                          )

        st.info("💡 **Tip:** Navigate to the 'Live Diagnosis' section to run the AI model.")

    # --- MAIN LAYOUT ---

    # 1. Top Navigation Tabs (Moved to Top Most)
    tab_home, tab_about = st.tabs(["  Home  ", " About HeartCast "])

    # --- TAB A: HOME (Original Dashboard) ---
    with tab_home:

        # --- HEADER SECTION ---
        col_text, col_image = st.columns([2, 1])

        with col_text:
            st.markdown("""
                <h1 style="font-size:64px; margin-bottom:0;">HeartCast</h1>
                <div class="slogan">Advanced Insight for Your Next Beat</div>
                <div class="credits">
                    Developed by J.J. Boladola, M.A. Hernandez and J.B. Supnet
                </div>
            """, unsafe_allow_html=True)

        with col_image:
            st.image("header_image.png", use_container_width=True)

        # SUB-TAB 1: DASHBOARD / OVERVIEW
        if selected_dashboard_tab == "Dashboard":

            st.markdown("### HEALTH DIAGNOSIS OVERVIEW")

            # --- TOP METRICS ROW ---
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="metric-card-orange">
                        <h3>{df_raw.height}</h3>
                        <p>Total Patients</p>
                        <div style="font-size: 24px;">📈</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                hd_count = df_raw.filter(pl.col("HeartDisease") == 1).height
                st.markdown(f"""
                    <div class="metric-card-orange" style="background: linear-gradient(135deg, #D32F2F, #B71C1C);">
                        <h3>{hd_count}</h3>
                        <p>Positive Cases</p>
                        <small>High Risk Detected</small>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                avg_age = df_raw.select(pl.col("Age").mean()).item()
                st.markdown(f"""
                    <div class="metric-card-blue">
                        <h3>{int(avg_age)} yrs</h3>
                        <p>Average Age</p>
                        <small>Patient Demographics</small>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="metric-card-green">
                        <h3>{df_raw.width}</h3>
                        <p>Vitals Tracked</p>
                        <small>Features Analyzed</small>
                    </div>
                """, unsafe_allow_html=True)

            # --- CHEST PAIN INDICATORS ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### CHEST PAIN TYPE INDICATORS")

            # Calculate counts
            cp_counts = df_raw.group_by("ChestPainType").count()
            # Convert to dictionary for easy access: {'ASY': 100, 'NAP': 50...}
            cp_dict = {row[0]: row[1] for row in cp_counts.iter_rows()}

            cp1, cp2, cp3, cp4 = st.columns(4)

            with cp1:
                st.markdown(f"""
                    <div class="stat-card-small">
                        <span>Asymptomatic</span>
                        <h4>{cp_dict.get('ASY', 0)} <small style="font-size:0.8rem">Cases</small></h4>
                        <div style="margin-top:5px; font-weight:bold; color:#FF5722;">ASY</div>
                    </div>
                """, unsafe_allow_html=True)

            with cp2:
                st.markdown(f"""
                    <div class="stat-card-small" style="border-left-color: #2196F3;">
                        <span>Non-Anginal Pain</span>
                        <h4>{cp_dict.get('NAP', 0)} <small style="font-size:0.8rem">Cases</small></h4>
                        <div style="margin-top:5px; font-weight:bold; color:#2196F3;">NAP</div>
                    </div>
                """, unsafe_allow_html=True)

            with cp3:
                st.markdown(f"""
                    <div class="stat-card-small" style="border-left-color: #FFC107;">
                        <span>Atypical Angina</span>
                        <h4>{cp_dict.get('ATA', 0)} <small style="font-size:0.8rem">Cases</small></h4>
                        <div style="margin-top:5px; font-weight:bold; color:#FFC107;">ATA</div>
                    </div>
                """, unsafe_allow_html=True)

            with cp4:
                st.markdown(f"""
                    <div class="stat-card-small" style="border-left-color: #E91E63;">
                        <span>Typical Angina</span>
                        <h4>{cp_dict.get('TA', 0)} <small style="font-size:0.8rem">Cases</small></h4>
                        <div style="margin-top:5px; font-weight:bold; color:#E91E63;">TA</div>
                    </div>
                """, unsafe_allow_html=True)

            # --- DATASET TOGGLE ---
            st.markdown("### DATA REGISTRY")

            # Using st.toggle for a cleaner look than a standard button
            show_data = st.toggle("Show Sample Patient Data", value=False)

            if show_data:
                st.dataframe(df_raw.head(20).to_pandas(), use_container_width=True)
            else:
                st.caption("Toggle the switch above to view the raw patient data table.")

        # SUB-TAB 2: ANALYTICS (EDA)
        elif selected_dashboard_tab == "Analytics (EDA)":
            st.markdown("### VITALS ANALYTICS")
            plot_df = df_raw.to_pandas()
            col1, col2 = st.columns(2)
            with col1:
                p1 = (ggplot(plot_df, aes(x='Age', fill='factor(HeartDisease)'))
                      + geom_histogram(bins=20, alpha=0.8, color=None)
                      + theme_void()
                      + theme(text=element_text(color="white"), plot_background=element_rect(fill="#1E1E1E", alpha=0),
                              panel_background=element_rect(fill="#1E1E1E", alpha=0), legend_position="bottom")
                      + scale_fill_manual(values=["#2196F3", "#FF5722"])
                      + labs(fill="Diagnosis (0=Neg, 1=Pos)")
                      )
                st.markdown("""
                <div class="css-card">
                    <h4>Age vs. Heart Disease Distribution</h4>
                    <p>
                        This chart shows how heart disease cases are distributed across different ages.
                        Taller orange bars indicate that heart disease becomes more common as age increases,
                        especially among older patients.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.pyplot(p1.draw())
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                p2 = (ggplot(plot_df, aes(x='Cholesterol', y='MaxHR', color='factor(HeartDisease)'))
                      + geom_point(alpha=0.7, size=2)
                      + theme_void()
                      + theme(text=element_text(color="white"), plot_background=element_rect(fill="#1E1E1E", alpha=0),
                              panel_background=element_rect(fill="#1E1E1E", alpha=0), legend_position="bottom")
                      + scale_color_manual(values=["#2196F3", "#FF5722"])
                      )
                st.markdown("""
                <div class="css-card">
                    <h4>Cholesterol Levels and Maximum Heart Rate</h4>
                    <p>
                        Each dot represents a patient.
                        Patients with heart disease (orange) tend to have lower maximum heart rates,
                        even when cholesterol levels are similar.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.pyplot(p2.draw())
                st.markdown('</div>', unsafe_allow_html=True)

        # SUB-TAB 3: MODEL & CLUSTERS
        elif selected_dashboard_tab == "Model & Clusters":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="css-card">
                    <h4>Prediction Model using Random Forest</h4>
                    <p>
                        This model analyzes patient health data to predict the likelihood of heart disease.
                        A higher accuracy means the model is better at identifying both healthy and high-risk patients.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                with st.status("Training Random Forest...", expanded=True) as status:
                    X = df_model.drop("HeartDisease").to_pandas()
                    y = df_model.select("HeartDisease").to_pandas().values.ravel()
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    y_pred = rf_model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    status.update(label="Training Complete", state="complete", expanded=False)
                st.metric("Model Accuracy", f"{acc:.2%}", delta="High Precision")
                cm = confusion_matrix(y_test, y_pred)
                st.markdown("""
                <div class="css-card">
                    <h4>Confusion Matrix</h4>
                    <p>
                        This table shows how many predictions were correct and incorrect.
                        Higher values on the diagonal indicate better model performance.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(
                    pd.DataFrame(
                        cm,
                        columns=["Predicted Normal", "Predicted Disease"],
                        index=["Actual Normal", "Actual Disease"]
                    ),
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="css-card">
                    <h4>Patient Clustering (K-Means)</h4>
                    <p>
                        Patients are grouped based on similarities in age, blood pressure,
                        cholesterol, and heart rate.
                        Each color represents a cluster with similar health characteristics.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                cluster_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
                X_cluster = df_model.select(cluster_cols).to_pandas()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                X_cluster['Cluster'] = clusters.astype(str)
                p_cluster = (ggplot(X_cluster, aes(x='Age', y='MaxHR', color='Cluster'))
                             + geom_point(size=3, alpha=0.7)
                             + theme_void()
                             + theme(text=element_text(color="white"),
                                     plot_background=element_rect(fill="#1E1E1E", alpha=0),
                                     panel_background=element_rect(fill="#1E1E1E", alpha=0), legend_position="bottom")
                             + scale_color_brewer(type='qual', palette='Set2')
                             )
                st.pyplot(p_cluster.draw())
                st.markdown('</div>', unsafe_allow_html=True)

        # SUB-TAB 4: LIVE DIAGNOSIS
        elif selected_dashboard_tab == "Live Diagnosis":
            st.markdown("""
            <div class="css-card">
                <h4>LIVE PATIENT CHECK</h4>
                <p>
                    Enter the patient's vitals below to generate a real-time risk assessment.
                </p>
            </div>
            """, unsafe_allow_html=True)
            X = df_model.drop("HeartDisease").to_pandas()
            y = df_model.select("HeartDisease").to_pandas().values.ravel()
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input("Age", 20, 90, 50)
                    sex = st.selectbox("Sex", ["Male", "Female"])
                    cp_type = st.selectbox("Chest Pain", ["ASY (Asymptomatic)", "NAP (Non-Aginal Pain)", "ATA (Atypical Angina)", "TA (Typical Angina)"])
                with c2:
                    resting_bp = st.number_input("Resting BP", 80, 200, 120)
                    cholesterol = st.number_input("Cholesterol", 0, 600, 200)
                    fasting_bs = st.selectbox("Fasting BS > 120", [0, 1])
                with c3:
                    max_hr = st.number_input("Max HR", 60, 220, 140)
                    ex_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
                    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
                resting_ecg = "Normal"
                oldpeak = 0.0
                submit = st.form_submit_button("Analyze Vitals")
            if submit:
                input_dict = {
                    "Age": age, "Sex": 1 if sex == "M" else 0, "RestingBP": resting_bp,
                    "Cholesterol": cholesterol, "FastingBS": fasting_bs, "MaxHR": max_hr,
                    "ExerciseAngina": 1 if ex_angina == "Y" else 0, "Oldpeak": oldpeak,
                    "ChestPainType_ASY": 0, "ChestPainType_ATA": 0, "ChestPainType_NAP": 0, "ChestPainType_TA": 0,
                    "RestingECG_LVH": 0, "RestingECG_Normal": 0, "RestingECG_ST": 0,
                    "ST_Slope_Down": 0, "ST_Slope_Flat": 0, "ST_Slope_Up": 0
                }
                input_dict[f"ChestPainType_{cp_type}"] = 1
                input_dict[f"RestingECG_{resting_ecg}"] = 1
                input_dict[f"ST_Slope_{st_slope}"] = 1
                input_df = pd.DataFrame([input_dict])[X.columns]
                prediction = rf_model.predict(input_df)[0]
                prob = rf_model.predict_proba(input_df)[0][1]
                st.write("---")
                if prediction == 1:
                    st.markdown(
                        f"""<div class="metric-card-orange" style="background-color: #D32F2F;"><h2>⚠️ HIGH RISK DETECTED</h2><p>Probability: {prob * 100:.1f}%</p><p>Action: Urgent Cardiology Referral Recommended</p></div>""",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"""<div class="metric-card-orange" style="background-color: #388E3C;"><h2>✅ LOW RISK</h2><p>Probability: {prob * 100:.1f}%</p><p>Action: Routine Checkup</p></div>""",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- PROGRESS BAR (Moved to Bottom) ---
        draw_progress_bar(selected_dashboard_tab)

    # --- TAB B: ABOUT (Context) ---
    with tab_about:
        # Also cleaned up the About Header to match
        st.markdown("<h1 style='text-align: center;'>About HeartCast</h1>", unsafe_allow_html=True)
        st.markdown('<p class="slogan" style="text-align:center;">Detailed breakdown of data sources and attributes.</p>', unsafe_allow_html=True)
        st.divider()

        st.header("Project Overview")
        st.write(
            "This application serves as a comprehensive Data Science solution for Heart Failure prediction. It demonstrates the full lifecycle from data cleaning and exploration to modeling and deployment.")

        st.subheader("Context")
        st.markdown("""
        Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. 
        Four out of 5 CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. 
        Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

        People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
        """)

        st.markdown("---")

        col_attr, col_source = st.columns(2)

        with col_attr:
            st.subheader("Attribute Information")
            st.markdown("""
            *   **Age**: Age of the patient [years]
            *   **Sex**: Sex of the patient [M: Male, F: Female]
            *   **ChestPainType**: [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
            *   **RestingBP**: Resting blood pressure [mm Hg]
            *   **Cholesterol**: Serum cholesterol [mm/dl]
            *   **FastingBS**: Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
            *   **RestingECG**: Resting electrocardiogram results [Normal, ST, LVH]
            *   **MaxHR**: Maximum heart rate achieved [60-202]
            *   **ExerciseAngina**: Exercise-induced angina [Y: Yes, N: No]
            *   **Oldpeak**: ST [Numeric value measured in depression]
            *   **ST_Slope**: Slope of the peak exercise ST segment [Up, Flat, Down]
            *   **HeartDisease**: Output class [1: heart disease, 0: Normal]
            """)

        with col_source:
            st.subheader("Data Source")
            st.markdown(
                "This dataset was created by combining different datasets already available independently but not combined before.")
            st.markdown("""
            *   Cleveland: 303 observations
            *   Hungarian: 294 observations
            *   Switzerland: 123 observations
            *   Long Beach VA: 200 observations
            *   Stalog (Heart) Data Set: 270 observations

            **Final dataset:** 918 observations (after removing duplicates).
            """)
            st.link_button("View Dataset on Kaggle",
                           "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Data file (heart.csv) not found. Please ensure the CSV file is in the same directory.")
