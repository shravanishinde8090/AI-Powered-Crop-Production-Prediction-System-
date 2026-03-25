import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import time
import requests
from streamlit_lottie import st_lottie
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Crop Prediction Pro", page_icon="🌱", layout="wide")

# -------- LOTTIE & CACHING --------
@st.cache_data
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_plant = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_success = load_lottie("https://assets3.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# -------- CSS THEME INJECTION --------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Premium Animated Background */
.stApp {
    background: linear-gradient(120deg, #f0fdf4 0%, #d1fae5 50%, #ecfdf5 100%);
    background-size: 200% 200%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glassmorphism Main Container */
.block-container {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
    padding: 2.5rem 3rem !important;
    margin-top: 2rem;
    margin-bottom: 5rem;
}

/* Glassmorphism Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255, 255, 255, 0.6);
}

/* Glowing Button */
.stButton>button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    border-radius: 12px;
    border: none;
    height: 3.5rem;
    width: 100%;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(16, 185, 129, 0.5);
    background: linear-gradient(135deg, #047857 0%, #059669 100%);
}

/* Glowing Result Card */
.result-card {
    padding: 2.5rem;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(255,255,255,0.9), rgba(240,253,244,0.9));
    text-align: center;
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.25), 
                inset 0 0 20px rgba(255,255,255,0.5);
    animation: glowPulse 2s ease-in-out infinite alternate, fadeInUp 0.8s ease-out;
    margin: 2rem 0;
}
@keyframes glowPulse {
    from { box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
    to { box-shadow: 0 0 35px rgba(16, 185, 129, 0.4); }
}

@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

.result-title {
    font-size: 1.5rem;
    color: #064e3b;
    margin-bottom: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.result-value {
    font-size: 3rem;
    font-weight: 800;
    color: #059669;
    background: -webkit-linear-gradient(45deg, #047857, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

/* Output Metrics Flex Container */
.output-metrics {
    display: flex;
    justify-content: space-around;
    align-items: center;
    margin-top: 1.5rem;
}
.metric-box {
    flex: 1;
    text-align: center;
}
.metric-divider {
    border-left: 2px dashed rgba(16, 185, 129, 0.3);
    height: 80px;
    margin: 0 1.5rem;
}
.metric-label {
    font-size: 1.1rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* Input Styles */
.stSelectbox>div>div, .stNumberInput>div>div, .stSlider>div>div {
    border-radius: 12px;
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(209, 213, 219, 0.5);
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
}

h1, h2, h3, h4, h5, h6 {
    color: #111827;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Custom Footer */
.professional-footer {
    position: fixed;
    bottom: 0px;
    left: 0px;
    width: 100%;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-top: 1px solid rgba(16, 185, 129, 0.2);
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px 0;
    z-index: 1000;
    box-shadow: 0 -5px 20px rgba(0,0,0,0.03);
}
.professional-footer a {
    color: #059669;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s ease;
    margin-left: 5px;
}
.professional-footer a:hover {
    color: #047857;
    text-decoration: underline;
}
.footer-text {
    color: #4b5563;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
.linkedin-icon {
    width: 20px;
    height: 20px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# -------- MODEL & DATA --------
@st.cache_resource
def load_model():
    return joblib.load("crop_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("crop_production.csv")

with st.spinner("Loading agricultural dataset..."):
    df = load_data()

# -------- HERO SECTION --------
hero_col1, hero_col2 = st.columns([1.5, 1])

with hero_col1:
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; line-height: 1.1;'>AI-Powered Crop<br><span style='color: #059669;'>Intelligence Platform</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.25rem; color: #4b5563; margin-bottom: 2rem;'>Advanced Machine Learning forecasting for agricultural yield optimization. Plan your harvests with data-driven precision.</p>", unsafe_allow_html=True)
    
with hero_col2:
    if lottie_plant:
         st_lottie(lottie_plant, height=220, key="plant_anim")

st.markdown("---")

# -------- SIDEBAR ENHANCEMENTS --------
st.sidebar.markdown("<h2 style='font-size: 1.5rem; margin-bottom: 1rem;'>🌍 Localization Inputs</h2>", unsafe_allow_html=True)

state = st.sidebar.selectbox("📍 State / Province", df['State_Name'].unique())
filtered_df_state = df[df['State_Name'] == state]
district = st.sidebar.selectbox("📍 District / County", filtered_df_state['District_Name'].unique())
season = st.sidebar.selectbox("🌤️ Harvesting Season", df['Season'].unique())

# -------- MAIN CONFIGURATION --------
st.markdown("### 🚜 Agricultural Parameters")

with st.expander("Configure Crop & Land Specifics", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        crop = st.selectbox("🌱 Target Crop", df['Crop'].unique())
    with col_b:
        year = st.slider("📅 Forecast Year", int(df['Crop_Year'].min()) if 'Crop_Year' in df.columns else 1990, 2030, 2024, help="Select timeline for prediction into the future or past.")
    with col_c:
        area = st.number_input("🏞️ Cultivation Area (Hectares)", min_value=1.0, value=100.0, step=10.0)

# -------- ENCODING & PREDICTION TARGETS --------
le = LabelEncoder()
df['State_Name_Enc'] = le.fit_transform(df['State_Name'])
state_mapper = dict(zip(df['State_Name'], df['State_Name_Enc']))

df['District_Name_Enc'] = le.fit_transform(df['District_Name'])
district_mapper = dict(zip(df['District_Name'], df['District_Name_Enc']))

df['Season_Enc'] = le.fit_transform(df['Season'])
season_mapper = dict(zip(df['Season'], df['Season_Enc']))

df['Crop_Enc'] = le.fit_transform(df['Crop'])
crop_mapper = dict(zip(df['Crop'], df['Crop_Enc']))

state_val = state_mapper[state]
district_val = district_mapper[district]
season_val = season_mapper[season]
crop_val = crop_mapper[crop]

st.markdown("<br>", unsafe_allow_html=True)

# -------- ACTION BUTTON --------
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_btn = st.button("🚀 Analyze & Predict Yield")

if predict_btn:
    with st.spinner("Loading 1.6GB ML Model into Memory & Processing Variables..."):
        model = load_model()
        
    result = model.predict([[state_val, district_val, year, season_val, crop_val, area]])
    predicted_production = result[0]
    predicted_yield = predicted_production / area if area > 0 else 0
    
    # Glowing Result Card
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">Forecasting Results Overview</div>
            <div class="output-metrics">
                <div class="metric-box">
                    <div class="metric-label">Estimated Total Production</div>
                    <div class="result-value">{predicted_production:,.2f} <span style="font-size: 1.5rem; font-weight: 600; -webkit-text-fill-color: #059669;">Tonnes</span></div>
                </div>
                <div class="metric-divider"></div>
                <div class="metric-box">
                    <div class="metric-label">Expected Yield Rate</div>
                    <div class="result-value" style="background: -webkit-linear-gradient(45deg, #0d9488, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {predicted_yield:,.2f} <span style="font-size: 1.5rem; font-weight: 600;">T/Ha</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if lottie_success:
        with st.columns([1.5, 1, 1.5])[1]:
            st_lottie(lottie_success, height=140, key="success_anim", loop=False)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Production Analytics")
    
    # Interactive Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Invested Area (Ha)", "Predicted Output (Tonnes)"],
        y=[area, predicted_production],
        text=[f"{area:,.1f} Ha", f"{predicted_production:,.1f} T"],
        textposition='auto',
        marker=dict(
            color=['#34d399', '#059669'],
            line=dict(color='rgba(0,0,0,0)', width=0),
            pattern_shape=""
        ),
        hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>",
        width=0.45
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=14, color="#4b5563"),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=False, tickfont=dict(weight="bold")),
        yaxis=dict(showgrid=True, gridcolor='rgba(16, 185, 129, 0.15)', griddash='dash'),
        hovermode="x unified"
    )
    
    with st.container():
         st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# -------- METRICS HIGHLIGHTS (TABBED) --------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📊 Regional Insights", "📋 Full Dataset Analytics"])

with tab1:
    dist_data = filtered_df_state[filtered_df_state['District_Name'] == district]
    if not dist_data.empty:
        st.markdown(f"<p style='font-size: 1.1rem; color: #4b5563; font-weight: 500;'>Macro-level intelligence for <b>{district}, {state}</b></p>", unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Historical Data Points", f"{len(dist_data):,}")
        col_m2.metric("Crop Variety Diversity", f"{len(dist_data['Crop'].unique())}")
        
        dist_data['Production'] = pd.to_numeric(dist_data['Production'], errors='coerce')
        max_prod = dist_data['Production'].max()
        if pd.notna(max_prod):
            col_m3.metric("Max Recorded Yield Output", f"{max_prod:,.0f} T")
        else:
            col_m3.metric("Max Recorded Yield Output", "N/A Data")
    else:
        st.info("Insufficient historical intelligence for selected region.")

with tab2:
    st.dataframe(df.head(250), use_container_width=True, height=400)

# -------- MULTI-LINE FOOTER WITH LINKEDIN --------
st.markdown(
    """
    <div class="professional-footer">
        <div class="footer-text">
            © 2024 Crop Yield Intelligence Framework | Engineered for ML Portfolios |  
            <svg class="linkedin-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#0A66C2">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            <a href="https://linkedin.com/in/YOUR_PROFILE_LINK" target="_blank" rel="noopener noreferrer">Let's Connect on LinkedIn</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br><br><br>", unsafe_allow_html=True)
