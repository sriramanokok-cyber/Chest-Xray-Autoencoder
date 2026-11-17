
# ==================================================
# CUSTOM UI DESIGN - PRESENTATION READY
# ==================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #0b1220 0%, #111c30 100%);
    color: #f1f5f9;
}

/* Use the available screen width */
.block-container {
    max-width: 100% !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

/* Main title */
h1 {
    color: #e2e8f0 !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
}

/* Section headings */
h2 {
    color: #60a5fa !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
}

h3 {
    color: #cbd5e1 !important;
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

/* General text */
p, label, .stCaption {
    color: #cbd5e1 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.8rem !important;
    font-weight: 750 !important;
}

/* Uploaded file area */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.65);
    border: 1px dashed #60a5fa;
    border-radius: 14px;
    padding: 15px;
}

/* Image containers */
[data-testid="stImage"] {
    background: #172337;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 6px;
    overflow: hidden;
}

/* Status messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #334155 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #3b82f6;
    background: #2563eb;
    color: white;
    font-weight: 600;
}

/* Attention chart */
.stPlotlyChart, [data-testid="stPyplot"] {
    background: rgba(30, 41, 59, 0.65);
    border-radius: 14px;
    padding: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}

/* Responsive layout for smaller screens */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    h1 {
        font-size: 1.65rem !important;
    }

    h2 {
        font-size: 1.3rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }
}

</style>
""", unsafe_allow_html=True)
