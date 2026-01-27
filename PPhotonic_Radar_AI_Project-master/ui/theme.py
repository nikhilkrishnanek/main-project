"""
Strategic Radar Theme System
============================

Defines the visual aesthetic for the professional radar dashboard.
Style: Military-grade, dark-mode, high-contrast tactical green.
"""

TACTICAL_CSS = """
<style>
    /* Global Tactical Style */
    .stApp {
        background-color: #050805;
        color: #4dfa4d;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Tactical Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a120a;
        border-right: 1px solid #1a331a;
    }
    
    /* Metrics / Status Units */
    [data-testid="stMetricValue"] {
        color: #4dfa4d !important;
        text-shadow: 0 0 5px #4dfa4d;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #1a0505;
        border: 1px solid #ff3333;
        color: #ff4d4d;
    }
    
    /* Grid / Cards */
    div.stButton > button {
        background-color: #1a331a;
        color: #4dfa4d;
        border: 1px solid #4dfa4d;
    }
    
    div.stButton > button:hover {
        background-color: #4dfa4d;
        color: #050805;
    }
</style>
"""

def apply_tactical_theme(st):
    """
    Applies the tactical CSS to the Streamlit app.
    """
    st.markdown(TACTICAL_CSS, unsafe_allow_html=True)
