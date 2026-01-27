import streamlit as st
from ui.layout import render_main_layout

# --- Page Configuration ---
st.set_page_config(
    page_title="PHOENIX-RADAR STRATEGIC COMMAND",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """
    Main Entry Point for the PHOENIX-RADAR Research Platform.
    Launches the tactical military-style dashboard.
    """
    render_main_layout()

if __name__ == "__main__":
    main()
