import time
import streamlit as st   # ✅ THIS WAS MISSING

TIMEOUT_SECONDS = 300  # 5 minutes

def check_timeout():
    """
    Auto logout user after inactivity
    """
    if "last_active" not in st.session_state:
        st.session_state.last_active = time.time()
        return False

    if time.time() - st.session_state.last_active > TIMEOUT_SECONDS:
        st.session_state.logged_in = False
        return True

    # Update activity time
    st.session_state.last_active = time.time()
    return False
