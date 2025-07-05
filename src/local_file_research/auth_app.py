"""
Standalone authentication app for the Local File Research application.
"""

import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set page config
st.set_page_config(
    page_title="Local File Research - Authentication & Security",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Import auth UI
from src.local_file_research.auth_ui import auth_page, initialize_auth_state

# Initialize authentication state
initialize_auth_state()

# Main function
def main():
    """
    Main function to render the authentication UI.
    """
    # Check if user is already authenticated
    if st.session_state.get("authenticated", False):
        st.success(f"You are logged in as {st.session_state.username}!")

        # Add explanation about the two apps
        st.info("""
        **You are now in the Authentication & Security app (port 8502).**

        This app is dedicated to user authentication and security settings, including:
        - Account management
        - Two-Factor Authentication (2FA) setup
        - Security settings

        The main research application is running on port 8501.
        """)

        # Create columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            # Make the return button more prominent
            if st.button("🔙 Return to Main App", use_container_width=True):
                # Use JavaScript to redirect to the main app
                js_code = """
                <script>
                    window.open('http://localhost:8501', '_self');
                </script>
                """
                st.components.v1.html(js_code, height=0)
                st.markdown("[Click here if not redirected automatically](http://localhost:8501)")

        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                # Clear session state
                st.session_state.authenticated = False
                st.session_state.auth_token = None
                st.session_state.username = None
                st.success("Logged out successfully!")
                st.rerun()

        # Add a divider before showing security settings
        st.markdown("---")
        st.subheader("Security Settings")
        st.markdown("Use the tabs below to manage your account and security settings.")
    else:
        # Show authentication page
        auth_page()

if __name__ == "__main__":
    main()
