"""
Authentication UI components for the local file research system.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def login_form():
    """
    Render login form.
    """
    st.subheader("Login")

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit and username and password:
            with st.spinner("Logging in..."):
                try:
                    # Import api_request function
                    try:
                        from .ui_llamaindex import api_request
                    except ImportError:
                        from src.local_file_research.ui_llamaindex import api_request

                    # Make API request to login
                    response = api_request(
                        "/auth/login",
                        method="POST",
                        data={"username": username, "password": password},
                        auth_type="form"
                    )

                    if "error" not in response:
                        # Store token in session state
                        st.session_state.auth_token = response["access_token"]
                        st.session_state.username = username
                        st.success(f"Welcome, {username}!")

                        # Set current page to Research
                        st.session_state.current_page = "Research"
                        st.rerun()
                    else:
                        st.error(f"Login failed: {response['error']}")
                except Exception as e:
                    st.error(f"Login error: {str(e)}")

def register_form():
    """
    Render registration form.
    """
    st.subheader("Register")

    # Registration form
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")

        if submit:
            # Validate form
            if not username or not email or not password:
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                with st.spinner("Registering..."):
                    try:
                        # Import api_request function
                        try:
                            from .ui_llamaindex import api_request
                        except ImportError:
                            from src.local_file_research.ui_llamaindex import api_request

                        # Make API request to register
                        response = api_request(
                            "/auth/register",
                            method="POST",
                            data={
                                "username": username,
                                "email": email,
                                "password": password
                            }
                        )

                        if "error" not in response:
                            st.success("Registration successful! You can now login.")
                            # Switch to login tab
                            st.session_state.show_login = True
                            st.rerun()
                        else:
                            st.error(f"Registration failed: {response['error']}")
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")

def logout():
    """
    Logout the current user.
    """
    if "auth_token" in st.session_state:
        try:
            # Import api_request function
            try:
                from .ui_llamaindex import api_request
            except ImportError:
                from src.local_file_research.ui_llamaindex import api_request

            # Make API request to logout
            api_request("/auth/logout", method="POST")
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")

        # Clear session state regardless of API response
        if "auth_token" in st.session_state:
            del st.session_state.auth_token
        if "username" in st.session_state:
            del st.session_state.username

        st.success("Logged out successfully!")

        # Set current page to Research
        st.session_state.current_page = "Research"
        st.rerun()

def auth_page():
    """
    Render authentication page.
    """
    # Initialize show_login if not in session state
    if "show_login" not in st.session_state:
        st.session_state.show_login = True

    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_form()

    with tab2:
        register_form()

def user_info():
    """
    Display user information in the sidebar.
    """
    if "auth_token" in st.session_state and st.session_state.auth_token and "username" in st.session_state and st.session_state.username:
        st.sidebar.subheader("User")
        st.sidebar.text(f"Logged in as: {st.session_state.username}")

        if st.sidebar.button("Logout"):
            try:
                logout()
            except Exception as e:
                st.sidebar.error(f"Logout error: {str(e)}")
                # Clear session state as fallback
                if "auth_token" in st.session_state:
                    del st.session_state.auth_token
                if "username" in st.session_state:
                    del st.session_state.username
                st.rerun()
    else:
        st.sidebar.subheader("Authentication")
        if st.sidebar.button("Login / Register"):
            st.session_state.current_page = "Auth"
            st.rerun()
