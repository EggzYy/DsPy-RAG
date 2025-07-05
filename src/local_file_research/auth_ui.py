"""
Standalone authentication UI for the Local File Research application.
"""

import streamlit as st
import requests
import logging
import os
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Get API URL from environment variable or use default
try:
    from .config import API_PORT
except ImportError:
    try:
        from src.local_file_research.config import API_PORT
    except ImportError:
        API_PORT = 8006

API_URL = os.environ.get("API_URL", f"http://localhost:{API_PORT}")

def make_api_request(endpoint, method="GET", data=None, files=None, auth_type="bearer"):
    """
    Make API request to the backend.

    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data
        files: Request files
        auth_type: Authentication type (bearer or form)
    """
    url = f"{API_URL}{endpoint}"
    headers = {}

    # Add authentication token if available
    if auth_type == "bearer" and "auth_token" in st.session_state:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, data=data, files=files)
            elif auth_type == "form":
                # For form-based authentication (login endpoint)
                response = requests.post(
                    url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data)
        elif method == "PUT":
            headers["Content-Type"] = "application/json"
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            logger.error(f"Unsupported method: {method}")
            return {"error": f"Unsupported method: {method}"}

        if response.status_code >= 200 and response.status_code < 300:
            return response.json()
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            return {"error": f"API request failed: {response.status_code} - {response.text}"}
    except Exception as e:
        logger.error(f"API request error: {e}")
        return {"error": f"API request error: {e}"}

def verify_2fa_form():
    """
    Render 2FA verification form.
    """
    st.subheader("Two-Factor Authentication")
    st.info("Please enter the verification code from your authenticator app")

    # 2FA verification form
    with st.form("2fa_form"):
        totp_token = st.text_input("Verification Code")
        submit = st.form_submit_button("Verify")

        if submit and totp_token:
            with st.spinner("Verifying..."):
                try:
                    # Make API request to verify 2FA
                    response = make_api_request(
                        "/auth/2fa/verify",
                        method="POST",
                        data={
                            "token": st.session_state.temp_token,
                            "totp_token": totp_token
                        }
                    )

                    if "error" not in response:
                        # Store token in session state
                        st.session_state.auth_token = response["access_token"]
                        st.session_state.authenticated = True
                        st.session_state.requires_2fa = False
                        st.session_state.temp_token = None
                        st.success("Two-factor authentication successful!")

                        # Provide a button to return to the main app
                        st.info("Login successful! You can now return to the main application.")

                        # Use JavaScript to redirect to the main app
                        js_code = """
                        <script>
                            // Wait 2 seconds before redirecting to give user time to see the success message
                            setTimeout(function() {
                                window.open('http://localhost:8501', '_self');
                            }, 2000);
                        </script>
                        """
                        st.components.v1.html(js_code, height=0)
                        st.markdown("[Click here if not redirected automatically](http://localhost:8501)")
                        st.rerun()
                    else:
                        st.error(f"Verification failed: {response['error']}")
                except Exception as e:
                    st.error(f"Verification error: {str(e)}")

def login_form():
    """
    Render login form.
    """
    # Check if 2FA verification is required
    if st.session_state.get("requires_2fa", False) and st.session_state.get("temp_token"):
        verify_2fa_form()
        return

    st.subheader("Login")

    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit and username and password:
            with st.spinner("Logging in..."):
                try:
                    # Make API request to login
                    response = make_api_request(
                        "/auth/login",
                        method="POST",
                        data={"username": username, "password": password},
                        auth_type="form"
                    )

                    if "error" not in response:
                        # Check if 2FA is required
                        if response.get("requires_2fa", False):
                            # Store temporary token and username
                            st.session_state.temp_token = response["access_token"]
                            st.session_state.username = username
                            st.session_state.requires_2fa = True
                            st.info("Two-factor authentication required")
                            st.rerun()  # Rerun to show 2FA form
                        else:
                            # Normal login without 2FA
                            st.session_state.auth_token = response["access_token"]
                            st.session_state.username = username
                            st.session_state.authenticated = True
                            st.success(f"Welcome, {username}!")

                            # Provide a button to return to the main app
                            st.info("Login successful! You can now return to the main application.")

                            # Use JavaScript to redirect to the main app
                            js_code = """
                            <script>
                                // Wait 2 seconds before redirecting to give user time to see the success message
                                setTimeout(function() {
                                    window.open('http://localhost:8501', '_self');
                                }, 2000);
                            </script>
                            """
                            st.components.v1.html(js_code, height=0)
                            st.markdown("[Click here if not redirected automatically](http://localhost:8501)")
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
                        # Make API request to register
                        response = make_api_request(
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
            # Make API request to logout
            make_api_request("/auth/logout", method="POST")
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")

        # Clear session state regardless of API response
        if "auth_token" in st.session_state:
            del st.session_state.auth_token
        if "username" in st.session_state:
            del st.session_state.username
        if "requires_2fa" in st.session_state:
            del st.session_state.requires_2fa
        if "temp_token" in st.session_state:
            del st.session_state.temp_token
        if "totp_secret" in st.session_state:
            del st.session_state.totp_secret
        if "totp_qr_code" in st.session_state:
            del st.session_state.totp_qr_code
        if "totp_uri" in st.session_state:
            del st.session_state.totp_uri
        if "show_2fa_setup" in st.session_state:
            del st.session_state.show_2fa_setup

        # Set authenticated to False
        st.session_state.authenticated = False

        st.success("Logged out successfully!")
        st.rerun()

def setup_2fa_page():
    """
    Render 2FA setup page.
    """
    st.subheader("Two-Factor Authentication (2FA)")

    # Add detailed explanation about 2FA
    st.info("""
    **What is Two-Factor Authentication?**

    Two-factor authentication (2FA) adds an extra layer of security to your account by requiring:
    1. Something you know (your password)
    2. Something you have (a code from your authenticator app)

    This means that even if someone gets your password, they still can't access your account without your authenticator app.

    **How to set up 2FA:**
    1. Click the "Set Up Two-Factor Authentication" button below
    2. Scan the QR code with an authenticator app like Google Authenticator, Microsoft Authenticator, or Authy
    3. Enter the verification code from your app to complete the setup
    """)

    # Add recommended apps
    st.markdown("**Recommended Authenticator Apps:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[Google Authenticator](https://play.google.com/store/apps/details?id=com.google.android.apps.authenticator2)")
    with col2:
        st.markdown("[Microsoft Authenticator](https://www.microsoft.com/en-us/security/mobile-authenticator-app)")
    with col3:
        st.markdown("[Authy](https://authy.com/download/)")

    st.markdown("---")

    # Check if user is authenticated
    if not st.session_state.get("authenticated", False) or not st.session_state.get("auth_token"):
        st.error("You must be logged in to set up 2FA")
        return

    # Get user info
    user_info = make_api_request("/auth/me")
    if "error" in user_info:
        st.error(f"Failed to get user information: {user_info['error']}")
        return

    # Check if 2FA is already enabled
    if user_info.get("totp_enabled", False):
        st.success("Two-factor authentication is already enabled for your account")

        # Disable 2FA option
        with st.form("disable_2fa_form"):
            st.warning("Disabling two-factor authentication will reduce the security of your account")
            password = st.text_input("Enter your password to confirm", type="password")
            submit = st.form_submit_button("Disable 2FA")

            if submit and password:
                with st.spinner("Disabling 2FA..."):
                    response = make_api_request(
                        "/auth/2fa/disable",
                        method="POST",
                        data={"password": password}
                    )

                    if "error" not in response:
                        st.success("Two-factor authentication has been disabled")
                        st.rerun()
                    else:
                        st.error(f"Failed to disable 2FA: {response['error']}")
    else:
        # Set up 2FA
        if st.button("Set Up Two-Factor Authentication"):
            with st.spinner("Setting up 2FA..."):
                response = make_api_request("/auth/2fa/setup", method="POST")

                if "error" not in response:
                    # Store setup information in session state
                    st.session_state.totp_secret = response["secret"]
                    st.session_state.totp_qr_code = response["qr_code"]
                    st.session_state.totp_uri = response["uri"]
                    st.session_state.show_2fa_setup = True
                else:
                    st.error(f"Failed to set up 2FA: {response['error']}")

        # Show 2FA setup instructions if available
        if st.session_state.get("show_2fa_setup", False):
            st.success("Two-factor authentication setup initiated")

            # Display QR code
            st.subheader("1. Scan this QR code with your authenticator app")
            qr_code = st.session_state.totp_qr_code
            st.image(f"data:image/png;base64,{qr_code}", width=300)

            # Display manual entry option
            st.subheader("2. Or enter this code manually")
            st.code(st.session_state.totp_secret)

            # Verify and enable 2FA
            st.subheader("3. Verify and enable 2FA")
            with st.form("enable_2fa_form"):
                totp_token = st.text_input("Enter the verification code from your app")
                submit = st.form_submit_button("Verify and Enable")

                if submit and totp_token:
                    with st.spinner("Verifying and enabling 2FA..."):
                        response = make_api_request(
                            "/auth/2fa/enable",
                            method="POST",
                            data={"totp_token": totp_token}
                        )

                        if "error" not in response:
                            # Clear setup information
                            if "totp_secret" in st.session_state:
                                del st.session_state.totp_secret
                            if "totp_qr_code" in st.session_state:
                                del st.session_state.totp_qr_code
                            if "totp_uri" in st.session_state:
                                del st.session_state.totp_uri
                            if "show_2fa_setup" in st.session_state:
                                del st.session_state.show_2fa_setup

                            st.success("Two-factor authentication has been enabled for your account")
                            st.rerun()
                        else:
                            st.error(f"Failed to enable 2FA: {response['error']}")

def auth_page():
    """
    Render authentication page.
    """
    st.title("📚 Local File Research - Authentication & Security")
    st.markdown("*User authentication and security settings*")

    # Add explanation about this page
    st.info("""
    **This is the authentication and security page (port 8502).**

    Here you can:
    - Register a new account
    - Login to your account
    - Set up Two-Factor Authentication (2FA)
    - Manage your security settings

    After logging in, you'll see additional security options in the Account and Security tabs.
    """)

    # Check if user is authenticated
    if st.session_state.get("authenticated", False):
        # Show user settings
        tab1, tab2 = st.tabs(["Account", "Security"])

        with tab1:
            st.subheader("Account Information")
            user_info = make_api_request("/auth/me")
            if "error" not in user_info:
                st.write(f"**Username:** {user_info['username']}")
                st.write(f"**Email:** {user_info['email']}")
                st.write(f"**Role:** {user_info['role']}")
            else:
                st.error(f"Failed to get user information: {user_info['error']}")

        with tab2:
            setup_2fa_page()
    else:
        # Initialize show_login if not in session state
        if "show_login" not in st.session_state:
            st.session_state.show_login = True

        # Create tabs for login and registration
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            login_form()

        with tab2:
            register_form()

def user_info_sidebar():
    """
    Display user information in the sidebar.
    """
    if st.session_state.get("authenticated", False) and st.session_state.get("username"):
        st.sidebar.subheader("User")
        st.sidebar.text(f"Logged in as: {st.session_state.username}")

        if st.sidebar.button("Logout"):
            logout()
    else:
        st.sidebar.subheader("Authentication")
        if st.sidebar.button("Login / Register"):
            st.session_state.authenticated = False
            st.rerun()

def is_authenticated():
    """
    Check if the user is authenticated.

    Returns:
        bool: True if authenticated, False otherwise
    """
    return st.session_state.get("authenticated", False)

def initialize_auth_state():
    """
    Initialize authentication state.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "requires_2fa" not in st.session_state:
        st.session_state.requires_2fa = False
    if "temp_token" not in st.session_state:
        st.session_state.temp_token = None
    if "totp_secret" not in st.session_state:
        st.session_state.totp_secret = None
    if "totp_qr_code" not in st.session_state:
        st.session_state.totp_qr_code = None
    if "totp_uri" not in st.session_state:
        st.session_state.totp_uri = None
    if "show_2fa_setup" not in st.session_state:
        st.session_state.show_2fa_setup = False
