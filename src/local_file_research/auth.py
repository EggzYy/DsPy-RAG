"""
User authentication module for Local File Deep Research.
"""

import os
import json
import time
import uuid
import hashlib
import logging
import secrets
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Import 2FA libraries
try:
    import pyotp
    import qrcode
    from io import BytesIO
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    logger.warning("pyotp or qrcode not available. 2FA functionality will be disabled.")

# Constants
AUTH_DIR = os.environ.get("AUTH_DIR", "auth_data")
USERS_FILE = os.path.join(AUTH_DIR, "users.json")
SESSIONS_FILE = os.path.join(AUTH_DIR, "sessions.json")
TOKEN_EXPIRY_DAYS = 7
PASSWORD_SALT_SIZE = 16
MIN_PASSWORD_LENGTH = 8
TOTP_ISSUER = "LocalFileResearch"
TEMP_TOKEN_EXPIRY_MINUTES = 5  # Temporary token expiry for 2FA verification

class AuthError(Exception):
    """Base exception for authentication errors."""
    pass

class UserExistsError(AuthError):
    """Exception raised when trying to create a user that already exists."""
    pass

class UserNotFoundError(AuthError):
    """Exception raised when a user is not found."""
    pass

class InvalidCredentialsError(AuthError):
    """Exception raised when credentials are invalid."""
    pass

class SessionExpiredError(AuthError):
    """Exception raised when a session has expired."""
    pass

def _ensure_auth_dir():
    """Ensure the authentication directory exists."""
    os.makedirs(AUTH_DIR, exist_ok=True)

    # Create users file if it doesn't exist
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)

    # Create sessions file if it doesn't exist
    if not os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, 'w') as f:
            json.dump({}, f)

def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password with a salt.

    Args:
        password: The password to hash
        salt: Optional salt to use (generates a new one if not provided)

    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(PASSWORD_SALT_SIZE)

    # Hash the password with the salt
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # Number of iterations
    ).hex()

    return hashed, salt

def _load_users() -> Dict[str, Dict[str, Any]]:
    """Load users from the users file."""
    _ensure_auth_dir()
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _save_users(users: Dict[str, Dict[str, Any]]):
    """Save users to the users file."""
    _ensure_auth_dir()
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def _load_sessions() -> Dict[str, Dict[str, Any]]:
    """Load sessions from the sessions file."""
    _ensure_auth_dir()
    try:
        with open(SESSIONS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _save_sessions(sessions: Dict[str, Dict[str, Any]]):
    """Save sessions to the sessions file."""
    _ensure_auth_dir()
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

def create_user(username: str, password: str, email: str, role: str = "user") -> Dict[str, Any]:
    """
    Create a new user.

    Args:
        username: Username
        password: Password
        email: Email address
        role: User role (default: "user")

    Returns:
        User information

    Raises:
        UserExistsError: If the username already exists
        ValueError: If the password is too short
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")

    users = _load_users()

    if username in users:
        raise UserExistsError(f"User '{username}' already exists")

    # Hash the password
    hashed_password, salt = _hash_password(password)

    # Create the user
    user = {
        "username": username,
        "email": email,
        "password_hash": hashed_password,
        "salt": salt,
        "role": role,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "totp_secret": None,  # TOTP secret for 2FA
        "totp_enabled": False  # Whether 2FA is enabled
    }

    # Save the user
    users[username] = user
    _save_users(users)

    # Return the user without sensitive information
    return {
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"]
    }

def authenticate(username: str, password: str) -> str:
    """
    Authenticate a user and return a session token.

    Args:
        username: Username
        password: Password

    Returns:
        Session token

    Raises:
        UserNotFoundError: If the user is not found
        InvalidCredentialsError: If the credentials are invalid
    """
    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Hash the password with the user's salt
    hashed_password, _ = _hash_password(password, user["salt"])

    # Check if the password is correct
    if hashed_password != user["password_hash"]:
        raise InvalidCredentialsError("Invalid password")

    # Create a session token
    token = str(uuid.uuid4())
    expiry = (datetime.now() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()

    # Save the session
    sessions = _load_sessions()
    sessions[token] = {
        "username": username,
        "created_at": datetime.now().isoformat(),
        "expires_at": expiry
    }
    _save_sessions(sessions)

    # Update last login time
    user["last_login"] = datetime.now().isoformat()
    _save_users(users)

    return token

def validate_token(token: str) -> Dict[str, Any]:
    """
    Validate a session token and return the user information.

    Args:
        token: Session token

    Returns:
        User information

    Raises:
        InvalidCredentialsError: If the token is invalid
        SessionExpiredError: If the session has expired
    """
    sessions = _load_sessions()

    if token not in sessions:
        raise InvalidCredentialsError("Invalid session token")

    session = sessions[token]

    # Check if the session has expired
    expiry = datetime.fromisoformat(session["expires_at"])
    if datetime.now() > expiry:
        # Remove the expired session
        del sessions[token]
        _save_sessions(sessions)
        raise SessionExpiredError("Session has expired")

    # Get the user information
    users = _load_users()
    username = session["username"]

    if username not in users:
        # This should not happen, but just in case
        del sessions[token]
        _save_sessions(sessions)
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Return the user without sensitive information
    return {
        "username": user["username"],
        "email": user["email"],
        "role": user["role"]
    }

def logout(token: str) -> bool:
    """
    Logout a user by invalidating their session token.

    Args:
        token: Session token

    Returns:
        True if the session was found and removed, False otherwise
    """
    sessions = _load_sessions()

    if token in sessions:
        del sessions[token]
        _save_sessions(sessions)
        return True

    return False

def change_password(username: str, current_password: str, new_password: str) -> bool:
    """
    Change a user's password.

    Args:
        username: Username
        current_password: Current password
        new_password: New password

    Returns:
        True if the password was changed successfully

    Raises:
        UserNotFoundError: If the user is not found
        InvalidCredentialsError: If the current password is invalid
        ValueError: If the new password is too short
    """
    if len(new_password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")

    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Verify the current password
    hashed_current, _ = _hash_password(current_password, user["salt"])
    if hashed_current != user["password_hash"]:
        raise InvalidCredentialsError("Invalid current password")

    # Hash the new password
    hashed_new, salt = _hash_password(new_password)

    # Update the user
    user["password_hash"] = hashed_new
    user["salt"] = salt
    _save_users(users)

    return True

def get_user(username: str) -> Dict[str, Any]:
    """
    Get user information.

    Args:
        username: Username

    Returns:
        User information

    Raises:
        UserNotFoundError: If the user is not found
    """
    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Return the user without sensitive information
    return {
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
        "totp_enabled": user.get("totp_enabled", False)  # Include 2FA status
    }

def list_users() -> List[Dict[str, Any]]:
    """
    List all users.

    Returns:
        List of users without sensitive information
    """
    users = _load_users()

    return [
        {
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
            "totp_enabled": user.get("totp_enabled", False)  # Include 2FA status
        }
        for user in users.values()
    ]

def delete_user(username: str) -> bool:
    """
    Delete a user.

    Args:
        username: Username

    Returns:
        True if the user was deleted, False otherwise
    """
    users = _load_users()

    if username not in users:
        return False

    # Delete the user
    del users[username]
    _save_users(users)

    # Delete any sessions for this user
    sessions = _load_sessions()
    sessions = {
        token: session
        for token, session in sessions.items()
        if session["username"] != username
    }
    _save_sessions(sessions)

    return True

def cleanup_expired_sessions():
    """Clean up expired sessions."""
    sessions = _load_sessions()
    now = datetime.now()

    # Filter out expired sessions
    valid_sessions = {}
    for token, session in sessions.items():
        expiry = datetime.fromisoformat(session["expires_at"])
        if now <= expiry:
            valid_sessions[token] = session

    # Save the valid sessions
    _save_sessions(valid_sessions)

    return len(sessions) - len(valid_sessions)  # Number of expired sessions removed

def get_current_user(token: Optional[str] = None) -> str:
    """
    Get the current user from a token.

    Args:
        token: Session token

    Returns:
        Username

    Raises:
        InvalidCredentialsError: If the token is invalid
        SessionExpiredError: If the session has expired
    """
    if token is None:
        return "default_user"

    try:
        # Validate the token and get user info
        user_info = validate_token(token)
        return user_info["username"]
    except (InvalidCredentialsError, SessionExpiredError):
        # If token validation fails, return default user
        # This allows backward compatibility with existing code
        return "default_user"

# --- Two-Factor Authentication Functions ---

def generate_totp_secret() -> str:
    """
    Generate a secret for TOTP (Time-based One-Time Password).

    Returns:
        TOTP secret

    Raises:
        ImportError: If pyotp is not available
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp is required for TOTP functionality")

    return pyotp.random_base32()

def generate_totp_uri(username: str, secret: str) -> str:
    """
    Generate a TOTP URI for QR code generation.

    Args:
        username: Username
        secret: TOTP secret

    Returns:
        TOTP URI

    Raises:
        ImportError: If pyotp is not available
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp is required for TOTP functionality")

    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(username, issuer_name=TOTP_ISSUER)

def generate_totp_qr_code(username: str, secret: str) -> bytes:
    """
    Generate a QR code for TOTP setup.

    Args:
        username: Username
        secret: TOTP secret

    Returns:
        QR code image as bytes

    Raises:
        ImportError: If pyotp or qrcode is not available
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp and qrcode are required for TOTP functionality")

    # Generate TOTP URI
    uri = generate_totp_uri(username, secret)

    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def verify_totp(secret: str, token: str) -> bool:
    """
    Verify a TOTP token.

    Args:
        secret: TOTP secret
        token: TOTP token

    Returns:
        True if the token is valid, False otherwise

    Raises:
        ImportError: If pyotp is not available
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp is required for TOTP functionality")

    totp = pyotp.TOTP(secret)
    return totp.verify(token)

def setup_2fa(username: str) -> Dict[str, Any]:
    """
    Set up 2FA for a user.

    Args:
        username: Username

    Returns:
        Dictionary with TOTP secret and QR code

    Raises:
        UserNotFoundError: If the user is not found
        ImportError: If pyotp or qrcode is not available
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp and qrcode are required for TOTP functionality")

    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Generate TOTP secret
    secret = generate_totp_secret()

    # Store the secret (but don't enable 2FA yet)
    user["totp_secret"] = secret
    user["totp_enabled"] = False
    _save_users(users)

    # Generate QR code
    qr_code = generate_totp_qr_code(username, secret)

    return {
        "secret": secret,
        "qr_code": base64.b64encode(qr_code).decode('utf-8'),
        "uri": generate_totp_uri(username, secret)
    }

def enable_2fa(username: str, token: str) -> bool:
    """
    Enable 2FA for a user after verifying the token.

    Args:
        username: Username
        token: TOTP token

    Returns:
        True if 2FA was enabled successfully

    Raises:
        UserNotFoundError: If the user is not found
        InvalidCredentialsError: If the token is invalid
        ImportError: If pyotp is not available
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp is required for TOTP functionality")

    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Check if the user has a TOTP secret
    if not user.get("totp_secret"):
        raise InvalidCredentialsError("2FA setup not initiated")

    # Verify the token
    if not verify_totp(user["totp_secret"], token):
        raise InvalidCredentialsError("Invalid TOTP token")

    # Enable 2FA
    user["totp_enabled"] = True
    _save_users(users)

    return True

def disable_2fa(username: str, password: str) -> bool:
    """
    Disable 2FA for a user.

    Args:
        username: Username
        password: Password (for verification)

    Returns:
        True if 2FA was disabled successfully

    Raises:
        UserNotFoundError: If the user is not found
        InvalidCredentialsError: If the password is invalid
    """
    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Verify the password
    hashed_password, _ = _hash_password(password, user["salt"])
    if hashed_password != user["password_hash"]:
        raise InvalidCredentialsError("Invalid password")

    # Disable 2FA
    user["totp_enabled"] = False
    user["totp_secret"] = None
    _save_users(users)

    return True

def authenticate_with_2fa(username: str, password: str) -> Dict[str, Any]:
    """
    First step of 2FA authentication.

    Args:
        username: Username
        password: Password

    Returns:
        Dictionary with temporary token and 2FA status

    Raises:
        UserNotFoundError: If the user is not found
        InvalidCredentialsError: If the password is invalid
    """
    users = _load_users()

    if username not in users:
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Hash the password with the user's salt
    hashed_password, _ = _hash_password(password, user["salt"])

    # Check if the password is correct
    if hashed_password != user["password_hash"]:
        raise InvalidCredentialsError("Invalid password")

    # Check if 2FA is enabled
    if not user.get("totp_enabled", False):
        # If 2FA is not enabled, authenticate normally
        token = authenticate(username, password)
        return {
            "token": token,
            "requires_2fa": False
        }

    # Create a temporary token for 2FA verification
    temp_token = str(uuid.uuid4())
    expiry = (datetime.now() + timedelta(minutes=TEMP_TOKEN_EXPIRY_MINUTES)).isoformat()

    # Save the temporary session
    sessions = _load_sessions()
    sessions[temp_token] = {
        "username": username,
        "created_at": datetime.now().isoformat(),
        "expires_at": expiry,
        "is_temporary": True,  # Mark as temporary for 2FA verification
        "requires_2fa": True
    }
    _save_sessions(sessions)

    return {
        "token": temp_token,
        "requires_2fa": True
    }

def verify_2fa_token(temp_token: str, totp_token: str) -> str:
    """
    Verify a TOTP token and complete the authentication.

    Args:
        temp_token: Temporary token from first authentication step
        totp_token: TOTP token

    Returns:
        Final session token

    Raises:
        InvalidCredentialsError: If the temporary token is invalid
        SessionExpiredError: If the temporary token has expired
        InvalidCredentialsError: If the TOTP token is invalid
    """
    if not TOTP_AVAILABLE:
        raise ImportError("pyotp is required for TOTP functionality")

    sessions = _load_sessions()

    if temp_token not in sessions:
        raise InvalidCredentialsError("Invalid temporary token")

    session = sessions[temp_token]

    # Check if the session is temporary and requires 2FA
    if not session.get("is_temporary") or not session.get("requires_2fa"):
        raise InvalidCredentialsError("Invalid temporary token")

    # Check if the session has expired
    expiry = datetime.fromisoformat(session["expires_at"])
    if datetime.now() > expiry:
        # Remove the expired session
        del sessions[temp_token]
        _save_sessions(sessions)
        raise SessionExpiredError("Temporary token has expired")

    # Get the user
    username = session["username"]
    users = _load_users()

    if username not in users:
        # This should not happen, but just in case
        del sessions[temp_token]
        _save_sessions(sessions)
        raise UserNotFoundError(f"User '{username}' not found")

    user = users[username]

    # Verify the TOTP token
    if not verify_totp(user["totp_secret"], totp_token):
        raise InvalidCredentialsError("Invalid TOTP token")

    # Remove the temporary session
    del sessions[temp_token]
    _save_sessions(sessions)

    # Create a new session token
    token = str(uuid.uuid4())
    expiry = (datetime.now() + timedelta(days=TOKEN_EXPIRY_DAYS)).isoformat()

    # Save the session
    sessions[token] = {
        "username": username,
        "created_at": datetime.now().isoformat(),
        "expires_at": expiry
    }
    _save_sessions(sessions)

    # Update last login time
    user["last_login"] = datetime.now().isoformat()
    _save_users(users)

    return token
