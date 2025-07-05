"""
Security module for Local File Deep Research.
"""

import os
import json
import time
import uuid
import logging
import hashlib
import secrets
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
SECURITY_DIR = os.environ.get("SECURITY_DIR", "security")
AUDIT_LOG_FILE = os.path.join(SECURITY_DIR, "audit.log")
KEYS_FILE = os.path.join(SECURITY_DIR, "keys.json")
JWT_SECRET = os.environ.get("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 7
PASSWORD_SALT_SIZE = 16
MIN_PASSWORD_LENGTH = 8
ENCRYPTION_KEY_SIZE = 32
PBKDF2_ITERATIONS = 100000

# --- Security Errors ---
class SecurityError(Exception):
    """Base exception for security errors."""
    pass

class AuthenticationError(SecurityError):
    """Exception raised for authentication errors."""
    pass

class AuthorizationError(SecurityError):
    """Exception raised for authorization errors."""
    pass

class EncryptionError(SecurityError):
    """Exception raised for encryption errors."""
    pass

# --- Security Utilities ---
def _ensure_security_dir():
    """Ensure the security directory exists."""
    os.makedirs(SECURITY_DIR, exist_ok=True)
    
    # Create keys file if it doesn't exist
    if not os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, 'w') as f:
            json.dump({}, f)

def _get_jwt_secret():
    """
    Get the JWT secret key.
    
    Returns:
        JWT secret key
    """
    global JWT_SECRET
    
    if JWT_SECRET:
        return JWT_SECRET
    
    # Load from keys file
    _ensure_security_dir()
    try:
        with open(KEYS_FILE, 'r') as f:
            keys = json.load(f)
            if "jwt_secret" in keys:
                JWT_SECRET = keys["jwt_secret"]
                return JWT_SECRET
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    
    # Generate new secret
    JWT_SECRET = secrets.token_hex(32)
    
    # Save to keys file
    try:
        with open(KEYS_FILE, 'r') as f:
            keys = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        keys = {}
    
    keys["jwt_secret"] = JWT_SECRET
    
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f)
    
    return JWT_SECRET

def _get_encryption_key():
    """
    Get the encryption key.
    
    Returns:
        Encryption key
    """
    # Load from keys file
    _ensure_security_dir()
    try:
        with open(KEYS_FILE, 'r') as f:
            keys = json.load(f)
            if "encryption_key" in keys:
                return base64.urlsafe_b64decode(keys["encryption_key"])
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    
    # Generate new key
    encryption_key = Fernet.generate_key()
    
    # Save to keys file
    try:
        with open(KEYS_FILE, 'r') as f:
            keys = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        keys = {}
    
    keys["encryption_key"] = base64.urlsafe_b64encode(encryption_key).decode()
    
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f)
    
    return encryption_key

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
        PBKDF2_ITERATIONS
    ).hex()
    
    return hashed, salt

def _derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive an encryption key from a password and salt.
    
    Args:
        password: Password to derive key from
        salt: Salt for key derivation
        
    Returns:
        Derived key
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=ENCRYPTION_KEY_SIZE,
        salt=salt,
        iterations=PBKDF2_ITERATIONS
    )
    
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

# --- Authentication Functions ---
def generate_token(user_id: str, username: str, role: str = "user", expiry_days: int = JWT_EXPIRY_DAYS) -> str:
    """
    Generate a JWT token for a user.
    
    Args:
        user_id: User ID
        username: Username
        role: User role
        expiry_days: Token expiry in days
        
    Returns:
        JWT token
    """
    # Get JWT secret
    secret = _get_jwt_secret()
    
    # Set expiry
    expiry = datetime.utcnow() + timedelta(days=expiry_days)
    
    # Create payload
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expiry,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    
    # Generate token
    token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
    
    # Log token generation
    log_audit_event("token_generated", {"user_id": user_id, "username": username, "role": role})
    
    return token

def validate_token(token: str) -> Dict[str, Any]:
    """
    Validate a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        Token payload
        
    Raises:
        AuthenticationError: If the token is invalid
    """
    # Get JWT secret
    secret = _get_jwt_secret()
    
    try:
        # Decode token
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        
        # Log token validation
        log_audit_event("token_validated", {"user_id": payload.get("sub"), "username": payload.get("username")})
        
        return payload
    except jwt.ExpiredSignatureError:
        log_audit_event("token_expired", {"token": token[:10] + "..."})
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        log_audit_event("token_invalid", {"token": token[:10] + "...", "error": str(e)})
        raise AuthenticationError(f"Invalid token: {e}")

def check_permission(user: Dict[str, Any], resource_type: str, resource_id: str, action: str) -> bool:
    """
    Check if a user has permission to perform an action on a resource.
    
    Args:
        user: User information
        resource_type: Type of resource
        resource_id: Resource ID
        action: Action to perform
        
    Returns:
        True if the user has permission, False otherwise
    """
    # Admin role has all permissions
    if user.get("role") == "admin":
        return True
    
    # Check specific permissions based on resource type and action
    if resource_type == "project":
        # Project permissions
        if action in ["view", "list"]:
            # Check if user is a member of the project or has access via shares
            from .collaboration import get_project
            try:
                project = get_project(resource_id)
                if user.get("username") in project.get("members", []):
                    return True
                
                # Check shares
                from .collaboration import get_shares
                shares = get_shares(resource_id, user.get("username"))
                if shares:
                    if action == "view" or action == "list":
                        return True
            except:
                pass
        elif action in ["edit", "delete"]:
            # Check if user is the owner of the project
            from .collaboration import get_project
            try:
                project = get_project(resource_id)
                if project.get("owner") == user.get("username"):
                    return True
                
                # Check shares with write permission
                from .collaboration import get_shares
                shares = get_shares(resource_id, user.get("username"))
                if shares and any(s.get("permission") == "write" for s in shares):
                    return True
            except:
                pass
    elif resource_type == "document":
        # Document permissions
        if action in ["view", "list"]:
            # Check if document is in a project the user has access to
            from .versioning import get_document
            try:
                document = get_document(resource_id)
                project_id = document.get("project_id")
                if project_id:
                    return check_permission(user, "project", project_id, "view")
            except:
                pass
        elif action in ["edit", "delete"]:
            # Check if document is in a project the user has edit permission for
            from .versioning import get_document
            try:
                document = get_document(resource_id)
                project_id = document.get("project_id")
                if project_id:
                    return check_permission(user, "project", project_id, "edit")
            except:
                pass
    
    # Default to no permission
    return False

def require_permission(user: Dict[str, Any], resource_type: str, resource_id: str, action: str):
    """
    Require a user to have permission to perform an action on a resource.
    
    Args:
        user: User information
        resource_type: Type of resource
        resource_id: Resource ID
        action: Action to perform
        
    Raises:
        AuthorizationError: If the user doesn't have permission
    """
    if not check_permission(user, resource_type, resource_id, action):
        log_audit_event("permission_denied", {
            "user_id": user.get("sub"),
            "username": user.get("username"),
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action
        })
        raise AuthorizationError(f"You don't have permission to {action} this {resource_type}")

# --- Encryption Functions ---
def encrypt_data(data: Union[str, bytes]) -> str:
    """
    Encrypt data.
    
    Args:
        data: Data to encrypt
        
    Returns:
        Encrypted data as a base64 string
        
    Raises:
        EncryptionError: If encryption fails
    """
    try:
        # Get encryption key
        key = _get_encryption_key()
        
        # Create Fernet cipher
        cipher = Fernet(key)
        
        # Encrypt data
        if isinstance(data, str):
            data = data.encode("utf-8")
        
        encrypted = cipher.encrypt(data)
        
        # Return as base64 string
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        raise EncryptionError(f"Failed to encrypt data: {e}")

def decrypt_data(encrypted_data: str) -> bytes:
    """
    Decrypt data.
    
    Args:
        encrypted_data: Encrypted data as a base64 string
        
    Returns:
        Decrypted data as bytes
        
    Raises:
        EncryptionError: If decryption fails
    """
    try:
        # Get encryption key
        key = _get_encryption_key()
        
        # Create Fernet cipher
        cipher = Fernet(key)
        
        # Decode base64
        encrypted = base64.urlsafe_b64decode(encrypted_data)
        
        # Decrypt data
        return cipher.decrypt(encrypted)
    except Exception as e:
        raise EncryptionError(f"Failed to decrypt data: {e}")

def encrypt_file(input_path: str, output_path: str):
    """
    Encrypt a file.
    
    Args:
        input_path: Path to the input file
        output_path: Path to the output file
        
    Raises:
        EncryptionError: If encryption fails
    """
    try:
        # Read input file
        with open(input_path, "rb") as f:
            data = f.read()
        
        # Encrypt data
        encrypted = encrypt_data(data)
        
        # Write output file
        with open(output_path, "w") as f:
            f.write(encrypted)
    except Exception as e:
        raise EncryptionError(f"Failed to encrypt file: {e}")

def decrypt_file(input_path: str, output_path: str):
    """
    Decrypt a file.
    
    Args:
        input_path: Path to the input file
        output_path: Path to the output file
        
    Raises:
        EncryptionError: If decryption fails
    """
    try:
        # Read input file
        with open(input_path, "r") as f:
            encrypted = f.read()
        
        # Decrypt data
        decrypted = decrypt_data(encrypted)
        
        # Write output file
        with open(output_path, "wb") as f:
            f.write(decrypted)
    except Exception as e:
        raise EncryptionError(f"Failed to decrypt file: {e}")

# --- Audit Logging ---
def log_audit_event(event_type: str, event_data: Dict[str, Any]):
    """
    Log an audit event.
    
    Args:
        event_type: Type of event
        event_data: Event data
    """
    # Ensure security directory exists
    _ensure_security_dir()
    
    # Create event
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "event_data": event_data,
        "ip_address": os.environ.get("REMOTE_ADDR", "unknown"),
        "user_agent": os.environ.get("HTTP_USER_AGENT", "unknown")
    }
    
    # Log to file
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def get_audit_logs(event_type: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get audit logs.
    
    Args:
        event_type: Optional event type to filter by
        start_time: Optional start time to filter by
        end_time: Optional end time to filter by
        limit: Maximum number of logs to return
        
    Returns:
        List of audit logs
    """
    # Ensure security directory exists
    _ensure_security_dir()
    
    # Check if audit log file exists
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    
    # Read logs
    logs = []
    with open(AUDIT_LOG_FILE, "r") as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                
                # Apply filters
                if event_type and log.get("event_type") != event_type:
                    continue
                
                if start_time:
                    log_time = datetime.fromisoformat(log.get("timestamp"))
                    if log_time < start_time:
                        continue
                
                if end_time:
                    log_time = datetime.fromisoformat(log.get("timestamp"))
                    if log_time > end_time:
                        continue
                
                logs.append(log)
                
                # Check limit
                if len(logs) >= limit:
                    break
            except:
                continue
    
    return logs

# --- Two-Factor Authentication ---
def generate_totp_secret() -> str:
    """
    Generate a secret for TOTP (Time-based One-Time Password).
    
    Returns:
        TOTP secret
    """
    try:
        import pyotp
    except ImportError:
        raise ImportError("pyotp is required for TOTP functionality")
    
    return pyotp.random_base32()

def generate_totp_uri(secret: str, username: str, issuer: str = "LocalFileResearch") -> str:
    """
    Generate a TOTP URI for QR code generation.
    
    Args:
        secret: TOTP secret
        username: Username
        issuer: Issuer name
        
    Returns:
        TOTP URI
    """
    try:
        import pyotp
    except ImportError:
        raise ImportError("pyotp is required for TOTP functionality")
    
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(username, issuer_name=issuer)

def verify_totp(secret: str, token: str) -> bool:
    """
    Verify a TOTP token.
    
    Args:
        secret: TOTP secret
        token: TOTP token
        
    Returns:
        True if the token is valid, False otherwise
    """
    try:
        import pyotp
    except ImportError:
        raise ImportError("pyotp is required for TOTP functionality")
    
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

# --- Rate Limiting ---
_rate_limit_data = {}

def check_rate_limit(key: str, limit: int, period: int) -> bool:
    """
    Check if a rate limit has been exceeded.
    
    Args:
        key: Rate limit key (e.g., IP address, username)
        limit: Maximum number of requests
        period: Time period in seconds
        
    Returns:
        True if the rate limit has not been exceeded, False otherwise
    """
    global _rate_limit_data
    
    # Get current time
    now = time.time()
    
    # Initialize rate limit data for key
    if key not in _rate_limit_data:
        _rate_limit_data[key] = {"count": 0, "reset_time": now + period}
    
    # Check if rate limit period has expired
    if now > _rate_limit_data[key]["reset_time"]:
        # Reset rate limit
        _rate_limit_data[key] = {"count": 1, "reset_time": now + period}
        return True
    
    # Increment count
    _rate_limit_data[key]["count"] += 1
    
    # Check if rate limit has been exceeded
    if _rate_limit_data[key]["count"] > limit:
        return False
    
    return True

def get_rate_limit_remaining(key: str) -> Tuple[int, int]:
    """
    Get the remaining rate limit for a key.
    
    Args:
        key: Rate limit key
        
    Returns:
        Tuple of (remaining requests, seconds until reset)
    """
    global _rate_limit_data
    
    # Get current time
    now = time.time()
    
    # Check if key exists
    if key not in _rate_limit_data:
        return 0, 0
    
    # Calculate remaining requests and time
    data = _rate_limit_data[key]
    remaining = max(0, data["count"])
    reset_time = max(0, int(data["reset_time"] - now))
    
    return remaining, reset_time
