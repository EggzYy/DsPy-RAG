"""
API endpoints for user authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
import os

# Import local modules
from src.local_file_research.auth import (
    create_user, authenticate, validate_token, logout,
    change_password, get_user, list_users, delete_user,
    UserExistsError, UserNotFoundError, InvalidCredentialsError, SessionExpiredError,
    # 2FA functions
    setup_2fa, enable_2fa, disable_2fa, authenticate_with_2fa, verify_2fa_token,
    TOTP_AVAILABLE
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Define request models
class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    role: Optional[str] = "user"

class UserLogin(BaseModel):
    username: str
    password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class TOTPSetup(BaseModel):
    token: str

class TOTPVerify(BaseModel):
    token: str
    totp_token: str

class TOTPEnable(BaseModel):
    totp_token: str

class TOTPDisable(BaseModel):
    password: str

# Helper function to get current user from token
async def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Get the current user from a token.

    Args:
        token: Session token

    Returns:
        User information

    Raises:
        HTTPException: If the token is invalid or expired
    """
    try:
        return validate_token(token)
    except InvalidCredentialsError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except SessionExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Session has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )

@router.post("/register", response_model=Dict[str, Any])
async def register_user(user: UserCreate):
    """
    Register a new user.
    """
    try:
        user_info = create_user(
            username=user.username,
            password=user.password,
            email=user.email,
            role=user.role
        )
        return {
            "message": "User created successfully",
            "user": user_info
        }
    except UserExistsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in register_user endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/login", response_model=Dict[str, Any])
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login and get an authentication token.
    If 2FA is enabled, returns a temporary token that requires verification.
    """
    try:
        # Use the 2FA-aware authentication function
        auth_result = authenticate_with_2fa(
            username=form_data.username,
            password=form_data.password
        )

        if auth_result["requires_2fa"]:
            # If 2FA is required, return a temporary token
            return {
                "access_token": auth_result["token"],
                "token_type": "bearer",
                "requires_2fa": True
            }
        else:
            # If 2FA is not required, return a normal token
            return {
                "access_token": auth_result["token"],
                "token_type": "bearer",
                "requires_2fa": False
            }
    except UserNotFoundError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Error in login_user endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.post("/logout", response_model=Dict[str, Any])
async def logout_user(token: str = Depends(oauth2_scheme)):
    """
    Logout and invalidate the current token.
    """
    try:
        success = logout(token)
        if success:
            return {"message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid token")
    except Exception as e:
        logger.error(f"Error in logout_user endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

@router.get("/me", response_model=Dict[str, Any])
async def get_current_user(user: Dict[str, Any] = Depends(get_current_user_from_token)):
    """
    Get information about the current user.
    """
    return user

@router.post("/change-password", response_model=Dict[str, Any])
async def update_password(
    password_data: PasswordChange,
    user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """
    Change the current user's password.
    """
    try:
        success = change_password(
            username=user["username"],
            current_password=password_data.current_password,
            new_password=password_data.new_password
        )
        if success:
            return {"message": "Password changed successfully"}
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in update_password endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Password change failed: {str(e)}")

@router.get("/users", response_model=List[Dict[str, Any]])
async def get_users(user: Dict[str, Any] = Depends(get_current_user_from_token)):
    """
    List all users. Only available to admin users.
    """
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to access user list")

    try:
        return list_users()
    except Exception as e:
        logger.error(f"Error in get_users endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")

@router.delete("/users/{username}", response_model=Dict[str, Any])
async def remove_user(
    username: str,
    user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """
    Delete a user. Only available to admin users or the user themselves.
    """
    if user["role"] != "admin" and user["username"] != username:
        raise HTTPException(status_code=403, detail="Not authorized to delete this user")

    try:
        success = delete_user(username)
        if success:
            return {"message": f"User {username} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
    except Exception as e:
        logger.error(f"Error in remove_user endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")

@router.post("/2fa/verify", response_model=Dict[str, Any])
async def verify_2fa(data: TOTPVerify):
    """
    Verify a TOTP token and complete the authentication.
    """
    try:
        # Verify the TOTP token
        token = verify_2fa_token(
            temp_token=data.token,
            totp_token=data.totp_token
        )
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except SessionExpiredError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Error in verify_2fa endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"2FA verification failed: {str(e)}")

@router.post("/2fa/setup", response_model=Dict[str, Any])
async def setup_2fa_endpoint(user: Dict[str, Any] = Depends(get_current_user_from_token)):
    """
    Set up 2FA for the current user.
    """
    if not TOTP_AVAILABLE:
        raise HTTPException(status_code=501, detail="2FA functionality is not available")

    try:
        # Set up 2FA
        setup_result = setup_2fa(user["username"])
        return {
            "message": "2FA setup initiated",
            "secret": setup_result["secret"],
            "qr_code": setup_result["qr_code"],
            "uri": setup_result["uri"]
        }
    except Exception as e:
        logger.error(f"Error in setup_2fa_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"2FA setup failed: {str(e)}")

@router.post("/2fa/enable", response_model=Dict[str, Any])
async def enable_2fa_endpoint(
    data: TOTPEnable,
    user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """
    Enable 2FA for the current user after verifying the token.
    """
    if not TOTP_AVAILABLE:
        raise HTTPException(status_code=501, detail="2FA functionality is not available")

    try:
        # Enable 2FA
        success = enable_2fa(
            username=user["username"],
            token=data.totp_token
        )
        if success:
            return {"message": "2FA enabled successfully"}
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Error in enable_2fa_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"2FA enablement failed: {str(e)}")

@router.post("/2fa/disable", response_model=Dict[str, Any])
async def disable_2fa_endpoint(
    data: TOTPDisable,
    user: Dict[str, Any] = Depends(get_current_user_from_token)
):
    """
    Disable 2FA for the current user.
    """
    try:
        # Disable 2FA
        success = disable_2fa(
            username=user["username"],
            password=data.password
        )
        if success:
            return {"message": "2FA disabled successfully"}
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Error in disable_2fa_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"2FA disablement failed: {str(e)}")

def setup_auth_routes(app):
    """
    Set up authentication routes for the FastAPI app.
    """
    app.include_router(router, prefix="/auth", tags=["Authentication"])
    logger.info("Set up authentication routes")
