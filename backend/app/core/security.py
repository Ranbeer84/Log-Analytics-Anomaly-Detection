"""
Security utilities for authentication and authorization
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()


class SecurityManager:
    """Manager for security operations"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_minutes = 60
        self.refresh_token_expire_days = 7
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create JWT refresh token
        
        Args:
            data: Data to encode in token
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and verify JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token data
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}"
            )
    
    def generate_api_key(self, length: int = 32) -> str:
        """
        Generate a random API key
        
        Args:
            length: Length of the API key
            
        Returns:
            Random API key string
        """
        return secrets.token_urlsafe(length)
    
    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for storage
        
        Args:
            api_key: Plain text API key
            
        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        """
        Verify an API key against hash
        
        Args:
            plain_key: Plain text API key
            hashed_key: Hashed API key
            
        Returns:
            True if API key matches
        """
        return self.hash_api_key(plain_key) == hashed_key


# Global security manager instance
security_manager = SecurityManager(
    secret_key=settings.SECRET_KEY
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to get current user from JWT token
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        Current user data
        
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    
    try:
        payload = security_manager.decode_token(token)
        
        # Check token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=401,
                detail="Invalid token type"
            )
        
        # Extract user data
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload"
            )
        
        return {
            "user_id": user_id,
            "username": payload.get("username"),
            "roles": payload.get("roles", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to verify API key
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        API key metadata
        
    Raises:
        HTTPException: If API key is invalid
    """
    api_key = credentials.credentials
    
    # In production, this would check against database
    # For now, we'll just validate the format
    if not api_key or len(api_key) < 20:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return {
        "api_key": api_key,
        "valid": True
    }


def require_role(required_role: str):
    """
    Dependency to require specific role
    
    Args:
        required_role: Required role name
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        user_roles = current_user.get("roles", [])
        
        if required_role not in user_roles and "admin" not in user_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker


def require_any_role(*required_roles: str):
    """
    Dependency to require any of the specified roles
    
    Args:
        *required_roles: Required role names
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        user_roles = current_user.get("roles", [])
        
        if "admin" in user_roles:
            return current_user
        
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required one of: {', '.join(required_roles)}"
            )
        
        return current_user
    
    return role_checker


class RateLimitExceeded(HTTPException):
    """Exception for rate limit exceeded"""
    def __init__(self):
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


class IPWhitelist:
    """IP whitelist manager"""
    
    def __init__(self, allowed_ips: list[str]):
        self.allowed_ips = set(allowed_ips)
    
    def is_allowed(self, ip: str) -> bool:
        """
        Check if IP is whitelisted
        
        Args:
            ip: IP address to check
            
        Returns:
            True if IP is allowed
        """
        return ip in self.allowed_ips or "*" in self.allowed_ips
    
    def add_ip(self, ip: str):
        """Add IP to whitelist"""
        self.allowed_ips.add(ip)
    
    def remove_ip(self, ip: str):
        """Remove IP from whitelist"""
        self.allowed_ips.discard(ip)


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip whitespace
    text = text.strip()
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """
    Mask sensitive data (e.g., API keys, passwords)
    
    Args:
        data: Sensitive data to mask
        visible_chars: Number of characters to leave visible
        
    Returns:
        Masked string
    """
    if not data or len(data) <= visible_chars:
        return "***"
    
    return data[:visible_chars] + "*" * (len(data) - visible_chars)


# Helper functions for common operations
def create_user_token(user_id: str, username: str, roles: list[str]) -> Dict[str, str]:
    """
    Create access and refresh tokens for user
    
    Args:
        user_id: User ID
        username: Username
        roles: User roles
        
    Returns:
        Dictionary with access_token and refresh_token
    """
    token_data = {
        "sub": user_id,
        "username": username,
        "roles": roles
    }
    
    access_token = security_manager.create_access_token(token_data)
    refresh_token = security_manager.create_refresh_token(token_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }