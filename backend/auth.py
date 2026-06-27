"""
auth.py — Password hashing and login tokens.

- Passwords are hashed with bcrypt (never stored in plain text).
- On login we issue a JWT (a signed token). The frontend sends it back on every
  request via the `Authorization: Bearer <token>` header to prove who it is.
- `get_current_user` is a FastAPI dependency that decodes that token and loads
  the matching user, rejecting invalid/expired tokens with 401.
"""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

import config
import database
from database import User

# bcrypt rejects passwords longer than 72 bytes, so we always truncate to 72.
_MAX_PW_BYTES = 72


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")[:_MAX_PW_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    pw = password.encode("utf-8")[:_MAX_PW_BYTES]
    try:
        return bcrypt.checkpw(pw, password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=config.TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    session: Session = Depends(database.get_session),
) -> User:
    """Decode the Bearer token and return the logged-in user (or 401)."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired login. Please sign in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            creds.credentials, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM]
        )
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise credentials_error

    user = session.get(User, user_id)
    if user is None:
        raise credentials_error
    return user
