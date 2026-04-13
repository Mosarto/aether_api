from fastapi import APIRouter, HTTPException, Depends

from app.auth import get_current_user
from app.profile import fetch_user_profile
from app.quota import get_quota_remaining

router = APIRouter(tags=["Perfil"])


@router.get("/user/profile")
async def get_user_profile(user: dict = Depends(get_current_user)):
    profile = fetch_user_profile(user["uid"])
    if profile is None:
        raise HTTPException(status_code=404, detail="Perfil não encontrado")
    return profile


@router.get("/user/quota")
async def get_user_quota(user: dict = Depends(get_current_user)):
    """Read-only quota check for UI sync on app startup."""
    return await get_quota_remaining(user)
