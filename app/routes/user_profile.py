from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import Response

from app.account import purge_user_data
from app.auth import get_current_user
from app.profile import fetch_user_profile
from app.quota import get_quota_remaining
from app.rate_limit import check_rate_limit

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


@router.delete("/user/account", status_code=204)
async def delete_user_account(user: dict = Depends(get_current_user)):
    """Permanently erase the authenticated user's data.

    Deletes semantic memories, conversations, and the enriched profile from
    Qdrant, the Firestore user tree, and the Firebase Auth account. The uid
    comes from the verified token, so a user can only delete themselves.

    Returns 500 when any store failed, so the client never tells the user
    their data is gone while part of it remains.
    """
    await check_rate_limit(user["uid"])

    report = purge_user_data(user["uid"])
    if not report["complete"]:
        raise HTTPException(
            status_code=500,
            detail={"error": "account_deletion_incomplete"},
        )

    return Response(status_code=204)
