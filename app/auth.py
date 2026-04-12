import firebase_admin.auth
from fastapi import Depends, HTTPException, Request

from app.firebase import fetch_firestore_user

# AuthUser fields:
# - uid: str
# - email: str | None
# - subscription_tier: str
# - is_anonymous: bool
AuthUser = dict


async def get_current_user(request: Request) -> dict:
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Token de autenticação não fornecido",
        )

    token = authorization[len("Bearer "):].strip()

    try:
        decoded_token = firebase_admin.auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail="Token inválido ou expirado",
        ) from exc

    uid = decoded_token.get("uid")
    user_data = fetch_firestore_user(uid)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")

    return {
        "uid": uid,
        "email": decoded_token.get("email"),
        "subscription_tier": user_data.get("subscriptionTier", "free"),
        "is_anonymous": user_data.get("isAnonymous", False),
    }


CurrentUser = Depends(get_current_user)
