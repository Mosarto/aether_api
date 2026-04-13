from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from google.cloud.firestore_v1 import transactional

from app.config import DAILY_QUOTA_FREE, DAILY_QUOTA_PREMIUM, QUOTA_TIMEZONE, logger
from app.firebase import get_firestore_db


def _today_brt() -> str:
    """Get today's date string in BRT (America/Sao_Paulo) as YYYY-MM-DD."""
    return datetime.now(ZoneInfo(QUOTA_TIMEZONE)).strftime("%Y-%m-%d")


def _next_midnight_brt_iso() -> str:
    """Get next midnight BRT as ISO 8601 string."""
    tz = ZoneInfo(QUOTA_TIMEZONE)
    now = datetime.now(tz)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.isoformat()


async def check_and_reserve_quota(user: dict) -> dict:
    """
    Atomically check AND reserve one quota slot in a single Firestore transaction.

    Eliminates the TOCTOU race between separate check/increment calls:
    the transaction reads current usage, verifies the limit, and increments
    in one atomic operation.

    Args:
        user: AuthUser dict with uid, subscription_tier fields.

    Returns:
        dict with {remaining: int} (AFTER reservation).

    Raises:
        HTTPException 429: When daily limit exhausted for free users.
    """
    if user.get("subscription_tier") == "premium":
        return {"remaining": DAILY_QUOTA_PREMIUM}

    uid = user["uid"]
    db = get_firestore_db()
    if not db:
        logger.warning("Firestore indisponível para quota check, permitindo request")
        return {"remaining": DAILY_QUOTA_FREE}

    today = _today_brt()
    quota_ref = db.collection("users").document(uid).collection("quota").document("daily")

    @transactional
    def _check_and_increment(transaction):
        snapshot = quota_ref.get(transaction=transaction)
        used = 0
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            if data.get("date") == today:
                used = int(data.get("used", 0))

        if used >= DAILY_QUOTA_FREE:
            return -1  # signal: limit exceeded

        # Atomically reserve the slot by incrementing in the same transaction
        transaction.set(quota_ref, {"date": today, "used": used + 1})
        return DAILY_QUOTA_FREE - used - 1

    try:
        remaining = _check_and_increment(db.transaction())
    except Exception as e:
        logger.warning("Falha na transação de quota do usuário %s: %s", uid, e)
        # Fail-open: allow request if Firestore transaction fails
        return {"remaining": DAILY_QUOTA_FREE}

    if remaining < 0:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_exceeded",
                "remaining": 0,
                "resetsAt": _next_midnight_brt_iso(),
            },
        )

    return {"remaining": remaining}


# Keep aliases for backward compatibility with imports
check_quota = check_and_reserve_quota


async def get_quota_remaining(user: dict) -> dict:
    """
    Read-only quota check — returns remaining messages without incrementing.

    Used by the app on startup to sync the FreemiumService counter
    before the user sends any message.
    """
    if user.get("subscription_tier") == "premium":
        return {"remaining": DAILY_QUOTA_PREMIUM}

    uid = user["uid"]
    db = get_firestore_db()
    if not db:
        logger.warning("Firestore indisponível para quota read, retornando max")
        return {"remaining": DAILY_QUOTA_FREE}

    today = _today_brt()
    quota_ref = db.collection("users").document(uid).collection("quota").document("daily")

    try:
        snapshot = quota_ref.get()
        used = 0
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            if data.get("date") == today:
                used = int(data.get("used", 0))
        remaining = max(DAILY_QUOTA_FREE - used, 0)
    except Exception as e:
        logger.warning("Falha ao ler quota do usuário %s: %s", uid, e)
        return {"remaining": DAILY_QUOTA_FREE}

    return {"remaining": remaining}


async def increment_quota(uid: str) -> None:
    """
    No-op. Quota is now reserved atomically in check_and_reserve_quota().
    Kept for backward compatibility — callers can safely remove this call.
    """
    pass
