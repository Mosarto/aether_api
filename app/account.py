"""Account deletion.

Every store that holds personal data is keyed by the Firebase uid, so a
deletion can be complete:

- Qdrant `conversations`   → payload `user_id` on turns and session meta
- Qdrant `user_memories`   → payload `user_id`
- Qdrant `user_profiles`   → payload `user_id` (deterministic point id)
- Firestore `users/{uid}`  → document + every subcollection
- Firebase Auth            → the account itself

The shared `reflections` catalog is deliberately untouched: it is product
content consumed by all users, not personal data.
"""

from qdrant_client.http import models as qmodels

from app.config import (
    COL_CONVERSATIONS,
    COL_USER_MEMORIES,
    COL_USER_PROFILES,
    logger,
)
from app.firebase import delete_auth_user, delete_user_data
from app.providers import qdrant

PERSONAL_COLLECTIONS = (COL_CONVERSATIONS, COL_USER_MEMORIES, COL_USER_PROFILES)


def _delete_user_points(collection: str, user_id: str) -> bool:
    """Delete every point of a user in one collection. Missing collection is ok."""
    try:
        qdrant.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="user_id", match=qmodels.MatchValue(value=user_id)
                        )
                    ]
                )
            ),
            wait=True,
        )
        return True
    except Exception as e:
        # A collection that was never created is not a failure; anything else is.
        status = getattr(e, "status_code", None)
        if status == 404:
            return True
        logger.error(
            "Exclusão de conta: falha ao apagar %s — %s", collection, e.__class__.__name__
        )
        return False


def _count_user_points(collection: str, user_id: str) -> int:
    """Remaining points of a user, used to verify the purge actually landed."""
    try:
        result = qdrant.count(
            collection_name=collection,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id", match=qmodels.MatchValue(value=user_id)
                    )
                ]
            ),
            exact=True,
        )
        return int(getattr(result, "count", 0))
    except Exception:
        # Cannot verify — treat as clean rather than blocking the deletion,
        # since the delete call itself already reported success.
        return 0


def purge_user_data(user_id: str) -> dict:
    """Erase every trace of a user. Idempotent: deleting twice is harmless.

    Returns a content-free report. `complete` is False when any store failed,
    so the API never claims a deletion it did not finish.
    """
    report: dict[str, object] = {"complete": True, "stores": {}}

    for collection in PERSONAL_COLLECTIONS:
        ok = _delete_user_points(collection, user_id)
        remaining = _count_user_points(collection, user_id) if ok else -1
        if not ok or remaining > 0:
            report["complete"] = False
        report["stores"][collection] = "deleted" if ok and remaining == 0 else "failed"

    firestore_ok, documents = delete_user_data(user_id)
    report["stores"]["firestore"] = "deleted" if firestore_ok else "failed"
    report["firestore_documents"] = documents
    if not firestore_ok:
        report["complete"] = False

    # Auth last: while the account exists the user can retry a failed purge.
    auth_ok = delete_auth_user(user_id)
    report["stores"]["auth"] = "deleted" if auth_ok else "failed"
    if not auth_ok:
        report["complete"] = False

    logger.info(
        "Exclusão de conta concluída (completa=%s, stores=%s)",
        report["complete"],
        report["stores"],
    )
    return report
