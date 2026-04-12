from fastapi import APIRouter

router = APIRouter(prefix="/daily-verse", tags=["daily-verse"])

# Routes removed — daily verse runs only via background job (startup.py)
