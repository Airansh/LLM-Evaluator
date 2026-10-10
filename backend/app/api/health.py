from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import __version__
from app.db.session import get_session
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(session: Session = Depends(get_session)) -> HealthOut:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return HealthOut(status="degraded", version=__version__, database="error")
    return HealthOut(status="ok", version=__version__, database="ok")
