
from fastapi import APIRouter

from .detect import router as detection_router
from .agent import router as agent_router
from .auth import router as auth_router


router = APIRouter()
router.include_router(detection_router, prefix="/detect", tags=["Detection"])
router.include_router(agent_router, prefix="/agent", tags=["Agent"])
router.include_router(auth_router, prefix="/auth", tags=["Auth"])


