
from fastapi import APIRouter

from .detect import router as detection_router
from .agent import router as agent_router


router = APIRouter()
router.include_router(detection_router, tags=["Detection"])
router.include_router(agent_router, tags =["Agent"])


