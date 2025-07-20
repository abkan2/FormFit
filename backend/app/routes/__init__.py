
from fastapi import APIRouter

from .detect import router as detection_router



router = APIRouter()
router.include_router(detection_router, tags=["Detection"])


