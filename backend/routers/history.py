"""
routers/history.py — Search History API Router
===============================================
Endpoints for retrieving and clearing accurate user search history.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends

from database.db import clear_user_search_history, get_user_search_history
from models.auth import SearchHistoryItemResponse, SearchHistoryListResponse
from security.auth import get_current_user_optional

logger = logging.getLogger(__name__)

history_router = APIRouter(
    prefix="/api/v1/history",
    tags=["Search History"],
)


@history_router.get(
    "",
    response_model=SearchHistoryListResponse,
    summary="Get recent search history",
)
def get_history(current_user: Optional[dict] = Depends(get_current_user_optional)) -> SearchHistoryListResponse:
    """
    Returns up to 50 recent searches recorded for the user.
    If authenticated, returns user's personal search history.
    """
    user_id = current_user["id"] if current_user else None
    history_items = get_user_search_history(user_id=user_id, limit=50)
    
    parsed_items = [
        SearchHistoryItemResponse(
            id=item["id"],
            ticker=item["ticker"],
            exchange=item["exchange"],
            company_name=item["company_name"],
            score=item["score"],
            grade=item["grade"],
            recommendation=item["recommendation"],
            searched_at=item["searched_at"],
        )
        for item in history_items
    ]
    return SearchHistoryListResponse(history=parsed_items, total_count=len(parsed_items))


@history_router.delete(
    "",
    summary="Clear search history",
)
def clear_history(current_user: Optional[dict] = Depends(get_current_user_optional)):
    """Deletes search history records for the current user or guest."""
    user_id = current_user["id"] if current_user else None
    count = clear_user_search_history(user_id=user_id)
    return {"status": "success", "deleted_count": count}
