"""
Chat API endpoints.
POST /chat, GET /chat/stream (WebSocket), GET /chat/history, POST /chat/feedback
"""
import uuid
import json
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, async_session_maker
from app.services.chat_service import ChatService
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationListResponse,
    MessageResponse,
    FeedbackRequest,
    SourceCitation,
)
from app.api.deps import get_current_user, CurrentUser
from app.middleware.error_handler import NotFoundError

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a chat message and get a RAG-enhanced response.
    
    - Retrieves relevant document chunks
    - Generates response using local LLM
    - Returns response with source citations
    """
    chat_service = ChatService(db)
    
    result = await chat_service.chat(
        query=request.message,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        conversation_id=request.conversation_id,
        document_ids=request.document_ids,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    # Convert sources to SourceCitation objects
    sources = [
        SourceCitation(
            id=s["id"],
            name=s["name"],
            page=s.get("page"),
            section=s.get("section"),
            relevance=s.get("relevance", "Medium"),
            snippet=s.get("snippet"),
        )
        for s in result.get("sources", [])
    ]
    
    return ChatResponse(
        conversation_id=result["conversation_id"],
        message_id=result["message_id"],
        content=result["content"],
        sources=sources,
        model=result["model"],
        tokens_used=result["tokens_used"],
        latency_ms=result["latency_ms"],
    )


@router.get("/stream")
async def chat_stream_sse(
    message: str = Query(...),
    conversation_id: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Stream chat response using Server-Sent Events (SSE).
    
    Use this for real-time token-by-token streaming in the frontend.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Capture user info before entering the generator
    user_id = current_user.id
    tenant_id = current_user.tenant_id
    
    logger.info(f"[ChatSSE] Starting stream for user {user_id}, message: {message[:50]}...")
    
    async def event_generator():
        # Create a fresh DB session inside the generator
        # This ensures the session lives for the entire stream duration
        async with async_session_maker() as db:
            try:
                chat_service = ChatService(db)
                event_count = 0
                async for event in chat_service.chat_stream(
                    query=message,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                ):
                    event_count += 1
                    event_type = event["event"]
                    data = event["data"]
                    if event_count <= 5:
                        logger.debug(f"[ChatSSE] Event {event_count}: {event_type} = {repr(data[:50] if len(data) > 50 else data)}")
                    yield f"event: {event_type}\ndata: {data}\n\n"
                logger.info(f"[ChatSSE] Stream complete. Total events: {event_count}")
            except Exception as e:
                logger.error(f"[ChatSSE] Error: {e}")
                yield f"event: error\ndata: {str(e)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket endpoint for real-time chat with streaming.
    
    Protocol:
    - Client sends: {"message": "...", "conversation_id": "..."}
    - Server sends: {"event": "token|sources|done|error", "data": "..."}
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            message = data.get("message", "")
            conversation_id = data.get("conversation_id")
            user_id = data.get("user_id")  # In production, get from auth token
            
            if not message:
                await websocket.send_json({"event": "error", "data": "Message required"})
                continue
            
            if not user_id:
                await websocket.send_json({"event": "error", "data": "Authentication required"})
                continue
            
            # Stream response
            chat_service = ChatService(db)
            
            async for event in chat_service.chat_stream(
                query=message,
                user_id=uuid.UUID(user_id),
                conversation_id=conversation_id,
            ):
                await websocket.send_json(event)
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"event": "error", "data": str(e)})


@router.get("/history", response_model=ConversationListResponse)
async def get_chat_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get list of user's conversations.
    """
    chat_service = ChatService(db)
    
    conversations, total = await chat_service.list_conversations(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )
    
    return ConversationListResponse(
        conversations=[
            {
                "id": str(c.id),
                "title": c.title,
                "message_count": c.message_count,
                "created_at": c.created_at.isoformat(),
                "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
            }
            for c in conversations
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a conversation with its messages.
    """
    chat_service = ChatService(db)
    
    conversation = await chat_service.get_conversation(
        conversation_id=uuid.UUID(conversation_id),
        user_id=current_user.id,
    )
    
    if not conversation:
        raise NotFoundError("Conversation not found")
    
    messages = await chat_service.get_messages(
        conversation_id=uuid.UUID(conversation_id),
        user_id=current_user.id,
    )
    
    return ConversationResponse(
        id=str(conversation.id),
        title=conversation.title,
        messages=[
            MessageResponse(
                id=str(m.id),
                role=m.role.value,
                content=m.content,
                sources=json.loads(m.sources_json) if m.sources_json else None,
                created_at=m.created_at,
            )
            for m in messages
        ],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=conversation.message_count,
    )


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submit feedback on a chat response.
    """
    chat_service = ChatService(db)
    
    await chat_service.save_feedback(
        message_id=uuid.UUID(request.message_id),
        user_id=current_user.id,
        rating=request.rating,
        is_helpful=request.is_helpful,
        is_accurate=request.is_accurate,
        comment=request.comment,
        issues=request.issues,
    )
    
    return {"message": "Feedback submitted successfully"}


# ============================================================================
# Phase 5: Stream Cancellation Endpoints
# ============================================================================

@router.post("/stream/cancel/{stream_id}")
async def cancel_stream(
    stream_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Cancel an active streaming response.
    
    Use this to stop a long-running generation before it completes.
    """
    from app.services.stream_cancellation_service import get_stream_cancellation_service
    
    service = get_stream_cancellation_service()
    
    # Verify the stream belongs to this user
    stream_info = service.get_stream_info(stream_id)
    if not stream_info:
        return {"success": False, "message": "Stream not found or already completed"}
    
    if stream_info["user_id"] != str(current_user.id):
        return {"success": False, "message": "Stream does not belong to this user"}
    
    cancelled = await service.cancel(stream_id, "User requested cancellation")
    
    return {
        "success": cancelled,
        "message": "Stream cancelled" if cancelled else "Stream not found",
        "stream_id": stream_id,
    }


@router.post("/stream/cancel-all")
async def cancel_all_user_streams(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Cancel all active streams for the current user.
    """
    from app.services.stream_cancellation_service import get_stream_cancellation_service
    
    service = get_stream_cancellation_service()
    cancelled = await service.cancel_user_streams(
        current_user.id,
        "User cancelled all streams",
    )
    
    return {
        "success": True,
        "cancelled_count": cancelled,
    }


@router.get("/stream/active")
async def list_active_streams(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    List active streaming sessions for the current user.
    """
    from app.services.stream_cancellation_service import get_stream_cancellation_service
    
    service = get_stream_cancellation_service()
    streams = service.get_active_streams(user_id=current_user.id)
    
    return {
        "active_streams": streams,
        "count": len(streams),
    }
