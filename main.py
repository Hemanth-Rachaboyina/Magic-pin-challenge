from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import time
from datetime import datetime, timezone
import uuid

from schemas import (
    HealthzResponse, ContextCounts, MetadataResponse,
    ContextPushRequest, ContextPushResponse,
    TickRequest, TickResponse, Action,
    ReplyRequest, ReplyResponse
)
from store import store
from engine import compose_tick_message, compose_reply_action
import logging
from models import openai_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vera Message Engine")
start_time = time.time()

# -----------------------------------------------------------------------------
# WARMUP ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/v1/healthz", response_model=HealthzResponse)
async def healthz():
    counts = store.get_counts()
    return HealthzResponse(
        status="ok",
        uptime_seconds=int(time.time() - start_time),
        contexts_loaded=ContextCounts(**counts)
    )

@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata():
    return MetadataResponse(
        team_name="AI Architects",
        team_members=["Hemanth"],
        model=openai_model,
        approach="Deterministic structured outputs with aggressive prompt isolation.",
        contact_email="hemanthtoyall222@gmail.com",
        version="1.0.0",
        submitted_at=datetime.now(timezone.utc).isoformat()
    )

@app.post("/v1/context", response_model=ContextPushResponse)
async def push_context(request: ContextPushRequest):
    accepted, result = store.upsert_context(request)
    
    if accepted:
        return ContextPushResponse(
            accepted=True,
            ack_id=result,
            stored_at=datetime.now(timezone.utc).isoformat()
        )
    else:
        # Returning 409 Conflict as per api-call-examples.md for stale version
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": result,
                "current_version": store.versions.get(request.context_id)
            }
        )

# -----------------------------------------------------------------------------
# CORE LOGIC ENDPOINTS (Mocked for now)
# -----------------------------------------------------------------------------
@app.post("/v1/tick", response_model=TickResponse)
def tick(request: TickRequest):
    actions = []
    
    # Process up to 10 triggers concurrently to stay well within 30s timeout
    triggers_to_process = request.available_triggers[:10]
    
    if triggers_to_process:
        import concurrent.futures
        
        def process_trigger(trigger_id):
            try:
                logger.info(f"Generating tick action for trigger: {trigger_id}")
                return compose_tick_message(trigger_id)
            except Exception as e:
                logger.error(f"Error generating action for {trigger_id}: {str(e)}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(process_trigger, triggers_to_process)
            for action in results:
                if action is not None:
                    actions.append(action)
            
    return TickResponse(actions=actions)


@app.post("/v1/reply", response_model=ReplyResponse)
def reply(request: ReplyRequest):
    # Store incoming message
    store.append_conversation(
        request.conversation_id, 
        {"role": request.from_role, "message": request.message, "time": request.received_at}
    )
    
    try:
        logger.info(f"Generating reply action for conversation: {request.conversation_id}")
        reply_response = compose_reply_action(request.conversation_id, request.message, request.from_role)
        
        # Store our own response back to history if we send one
        if reply_response.action == "send" and reply_response.body:
             store.append_conversation(
                 request.conversation_id,
                 {"role": "vera", "message": reply_response.body, "time": datetime.now(timezone.utc).isoformat()}
             )
             
        return reply_response
    except Exception as e:
        logger.error(f"Error generating reply action: {str(e)}")
        # Fallback to safe action
        return ReplyResponse(
            action="end",
            rationale="System encountered an error, closing safely."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
