from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# -----------------------------------------------------------------------------
# /v1/healthz
# -----------------------------------------------------------------------------
class ContextCounts(BaseModel):
    category: int = 0
    merchant: int = 0
    customer: int = 0
    trigger: int = 0


class HealthzResponse(BaseModel):
    status: str
    uptime_seconds: int
    contexts_loaded: ContextCounts


# -----------------------------------------------------------------------------
# /v1/metadata
# -----------------------------------------------------------------------------
class MetadataResponse(BaseModel):
    team_name: str
    team_members: List[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str


# -----------------------------------------------------------------------------
# /v1/context
# -----------------------------------------------------------------------------
class ContextPushRequest(BaseModel):
    scope: str = Field(description="Must be one of: category, merchant, customer, trigger")
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


class ContextPushResponse(BaseModel):
    accepted: bool
    ack_id: Optional[str] = None
    stored_at: Optional[str] = None
    reason: Optional[str] = None
    current_version: Optional[int] = None


# -----------------------------------------------------------------------------
# /v1/tick
# -----------------------------------------------------------------------------
class TickRequest(BaseModel):
    now: str
    available_triggers: List[str]


class Action(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str = Field(description="'vera' or 'merchant_on_behalf'")
    trigger_id: str
    template_name: Optional[str] = None
    template_params: Optional[List[str]] = None
    body: str = Field(description="The actual message to send")
    cta: str = Field(description="Call to action type, e.g., 'open_ended', 'binary_yes_no'")
    suppression_key: str = Field(description="Key used to prevent spamming")
    rationale: str = Field(description="Explanation for why this action was chosen")


class TickResponse(BaseModel):
    actions: List[Action]


# -----------------------------------------------------------------------------
# /v1/reply
# -----------------------------------------------------------------------------
class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"] = Field(description="Must be exactly 'send', 'wait', or 'end'")
    body: Optional[str] = Field(default=None, description="The message body, required if action is 'send'")
    cta: Optional[str] = Field(default=None, description="The CTA type, required if action is 'send'")
    wait_seconds: Optional[int] = Field(default=None, description="Seconds to wait, required if action is 'wait'")
    rationale: str = Field(description="Explanation for why this reply action was chosen")

