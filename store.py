from typing import Any, Dict, List, Tuple
from schemas import ContextPushRequest
from datetime import datetime

class ContextStore:
    def __init__(self):
        # Maps context_id -> payload for fast lookups
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.merchants: Dict[str, Dict[str, Any]] = {}
        self.customers: Dict[str, Dict[str, Any]] = {}
        self.triggers: Dict[str, Dict[str, Any]] = {}
        
        # Maps context_id -> version
        self.versions: Dict[str, int] = {}
        
        # Maps conversation_id -> List of message dictionaries
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

    def get_counts(self) -> Dict[str, int]:
        return {
            "category": len(self.categories),
            "merchant": len(self.merchants),
            "customer": len(self.customers),
            "trigger": len(self.triggers)
        }

    def upsert_context(self, request: ContextPushRequest) -> Tuple[bool, str]:
        """
        Idempotent upsert logic. 
        Returns (is_accepted, ack_id_or_reason)
        """
        cid = request.context_id
        incoming_version = request.version
        
        # 1. Check idempotency/versioning
        current_version = self.versions.get(cid)
        if current_version is not None:
            if incoming_version == current_version:
                return False, "stale_version"
            if incoming_version < current_version:
                return False, "stale_version"
                
        # 2. Store the payload based on scope
        scope = request.scope
        payload = request.payload
        
        if scope == "category":
            self.categories[cid] = payload
        elif scope == "merchant":
            self.merchants[cid] = payload
        elif scope == "customer":
            self.customers[cid] = payload
        elif scope == "trigger":
            self.triggers[cid] = payload
        else:
            return False, "invalid_scope"
            
        # 3. Update version
        self.versions[cid] = incoming_version
        
        ack_id = f"ack_{cid}_v{incoming_version}"
        return True, ack_id

    def append_conversation(self, conversation_id: str, message_data: Dict[str, Any]):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(message_data)

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.conversations.get(conversation_id, [])

# Global singleton instance for the FastAPI app to use
store = ContextStore()
