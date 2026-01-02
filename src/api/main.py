"""
AfriMed CHW Assistant API

FastAPI-based API for serving the fine-tuned model to CHWs via:
- REST API for mobile apps
- Webhook endpoints for WhatsApp/SMS integrations
"""

import os
from datetime import datetime
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langdetect import detect
import structlog

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="AfriMed CHW Assistant",
    description="AI-powered decision support for Community Health Workers in maternal healthcare",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vertex AI
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-project-id")
REGION = os.environ.get("GCP_REGION", "us-central1")
MODEL_ENDPOINT = os.environ.get("AFRIMED_MODEL_ENDPOINT", "gemini-1.5-flash")  # Will be fine-tuned model

vertexai.init(project=PROJECT_ID, location=REGION)


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SWAHILI = "sw"
    AUTO = "auto"


class UrgencyLevel(str, Enum):
    """Response urgency levels."""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"


class QueryRequest(BaseModel):
    """Request model for CHW queries."""
    query: str = Field(..., min_length=5, max_length=2000, description="The CHW's question")
    language: Language = Field(default=Language.AUTO, description="Preferred response language")
    chw_id: Optional[str] = Field(default=None, description="CHW identifier for logging")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    context: Optional[str] = Field(default=None, description="Additional context about the patient")


class QueryResponse(BaseModel):
    """Response model for CHW queries."""
    response: str = Field(..., description="The assistant's response")
    urgency: UrgencyLevel = Field(..., description="Urgency level of the situation")
    detected_language: str = Field(..., description="Detected query language")
    response_language: str = Field(..., description="Response language")
    danger_signs_detected: list[str] = Field(default=[], description="Any danger signs identified")
    recommended_actions: list[str] = Field(default=[], description="Recommended actions")
    refer_to_facility: bool = Field(..., description="Whether facility referral is recommended")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence score")
    timestamp: str = Field(..., description="Response timestamp")
    request_id: str = Field(..., description="Unique request identifier")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


# Danger signs keywords for detection
DANGER_SIGNS_KEYWORDS = {
    "en": [
        "heavy bleeding", "severe bleeding", "convulsion", "seizure", "fit",
        "severe headache", "blurred vision", "high fever", "not breathing",
        "unconscious", "severe pain", "swelling", "yellow", "jaundice",
        "not feeding", "not moving", "cord bleeding", "fast breathing"
    ],
    "sw": [
        "kutoka damu nyingi", "degedege", "mshtuko", "kichwa kuuma sana",
        "macho kuona vibaya", "homa kali", "kupumua vibaya", "kupoteza fahamu",
        "maumivu makali", "kuvimba", "manjano", "kunyonya vibaya", "kutosogea"
    ]
}

SYSTEM_PROMPT = """You are AfriMed, a medical assistant helping Community Health Workers (CHWs) 
in East Africa provide maternal and newborn healthcare. 

Your role:
- Provide clear, actionable guidance for maternal health issues
- Identify danger signs that require immediate facility referral
- Support antenatal, delivery, postpartum, and newborn care decisions
- Communicate in simple, clear language (respond in the same language as the query)

Safety rules:
- NEVER diagnose conditions - only support CHW decision-making
- ALWAYS recommend facility referral for danger signs
- When uncertain, advise consulting a supervisor or health facility
- Include clear action steps in every response"""


def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        detected = detect(text)
        if detected in ["sw", "swahili"]:
            return "sw"
        return "en"
    except:
        return "en"


def check_danger_signs(text: str, language: str) -> list[str]:
    """Check for danger signs in the query text."""
    text_lower = text.lower()
    detected_signs = []
    
    keywords = DANGER_SIGNS_KEYWORDS.get(language, DANGER_SIGNS_KEYWORDS["en"])
    for keyword in keywords:
        if keyword.lower() in text_lower:
            detected_signs.append(keyword)
    
    # Also check English keywords if input was Swahili
    if language == "sw":
        for keyword in DANGER_SIGNS_KEYWORDS["en"]:
            if keyword.lower() in text_lower:
                detected_signs.append(keyword)
    
    return list(set(detected_signs))


def determine_urgency(danger_signs: list[str], response_text: str) -> UrgencyLevel:
    """Determine urgency level based on danger signs and response."""
    if danger_signs:
        return UrgencyLevel.EMERGENCY
    
    emergency_keywords = ["immediately", "urgent", "emergency", "refer now", "haraka", "dharura"]
    if any(kw in response_text.lower() for kw in emergency_keywords):
        return UrgencyLevel.URGENT
    
    return UrgencyLevel.ROUTINE


def extract_actions(response_text: str) -> list[str]:
    """Extract recommended actions from the response."""
    actions = []
    lines = response_text.split("\n")
    
    for line in lines:
        line = line.strip()
        # Look for numbered items or bullet points
        if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
            # Clean up the line
            action = line.lstrip("0123456789.-•) ").strip()
            if action and len(action) > 10:
                actions.append(action)
    
    return actions[:5]  # Return top 5 actions


async def generate_response(query: str, language: str, context: Optional[str] = None) -> str:
    """Generate response using the fine-tuned model."""
    
    # Build the prompt
    full_query = query
    if context:
        full_query = f"Context: {context}\n\nQuestion: {query}"
    
    if language == "sw":
        full_query += "\n\n(Please respond in Swahili / Tafadhali jibu kwa Kiswahili)"
    
    try:
        model = GenerativeModel(
            MODEL_ENDPOINT,
            system_instruction=SYSTEM_PROMPT
        )
        
        response = model.generate_content(full_query)
        return response.text
        
    except Exception as e:
        logger.error("Model inference failed", error=str(e))
        raise HTTPException(status_code=500, detail="Model inference failed")


async def log_interaction(request: QueryRequest, response: QueryResponse):
    """Log the interaction for analytics and improvement."""
    log_entry = {
        "timestamp": response.timestamp,
        "request_id": response.request_id,
        "chw_id": request.chw_id,
        "query_language": response.detected_language,
        "urgency": response.urgency,
        "danger_signs": response.danger_signs_detected,
        "refer_to_facility": response.refer_to_facility,
        "confidence": response.confidence,
    }
    logger.info("Interaction logged", **log_entry)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        model_loaded=True,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process a CHW query and return guidance.
    
    This is the main endpoint for CHW interactions.
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]
    
    logger.info("Processing query", request_id=request_id, chw_id=request.chw_id)
    
    # Detect language
    if request.language == Language.AUTO:
        detected_lang = detect_language(request.query)
    else:
        detected_lang = request.language.value
    
    # Check for danger signs
    danger_signs = check_danger_signs(request.query, detected_lang)
    
    # Generate response
    response_text = await generate_response(request.query, detected_lang, request.context)
    
    # Determine urgency
    urgency = determine_urgency(danger_signs, response_text)
    
    # Extract actions
    actions = extract_actions(response_text)
    
    # Determine if referral is needed
    refer = urgency == UrgencyLevel.EMERGENCY or bool(danger_signs)
    
    # Calculate confidence (simplified - would be more sophisticated in production)
    confidence = 0.9 if not danger_signs else 0.95  # Higher confidence for danger sign detection
    
    response = QueryResponse(
        response=response_text,
        urgency=urgency,
        detected_language=detected_lang,
        response_language=detected_lang,
        danger_signs_detected=danger_signs,
        recommended_actions=actions,
        refer_to_facility=refer,
        confidence=confidence,
        timestamp=datetime.utcnow().isoformat(),
        request_id=request_id
    )
    
    # Log interaction in background
    background_tasks.add_task(log_interaction, request, response)
    
    return response


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(payload: dict):
    """
    Webhook endpoint for WhatsApp Business API integration.
    
    This handles incoming messages from WhatsApp and returns responses.
    """
    # Extract message from WhatsApp payload format
    # This is a simplified example - actual implementation depends on provider
    try:
        message = payload.get("message", {}).get("text", "")
        sender = payload.get("from", "unknown")
        
        if not message:
            return {"status": "no_message"}
        
        # Process as a query
        request = QueryRequest(query=message, chw_id=sender)
        response = await process_query(request, BackgroundTasks())
        
        # Format for WhatsApp
        whatsapp_response = {
            "to": sender,
            "type": "text",
            "text": {
                "body": response.response
            }
        }
        
        # Add urgent prefix if needed
        if response.urgency == UrgencyLevel.EMERGENCY:
            whatsapp_response["text"]["body"] = f"⚠️ URGENT ⚠️\n\n{response.response}"
        
        return whatsapp_response
        
    except Exception as e:
        logger.error("WhatsApp webhook error", error=str(e))
        return {"status": "error", "message": str(e)}


@app.post("/webhook/sms")
async def sms_webhook(payload: dict):
    """
    Webhook endpoint for SMS gateway integration (e.g., Africa's Talking).
    
    Handles incoming SMS and returns responses formatted for SMS length limits.
    """
    try:
        message = payload.get("text", "")
        sender = payload.get("from", "unknown")
        
        if not message:
            return {"status": "no_message"}
        
        # Process as a query
        request = QueryRequest(query=message, chw_id=sender)
        background_tasks = BackgroundTasks()
        response = await process_query(request, background_tasks)
        
        # Truncate for SMS (160 char limit for single SMS)
        sms_response = response.response[:500]  # Allow for multi-part SMS
        
        if response.refer_to_facility:
            sms_response = f"⚠️REFER TO FACILITY⚠️\n{sms_response}"
        
        return {
            "to": sender,
            "message": sms_response
        }
        
    except Exception as e:
        logger.error("SMS webhook error", error=str(e))
        return {"status": "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
