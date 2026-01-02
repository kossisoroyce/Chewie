import streamlit as st
import requests
import json
import uuid
from datetime import datetime
import pandas as pd
import time

# Page config
st.set_page_config(
    page_title="AfriMed CHW Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8000/query"
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

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=60)
    st.title("AfriMed CHW")
    st.caption(f"Session: {st.session_state.session_id}")
    
    st.divider()
    
    language = st.radio("Language / Lugha", ["English", "Kiswahili"])
    lang_code = "sw" if language == "Kiswahili" else "en"
    
    mode = st.toggle("Use Live API", value=False, help="Toggle between Mock Mode and Live API (port 8000)")
    
    st.divider()
    st.markdown("### 🚨 Quick Triage")
    st.info("Check for danger signs immediately in every case.")

# Helper Functions
def check_danger_signs_local(text, lang):
    """Local implementation of danger sign detection for mock mode"""
    found = []
    text_lower = text.lower()
    keywords = DANGER_SIGNS_KEYWORDS.get(lang, DANGER_SIGNS_KEYWORDS['en'])
    for kw in keywords:
        if kw in text_lower:
            found.append(kw)
    return list(set(found))

def get_mock_response(query, lang):
    """Generate a mock response for demonstration"""
    danger_signs = check_danger_signs_local(query, lang)
    is_danger = len(danger_signs) > 0
    
    time.sleep(1.5) # Simulate latency
    
    if lang == "sw":
        if is_danger:
            return {
                "response": f"⚠️ HATARI: Hii ni dharura. Mgonjwa anaonyesha dalili za hatari: {', '.join(danger_signs)}. \n\nHatua za kuchukua:\n1. Mpe rufaa kwenda kituo cha afya mara moja.\n2. Mpe huduma ya kwanza ikibidi.\n3. Hakikisha anasindikizwa.",
                "urgency": "emergency",
                "danger_signs": danger_signs,
                "refer_to_facility": True
            }
        else:
            return {
                "response": "Hii inaonekana ni hali ya kawaida. \n\nUshauri:\n1. Mpe dawa za kupunguza maumivu (Paracetamol).\n2. Mhimize anywe maji mengi.\n3. Rudi kituoni ikiwa hali itabadilika.",
                "urgency": "routine",
                "danger_signs": [],
                "refer_to_facility": False
            }
    else:
        if is_danger:
            return {
                "response": f"⚠️ DANGER: This is an emergency. Danger signs detected: {', '.join(danger_signs)}. \n\nActions:\n1. Refer to health facility IMMEDIATELY.\n2. Stabilize patient if possible.\n3. Arrange transport.",
                "urgency": "emergency",
                "danger_signs": danger_signs,
                "refer_to_facility": True
            }
        else:
            return {
                "response": "This appears to be a routine case. \n\nAdvice:\n1. Manage fever/pain with Paracetamol.\n2. Ensure plenty of fluids.\n3. Monitor for 24 hours.",
                "urgency": "routine",
                "danger_signs": [],
                "refer_to_facility": False
            }

def query_api(text, lang):
    """Call the backend API"""
    try:
        payload = {
            "query": text,
            "language": lang,
            "chw_id": "demo-user",
            "session_id": st.session_state.session_id
        }
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Main Chat Interface
st.markdown("### 👩🏾‍⚕️ Maternal & Child Health Assistant")
st.markdown("Ask about symptoms, danger signs, or treatment guidelines.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            if meta.get("urgency") == "emergency":
                st.error(f"🚨 URGENCY: EMERGENCY - Refer to Facility: {meta.get('refer_to_facility')}")
            elif meta.get("urgency") == "urgent":
                st.warning(f"⚠️ URGENCY: URGENT - Refer to Facility: {meta.get('refer_to_facility')}")
            
            if meta.get("danger_signs"):
                st.markdown(f"**Danger Signs:** {', '.join(meta['danger_signs'])}")

# Chat Input
if prompt := st.chat_input("Describe the patient's condition..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            if mode:
                # Live API
                api_response = query_api(prompt, lang_code)
                if "error" in api_response:
                    st.error(f"API Error: {api_response['error']}")
                    result = None
                else:
                    result = api_response
            else:
                # Mock Mode
                result = get_mock_response(prompt, lang_code)
            
            if result:
                response_text = result.get("response", "")
                
                # Format output
                st.markdown(response_text)
                
                # Metadata display
                urgency = result.get("urgency", "routine")
                danger_signs = result.get("danger_signs", []) or result.get("danger_signs_detected", [])
                refer = result.get("refer_to_facility", False)
                
                if urgency == "emergency":
                    st.error("🚨 EMERGENCY CASE detected")
                
                if danger_signs:
                    st.warning(f"⚠️ Danger Signs: {', '.join(danger_signs)}")
                
                # Save context
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "metadata": {
                        "urgency": urgency,
                        "danger_signs": danger_signs,
                        "refer_to_facility": refer
                    }
                })

# Footer / Disclaimer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This AI tool is for decision support only. Always follow national clinical guidelines and consult a supervisor for complex cases.")
