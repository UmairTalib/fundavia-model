import json
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, START, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None
from langgraph.checkpoint.memory import MemorySaver
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "openai/gpt-oss-120b"
DB_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FIELDS = ["state", "employees", "project_status", "company_age"]
RD_EXTRA_FIELDS = ["trl_level", "rd_cooperation"]

# ─── PROMPTS ──────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """Du bist ein Datenextraktions-Assistent für deutsche Fördermittelanfragen.
DEINE EINZIGE AUFGABE: Extrahiere strukturierte Daten aus dem Nutzertext.
- Erfinde KEINE Werte.
- Wenn eine Information nicht explizit genannt wird, setze das Feld auf null.
- Wenn der Nutzer "weiß nicht" sagt, setze das Feld auf "unknown".

EXTRAHIERE FOLGENDES ALS STRICT JSON:
{
  "state": "Bundesland (z.B. Berlin, Bayern)",
  "industry": "Branche (z.B. Handwerk, IT)",
  "employees": "Mitarbeiterzahl (Integer)",
  "project_desc": "Kurze Beschreibung des Vorhabens",
  "project_status": "already_started, not_started, unknown",
  "company_age": "< 5 Jahre, > 5 Jahre",
  "inferred_project_type": "Digitalisierung, FuE & Innovation, Energieeffizienz",
  "needs_confirmation": ["Liste der abgeleiteten Felder die bestätigt werden müssen"],
  "is_off_topic": true/false
}
"""

PLANNING_PROMPT = """Du bist der KI-Förderberater von Fundavia.
Aktuelles Profil: {profile_summary}
Felder die noch fehlen (null): {missing_fields}
Felder die bestätigt werden müssen: {pending_confirmations}
Letzter User-Satz: "{last_message}"

AUFGABEN:
1. Falls "pending_confirmations" nicht leer ist: Bestätige deinen Vorschlag auf natürliche Weise.
2. Falls der User "weiß nicht" gesagt hat: Akzeptiere es freundlich.
3. Fasse mehrere Fragen ZUSAMMEN wenn es natürlich wirkt (max. 2).
4. Beziehe dich auf das was der User gesagt hat.
5. NIEMALS nach Branche oder Projektkategorie fragen — das leitest du ab.
"""

# ─── LANGGRAPH STATE ──────────────────────────────────────────────────────────
class GraphState(TypedDict):
    history: Annotated[list, operator.add]
    profile: dict
    flags: dict
    pending_confirmations: list
    missing_fields: list
    user_msg: str
    assistant_reply: str
    extracted: dict

def init_state() -> dict:
    return {
        "history": [],
        "profile": {},
        "flags": {"rd_detected": False, "retroactive_detected": False, "profile_complete": False, "is_off_topic": False},
        "pending_confirmations": [],
        "missing_fields": REQUIRED_FIELDS.copy(),
        "user_msg": "",
        "assistant_reply": "",
        "extracted": {}
    }

# ─── NODES ────────────────────────────────────────────────────────────────────
def extract_node(state: GraphState):
    client = Groq(api_key=GROQ_API_KEY)
    user_msg = state.get("user_msg", "")
    
    # Check confirmations (if negative, clear field)
    negative = any(w in user_msg.lower() for w in ["nein", "falsch", "nicht", "no ", "stimmt nicht"])
    profile = dict(state.get("profile", {}))
    pending = list(state.get("pending_confirmations", []))
    
    if negative and pending:
        for p in pending:
            profile[p] = None
        pending = []

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        extracted = json.loads(res.choices[0].message.content)
    except Exception:
        extracted = {}
        
    return {"extracted": extracted, "profile": profile, "pending_confirmations": pending}

def validate_node(state: GraphState):
    extracted = state.get("extracted", {})
    profile = dict(state.get("profile", {}))
    flags = dict(state.get("flags", {}))
    pending = list(state.get("pending_confirmations", []))
    
    flags["is_off_topic"] = bool(extracted.get("is_off_topic", False))
    flags["rd_detected"] = bool(flags.get("rd_detected", False))
    flags["retroactive_detected"] = bool(flags.get("retroactive_detected", False))
    
    needs_conf = extracted.get("needs_confirmation", []) or []
    for field in needs_conf:
        if field not in pending:
            pending.append(field)
            
    for k, v in extracted.items():
        if k in ["needs_confirmation", "is_off_topic"]:
            continue
        if v is not None:
            profile[k] = v
            
    # Legal Firewalls
    desc = str(profile.get("project_desc") or "").lower()
    inferred = str(profile.get("inferred_project_type") or "").lower()
    if any(w in desc for w in ["forschung", "entwicklung", "ki ", "software"]) or "fue" in inferred or "forschung" in inferred:
        flags["rd_detected"] = True
        
    if profile.get("project_status") == "already_started":
        flags["retroactive_detected"] = True
        
    # Completeness Check
    req = list(REQUIRED_FIELDS)
    if flags.get("rd_detected"):
        req += RD_EXTRA_FIELDS
        
    missing = [f for f in req if profile.get(f) is None]
    flags["profile_complete"] = (len(missing) == 0)
    
    return {"profile": profile, "flags": flags, "pending_confirmations": pending, "missing_fields": missing}

def reply_node(state: GraphState):
    client = Groq(api_key=GROQ_API_KEY)
    flags = state.get("flags", {})
    
    if flags.get("is_off_topic"):
        return {"assistant_reply": "Ich bin ein spezialisierter KI-Förderberater und kann leider nicht zu anderen Themen Auskunft geben. Könnten wir zu Ihrem Vorhaben zurückkehren?"}
        
    if flags.get("retroactive_detected") and flags.get("profile_complete"):
        return {"assistant_reply": ""} # Will trigger engine directly
        
    if flags.get("profile_complete"):
        return {"assistant_reply": ""} # Will trigger engine directly
        
    parts = []
    profile = state["profile"]
    if profile.get("industry"): parts.append(f"Branche: {profile['industry']}")
    if profile.get("state"): parts.append(f"Bundesland: {profile['state']}")
    if profile.get("employees"): parts.append(f"Mitarbeiter: {profile['employees']}")
    
    prompt = PLANNING_PROMPT.format(
        profile_summary=", ".join(parts) if parts else "Noch keine",
        missing_fields=", ".join(state["missing_fields"]),
        pending_confirmations=", ".join(state["pending_confirmations"]),
        last_message=state["user_msg"]
    )
    
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4
    )
    return {"assistant_reply": res.choices[0].message.content.strip()}

# ─── GRAPH DEFINITION ─────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)
workflow.add_node("extract", extract_node)
workflow.add_node("validate", validate_node)
workflow.add_node("reply", reply_node)

workflow.add_edge(START, "extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", "reply")
workflow.add_edge("reply", END)

_checkpointer = None

def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer

class FundaviaAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.saver = get_checkpointer()
        self.app = workflow.compile(checkpointer=self.saver)
        
    def process_message(self, user_msg: str):
        config = {"configurable": {"thread_id": self.session_id}}
        
        curr = self.app.get_state(config)
        state = dict(curr.values) if (curr and curr.values) else init_state()
            
        state["user_msg"] = user_msg
        state["history"] = [{"role": "user", "content": user_msg}]
        
        out_state = self.app.invoke(state, config)
        
        reply = out_state.get("assistant_reply", "")
        if reply:
            try:
                self.app.update_state(config, {"history": [{"role": "assistant", "content": reply}]})
            except Exception:
                pass
            
        return out_state
        
    def get_form_data(self, state: dict):
        p = state.get("profile", {})
        return {
            "employees": p.get("employees"),
            "state": p.get("state"),
            "district": p.get("district"),
            "company_age": p.get("company_age"),
            "project_type": p.get("inferred_project_type"),
            "project_status": p.get("project_status")
        }
        
    def get_query_text(self, state: dict):
        p = state.get("profile", {})
        return p.get("project_desc") or p.get("industry") or "Allgemein"
