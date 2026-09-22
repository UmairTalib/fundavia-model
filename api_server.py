from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os
import uuid

# Use relative paths for cloud hosting compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import recommendation_engine as eng
import conversation_agent

# Point engine to local sqlite copy
DB_PATH = os.path.join(BASE_DIR, 'fundavia.db')
eng.DB_PATH = DB_PATH

app = FastAPI(title="Fundavia Recommendation Engine API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    form_data: Optional[Dict[str, Any]] = None

@app.post("/api/recommend")
async def api_recommend(req: QueryRequest):
    try:
        results = eng.recommend(req.query, DB_PATH, top_k=5, form_data=req.form_data)
        return {"status": "success", **results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = Query(None)):
    await websocket.accept()
    
    if not session_id:
        session_id = str(uuid.uuid4())
        
    agent = conversation_agent.FundaviaAgent(session_id)
    
    # Check if there is existing history
    config = {"configurable": {"thread_id": session_id}}
    state = agent.app.get_state(config).values
    history = state.get("history", []) if state else []
    
    if not history:
        welcome_msg = "Guten Tag! Ich bin Ihr KI-Förderberater von Fundavia. Erzählen Sie mir einfach: Was macht Ihr Unternehmen, und was haben Sie vor?"
        agent.app.update_state(config, {"history": [{"role": "assistant", "content": welcome_msg}]})
        await websocket.send_json({"role": "agent", "text": welcome_msg})
    else:
        # Send history back to client
        for msg in history:
            role = "user" if msg["role"] == "user" else "agent"
            await websocket.send_json({"role": role, "text": msg["content"]})
            
        # Check if profile was already complete from previous session
        if state and state.get("flags", {}).get("profile_complete"):
             await websocket.send_json({
                 "role": "agent",
                 "text": "(Fortsetzung) Ich lade Ihre passenden Programme..."
             })
             form_data = agent.get_form_data(state)
             query_text = agent.get_query_text(state)
             results = eng.recommend(query_text, DB_PATH, top_k=5, form_data=form_data)
             await websocket.send_json({"status": "success", "results": results.get("results", [])})

    try:
        while True:
            user_msg = await websocket.receive_text()
            
            # Process via LangGraph
            final_state = agent.process_message(user_msg)
            reply = final_state.get("assistant_reply", "")
            
            if reply:
                await websocket.send_json({"role": "agent", "text": reply})
                
            flags = final_state.get("flags", {})
            if flags.get("profile_complete"):
                await websocket.send_json({
                    "role": "agent",
                    "text": "Ich habe alle notwendigen Angaben. Einen Moment — ich gleiche Ihr Profil jetzt mit der Förderdatenbank ab..."
                })
                form_data = agent.get_form_data(final_state)
                query_text = agent.get_query_text(final_state)
                results = eng.recommend(query_text, DB_PATH, top_k=5, form_data=form_data)
                
                await websocket.send_json({"status": "success", "results": results.get("results", [])})
                
    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")

@app.get("/")
async def serve_landing():
    return FileResponse(os.path.join(BASE_DIR, "static", "landing.html"))

@app.get("/chat")
async def serve_chat():
    return FileResponse(os.path.join(BASE_DIR, "static", "chat.html"))

@app.get("/form")
async def serve_form():
    return FileResponse(os.path.join(BASE_DIR, "static", "form.html"))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=10000, reload=True)
