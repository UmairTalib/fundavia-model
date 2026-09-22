from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os
import uuid
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import recommendation_engine as eng
import conversation_agent

DB_PATH = os.path.join(BASE_DIR, 'fundavia.db')
eng.DB_PATH = DB_PATH

app = FastAPI(title="Fundavia Recommendation Engine API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    employees: Optional[Any] = None
    state: Optional[str] = None
    district: Optional[str] = None
    company_age: Optional[str] = None
    project_type: Optional[str] = None
    project_status: Optional[str] = None
    budget: Optional[float] = None
    form_data: Optional[Dict[str, Any]] = None

@app.post("/api/test_chat")
async def api_test_chat(req: QueryRequest):
    agent = conversation_agent.FundaviaAgent("debug_" + str(uuid.uuid4()))
    try:
        state = agent.process_message(req.query)
        res = {"state": state}
        if state.get("flags", {}).get("profile_complete"):
            fd = agent.get_form_data(state)
            q = agent.get_query_text(state)
            recs = eng.recommend(q, DB_PATH, top_k=5, form_data=fd)
            res["recommendations"] = recs
        return res
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/api/recommend")
async def api_recommend(req: QueryRequest):
    try:
        fd = req.form_data.copy() if req.form_data else {}
        if req.employees is not None: fd["employees"] = req.employees
        if req.state is not None: fd["state"] = req.state
        if req.district is not None: fd["district"] = req.district
        if req.company_age is not None: fd["company_age"] = req.company_age
        if req.project_type is not None: fd["project_type"] = req.project_type
        if req.project_status is not None: fd["project_status"] = req.project_status
        if req.budget is not None: fd["budget"] = req.budget

        results = eng.recommend(req.query, DB_PATH, top_k=5, form_data=fd)
        return {"status": "success", **results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = Query(None)):
    await websocket.accept()
    
    if not session_id:
        session_id = str(uuid.uuid4())
        
    agent = conversation_agent.FundaviaAgent(session_id)
    
    config = {"configurable": {"thread_id": session_id}}
    state = agent.app.get_state(config).values
    history = state.get("history", []) if state else []
    
    if not history:
        welcome_msg = "Guten Tag! Ich bin Ihr KI-Förderberater von Fundavia. Erzählen Sie mir einfach: Was macht Ihr Unternehmen, und was haben Sie vor?"
        agent.app.update_state(config, {"history": [{"role": "assistant", "content": welcome_msg}]})
        await websocket.send_json({"role": "agent", "text": welcome_msg})
    else:
        for msg in history:
            role = "user" if msg["role"] == "user" else "agent"
            await websocket.send_json({"role": role, "text": msg["content"]})
            
        if state and state.get("flags", {}).get("profile_complete"):
            await websocket.send_json({
                "role": "agent",
                "text": "(Fortsetzung) Ich lade Ihre passenden Programme..."
            })
            form_data = agent.get_form_data(state)
            query_text = agent.get_query_text(state)
            results = eng.recommend(query_text, DB_PATH, top_k=5, form_data=form_data)
            await websocket.send_json({
                "role": "results",
                "status": "success",
                "data": results,
                "results": results.get("results", [])
            })

    try:
        while True:
            user_msg = await websocket.receive_text()
            try:
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
                    await websocket.send_json({
                        "role": "results",
                        "status": "success",
                        "data": results,
                        "results": results.get("results", [])
                    })
            except Exception as e:
                err = traceback.format_exc()
                print("WS EXCEPTION:", err)
                await websocket.send_json({"role": "agent", "text": f"SYSTEM FEHLER: {err}"})
                
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
