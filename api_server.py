from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os

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

from typing import Union

class QueryRequest(BaseModel):
    query: str
    employees: Union[int, str, None] = None
    state: Union[str, None] = None
    district: Union[str, None] = None
    company_age: Union[str, None] = None
    project_type: Union[str, None] = None
    project_status: Union[str, None] = None
    budget: Union[int, str, None] = None

@app.post("/api/recommend")
async def recommend_endpoint(req: QueryRequest):
    if not req.query or len(req.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Bitte beschreiben Sie Ihr Vorhaben ausführlicher.")
    try:
        form_data = {
            "employees": req.employees,
            "state": req.state,
            "district": req.district,
            "company_age": req.company_age,
            "project_type": req.project_type,
            "project_status": req.project_status
        }
        result = eng.recommend(req.query, DB_PATH, top_k=3, form_data=form_data)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_landing():
    return FileResponse(os.path.join(BASE_DIR, "static", "landing.html"))

@app.get("/form")
async def serve_form():
    return FileResponse(os.path.join(BASE_DIR, "static", "form.html"))

# Mount static files

@app.get("/chat")
async def serve_chat():
    return FileResponse(os.path.join(BASE_DIR, "static", "chat.html"))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    agent = conversation_agent.ConversationAgent()
    await websocket.send_json({
        "role": "agent",
        "text": "Guten Tag! Ich bin Ihr KI-Förderberater von Fundavia. Erzählen Sie mir einfach: Was macht Ihr Unternehmen, und was haben Sie vor?"
    })

    try:
        while True:
            user_msg = await websocket.receive_text()

            # ── STEP 1: Extract facts silently ─────────────────────────────
            extracted = agent.extract(user_msg)

            # ── STEP 2: Handle confirmation of derived fields ───────────────
            for field in list(agent.pending_confirmations):
                agent.handle_confirmation(user_msg, field)

            # ── STEP 3: Update profile state + run legal firewalls ──────────
            agent.update_profile(extracted, user_msg)

            # ── STEP 4: Check if profile is complete ────────────────────────
            if agent.flags["profile_complete"]:
                # INTERMEDIATE: Show thinking message first
                await websocket.send_json({
                    "role": "agent",
                    "text": "Ich habe alle notwendigen Angaben. Einen Moment — ich gleiche Ihr Profil jetzt mit der Förderdatenbank ab..."
                })

                # FIRE the SQL engine
                form_data = agent.get_form_data()
                query_text = agent.get_query_text()
                results = eng.recommend(query_text, DB_PATH, top_k=5, form_data=form_data)

                # INTERMEDIATE FIT SUMMARY before showing full cards
                programmes = results.get("results", [])
                if programmes:
                    top_names = ", ".join(
                        f"**{p.get('Programm_Name', 'Unbekannt')}**"
                        for p in programmes[:3]
                    )
                    summary_msg = (
                        f"Ich habe {len(programmes)} passende Förderprogramme gefunden. "
                        f"Die vielversprechendsten sind: {top_names}. "
                        f"Hier ist eine Übersicht:"
                    )
                else:
                    summary_msg = (
                        "Leider konnte ich mit den aktuellen Angaben keine passenden "
                        "Programme finden. Möchten Sie die Angaben anpassen?"
                    )

                await websocket.send_json({"role": "agent", "text": summary_msg})

                # Send the full result cards
                await websocket.send_json({"role": "results", "data": results})

                # Break out of the loop — session complete
                break

            else:
                # ── STEP 5: LLM Planning Node — contextual next question ────
                reply = agent.plan_next_reply(user_msg)
                if reply:
                    await websocket.send_json({"role": "agent", "text": reply})

    except WebSocketDisconnect:
        print("Client disconnected")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)


@app.post("/api/draft")
def draft_endpoint(req: SearchRequest):
    form_data = {
        "employees": req.employees,
        "state": req.state
    }
    draft = eng.generate_draft(req.query, req.program_name, form_data)
    return {"draft": draft}
