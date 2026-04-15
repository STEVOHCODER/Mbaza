import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import bcrypt
import httpx
import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, EmailStr
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mbaza-backend")

# ---------------- ENV ----------------
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

# Fix Render / old Postgres format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ---------------- GEMINI / OLLAMA ----------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret-before-deploy")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "7"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "6"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))

MBAZA_SYSTEM_PROMPT = (
    "You are Mbaza, an expert youth sexual and reproductive health consultant. "
    "Answer with practical, medically-sound sexology guidance. "
    "Core areas: puberty, consent, contraception, STI prevention/testing, safe sex, relationships, menstrual health, "
    "fertility awareness, sexual wellbeing, and healthy communication. "
    "If a question is outside sexology/reproductive health, do not answer that topic; "
    "briefly say you only handle sexology topics and ask the user to reframe within that scope. "
    "Rules: never roleplay as the user, never claim personal emotions or personal life events, "
    "never output self-therapy monologues, and do not invent clinical facts. "
    "Be clear, respectful, and direct; use short structured guidance with actionable steps."
)
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# ---------------- FASTAPI ----------------
app = FastAPI(title="Mbaza Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ---------------- DATABASE ----------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

# ---------------- MODELS ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    title = Column(String)
    event_date = Column(DateTime)
    category = Column(String)

class PeriodTracker(Base):
    __tablename__ = "period_tracker"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    start_date = Column(DateTime)
    intensity = Column(String)
    notes = Column(String)

Base.metadata.create_all(bind=engine)

# ---------------- SCHEMAS ----------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class CalendarCreateRequest(BaseModel):
    title: str
    event_date: str
    category: str = "health"

class PeriodCreateRequest(BaseModel):
    start_date: str
    intensity: str
    notes: Optional[str] = None

# ---------------- DB SESSION ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- UTILS ----------------
def hash_password(p: str):
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()

def verify_password(p: str, h: str):
    return bcrypt.checkpw(p.encode(), h.encode())

def create_token(user_id: int, email: str):
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def parse_date(v: str):
    if "T" not in v and len(v) == 10:
        return datetime.combine(date.fromisoformat(v), datetime.min.time())
    return datetime.fromisoformat(v)

# ---------------- MEMORY ----------------
session_memories: Dict[str, List[Dict[str, str]]] = {}
MAX_CHAT_HISTORY = 10

def build_chat_prompt(sid: str, user_message: str) -> str:
    history = session_memories.get(sid, [])[-MAX_CHAT_HISTORY:]
    prompt_parts = [f"Session ID: {sid}"]

    if history:
        prompt_parts.append("Recent conversation:")
        for item in history:
            role = item.get("role", "user").capitalize()
            content = item.get("content", "").strip()
            if content:
                prompt_parts.append(f"{role}: {content}")

    prompt_parts.append(f"User: {user_message.strip()}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

def store_message(sid: str, role: str, content: str) -> None:
    session_memories.setdefault(sid, []).append({"role": role, "content": content})
    session_memories[sid] = session_memories[sid][-MAX_CHAT_HISTORY:]

def get_gemini_reply(prompt: str) -> str:
    if not client:
        raise RuntimeError("Gemini API key is not configured")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            systemInstruction=MBAZA_SYSTEM_PROMPT,
            temperature=0.4,
            maxOutputTokens=400,
        ),
    )

    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()

    raise RuntimeError("Gemini returned an empty response")

async def get_ollama_reply(prompt: str) -> str:
    payload = {
        "model": LOCAL_MODEL,
        "prompt": f"{MBAZA_SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
    }

    timeout = httpx.Timeout(OLLAMA_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        response = await http_client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    text = (data.get("response") or "").strip()
    if text:
        return text

    raise RuntimeError("Ollama returned an empty response")

async def generate_chat_reply(sid: str, user_message: str) -> str:
    prompt = build_chat_prompt(sid, user_message)
    errors: List[str] = []

    if client:
        try:
            return await asyncio.to_thread(get_gemini_reply, prompt)
        except Exception as exc:
            logger.exception("Gemini chat request failed for session %s", sid)
            errors.append(f"Gemini: {exc}")

    try:
        return await get_ollama_reply(prompt)
    except Exception as exc:
        logger.exception("Ollama chat request failed for session %s", sid)
        errors.append(f"Ollama: {exc}")

    logger.error("All chat providers failed for session %s: %s", sid, " | ".join(errors))
    return (
        "I could not reach the AI provider right now. "
        "Please check `GEMINI_API_KEY` or make sure Ollama is running, then try again."
    )

# ---------------- AUTH ----------------
@app.post("/api/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower()

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already exists")

    user = User(email=email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "access_token": create_token(user.id, user.email),
        "user": {"id": user.id, "email": user.email},
    }

@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")

    return {
        "access_token": create_token(user.id, user.email),
        "user": {"id": user.id, "email": user.email},
    }

# ---------------- CALENDAR ----------------
@app.post("/api/calendar/{sid}")
def add_event(sid: str, payload: CalendarCreateRequest, db: Session = Depends(get_db)):
    event = CalendarEvent(
        session_id=sid,
        title=payload.title,
        event_date=parse_date(payload.event_date),
        category=payload.category,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

@app.get("/api/calendar/{sid}")
def get_events(sid: str, db: Session = Depends(get_db)):
    return db.query(CalendarEvent).filter(CalendarEvent.session_id == sid).all()

# ---------------- PERIOD ----------------
@app.post("/api/period/{sid}")
def add_period(sid: str, payload: PeriodCreateRequest, db: Session = Depends(get_db)):
    entry = PeriodTracker(
        session_id=sid,
        start_date=parse_date(payload.start_date),
        intensity=payload.intensity,
        notes=payload.notes or "",
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

@app.get("/api/period/{sid}")
def get_period(sid: str, db: Session = Depends(get_db)):
    return db.query(PeriodTracker).filter(PeriodTracker.session_id == sid).all()

# ---------------- HEALTH ----------------
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ---------------- WEBSOCKET ----------------
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, sid: str):
        await ws.accept()
        self.active.setdefault(sid, []).append(ws)

    def disconnect(self, ws: WebSocket, sid: str):
        if sid in self.active and ws in self.active[sid]:
            self.active[sid].remove(ws)

manager = ConnectionManager()

@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await manager.connect(ws, sid)
    try:
        while True:
            msg = await ws.receive_text()
            cleaned_msg = msg.strip()
            if not cleaned_msg:
                await ws.send_json({"reply": "Please send a message so I can help."})
                continue

            store_message(sid, "user", cleaned_msg)
            reply = await generate_chat_reply(sid, cleaned_msg)
            store_message(sid, "assistant", reply)
            await ws.send_json({"reply": reply})
    except WebSocketDisconnect:
        manager.disconnect(ws, sid)

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
