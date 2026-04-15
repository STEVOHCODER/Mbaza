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
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, EmailStr
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mbaza-backend")

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_KEY and env_path.exists():
    # Handle UTF-8 BOM in .env (common on Windows editors)
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("GEMINI_API_KEY="):
            GEMINI_KEY = line.split("=", 1)[1].strip().strip("\"'")
            break

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret-before-deploy")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "7"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./autodev_enterprise.db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "6"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
GEMINI_BACKOFF_SECONDS = int(os.getenv("GEMINI_BACKOFF_SECONDS", "120"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "180"))
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

if JWT_SECRET == "change-this-secret-before-deploy":
    logger.warning("JWT_SECRET is using default value. Set a strong secret before deployment.")

client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

app = FastAPI(title="Mbaza - AI Consultant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    event_date = Column(DateTime, nullable=False)
    category = Column(String, nullable=False)


class PeriodTracker(Base):
    __tablename__ = "period_tracker"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True, nullable=False)
    start_date = Column(DateTime, nullable=False)
    intensity = Column(String, nullable=False)
    notes = Column(String, default="", nullable=False)


Base.metadata.create_all(bind=engine)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthUser(BaseModel):
    id: str
    email: str
    name: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


class CalendarCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    event_date: str
    category: str = "health"


class PeriodCreateRequest(BaseModel):
    start_date: str
    intensity: str
    notes: Optional[str] = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def parse_iso_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Date value cannot be empty")

    if "T" not in cleaned and len(cleaned) == 10:
        return datetime.combine(date.fromisoformat(cleaned), datetime.min.time())
    return datetime.fromisoformat(cleaned)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(user_id: int, email: str) -> str:
    expires_at = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "email": email, "exp": expires_at}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


session_memories: Dict[str, List[Dict[str, str]]] = {}
gemini_backoff_until: Optional[datetime] = None


def infer_topic(prompt: str, history: List[dict]) -> str:
    prompt_l = prompt.lower().strip()
    recent = " ".join([m.get("content", "") for m in history[-4:]]).lower()

    # Detect topic directly from current prompt first.
    if any(k in prompt_l for k in ["menstru", "mentru", "period", "cycle", "cramp", "pms", "flow"]):
        return "menstruation"
    if any(k in prompt_l for k in ["unprotected", "pregnan", "emergency contraception", "plan b"]):
        return "pregnancy-risk"
    if any(k in prompt_l for k in ["sti", "std", "hiv", "exposure", "test"]):
        return "sti-risk"

    # Use recent context only for clear follow-up phrasing.
    is_followup = bool(
        re.search(r"\b(it|this|that)\b", prompt_l)
        or "those symptoms" in prompt_l
        or "control it" in prompt_l
        or "manage it" in prompt_l
    )
    if is_followup and any(k in recent for k in ["menstru", "mentru", "period", "cycle", "cramp", "pms", "flow"]):
        return "menstruation"
    if is_followup and any(k in recent for k in ["unprotected", "pregnan", "emergency contraception", "plan b"]):
        return "pregnancy-risk"
    if is_followup and any(k in recent for k in ["sti", "std", "hiv", "exposure", "test"]):
        return "sti-risk"

    return "general"


def is_sexology_prompt(prompt: str, history: List[dict]) -> bool:
    prompt_l = prompt.lower().strip()
    if re.fullmatch(r"(h+i+|he+l+o+|hey+|yo+|hola+)[!?. ]*", prompt_l):
        return True
    return infer_topic(prompt, history) != "general"


def out_of_scope_response() -> str:
    return (
        "I only answer sexology and reproductive-health questions. "
        "Please ask about topics like puberty, menstruation, contraception, STI risk/testing, consent, or relationships."
    )


def local_fallback_response(prompt: str, history: Optional[List[dict]] = None) -> str:
    return (
        "Live AI is temporarily unavailable. Please retry your sexology question in a moment."
    )


def sanitize_assistant_response(response_text: str, prompt: str) -> str:
    lower = response_text.lower()

    invalid_markers = [
        "i'm feeling",
        "i feel overwhelmed",
        "thank you for offering me the chance to be here",
        "i appreciate you trusting me",
    ]
    if any(marker in lower for marker in invalid_markers):
        return local_fallback_response(prompt)
    return response_text


async def call_ollama(prompt: str, history: List[dict]) -> str:
    full_prompt = f"System: {MBAZA_SYSTEM_PROMPT}\n\n"
    for message in history:
        full_prompt += f"{message['role']}: {message['content']}\n"
    full_prompt += f"user: {prompt}\nmodel:"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(OLLAMA_TIMEOUT_SECONDS, connect=2.0),
            trust_env=False,
        ) as http_client:
            response = await http_client.post(
                OLLAMA_URL,
                json={
                    "model": LOCAL_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"num_predict": OLLAMA_NUM_PREDICT, "temperature": 0.4},
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("response", "No response from local AI.")
            return sanitize_assistant_response(text, prompt)
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning("Ollama fallback failed: %s", exc)
        return local_fallback_response(prompt, history)


async def call_llm(prompt: str, session_id: str) -> str:
    global gemini_backoff_until

    if session_id not in session_memories:
        session_memories[session_id] = []
    history = session_memories[session_id][-4:]
    if not is_sexology_prompt(prompt, history):
        response_text = out_of_scope_response()
        session_memories[session_id].append({"role": "user", "content": prompt})
        session_memories[session_id].append({"role": "assistant", "content": response_text})
        return response_text

    can_try_gemini = client and (not gemini_backoff_until or datetime.utcnow() >= gemini_backoff_until)
    if can_try_gemini:
        try:
            contents = []
            for message in history:
                role = "model" if message["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": message["content"]}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})

            gemini_call = client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config={"system_instruction": MBAZA_SYSTEM_PROMPT},
            )
            response = await asyncio.wait_for(gemini_call, timeout=GEMINI_TIMEOUT_SECONDS)
            if response and response.text:
                clean_text = sanitize_assistant_response(response.text, prompt)
                session_memories[session_id].append({"role": "user", "content": prompt})
                session_memories[session_id].append({"role": "assistant", "content": clean_text})
                return clean_text
        except (asyncio.TimeoutError, httpx.HTTPError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Gemini call failed: %s", exc)
            gemini_backoff_until = datetime.utcnow() + timedelta(seconds=GEMINI_BACKOFF_SECONDS)
    elif client:
        logger.info("Skipping Gemini until backoff ends at %s", gemini_backoff_until.isoformat())

    logger.info("Falling back to Ollama...")
    fallback_response = await call_ollama(prompt, history)
    session_memories[session_id].append({"role": "user", "content": prompt})
    session_memories[session_id].append({"role": "assistant", "content": fallback_response})
    return fallback_response


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections and websocket in self.active_connections[session_id]:
            self.active_connections[session_id].remove(websocket)

    async def send(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError) as exc:
            logger.warning("WebSocket send failed: %s", exc)


manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            message = payload.get("message")
            if not message or not isinstance(message, str):
                await manager.send({"type": "error", "content": "Invalid message payload."}, websocket)
                continue
            response = await call_llm(message, session_id)
            await manager.send({"type": "result", "content": response}, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except json.JSONDecodeError:
        await manager.send({"type": "error", "content": "Malformed JSON message."}, websocket)
        manager.disconnect(websocket, session_id)


@app.get("/api/health")
def health():
    return {"status": "ok", "name": "Mbaza"}


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    if len(payload.password) < 6:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 6 characters")

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(email=email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return AuthResponse(
        access_token=token,
        user=AuthUser(id=str(user.id), email=user.email, name=payload.name),
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)
    return AuthResponse(
        access_token=token,
        user=AuthUser(id=str(user.id), email=user.email),
    )


@app.get("/api/period/{sid}")
def get_period_entries(sid: str, db: Session = Depends(get_db)):
    return db.query(PeriodTracker).filter(PeriodTracker.session_id == sid).all()


@app.post("/api/period/{sid}")
def add_period_entry(sid: str, payload: PeriodCreateRequest, db: Session = Depends(get_db)):
    try:
        start_date = parse_iso_datetime(payload.start_date)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid start_date format") from exc

    entry = PeriodTracker(
        session_id=sid,
        start_date=start_date,
        intensity=payload.intensity,
        notes=(payload.notes or "").strip(),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.get("/api/calendar/{sid}")
def get_calendar_events(sid: str, db: Session = Depends(get_db)):
    return db.query(CalendarEvent).filter(CalendarEvent.session_id == sid).all()


@app.post("/api/calendar/{sid}")
def add_calendar_event(sid: str, payload: CalendarCreateRequest, db: Session = Depends(get_db)):
    try:
        event_date = parse_iso_datetime(payload.event_date)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid event_date format") from exc

    event = CalendarEvent(
        session_id=sid,
        title=payload.title.strip(),
        event_date=event_date,
        category=(payload.category or "health").strip(),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
