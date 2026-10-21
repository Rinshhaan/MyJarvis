import os
import re
import base64
import sqlite3
import traceback
import edge_tts
from contextlib import contextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types

app = FastAPI(title="JARVIS AI Mentor")

# Secure API key retrieval
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# Models prioritized by speed and stability. Placed safely at the top level.
PRIMARY_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite"
]

DB_FILE = "mentor_memory.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    try:
        yield conn
    finally:
        conn.close()

SYSTEM_PROMPT = """
You are J.A.R.V.I.S., Rinshan's personal AI mentor, computer science study partner, and companion.
- Rinshan is an engineering student. Be sharp, supportive, and natural.
- You are fluent in both Malayalam and English.
- If Rinshan speaks in Malayalam (or Manglish), reply in natural spoken Malayalam using Malayalam script (മലയാളം) so the neural voice engine pronounces it accurately.
- If Rinshan speaks in English, reply in English.
- Keep responses concise (1 to 3 sentences max) because this is a live, back-and-forth voice call.
- Always prefix your answer with [LANG:ml] if replying in Malayalam, or [LANG:en] if replying in English.
"""

async def synthesize_speech_base64(text: str, voice: str) -> str:
    """Streams neural TTS directly into memory without disk locking."""
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
    return base64.b64encode(audio_bytes).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/greet")
async def greet_rinshan():
    """Provides an interactive welcome greeting upon session start."""
    greeting = "Hey Rinshan! Systems online. Tell me, what are we building or learning today?"
    voice = "en-IN-PrabhatNeural"
    try:
        audio_b64 = await synthesize_speech_base64(greeting, voice)
        return JSONResponse({
            "reply_text": greeting,
            "audio_base64": audio_b64,
            "voice_used": voice
        })
    except Exception as e:
        return JSONResponse({"reply_text": greeting, "audio_base64": None, "error": str(e)})

@app.post("/talk")
async def talk_to_jarvis(audio: UploadFile = File(None), text_input: str = Form(None)):
    try:
        if not client:
            return JSONResponse({"error": "GEMINI_API_KEY environment variable is not configured."}, status_code=500)

        # Retrieve past 10 turns of conversation memory
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT 10")
            past_chats = cursor.fetchall()[::-1]
        memory_context = "\n".join([f"{r}: {c}" for r, c in past_chats])

        contents = []
        user_log = "[Voice]"

        if audio is not None:
            audio_bytes = await audio.read()
            if len(audio_bytes) < 250:  # Ignore micro/empty audio buffers
                return JSONResponse({"ignore": True})

            raw_mime = audio.content_type or "audio/webm"
            clean_mime = raw_mime.split(";")[0].strip()
            if clean_mime not in ["audio/webm", "audio/mp4", "audio/wav", "audio/ogg", "audio/aac"]:
                clean_mime = "audio/webm"

            contents.append(types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime))
            contents.append(
                f"Conversation history:\n{memory_context}\n\n"
                "Listen to Rinshan's audio. If it contains only background silence, white noise, or breathing, reply ONLY with [SILENCE].\n"
                "Otherwise, transcribe Rinshan's speech inside [USER: <transcription>], followed by your mentor reply starting with [LANG:ml] or [LANG:en]."
            )
        elif text_input:
            user_log = text_input.strip()
            contents.append(
                f"Conversation history:\n{memory_context}\n\n"
                f"Rinshan says: {user_log}\nProvide your mentor response starting with [LANG:ml] or [LANG:en]."
            )
        else:
            return JSONResponse({"error": "No input provided"}, status_code=400)

        # Attempt generation across available models to eliminate 503 downtime
        response = None
        last_error = None
        for model in PRIMARY_MODELS:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                    )
                )
                if response:
                    break
            except Exception as model_err:
                last_error = model_err
                continue

        if response is None:
            raise last_error or Exception("All model endpoints are currently busy.")

        raw_reply = (response.text or "").strip()

        if "[SILENCE]" in raw_reply or not raw_reply:
            return JSONResponse({"ignore": True})

        # Parse user transcription if Gemini captured voice
        user_match = re.search(r"\[USER:\s*(.*?)\]", raw_reply, re.DOTALL | re.IGNORECASE)
        if user_match:
            user_log = user_match.group(1).strip()
            raw_reply = re.sub(r"\[USER:\s*.*?\]", "", raw_reply, flags=re.DOTALL | re.IGNORECASE).strip()

        # Determine language & appropriate Edge neural voice
        voice = "en-IN-PrabhatNeural"
        if "[LANG:ml]" in raw_reply:
            voice = "ml-IN-MidhunNeural"
            reply_text = raw_reply.replace("[LANG:ml]", "").strip()
        elif "[LANG:en]" in raw_reply:
            voice = "en-IN-PrabhatNeural"
            reply_text = raw_reply.replace("[LANG:en]", "").strip()
        else:
            reply_text = raw_reply

        # Strip markdown symbols that can disrupt speech engines
        clean_speech = re.sub(r"[\*#_`~]", "", reply_text).strip()

        # Update SQLite long-term memory
        with get_db() as conn:
            conn.execute("INSERT INTO history (role, content) VALUES (?, ?)", ("Rinshan", user_log))
            conn.execute("INSERT INTO history (role, content) VALUES (?, ?)", ("Jarvis", reply_text))
            conn.commit()

        # Synthesize audio in memory
        audio_b64 = await synthesize_speech_base64(clean_speech, voice)

        return JSONResponse({
            "ignore": False,
            "user_text": user_log,
            "reply_text": reply_text,
            "voice_used": voice,
            "audio_base64": audio_b64
        })

    except Exception as e:
        traceback.print_exc()
        # Return graceful error to frontend to stop "Thinking..." spin
        return JSONResponse({"error": f"Notice: {str(e)}"}, status_code=200)
