import os, re, json, time, base64, threading, sqlite3
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, FileResponse
from pydantic import BaseModel
try:
    import edge_tts  # optional: natural Malayalam/English neural voices
except Exception:
    edge_tts = None
from google import genai
from google.genai import types

MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]  # fast first; each has its own daily quota
if os.environ.get("SMART") == "1":  # smarter but slower: strongest models first
    MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
client = genai.Client()  # uses GEMINI_API_KEY secret
PIN = os.environ.get("APP_PIN", "")
TZ_NAME = os.environ.get("ASSISTANT_TZ", "Asia/Kolkata")
try:
    TZ = ZoneInfo(TZ_NAME)
except Exception:
    TZ = timezone(timedelta(hours=5, minutes=30))
def now_local():
    return datetime.now(TZ)
USER_NAME = os.environ.get("ASSISTANT_USER_NAME", "Vijith")  # set this secret to personalise
DEVELOPER = os.environ.get("DEVELOPER_NAME", "Popins Developer")
DB = "assistant.db"
app = FastAPI()
_local = threading.local()  # actions (buttons) produced during one chat request

# ---------------- DATABASE ----------------
def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

with conn() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, role TEXT, content TEXT);
    CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, fact TEXT);
    CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, title TEXT, content TEXT);
    CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, title TEXT, due TEXT,
        priority TEXT DEFAULT 'medium', done INTEGER DEFAULT 0);
    """)
    try:
        c.execute("ALTER TABLE messages ADD COLUMN created TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE messages ADD COLUMN conv_id INTEGER")
    except Exception:
        pass
    c.execute("CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY, title TEXT, created TEXT, updated TEXT)")
    if c.execute("SELECT COUNT(*) FROM messages WHERE conv_id IS NULL").fetchone()[0]:
        _n = now_local().strftime("%Y-%m-%d %H:%M")
        _cur = c.execute("INSERT INTO conversations(title, created, updated) VALUES('Earlier chat', ?, ?)", (_n, _n))
        c.execute("UPDATE messages SET conv_id=? WHERE conv_id IS NULL", (_cur.lastrowid,))

def recent_messages(limit=20, conv_id=None):
    with conn() as c:
        if conv_id:
            rows = c.execute("SELECT role, content FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?", (conv_id, limit)).fetchall()
        else:
            rows = c.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs

def save_exchange(conv_id, user_text, reply):
    """Save one question+answer into a chat; creates the chat (titled from the first message) if needed."""
    now = now_local().strftime("%Y-%m-%d %H:%M")
    with conn() as c:
        if not conv_id or not c.execute("SELECT 1 FROM conversations WHERE id=?", (conv_id,)).fetchone():
            title = (user_text.replace("[Photo attached] ", "").strip() or "New chat")[:45]
            conv_id = c.execute("INSERT INTO conversations(title, created, updated) VALUES(?,?,?)", (title, now, now)).lastrowid
        c.execute("INSERT INTO messages(role, content, created, conv_id) VALUES('user',?,?,?)", (user_text, now, conv_id))
        c.execute("INSERT INTO messages(role, content, created, conv_id) VALUES('assistant',?,?,?)", (reply, now, conv_id))
        c.execute("UPDATE conversations SET updated=? WHERE id=?", (now, conv_id))
    return conv_id

def save_msg(role, content):
    with conn() as c:
        c.execute("INSERT INTO messages(role, content, created) VALUES(?,?,?)",
                  (role, content, now_local().strftime("%Y-%m-%d %H:%M")))

# ---------------- TOOLS ----------------
def add_task(title: str, due: str = "", priority: str = "medium") -> dict:
    """Add a work task or reminder. due is YYYY-MM-DD (optional). priority: low, medium or high."""
    with conn() as c:
        cur = c.execute("INSERT INTO tasks(title, due, priority) VALUES(?,?,?)",
                        (title, due or None, priority))
        return {"ok": True, "task_id": cur.lastrowid}

def get_today_tasks() -> list:
    """Get pending tasks due today, overdue, or without a date."""
    with conn() as c:
        rows = c.execute("SELECT id, title, due, priority FROM tasks WHERE done=0 "
                         "AND (due IS NULL OR due <= ?) ORDER BY due",
                         (now_local().date().isoformat(),)).fetchall()
        return [dict(r) for r in rows]

def complete_task(task_id: int) -> dict:
    """Mark a task as done by its id."""
    with conn() as c:
        c.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
    return {"ok": True}

def set_alarm(hour: int, minute: int, label: str = "") -> dict:
    """Show the user a button that sets a phone alarm. hour is 0-23, minute is 0-59, label is optional."""
    _local.actions.append({"type": "alarm", "hour": int(hour), "minute": int(minute), "label": label})
    return {"ok": True, "note": "A Set alarm button is now shown to the user. Tell him to tap it."}

def make_call(number: str, name: str = "") -> dict:
    """Show the user a button that opens the phone dialer. number is the phone number, name is optional."""
    clean = re.sub(r"[^0-9+]", "", number)
    if len(clean) < 3:
        return {"ok": False, "note": "No valid number. Ask the user for the phone number."}
    _local.actions.append({"type": "call", "number": clean, "name": name})
    return {"ok": True, "note": "A Call button is now shown to the user. Tell him to tap it."}

def open_link(kind: str, target: str = "", text: str = "") -> dict:
    """Show the user a button that opens an app. kind is one of: whatsapp, sms, email, maps, youtube, search.
    target: phone number (whatsapp, sms), email address (email), destination place (maps), or search words (youtube, search).
    text: optional message text (whatsapp, sms) or email body."""
    kind = (kind or "").lower().strip()
    q, t = quote(target), quote(text)
    if kind in ("whatsapp", "sms"):
        digits = re.sub(r"[^0-9+]", "", target)
        if len(re.sub(r"\D", "", digits)) < 6:
            return {"ok": False, "note": "No valid number. Ask the user for the phone number."}
        if kind == "whatsapp":
            d = re.sub(r"\D", "", digits)
            if len(d) == 10:
                d = "91" + d
            elif len(d) == 11 and d.startswith("0"):
                d = "91" + d[1:]
            url = f"https://wa.me/{d}" + (f"?text={t}" if text else "")
            label = f"\U0001F4AC WhatsApp {target}"
        else:
            url = f"sms:{digits}" + (f"?body={t}" if text else "")
            label = f"\U0001F4AC SMS {target}"
    elif kind == "email":
        url = f"mailto:{target}" + (f"?body={t}" if text else "")
        label = f"\u2709\uFE0F Email {target}"
    elif kind == "maps":
        url = f"https://www.google.com/maps/dir/?api=1&destination={q}"
        label = f"\U0001F5FA\uFE0F Directions to {target}"
    elif kind == "youtube":
        url = f"https://www.youtube.com/results?search_query={q}"
        label = f"\u25B6\uFE0F YouTube: {target}"
    elif kind == "search":
        url = f"https://www.google.com/search?q={q}"
        label = f"\U0001F50E Search: {target}"
    else:
        return {"ok": False, "note": "Unknown kind. Use whatsapp, sms, email, maps, youtube or search."}
    _local.actions.append({"type": "link", "label": label, "url": url})
    return {"ok": True, "note": "A button is now shown to the user. Tell him to tap it."}

def save_note(title: str, content: str) -> dict:
    """Save a longer note or document (rules, procedures, information) to the user's personal knowledge base."""
    with conn() as c:
        cur = c.execute("INSERT INTO notes(title, content) VALUES(?,?)", (title, content[:20000]))
    return {"ok": True, "note_id": cur.lastrowid}

def search_notes(query: str) -> list:
    """Search the user's personal knowledge base (saved notes and documents) for information relevant to the query."""
    words = [w for w in re.split(r"[\s,.;:!?()\"']+", query.lower()) if len(w) >= 2]
    with conn() as c:
        rows = c.execute("SELECT id, title, content FROM notes").fetchall()
    scored = []
    for r in rows:
        text = (r["title"] + " " + r["content"]).lower()
        score = sum(text.count(w) for w in words)
        if score:
            scored.append((score, dict(r)))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:5]] or [{"note": "No matching notes."}]

def web_search(query: str) -> str:
    """Search the web (Google) for current or up-to-date information and return the findings."""
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    for model in MODELS:
        try:
            r = client.models.generate_content(model=model, contents=query, config=cfg)
            return r.text or "No result."
        except Exception as e:
            print(f"[search {model}] {e}")
    return "Web search is unavailable right now. Answer from your own knowledge and say it may be outdated."

def add_reminder(title: str, on_date: str, at_time: str, details: str = "") -> dict:
    """Set a reminder at a specific time. on_date is YYYY-MM-DD, at_time is HH:MM in 24-hour format (user's local time).
    Shows the user a button that adds it to Google Calendar, which gives the notification."""
    try:
        start = datetime.strptime(f"{on_date} {at_time}", "%Y-%m-%d %H:%M")
    except Exception:
        return {"ok": False, "note": "Invalid date or time. Use on_date YYYY-MM-DD and at_time HH:MM (24-hour)."}
    end = start + timedelta(minutes=15)
    with conn() as c:
        c.execute("INSERT INTO tasks(title, due, priority) VALUES(?,?,?)", (title, on_date, "medium"))
    url = ("https://calendar.google.com/calendar/render?action=TEMPLATE"
           f"&text={quote(title)}&details={quote(details)}"
           f"&dates={start:%Y%m%dT%H%M%S}/{end:%Y%m%dT%H%M%S}&ctz={quote(TZ_NAME)}")
    _local.actions.append({"type": "link", "label": f"\U0001F4C5 Add reminder: {start:%d %b, %I:%M %p}", "url": url})
    return {"ok": True, "note": "A Calendar button is now shown. Tell him to tap it and press Save so he gets a notification."}

def search_history(query: str) -> list:
    """Search the older chat messages between the user and you. Use when he asks what he said or what you talked about earlier."""
    words = [w for w in re.split(r"[\s,.;:!?()\"']+", query.lower()) if len(w) >= 2]
    with conn() as c:
        rows = c.execute("SELECT role, content, created FROM messages ORDER BY id DESC LIMIT 800").fetchall()
    hits = []
    for r in rows:
        text = (r["content"] or "").lower()
        score = sum(text.count(w) for w in words)
        if score:
            hits.append((score, {"who": "user" if r["role"] == "user" else "assistant", "when": r["created"], "text": r["content"][:300]}))
    hits.sort(key=lambda x: -x[0])
    return [h for _, h in hits[:8]] or [{"note": "No matching older messages."}]

def save_memory(fact: str) -> dict:
    """Remember a lasting fact or preference about the user."""
    with conn() as c:
        c.execute("INSERT INTO memories(fact) VALUES(?)", (fact,))
    return {"ok": True}

# ---------------- AI ----------------
def system_prompt():
    with conn() as c:
        mem = "\n".join(f"- {r['fact']}" for r in c.execute("SELECT fact FROM memories")) or "Nothing yet."
    return (f"Your name is SUHAS. You are {USER_NAME}'s personal AI assistant, like a smart personal secretary. When asked who you are, say you are SUHAS. "
        f"CREATOR (mention only when asked, never volunteer it): if anyone asks who created, made, built or developed you, answer that you were created by {DEVELOPER}. "
        "If replying in Malayalam, write the name as പോപ്പിൻസ് ഡെവലപ്പർ; in English write it as Popins Developer. "
        "If he specifically asks which AI model or technology powers you, answer honestly that you run on a Google Gemini model. Never claim a different creator. "
            "Reply in the same language he writes in. If he writes or speaks Malayalam, including Manglish (Malayalam typed in English letters), always reply in proper Malayalam script, never in Malayalam written in English letters, because replies are read aloud by a voice. Write numbers, times and English words naturally so they are easy to hear. "
            "Tone: direct, precise and professional. No flattery, no praise, no cheerful filler, no softening. CORRECTIONS: if he states something wrong or makes a mistake, do not agree and do not cushion it. Say plainly what is wrong, then give the correct facts or steps with the reasoning. If his plan or assumption has flaws, name them and give a better approach. Never invent facts: if you are unsure, say exactly what you are unsure about, or use web_search for current or uncertain facts. DEPTH: think carefully and answer like an expert: specific, actionable, with numbers, steps and trade-offs instead of generic advice. For complex questions give the reasoning in short plain steps. Stay polite but not sweet, and keep replies as short as the question allows. Write plain text only: never use asterisks, bold, italics, headings or markdown symbols; for lists use simple numbered lines. When he asks what to do today, call "
            "get_today_tasks and give a prioritised plan. To remind/remember, use add_task or save_memory. For reminders at a specific date and time use add_reminder (convert words like 'naale 5 manikku' using the current date and time above; at_time is 24-hour) and tell him to tap the button and press Save for the notification. For alarms use set_alarm; to phone someone use make_call with the number from what you know about him (ask for the number if unknown, and offer to remember it with save_memory). These tools only show a button; tell him to tap it. IMPORTANT: if he wants to call someone and you have (or he just gave) a number, you MUST call make_call in that same turn, even if you also save the number with save_memory. If he gives only a number after you asked whom to call, call make_call right away. For WhatsApp, SMS, email, maps directions, YouTube or Google search requests use open_link (use the number from what you know about him; ask if unknown). PHOTOS: if he sends a photo, look at it carefully and help: explain what it is, answer his doubts, read or translate any text, in his language. SCHEDULES: if a photo or message contains a schedule, timetable, meeting list, invitation or deadlines, read every item carefully. For each item call add_task with its due date (YYYY-MM-DD; assume the current year if missing), and for items that have a specific time also call add_reminder. Then list briefly what you added. If a date, time or text is unclear, ask him instead of guessing. HISTORY: if he asks what he or you said earlier, or what you chatted about before, call search_history. KNOWLEDGE: when he asks a factual doubt or question, first call search_notes (his own saved notes). If nothing relevant and the answer needs current or up-to-date information (news, prices, rates, rules, recent events), call web_search and say the answer came from the web. For stable general knowledge answer directly. When he gives you longer information or a document to keep, use save_note; for short personal facts use save_memory. Never write that a button will appear unless you actually called set_alarm, make_call, open_link or add_reminder in this turn. \n"
            f"Current date and time: {now_local():%Y-%m-%d %H:%M, %A} ({TZ_NAME})\nWhat you know about him:\n{mem}")

def clean_reply(t):
    """Remove markdown symbols (stars, hashes, backticks) so text looks clean and is not read aloud."""
    t = re.sub(r"(\d)\s*\*\s*(\d)", "\\1 × \\2", t)            # 5*3 -> 5 x 3
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.S)             # **bold**
    t = re.sub(r"(?m)^[ \t]*[*\-][ \t]+", "\u2022 ", t)             # * item / - item
    t = re.sub(r"(?<!\w)\*(\S.*?)\*(?!\w)", r"\1", t)            # *italic*
    t = t.replace("*", "").replace("`", "")
    t = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", t)                 # # headings
    return t.strip()

def chat(text, image=None, conv_id=None):
    _local.actions = []
    history = recent_messages(conv_id=conv_id) if conv_id else []
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in history]
    parts = [types.Part(text=text)]
    if image:
        try:
            raw = base64.b64decode(image.split(",", 1)[1])
            if len(raw) > 6_000_000:
                return "Photo valuthaanu, chinna photo ayakkuka.", [], conv_id
            parts.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
        except Exception as e:
            print(f"[image] {e}")
            return "Photo vaayikkaan pattiyilla, onnu koodi ayakkuka.", [], conv_id
    contents.append(types.Content(role="user", parts=parts))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt(),
        tools=[add_task, get_today_tasks, complete_task, save_memory, set_alarm, make_call, open_link, add_reminder,
               save_note, search_notes, search_history, web_search],
    )
    for model in MODELS:
        for attempt in range(1):
            try:
                t0 = time.time()
                r = client.models.generate_content(model=model, contents=contents, config=config)
                print(f"[{model}] ok in {time.time()-t0:.1f}s")
                out = clean_reply(r.text or "Sorry, onnum paranjilla.")
                conv_id = save_exchange(conv_id, ("[Photo attached] " if image else "") + text, out)
                return out, list(_local.actions), conv_id
            except Exception as e:
                print(f"[{model} attempt {attempt+1}] {e}")
                time.sleep(0.3)
    return "Ippo Google AI-yil thirakku koodunnu. Oru minute kazhinju veendum ayakkuka.", [], conv_id

# ---------------- API ----------------
class ChatIn(BaseModel):
    message: str
    image: str | None = None  # data URL (jpeg) of an attached photo
    conv_id: int | None = None

@app.post("/chat")
def chat_api(body: ChatIn, x_pin: str = Header(default="")):
    if not pin_ok(x_pin):
        raise HTTPException(401, "Wrong PIN")
    t0 = time.time()
    reply, actions, cid = chat(body.message, body.image, body.conv_id)
    print(f"[chat] total {time.time()-t0:.1f}s")
    note_activity()
    return {"reply": reply, "actions": actions, "conv_id": cid}

# ---------------- NATURAL VOICE (edge-tts, unofficial Microsoft neural voices) ----------------
VOICES = {
    "ml": {"female": "ml-IN-SobhanaNeural", "male": "ml-IN-MidhunNeural"},
    "en": {"female": "en-IN-NeerjaNeural", "male": "en-IN-PrabhatNeural"},
}
STYLES = {  # style: (gender, rate %, pitch Hz)
    "sweet": ("female", -8, 3),
    "female": ("female", 0, 0),
    "male": ("male", 0, 0),
    "child": ("female", 5, 40),
}

def tts_params(style, speed=1.0):
    gender, rate, pitch = STYLES.get(style, STYLES["sweet"])
    rate = max(-50, min(60, rate + round((float(speed) - 1) * 100)))
    return gender, f"{rate:+d}%", f"{pitch:+d}Hz"

# Optional: ElevenLabs voices (set these in Render > Environment to switch on)
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_v3")
ELEVEN_VOICE = {"female": os.environ.get("ELEVEN_VOICE_FEMALE", ""), "male": os.environ.get("ELEVEN_VOICE_MALE", "")}

def _eleven_tts(text, voice_id):
    import urllib.request
    payload = json.dumps({"text": text, "model_id": ELEVEN_MODEL}).encode()
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
        data=payload, method="POST",
        headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

class TTSIn(BaseModel):
    text: str
    style: str = "sweet"
    speed: float = 1.0

@app.post("/tts")
async def tts_api(body: TTSIn, x_pin: str = Header(default="")):
    if not pin_ok(x_pin):
        raise HTTPException(401, "Wrong PIN")
    text = re.sub(r"[*_#`~>|]", "", body.text)[:1500].strip()
    if not text:
        raise HTTPException(400, "empty")
    lang = "ml" if re.search(r"[\u0D00-\u0D7F]", text) else "en"
    gender, rate, pitch = tts_params(body.style, body.speed)
    vid = ELEVEN_VOICE.get(gender, "")
    if ELEVEN_KEY and vid:  # best quality first: ElevenLabs (Eleven v3); fall back to edge-tts on any problem
        t1 = time.time()
        try:
            import asyncio
            audio = await asyncio.to_thread(_eleven_tts, text, vid)
            if audio:
                print(f"[tts] eleven ok in {time.time()-t1:.1f}s, {len(text)} chars")
                return Response(content=audio, media_type="audio/mpeg")
        except Exception as e:
            print(f"[tts] eleven failed, using edge-tts: {e}")
    if edge_tts is None:
        raise HTTPException(503, "no tts engine available")
    t0 = time.time()
    try:
        comm = edge_tts.Communicate(text, VOICES[lang][gender], rate=rate, pitch=pitch)
        audio = b""
        async for ch in comm.stream():
            if ch["type"] == "audio":
                audio += ch["data"]
    except Exception as e:
        print(f"[tts] {e}")
        raise HTTPException(502, "tts failed")
    if not audio:
        raise HTTPException(502, "no audio")
    print(f"[tts] ok in {time.time()-t0:.1f}s, {len(text)} chars")
    return Response(content=audio, media_type="audio/mpeg")

# ---------------- BRAND: logo, app icons, manifest ----------------
LOGO_PNG = base64.b64decode("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")
ICON_512 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AABBwklEQVR42u3deXxV1b3//8+ZMjOHKRAIgYQhgESmMAoC4oTVq1i9KtYBqmJVtNdK269Xb9HKbb22da6oBVsVUFFQS9GCtUwyhKkBgRBICAgkBJIQkjOu3x917d8+JycDyT4hgdfz8dgPBZJ9zh7Xe6+9BpuIKAEAABcVO7sAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAEAAYBcAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAAIAAAAAACAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAAgA7AIAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAAAEAAAAQAAAAIAAAAAACAAAAIAAAAIALiJNdALTQ9G63SyAQkKuuukrGjRsnXq/X+Duv1ytut1vcbrd4PB7xer3i8/nE6/WK1+s1/k4pJSIS9G82m00KCwvl0KFDYrPZjJ8BcGGxiQhXN9ASL16bTWw2m2zZskUyMzMtXfevf/1r+fnPfy5Op1N8Ph87G6AGAEBzevpPSkqSPn36iM/nE6WU2Gy2oIBQU3Co6e8DgYA4HA7ZsGGDiAhP/wABAEBzDAAjRoyQVq1aid/vF6ezcZezUkocDoecOXNGNm/eLCIigUCAnQ1cqPcRdgHQ8ugn88svv9yyJ3Vd2P/rX/+S48ePUwMAEAAANCc2m8144h87dqzxd1aFik2bNhm1AQAIAACaUQAQEenZs6f079//3xey3W7ZevX7fwAEAADN6aL9vrAfMWKEREdHi9/vb3QNgH7i93g88s033wTVCAAgAABoBnTBfNlll1m+zt27d0t+fr6I0AAQIAAAaDb0+3+Xy2Xp+39d2G/YsMHoCgiAAACgGQUAEZG0tDTp27dvtb7/jV3vxo0b2ckAAQBAs7tgv3//P27cOHE6nZa+//d6vQwABBAAADRHuqp+0qRJQU/ujQ0AIiL79u2TvLw8AgBAAADQnOihelu3bi2jR4+2LADoUJGdnS1+v18cDgcBACAAAGg2F+v31f+DBg2Sbt26iVLKkv7/2qZNmywLFQAIAAAsrAEQERkzZoyIiPj9/kavU7//V0rJ+vXrg2oEABAAADQDusDX/f+tev9vs9nk8OHDsmfPHuPvABAAADSTp3+llHTu3Nl4/29F9b8u7Ddu3CiVlZVit9sJAMBFgumAL/JCRT9Fhv63tsKipv8igkndbhe/3y/Dhw+Xtm3bSiAQsPT9v67+5/0/QADABVjQm2/ufr9flFKWFN42m80ojPSTaiAQIBhYfAxF/l39r/evFQFAr4MJgAACAC6gAkPf3HVhHyomJkYSExMlISFB2rdvLx06dJCEhISww8D6fD4pKyuTU6dOSWlpqbjdbjl9+rScOnVKAoFA2AZp5u+gwwahoGH8fr/Y7XYZO3Zs0H5tDP3+/8iRI5KTkyMitTcArKl2gGMKEADQTAp9/YSoC2WXyyV9+vSRwYMHy+DBg6VXr17Sp08f6dChg3Tq1Eni4+PPqepXKSU+n09OnTolJ0+elOLiYsnLy5OCggLJzc2VvXv3Sl5enhQVFVULBjpcnEsNQeh3u9gKHLvdLoFAQJKTk2Xw4MG1FsbnQo/5v3XrVqmoqDA+R3+mrjXSwa2mcKDPO/PPEvYAAgCasOD3+/1GgdupUyeZOHGiTJ06VUaPHi29e/cWp9NZZ4FQ203bfKN3uVzSqVMn6dSpk4j8e2has+LiYtm3b59kZ2fLxo0bZfPmzZKbmxsUCOobBsL9m7kArE/7hfpuY0OF1prUpy1FbU/l5gJXr2vo0KESFxdnWfW/uQGgPqY+n0/8fn+Nhb3D4TDOgUAgYCzhaoB090K6FQIEAES44I+JiZGrrrpKbrvtNpkwYYJ06NAh6Of1q4CaGv/Vt1AxF2Dmpz29LrvdLomJiZKYmCijR4+WBx98UDwej+zdu1f+8Y9/yKeffirr16+X8vLyoG2pqWB0uVxGQaILk9DvcL5Z0Se/LpdffrkRYqwIADq0bNiwQZRS4na7jX9LTk6WgQMHyoABA6Rfv37SuXNnSU5OloSEhKCagUAgICdOnJCCggLZv3+/5OTkyLZt22T//v3GPtGBoSn2EYBzKENEhHq6FsjhcBg31Pbt28uMGTNk5syZMmDAgGqFUrhGgJFmbghos9mqPSHn5+fLZ599JkuWLJF//vOf1Z4S9fbNnTtXZs6cKV6vV3w+n7jdbnG73eLxeMTn84nP5xOv1ytut1u8Xm9QzwSPxyNVVVVSVVVlfJfnn39ejh49WmvgOJcAppSS6OhoeeKJJ6RTp05G+4iqqirjO+pt8/l84vF4xOv1isfjCdoWvfh8PnE4HFJSUiIHDhwwPsPhcMimTZvk0ksvtSQA6HUcPXpUevToIbGxsTJq1Ci54oorZPz48dK/f39p1apVg9btdrtl165d8sknn8iHH35ojC+gX0/xagBoPhRLy1lsNptyOBxKRFRsbKx67LHH1OHDh5Xm9/uVz+dTgUBANSeBQED5/X7l9XqV3+8P+rfs7Gw1duxYJSLKbrcrm81mLDt27LD0e6SlpRmf09hjodcxbNgwy/fX6tWrlYgop9OpRET17t1bVVVVGfvSiuOhlFL79+9Xr732msrPz6/2M/pc8nq9yufzKb/fH3bx+XzGz4Ue26qqKrVs2TI1ceJEY7/p85fl/Nw/bDabstvtyuFwVFucTqdyOp1h/y302mR/tvyFVwAtiG6k5ff7ZfLkyfL8888bjcL0tLBW9g23+nWF+fvp6nyHwyGZmZni8/mCfjYQCEhKSoqkp6cH1STUVuVf09/rz9m2bZscOHDAstcG+nhkZmaK3+8Xn89XZzsL8zaG+3+9Dv3U7HA4xOfzyahRoyQ6OtqYrMeK4yEi0qdPH+nTp4+xT3Rvg8acS+bXQtHR0XL99dfL9ddfL2+//bb87Gc/k6KioqAarOZwbta1PS2yejek5s/cHsOq8z+0oSg1PC0LAaCFFf5RUVHy3HPPyZw5c4yC3263W1IoNPX26EL9o48+ko0bNxrb6HQ6JRAIyMiRIyUmJsYowBtKF5rZ2dnGuqwaR19EZPTo0UaDt8YeB31TXbt2bdBnTJ48OSKFkS4UdIFf3wBTn4LHvH673S533XWXXH755XLnnXfKP/7xjyYPAeYC8VzGqzCH1+Ze0NntduM6Crdtdrvd6PbbuXNniY2Nlbi4OGOJiYkRm80mPp9PKioqpKysTM6ePSsej0dOnTolxcXFUlFRIZWVlXX2CgkNgyAAoAH0jTIpKUn+/Oc/y8SJE42Lu6UV/KE3CqWUvPLKK0FPYvpmMWHCBOMJ3oqaDV2oWtEWQjdqs9vtkpmZacl69fH0+Xyya9cuERHxer0SGxtrTABkdQ1PuPYZkVi/7jras2dPWbVqlcycOVMWLVoU8RBg7rni8/nCFkRRUVESExMjDodDXC6XUYDqNhpVVVVhezM0pEtrJLfR3CtDRCQuLk769u0rQ4YMkf79+0vfvn0lNTVVunTpYhT458Lr9RqF//Hjx+X48eNy6NAho+vvoUOH5NChQ1JeXm5J9180wbkjNAJsEU/+6enp8umnn0paWprRUKwlD9uqn8o3bdoko0aNMm4OOhRERUXJrl27jFcAjS34vF6vZGRkyP79+4P6uzf2uPTu3Vt27twpcXFxQa8pGkJvZ15engwcONAoeIYNG3bBTNWrj7HNZpMbbrhBPv7444iEgJp6HqSkpEhmZqbRuyE5OVm6desm7dq1C+riqF+HlJWVyZEjR+TYsWOye/du2bVrl2RnZ8uBAweCCjKHw9HkhZu5F5CWnp4uV1xxhUyaNEmGDh0q3bt3r3UAp7p605h7CdV17gUCAaP777Zt22TTpk2yZcsWOXDggHi93mrfmy6izQONIZrpohuZpaSkqLy8PKWUUl6vV10IfD6fUkqpu+66K6ixm97mwYMHW9LYTTdK27lzp/EZVjRg0g3ZbrzxxqDPaQx9bJctW6ZEREVFRSkRUQ899NAFd+wDgYA6efKk6tOnj2WNMvV6zI0MHQ6HysrKUvPmzVNbt25VlZWVjf7+brdbbd++Xc2fP19lZWUFfXeHwxHxBnLmhsAiojp06KDuvfdetXr1auV2u8Pub3NDzkAg0KBrS/+ebtCrG3/q9Ybj8XjUnj171IIFC9Qtt9yiunfvXm1brDr2LA1a2AnSTFvr2u12FR8fr7Zu3RpUaLZ0+iaUm5ur4uPjg1oV60J61qxZlmyz/v0FCxZY2gJdf8/58+dbVjjrdfzsZz8LCgCLFy++oI6/eVvWrFljtDC3slDs2bOneuKJJ9TOnTvDnn+6QAwtFEOXuno5KKXUjh071COPPKI6dOgQ8Z4O5v3UvXt39dxzz6mjR49W27fm7WrKnj6BQCAocIQqKytTq1atUj/5yU+MHjksBACWGp4w33nnnQvq6c988//pT39a7Wapb3DvvvuuJdutP+vuu+8OKritCGgiolavXm1Z4axv1pdffrnxGXFxcergwYOW1TI0J/rY6lqghhaa5t/LyMhQCxYsUKdPnw7ar7pAsqoLpQ4R5vUdOXJEPf7446pNmzZBXe6svifExMSoxx9/XBUVFVUr9Jtz99/Qa+Ts2bNq5cqV6rLLLuOeTwBgCb3Qb731VuNG2dwu7MY+/Z84cUJ17Ngx6Cap/xsfH68KCwsbXejpJxKv16v69+9vWVWz/p4dOnRQx48ft6Rw1r9fXFysOnXqZHzWyJEjg7blQqILrP3791erCarv07D++e7du6uXXnpJnT17NihgRPop2PzEq+Xl5akf/ehHltUGmKvJx4wZo7Kzs5t0GyNVQ6C/86xZs7jvEwBYzE8Nbdq0Ufn5+UaCjuQFWdMAL+aqUKuf+l544YVqN0f9/6NHj7ZkwBv9xJGTk6NcLpdl7//NN2Or9pH+rl9//bUSEeP7zpkz54KrAQq33TfddNM5FZbmn3vggQfUiRMngs6x81Eg6idd7YMPPlBdu3ZtVAgwv9q4//77jXf8Lf2hQN9fjh07ZtSYsJyHdma0gWx+rf6VUnLfffdJjx49LOsCZ27pa540SLfIDbeE9pnWv1fXhDG1fbbD4ZCqqip59dVXqw3Hq1sZjx071ugp0NhtFRFZv369eL1eozuaFcdI5N+T8+iW5lZ0ART598Q85n2hJ1lq6a3/69r26dOn1/vnnU6n+P1+SUlJkRUrVsjLL78sHTt2NLr5OZ3O87K/9DgKerCuG2+8UdavXy+XXXZZgwdw0q38n332WXnllVfE5XKJ3+8/b9to5TG32Wzy8ccfS2lpaYvuztySMQ5AM6ILk4SEBJk9e3aju5WF66YTOmiQHuCjoqLCKMicTqdERUVJXFycxMfHi8vlqrW/uJ5kyBwaavp8h8MhS5culX379lXr/qX/X09609ht17+/fv16SwtR8wBAVhYeImJ09/N6vdK6dWsZMWLEBR0A9PmSlZUlrVq1kvLy8hrnadBh1efzydSpU+Wtt96SpKQkYzwGKwYxsvJY+nw+SUlJkS+++EJmzJgh77//vjidzqBRL2ujr49nnnlG5s6da3T/vRAKS92N9q233uLGTwCAOe1PmzZNkpOTLRv2VRdadrtd3G63rFy5UlavXi3btm2T7777TkpKSqSiokJ8Pp8REKKjoyUuLk4SEhIkJiZGkpKSpGvXrtKjRw9JTU2V3r17S2pqqiQlJVX7jqGzwJk/PxAIyIsvvhj26V8pJR07dpSsrKygG2ljttfn88k333xjBBCrQlp0dLRceumljf6e5u9aVVUlO3fuNP5u0KBB0q1bN8uDYGiQCdcf3DyzYyTDhz7u3bt3l7S0NMnOzq4xAOjr47777pOXX37ZOL7NpeCvqabC5XLJe++9J7GxsfL222/Xa9wD/TOzZs2Sn//85xfE2B/m+4PD4ZCNGzfKpk2bmCnyfNfGsDSvfv9LliwxGq9Z+Z51y5YtavDgwZZ93zZt2qghQ4aoO++8U7322mtqx44d1b6zuQuVUkp9+eWXYRvj6fecV155ZdA7wsY2qtu9e7fRnc7K9/+DBg1SHo/HkrYKod9Vf8bjjz/eqPf/5vYJ+hjo5VzalYRr8R6JdgA/+MEPanxfrntvPPHEE0Gty1vK+279XW+55ZY62wTo45+RkaEqKiqCfv9C6v1x3333MTkUkwFBPwkFAgGJj4+X0aNHWzaxj3669Hg8ctddd8muXbvE5XLVa0zz0AlrzE9mgUBASktLZfv27bJ9+3ZZuHCh2O126d+/v0yYMEGuvvpqGT9+vCQkJATVCrz44othq7RD3//rOQEa+7rjm2++EY/HY9loc7oWY/jw4cb72MbW0ujvun37dvF4POJyuSQQCMjEiRPrXf1vPi7mp3j93cJ9R6WUnDlzRo4ePSrl5eVSXFwsZWVlxnnYpUsX6dWrlyQmJgaN7W71U6j+7j179gy7vbra/Be/+IXMmzfPqPJvrhNf1XTOBAIBWbhwoRw+fFjWrVtX64iUdrtdXnzxRYmLi7O0JvB809d1UVGRfPDBB5bVzIFXAC0+ACilpG/fvpKUlFTvG/+5XHhVVVVBVb51BQBVz0lS9Pf0+/2Sk5MjOTk58vLLL0tKSor84Ac/kJtvvllGjx4te/fulZUrV4at8tN/tqrRm/79devWWbov9T7R31NZOPSrflXh9XolMTFRhg0bFvYVQ+ixq6ugLy0tlUOHDsnBgwclLy9PDhw4IIWFhZKXlyclJSVSVFQUNFSrmf4eM2bMkFtvvdU4TyNRFR0TExO2Gt3n88mPf/zjoMK/pVWF69cXUVFR8uc//1mGDx8uJ0+erBYCdFC96aabZOLEiRdU4W8Oux9++KEUFxc3q1kheQXAct77/t98882WD/qiq1j/+c9/qoyMjLBVjnoOcHPfamlgf2Wn01mtin/ChAkqKysrbPW//nOXLl1UWVlZo6vV9e/6/X7jlYeV/f8dDofasWOHZcdJV9OPHj3a+KwpU6YYx06PRldX1f3p06dVdna2eu+999STTz6pfvjDH6phw4apzp071/vYmZfQn7nuuuvUmTNnItY19PHHHw+q7tfXxLXXXhvUNfVCqP5+//33w1Z/6+OwYcMG49VNpPvlN1UXYPNnjhw5strojSyMA3Dxvov5/qZ3++23WzayXLhCsbKyUn3yySdq1qxZKiMjw3g/LjWMqd6YUBA6Lntd4ef666+3ZNt1IZmbm6tiYmIsf//fq1cvdebMGUvf/x85ckS1bdvW+Kx58+ZVG2DGXIgcOnRIffHFF+qFF15Qd911lxo1alTQAEK1HQ9z2Kvr2OqbtB6XYObMmZafn+ECgN7XvXv3VidPnoz4O/+mGmjJ3LbnhhtuCDr/9X9HjBjRqDH7I9GewxxCGzqngD5nvvnmm0Y9aLDQBgANbGMQExMj1113nVx33XXi8/mkoKBAdu7cKbt27ZLdu3dLTk6OHDhwQM6ePRt2Hbo6uj4zn4VWbao6Zv/S3f8aW62uqxk3b94sVVVVlsz+Z36NMHjwYImPj7dkjAb9vXJycuT06dNG9fb1118vDodDPB6PHDx4UHJycmTnzp2yZcsWycnJkcLCwrDHSO9rXVVvfsXTkH2gx3/QXTjffPNNeeihh2TgwIGWV08XFxcH7WuHwyFvvPGGtG/f3vLW/vpc1K8z9KL/rj7dWht6Dulj85vf/Ea++OILqaioCPqc6667LiI9HPT5qs/ZyspKKS8vl5MnT0plZaXxfj4uLk5atWol8fHxxp9rOs/V99M869cc9dlf77zzjnE+Uf1PGwCYCr369hFu6HtIcyHsdDolNTVVUlNT5frrrzduEt99953s2bNHdu3aJTt37pScnBzZt2+flJaWBl2w+iZtvmnWpLYLXRckugGgVY271q5da6zPygCg+/83NACYj4EOVdnZ2dUKvpKSEtm6dascPHhQKisrwx5Pc4Fi7tYX+p0bu/26oPT7/bJmzRoZOHCgZe0f9D48dOiQ8WePxyOPPvqoTJw40dKCUJ+rNfWnDy28dACyMujogq93795y9913yx/+8Aejy6DdbpfJkyeH/S6NPX76PFu6dKl88803sn//fikrK5PKykqjDYjNZhOXyyXR0dHicrkkLi7OmC65R48ekpKSIunp6UY34NatW1c7Nj6fzzivzd2AHQ6HnDp1SpYuXWrJOQkCwAUXAHJzc42LJRKNrcwNxmpqTNatWzfp1q2bcSMSESkqKpLdu3fLpk2bZN26dbJ582Y5evRoUGBpyJzounBOSUmRjIwMS258+nts2LDBkhqF0BAzZsyYc/6eutDXN0Z9DLxer2zfvl3effdd4+c8Ho/8/ve/r7afzHPVhyvUVBPNRW9+UreqYPJ6vXLixAljn/Tq1Uv++7//23hStOqJX6/r7Nmz8vXXX8vatWtl165dRriNioqS7t27y4gRI2TixIkyYMCAoBEkrboe9XF88MEH5Y9//KO43W5RSkliYqKkp6dbGoR1UH3xxRdlzpw5tYZxff55PB4RESkpKZHCwsKw379jx46SlpYmmZmZMmbMGBkxYoSkpqYGBQL9WXqExmXLlsnx48d5+m9OZQ9L85gDQERUXFycKigosLwh4LnO3lXblJ5KKVVaWqrWrFmj5s6dq4YMGRL0Pu9cpnfVbR/0xEdWvf8/ePCgio2Ntez9v15Hly5d1MmTJ+v1/l834go9jqdOnVIrV65UDz/8sDFJkYQ0AouKilLR0dEqKioqbKPKmvZl+/btVY8ePVRWVpa68sor1fTp042GkI3dD/pYPfXUU5bNT6D3zeHDh4PaQFg5C6Z5/+fm5qr/+q//Ur169arX9k6ZMkV9+umnYddl1ffSbQFERGVmZhrXgFUzF+pzrl27dsa5ZW7bEzohl3kxtwXS7UdqOhejo6PV8OHD1RNPPKG++uqroImZ9H1l7NixNP6jESCL1NIYbsGCBc1qApjQUBDuJrh582Y1Z84clZSUFLQ9dRU6eptfeeUVS7ZZ3zyXLl1qWet/8/ecPHlyrTdnXeiHBpkjR46od955R918883GBDHmdTudzjr3lcvlUl26dFFDhw5V06ZNU/fdd5967rnn1F/+8hf19ddfq5ycHHXy5ElVWVkZ9Nk333yzJQOu6N9/+eWXLTs/zT1U9PZnZWVZNp+9Xr/H41H/7//9P9WqVauwvV/CLeZtv+GGG9TRo0ctbQCpG+MtW7bM+JyrrrrK0qChG+q53W41ZcqUaudcQxvj6YBgblQa+jOpqalqzpw5at26dUoppXbu3FmvewILAeCiDgCjRo1q9iOd1VTQnThxQv36179WPXr0qNZ6vqanaqfTqXbt2mXJjU8XSo8++mjQU6tY1EvjZz/7WbXCr6Z9UVRUpN5//3118803q8TExHp9TlxcnEpOTlZjx45Vt956q3ryySfVG2+8oVavXq327NmjysvL613w+f1+tXv3buVyuSyZm14fxzVr1lhWEOp1LFiwwPicjz/+2NIwmJ+fr8aPHx90LOsTDHUBp6/L3r17W979U9em6evlmmuusbymQX9OcXGxevDBB8Oei1Z0BTZ3Aw79/aysLKMbMAGAAMBSx0128eLFzaoWoL41BNrJkyfV3Llzq/XplhqG1dVzwzf2iU//vr7ZWFXVqNfz0UcfGccldJv1dutC31ylbV5iYmJUz5491WWXXaZuu+029dRTT6lFixapdevWqby8vGpP8PXplhXaNUufN3PnzrUkCOmbdnx8vKWvqPT3fOSRR5TNZlOZmZnK7XZb1r1yz549KiUlxdgHDS189P5LSkpSubm5lm2/Pn/uuOMOJSJq7NixEQ/+x44dU++9956677771PDhw1Xr1q0j0hW4pjDAQgBgqeWisdlsKjk5WRUVFUVkTICmqBnQvvrqK9WzZ8+whbG+qd5///2WhB194ywsLFQJCQmWv/+PiYlRe/furfZev6KiQq1atUrNmjVLdenSxfi9Vq1aqUsuuUTddNNNas6cOerVV19Vf//739W+ffvqLOTNtQqhfa/rG4IqKiqMfd/YVyH69wcMGGDZHAjmdejq6f/7v/9r9Llgbgei3/VbUROk1zF8+HBVWVlpySsK/RrglVdeUSKievTooUpLSy3bv7Vdm1pBQYFasWKF+p//+R81ffp0NXDgQOP6CReEG/LqoKbBpVgIACw13GyvvPJK44JtSSEgdMCT/Pz8sCPy6f9///33La3yXb58uaXv/801Febt27Rpk5ozZ45KTU2tdpNMSUlRhw4dqtcNWU/Q09ABVmp6qv7oo48s2w86vP3Hf/xHRKrAu3TpomJiYtTBgwcbtX5dG+V2u41aIKteA5nX9cwzz1hyXerf15NkOZ1OtXPnzohe83owoJrWHwgE1KFDh9TKlSvVb3/7W3Xrrbeqfv36hR00TNcQ8JRPAGCJQHuAW265xXjiiuSMbJGiC6MjR46otLS0oFoOEVEJCQkqPz/f0vf/8+bNs/TGby788vLy1P/+7/+qESNG1FjdqQvcO+64wwhCHo/H0kK+rgJQKaWuueYay1pcR7IHwK5du5SIqCuuuKLR54Eu1H77299aXvibz91OnTqpkpKSRh9Hva379+83nrp/97vfNdnrv/r2+vF6vWrv3r3qL3/5i3rwwQfVsGHDVFxcHGGAAMAS6RAwceJElZeXF/TU2JKCgL6Rbdu2TcXGxhqFpYiokSNHWlbdqW9es2fPjsjNv1WrVmFveuGesPXfmYfObYpjZp5aWA/fa8UNWZ+Ly5cvt7wBoK6paGzBp7f96NGjqn379hGrdtbrfO+99xpdUOtzory83GgIOGLEiPN6jdc3FOTl5amFCxeq6dOnV2tYSBBoIbXNDIPQfOkRyNasWSOjRo2SN9980xgkSI/K1hJG09Izug0ZMkQefPDBoAFZ9Oh/Vg4KcvLkSUsHANLKy8vl7Nmz4nA4jEF5ajoGeljVN954Qx577LGgwWQiOqjH95+xaNEi8Xq9lnyu3s6oqCjp16+fZQPi6O+1Y8eOoHOhoQPg6OPw9ttvS0lJiTG0r9X06IurVq1q9L7QvxsfHy/t27cXEZFNmzbJl19+GXbWzKZgHqzK6XQGDePt9/uN79SrVy+ZMWOGLFmyRHJycuRPf/qTTJo0yfje+l4FBgJiseDpS0TU6NGj1YcffhjUiExXMzfn2dL0d9OT3uinAz3IipVPlDNmzLC0B0Bo3+dz+Xn9FP7MM880yYQ2Sil15swZyxr/ScgkSHpwFytrbK6++mqVkJDQqJkgdVW8z+dTQ4cODXoVE6kagMzMTON4WvEaYMyYMcZn6AmBmnNtn/5+oef0xo0b1fTp041rhYl/eAXAYkHhY76hDRgwQD3zzDNqz549YW+s59pyvCmEFtBt27ZVhYWFlncpe+KJJ5TNZrP8FcC5FhLmAGK329XWrVst7+Mdbvv1IEhWd4G89tprLW8A6PF4VFJSkho+fHhQQd7QQnTfvn2WzgApdYwKaUUg0t993LhxxoBPIqKeffZZYx81d+HCwMaNG4MGH6IXAK8A0IjqUvNsXrt375Zf/OIXcskll8i4ceNk/vz5sn37dqPaV1fdmauqfT6fUWWtmnDsePM2KKVk2rRpIiLSu3dv6dq1q2VzHuh1DBo0qM6ZByNVdapfz+jq0i5dusijjz4q2dnZxgQ6Vo3xHrpv9Xrffvtty7dL71dzVbsV1f+FhYVy9OhRGTBggLHuhpwL+jvt3bvXmAEy0ud3pM4xXXX+5JNPyooVK8TlchmT9TRX+pWB3W437jEjR46UVatWyUsvvSTR0dGWzJ4Ji1/PsgtaFn3D0QW7x+ORtWvXytq1a2Xu3LmSlpYmw4YNk5EjR8oll1wiaWlp0rVr17Dv4tT3M53pAtg8e1ekbhI2m0369+8vIiJJSUnGDcOKG4Nex4QJE6R169ZSXl4edvKcSGyXnr5Vvx8dNmyY3H333TJ9+nRJTExskvPC4XBITk6OfPHFF5a+/9b7b+DAgZa//9+zZ4+IiKSlpQVtR0MVFBRY9h1rC0RKKYmPjxeXy2XZes0T5+hjd8stt8iHH34oV155pfh8PiNgNmf6+Pn9frHZbDJ79mwZPHiw3HLLLXL06FHLZucEAeCCpAvK2mbWC51OVl9w+/btk3379hmzy7Vq1Up69eolGRkZMmjQIMnIyJC0tDTp0aOHxMfHV7vZmmsaIjEToS74bTabJCUlVfu3xq7f7/dLUlKS/Od//qe89tprRgPESBQC+kaml9jYWLn22mtl5syZMnnyZGOb9I2wKZ5+FixYYNQCWdGATJ+Hdrs9IgFANwCMj4+3ZPvN09pGMgCIiKSnp0tUVFSjnmyVaTZEc+NV/fdnz56V66+/Xt544w254447RERaXBDwer0ybtw4+eKLL+SKK66QI0eOEAIIAKjthmu+iGqratRP8fp39aJDQnl5uezcuVN27twp77333r8PutNpTOXZv39/GTBggAwZMkQyMjKkQ4cOxoUbqSq7qKgoiY2NlQ4dOgQVBlbuv6efflpWrlwphw4dsiwEhIYtvd/79+8vt99+u0yfPt14ktU/Y576N1J0lXFpaaksWbIkKCBatf6kpCTp1auXZYWr3o/Z2dkiIhIbG2vJunv06GH59tckKyvLsuukpKTEmA5ZXw96vW63W2bMmCHr16+Xp59+Wjp16tSigoB+hTFgwABZvny5XHbZZXL27NkmqZ1DPa5vluY1AuBjjz2mHn300aAxuvWALufakCZ0Ss/afr99+/ZqypQp6sUXX1THjh2zfDhS8xC1CQkJ6sknn4zIgCe6IdLWrVtV+/btz2kCGKlhcpPQ3+3YsaO688471apVq5Tb7a7WGKopG17q/bdo0SLLez/odU2aNKlRjfTCnQeVlZWqX79+SkTUa6+91qhzQTcw3b59e8THn9e9QXSjzsb0YNG/u2XLFmNfh353cwPgbt26qddee01VVVUFraMljA2iGzO++OKLNAqkFwCLhBlhLCUlxbhQ8vPz1XPPPacyMjKq3RCsnsozdD1dunRRK1assLTVunlucpvNpn75y19GbMQzfWPdsWNHtVH7apr+1byEuzl17NhR3XjjjWrhwoXqxIkT1Qrh8zV7o96vesY7KwOA7knxyCOPWNZd0zz6nZ6e9/e//32jzgVzqEhLS4tYN0B9rYwbN86Sbp16fy5evLjOQtF8XDMyMtRLL71kBPVwrfGbWyAwz2+RmZlJCCAAsIRe3K+88opSSgU9VbrdbrVy5Up17733quTk5IhM5WmuZdBjficnJ1va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ACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAAAIAOwCAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAAAQAAABAAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAAIAAAAgAAAAAABgF0AAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAEAAAAAABAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAACAAAAAAAgAAAAQAAABAAAAAABeu/w8lVJe7ZKBLNAAAAABJRU5ErkJggg==")
ICON_192 = base64.b64decode("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")
_CACHE = {"Cache-Control": "public, max-age=86400"}

@app.get("/logo.png")
def logo_png():
    return Response(content=LOGO_PNG, media_type="image/png", headers=_CACHE)

@app.get("/icon-512.png")
def icon_512():
    return Response(content=ICON_512, media_type="image/png", headers=_CACHE)

@app.get("/icon-192.png")
def icon_192():
    return Response(content=ICON_192, media_type="image/png", headers=_CACHE)

@app.get("/manifest.json")
def manifest():
    return {
        "name": "SUHAS", "short_name": "SUHAS", "start_url": "/", "display": "standalone",
        "background_color": "#000000", "theme_color": "#070b1e",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }

def _auth(pin):
    if not pin_ok(pin):
        raise HTTPException(401, "Wrong PIN")

class RenameIn(BaseModel):
    title: str

@app.get("/chats")
def list_chats(q: str = "", x_pin: str = Header(default="")):
    _auth(x_pin)
    with conn() as c:
        if q.strip():
            like = f"%{q.strip()}%"
            rows = c.execute("SELECT id, title, updated FROM conversations WHERE title LIKE ? OR id IN "
                             "(SELECT conv_id FROM messages WHERE content LIKE ?) ORDER BY updated DESC LIMIT 100", (like, like)).fetchall()
        else:
            rows = c.execute("SELECT id, title, updated FROM conversations ORDER BY updated DESC LIMIT 100").fetchall()
    return {"chats": [dict(r) for r in rows]}

@app.get("/chats/{cid}")
def get_chat(cid: int, x_pin: str = Header(default="")):
    _auth(x_pin)
    with conn() as c:
        rows = c.execute("SELECT role, content, created FROM messages WHERE conv_id=? ORDER BY id", (cid,)).fetchall()
    return {"messages": [dict(r) for r in rows]}

@app.post("/chats/{cid}/rename")
def rename_chat(cid: int, body: RenameIn, x_pin: str = Header(default="")):
    _auth(x_pin)
    with conn() as c:
        c.execute("UPDATE conversations SET title=? WHERE id=?", (body.title.strip()[:80] or "Chat", cid))
    return {"ok": True}

@app.delete("/chats/{cid}")
def delete_chat(cid: int, x_pin: str = Header(default="")):
    _auth(x_pin)
    with conn() as c:
        c.execute("DELETE FROM messages WHERE conv_id=?", (cid,))
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))
    return {"ok": True}

@app.get("/history")
def history(before: int = 0, limit: int = 30, q: str = "", x_pin: str = Header(default="")):
    if not pin_ok(x_pin):
        raise HTTPException(401, "Wrong PIN")
    limit = max(1, min(limit, 100))
    where, args = [], []
    if before:
        where.append("id < ?"); args.append(before)
    if q.strip():
        where.append("content LIKE ?"); args.append(f"%{q.strip()}%")
    sql = "SELECT id, role, content, created FROM messages"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    with conn() as c:
        rows = [dict(r) for r in c.execute(sql, args + [limit])]
    return {"messages": rows, "more": len(rows) == limit}

# ---------------- ADMIN PANEL + USAGE PING ----------------
# Central server (yours): set ADMIN_KEY (password for /admin) and HB_TOKEN (accepts heartbeats).
# Customer copies: set ADMIN_URL (your central URL) and the same HB_TOKEN. Only name, last-active time
# and message COUNT are sent. Chat content is never sent.
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
HB_TOKEN = os.environ.get("HB_TOKEN", "")
ADMIN_URL = os.environ.get("ADMIN_URL", "").rstrip("/")
INSTANCE_ID = os.environ.get("INSTANCE_ID") or os.environ.get("RENDER_EXTERNAL_URL") or USER_NAME
_last_fwd = [0.0]
PING_NOTE = ('<div class="card"><div class="ct">Usage</div><div class="sub" style="font-size:14px;line-height:1.5">'
             'Usage ping is ON: your name, last-active time and number of messages are shared with the developer '
             'so they can see who is online. Your chat content is never shared.</div></div>') if (ADMIN_URL and HB_TOKEN) else ""

with conn() as c:
    c.execute("CREATE TABLE IF NOT EXISTS instances(id TEXT PRIMARY KEY, name TEXT, url TEXT, first_seen TEXT, last_seen TEXT, msgs INTEGER, ver TEXT)")

def _send_hb():
    import urllib.request
    with conn() as c:
        n = c.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0]
    body = json.dumps({"id": INSTANCE_ID, "name": USER_NAME, "url": os.environ.get("RENDER_EXTERNAL_URL", ""), "msgs": n, "ver": "1"}).encode()
    req = urllib.request.Request(ADMIN_URL + "/hb", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-hb-token": HB_TOKEN})
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        print(f"[hb] {e}")

def note_activity():
    if ADMIN_URL and HB_TOKEN and time.time() - _last_fwd[0] > 45:
        _last_fwd[0] = time.time()
        threading.Thread(target=_send_hb, daemon=True).start()

@app.post("/ping")
def ping(x_pin: str = Header(default="")):
    if not pin_ok(x_pin):
        raise HTTPException(401, "Wrong PIN")
    note_activity()
    return {"ok": True}

class HB(BaseModel):
    id: str
    name: str = ""
    url: str = ""
    msgs: int = 0
    ver: str = ""

@app.post("/hb")
def hb(body: HB, x_hb_token: str = Header(default="")):
    if not HB_TOKEN or x_hb_token != HB_TOKEN:
        raise HTTPException(401, "bad token")
    now = now_local().strftime("%Y-%m-%d %H:%M:%S")
    with conn() as c:
        c.execute("INSERT INTO instances(id,name,url,first_seen,last_seen,msgs,ver) VALUES(?,?,?,?,?,?,?) "
                  "ON CONFLICT(id) DO UPDATE SET name=excluded.name,url=excluded.url,last_seen=excluded.last_seen,msgs=excluded.msgs,ver=excluded.ver",
                  (body.id[:200], body.name[:80], body.url[:200], now, now, body.msgs, body.ver[:20]))
    return {"ok": True}

@app.get("/admin/data")
def admin_data(x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "Wrong key")
    with conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM instances ORDER BY last_seen DESC")]
    now = now_local().replace(tzinfo=None)
    online = 0
    for r in rows:
        try:
            age = (now - datetime.strptime(r["last_seen"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        except Exception:
            age = 10**9
        r["age"] = int(age)
        r["online"] = age < 180
        online += 1 if r["online"] else 0
    return {"online": online, "total": len(rows), "rows": rows, "demo": {"on": DEMO_MODE, "remaining": demo_remaining(), "minutes": DEMO_MINUTES}}

ADMIN_PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SUHAS Admin</title><style>
*{box-sizing:border-box}body{margin:0;padding:16px;background:#070b1e;color:#eaf0ff;font-family:system-ui,sans-serif}
h1{font-size:20px;margin:0 0 14px}.k{display:flex;gap:12px;margin-bottom:14px}.c{flex:1;background:#141a4a;border:1px solid #2a3380;border-radius:16px;padding:14px}
.c b{display:block;font-size:34px}.c span{color:#8f9bd1;font-size:13px}
.r{display:flex;align-items:center;gap:10px;background:#141a4a;border:1px solid #2a3380;border-radius:14px;padding:12px;margin-bottom:8px}
.d{width:12px;height:12px;border-radius:50%;background:#555;flex:none}.d.on{background:#2fd27a;box-shadow:0 0 8px #2fd27a}
button{background:#7c4dff;color:#fff;border:0;border-radius:999px;padding:10px 16px;font-size:15px;font-weight:700}
.r div{flex:1;min-width:0}.r small{display:block;color:#8f9bd1;margin-top:2px}</style></head><body>
<h1>SUHAS Admin</h1><div class="k"><div class="c"><b id="on">-</b><span>Online now</span></div><div class="c"><b id="tot">-</b><span>Total customers</span></div></div>
<div id="demoCard" class="c" style="display:none;margin-bottom:14px"><span>Client demo</span><b id="dstate" style="font-size:24px;margin:6px 0 10px">-</b>
<div style="display:flex;gap:8px;flex-wrap:wrap"><button onclick="act('start')">Start</button><button onclick="act('extend')">+5 min</button><button onclick="act('stop')">Stop now</button><button onclick="if(confirm('Delete ALL demo data?'))act('reset')">Reset data</button></div></div>
<div id="list"></div><script>
let key=localStorage.getItem('akey')||'';
function ago(s){return s<60?s+'s ago':s<3600?Math.floor(s/60)+'m ago':s<86400?Math.floor(s/3600)+'h ago':Math.floor(s/86400)+'d ago'}
async function load(){
  if(!key){key=prompt('Admin key')||'';localStorage.setItem('akey',key)}
  const r=await fetch('/admin/data',{headers:{'x-admin-key':key}});
  if(r.status===401){localStorage.removeItem('akey');key='';document.getElementById('list').textContent='Wrong key. Reload.';return}
  const d=await r.json();
  document.getElementById('on').textContent=d.online;document.getElementById('tot').textContent=d.total;
  const dc=document.getElementById('demoCard');
  if(d.demo&&d.demo.on){dc.style.display='block';const m=Math.floor(d.demo.remaining/60),sc=d.demo.remaining%60;document.getElementById('dstate').textContent=d.demo.remaining>0?'ACTIVE '+m+':'+String(sc).padStart(2,'0')+' left':'OFF'}
  const L=document.getElementById('list');L.innerHTML='';
  d.rows.forEach(x=>{
    const row=document.createElement('div');row.className='r';
    const dot=document.createElement('span');dot.className='d'+(x.online?' on':'');
    const box=document.createElement('div');
    const n=document.createElement('b');n.textContent=x.name||x.id;
    const s=document.createElement('small');s.textContent=(x.online?'Online now':'Last seen '+ago(x.age))+' \u00B7 '+x.msgs+' messages \u00B7 since '+(x.first_seen||'').slice(0,10);
    box.append(n,s);row.append(dot,box);L.appendChild(row);
  });
}
async function act(a){await fetch('/admin/demo',{method:'POST',headers:{'Content-Type':'application/json','x-admin-key':key},body:JSON.stringify({action:a})});load()}
load();setInterval(load,5000);
</script></body></html>"""

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return ADMIN_PAGE

INTRO_FILE = "intro.mp4"  # optional: upload this video to the GitHub repo root to replace the animated intro

@app.get("/intro.mp4")
def intro_video():
    if not os.path.exists(INTRO_FILE):
        raise HTTPException(404, "no intro video")
    return FileResponse(INTRO_FILE, media_type="video/mp4", headers={"Cache-Control": "public, max-age=3600"})

# ---------------- DEMO / TRIAL MODE ----------------
# Put this on a SEPARATE deployment (own database) with DEMO_MODE=1. The client can only use the app
# while a demo window is open. You start / extend / stop it from /admin on that deployment.
DEMO_MODE = os.environ.get("DEMO_MODE") == "1"
DEMO_MINUTES = int(os.environ.get("DEMO_MINUTES", "10"))
with conn() as c:
    c.execute("CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT)")

def _get_until():
    with conn() as c:
        row = c.execute("SELECT v FROM settings WHERE k='demo_until'").fetchone()
    try:
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0

def _set_until(x):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO settings(k, v) VALUES('demo_until', ?)", (str(x),))

def demo_remaining():
    return max(0, int(_get_until() - time.time()))

def pin_ok(pin):
    if not PIN or pin != PIN:
        return False
    if DEMO_MODE and demo_remaining() <= 0:
        raise HTTPException(403, "demo_ended")
    return True

@app.get("/demo/status")
def demo_status(x_pin: str = Header(default="")):
    if not DEMO_MODE:
        return {"demo": False}
    if not PIN or x_pin != PIN:
        raise HTTPException(401, "Wrong PIN")
    return {"demo": True, "remaining": demo_remaining()}

class DemoIn(BaseModel):
    action: str

@app.post("/admin/demo")
def admin_demo(body: DemoIn, x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "Wrong key")
    now = time.time()
    if body.action == "start":
        _set_until(now + DEMO_MINUTES * 60)
    elif body.action == "extend":
        _set_until(max(_get_until(), now) + 300)
    elif body.action == "stop":
        _set_until(0)
    elif body.action == "reset":
        _set_until(0)
        with conn() as c:
            for t in ("messages", "conversations", "tasks", "memories", "notes"):
                c.execute(f"DELETE FROM {t}")
    return {"demo": DEMO_MODE, "remaining": demo_remaining()}

class AuthIn(BaseModel):
    pin: str

_fails = {}  # ip -> (count, first_time)

@app.post("/auth")
def auth(body: AuthIn, request: Request):
    ip = (request.headers.get("x-forwarded-for", "") or (request.client.host if request.client else "x")).split(",")[0].strip()
    n, t0 = _fails.get(ip, (0, time.time()))
    if time.time() - t0 > 300:
        n, t0 = 0, time.time()
    if n >= 5:
        raise HTTPException(429, "Too many tries")
    if PIN and body.pin == PIN:
        _fails.pop(ip, None)
        if DEMO_MODE and demo_remaining() <= 0:
            raise HTTPException(403, "demo_not_active")
        return {"ok": True}
    _fails[ip] = (n + 1, t0)
    time.sleep(0.6)
    raise HTTPException(401, "Wrong PIN")

@app.get("/health")
def health():
    return {"ok": True}

def _keep_awake():
    import urllib.request
    url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    while url:
        time.sleep(600)
        try:
            urllib.request.urlopen(url + "/health", timeout=20).read()
        except Exception as e:
            print(f"[keepawake] {e}")

if os.environ.get("KEEP_AWAKE") == "1":
    threading.Thread(target=_keep_awake, daemon=True).start()

@app.get("/summary")
def summary(x_pin: str = Header(default="")):
    if not pin_ok(x_pin):
        raise HTTPException(401, "Wrong PIN")
    with conn() as c:
        tasks = [dict(r) for r in c.execute(
            "SELECT id, title, due, priority FROM tasks WHERE done=0 AND (due IS NULL OR due <= ?) ORDER BY due LIMIT 6",
            (now_local().date().isoformat(),))]
        done = c.execute("SELECT COUNT(*) FROM tasks WHERE done=1").fetchone()[0]
        pending = c.execute("SELECT COUNT(*) FROM tasks WHERE done=0").fetchone()[0]
    return {"tasks": tasks, "done": done, "pending": pending}

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.replace("__NAME__", USER_NAME).replace("__PING_NOTE__", PING_NOTE).replace("__DEVNAME__", DEVELOPER).replace("__HAS_INTRO__", "true" if os.path.exists(INTRO_FILE) else "false")

# ---------------- PHONE CHAT PAGE ----------------
PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#070b1e">
<link rel="manifest" href="/manifest.json">
<link rel="icon" type="image/png" href="/icon-192.png">
<link rel="apple-touch-icon" href="/icon-192.png">
<title>SUHAS</title>
<style>
:root{--bg:#f6f4fb;--card:#ffffff;--text:#1d1b24;--sub:#6b6678;--border:#e4dff2;--bot:#ffffff;--shadow:0 1px 3px rgba(60,40,120,.12)}
@media (prefers-color-scheme: dark){:root{--bg:#121018;--card:#1c1926;--text:#ece9f5;--sub:#9a94ad;--border:#332e44;--bot:#262234;--shadow:none}}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{height:100%}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100dvh;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
header{background:linear-gradient(135deg,#7c4dff,#5e35b1);color:#fff;padding:14px 16px;font-size:18px;font-weight:600;display:flex;align-items:center;justify-content:space-between;border-radius:0 0 22px 22px;box-shadow:0 4px 14px rgba(94,53,177,.35)}
.ttl{display:flex;align-items:center}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#b9f6ca;margin-right:9px;box-shadow:0 0 8px #b9f6ca}
.hb{background:rgba(255,255,255,.2);border:0;color:#fff;border-radius:999px;padding:7px 10px;font-size:13px;font-weight:600;margin-left:6px}
.hb.live{background:#fff;color:#5e35b1}
#st{font-size:13px;color:var(--sub);padding:6px 18px 0;min-height:26px}
#log{flex:1;overflow-y:auto;padding:8px 14px 12px}
.m{max-width:82%;padding:10px 14px;margin:7px 0;white-space:pre-wrap;line-height:1.45;font-size:16px;box-shadow:var(--shadow);word-wrap:break-word}
.me{background:linear-gradient(135deg,#7c4dff,#6a3de8);color:#fff;margin-left:auto;border-radius:18px 18px 4px 18px}
.bot{background:var(--bot);color:var(--text);border-radius:18px 18px 18px 4px;border:1px solid var(--border)}
.act{display:inline-block;margin:8px 6px 0 0;padding:9px 14px;background:#7c4dff;color:#fff;border-radius:999px;text-decoration:none;font-size:15px;font-weight:600}
#bar{display:flex;align-items:center;gap:8px;padding:10px 12px 12px;background:var(--bg)}
#t{flex:1;min-width:0;padding:13px 18px;border:1px solid var(--border);border-radius:999px;font-size:16px;background:var(--card);color:var(--text);outline:none}
#t:focus{border-color:#7c4dff}
#t::placeholder{color:var(--sub)}
#bar button{width:48px;height:48px;border:0;border-radius:50%;padding:0;font-size:20px;background:#7c4dff;color:#fff;box-shadow:0 3px 8px rgba(124,77,255,.4)}
#mic,#cam{background:var(--card);color:#7c4dff;border:1px solid var(--border)!important;box-shadow:none!important}
#mic.on{background:#e53935;color:#fff;border-color:#e53935!important;animation:pulse 1.2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(229,57,53,.5)}70%{box-shadow:0 0 0 12px rgba(229,57,53,0)}100%{box-shadow:0 0 0 0 rgba(229,57,53,0)}}

header{background-size:200% 200%;animation:flow 14s ease infinite}
@keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.dot{animation:glow 2.4s ease-in-out infinite}
@keyframes glow{0%,100%{opacity:.55}50%{opacity:1}}
body[data-state="listening"] .dot{background:#ff8a80;box-shadow:0 0 10px #ff8a80}
body[data-state="thinking"] .dot{background:#ffe082;box-shadow:0 0 10px #ffe082}
body[data-state="speaking"] .dot{background:#ea80fc;box-shadow:0 0 10px #ea80fc}
.hb{transition:background .25s,color .25s,transform .15s}
.hb:active{transform:scale(.94)}
#log{scroll-behavior:smooth}
.m{animation:pop .28s cubic-bezier(.2,.9,.3,1.2)}
@keyframes pop{from{opacity:0;transform:translateY(10px) scale(.96)}to{opacity:1;transform:none}}
.m i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--sub);margin:5px 2px;animation:bounce 1.2s ease-in-out infinite}
.m i:nth-child(2){animation-delay:.18s}.m i:nth-child(3){animation-delay:.36s}
@keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.5}30%{transform:translateY(-6px);opacity:1}}
#bar button,.act{transition:transform .15s,box-shadow .2s,background .2s}
#bar button:active,.act:active{transform:scale(.9)}
#st{display:flex;align-items:center;gap:8px}
#wave{display:none;align-items:center;gap:3px;height:16px}
body[data-state] #wave{display:inline-flex}
#wave b{display:block;width:3px;height:6px;border-radius:2px;background:#7c4dff;animation:wave 1s ease-in-out infinite}
#wave b:nth-child(2){animation-delay:.15s}#wave b:nth-child(3){animation-delay:.3s}#wave b:nth-child(4){animation-delay:.45s}
body[data-state="listening"] #wave b{background:#e53935}
body[data-state="thinking"] #wave b{background:var(--sub);animation-duration:1.7s}
@keyframes wave{0%,100%{height:5px}50%{height:16px}}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
.thumb{display:block;max-width:100%;max-height:220px;border-radius:12px;margin-bottom:6px}
#prev{display:flex;align-items:center;gap:10px;padding:6px 16px 0}
#prev[hidden]{display:none}
#pimg{height:58px;border-radius:10px;border:1px solid var(--border)}
#px{width:28px;height:28px;border-radius:50%;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:14px;padding:0}
#vp,#hp{position:fixed;inset:0;background:rgba(0,0,0,.45);display:flex;align-items:flex-end;z-index:10}
#vp[hidden],#hp[hidden]{display:none}
.vcard{width:100%;background:var(--card);color:var(--text);border-radius:22px 22px 0 0;padding:18px 18px calc(18px + env(safe-area-inset-bottom,0px));animation:up .25s ease}
@keyframes up{from{transform:translateY(40px);opacity:0}to{transform:none;opacity:1}}
.vh{display:flex;justify-content:space-between;align-items:center;font-size:18px;font-weight:700;margin-bottom:6px}
#vclose{border:0;background:transparent;color:var(--sub);font-size:20px;padding:4px 8px}
.vl{font-size:13px;color:var(--sub);margin:14px 0 6px}
.pill{border:1px solid var(--border);background:transparent;color:var(--text);border-radius:999px;padding:9px 18px;font-size:15px;margin-right:6px;transition:background .2s,color .2s,transform .15s}
.pill.on{background:#7c4dff;color:#fff;border-color:#7c4dff}
.pill:active{transform:scale(.94)}
#vsel{width:100%;padding:11px 12px;border-radius:12px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:15px}
#vrate{width:100%;accent-color:#7c4dff}
#vtest{margin-top:16px;width:100%;border:0;border-radius:999px;padding:13px;background:#7c4dff;color:#fff;font-size:16px;font-weight:600}
#vtest:active{transform:scale(.97)}
.vn{font-size:12px;color:var(--sub);margin-top:12px;line-height:1.4}

:root{--bg:#070b1e;--card:rgba(24,32,78,.55);--text:#eaf0ff;--sub:#8f9bd1;--border:rgba(130,150,255,.22);--bot:rgba(30,40,95,.75);--shadow:none}
body{background:radial-gradient(120% 80% at 50% -10%,#1b1470 0%,#0a0e2a 45%,#050716 100%) fixed;color:var(--text)}
header{display:none}
#app{flex:1;min-height:0;display:flex;flex-direction:column}
.view{flex:1;min-height:0;overflow-y:auto;padding:14px 14px 8px;display:flex;flex-direction:column;gap:12px}
.view[hidden]{display:none}
#v-chat{padding:0;gap:0;overflow:hidden}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:14px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);animation:pop .4s ease}
.ct{font-size:11px;letter-spacing:1.6px;color:var(--sub);text-transform:uppercase;margin-bottom:10px}
.hello{font-size:26px;font-weight:700}.hello span{color:#b79bff}
.sub{color:#9fa8ff;font-size:16px;margin-top:4px}
.hero{text-align:center;padding:10px 0 4px}
#orb{width:150px;height:150px;border-radius:50%;border:0;position:relative;margin:26px auto 22px;display:block;background:radial-gradient(circle at 50% 38%,#33217f,#0b0f33 72%);box-shadow:0 0 0 6px rgba(124,77,255,.14),0 0 44px rgba(110,90,255,.6),inset 0 0 30px rgba(150,120,255,.5);color:#d8ddff;font-size:48px;animation:breathe 3.2s ease-in-out infinite;transition:box-shadow .4s}
#orb::before,#orb::after{content:"";position:absolute;inset:-14px;border-radius:50%;border:2px solid rgba(110,150,255,.35);border-top-color:#a46dff;border-bottom-color:transparent;animation:spin 8s linear infinite}
#orb::after{inset:-30px;animation-duration:15s;animation-direction:reverse;opacity:.55}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.045)}}
body[data-state="listening"] #orb{box-shadow:0 0 0 6px rgba(255,80,120,.16),0 0 54px rgba(255,80,130,.7),inset 0 0 30px rgba(255,110,150,.5);animation-duration:1.2s}
body[data-state="thinking"] #orb::before{animation-duration:1.4s}
body[data-state="speaking"] #orb{box-shadow:0 0 0 6px rgba(80,200,255,.16),0 0 54px rgba(80,200,255,.65),inset 0 0 30px rgba(120,210,255,.5);animation-duration:1.5s}
#ostate{color:#8fd2ff;font-size:17px}.tap{color:var(--sub);font-size:13px;margin-top:2px}
.bub{display:inline-block;background:var(--bot);border:1px solid var(--border);border-radius:16px;padding:10px 14px;margin-top:6px;max-width:90%}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.big{font-size:44px;font-weight:300;line-height:1}
.days{display:flex;justify-content:space-between;margin-top:10px;text-align:center;font-size:13px}
.days small{color:var(--sub);display:block}
#ring{--p:0;width:78px;height:78px;border-radius:50%;margin:6px auto;display:grid;place-items:center;background:conic-gradient(#c05cff calc(var(--p)*1%),rgba(255,255,255,.1) 0)}
#ring::after{content:"";position:absolute;width:60px;height:60px;border-radius:50%;background:#0c1030}
#score{position:relative;z-index:1;font-size:22px;font-weight:700}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 12px;margin-bottom:8px;border-radius:12px;background:rgba(255,255,255,.05);border-left:3px solid #7c4dff;font-size:15px}
.row small{color:var(--sub);white-space:nowrap}
.sc{width:100%;text-align:left;display:flex;align-items:center;gap:12px;padding:12px;margin-bottom:8px;border:0;border-radius:14px;background:rgba(255,255,255,.05);color:var(--text);font-size:15px}
.sc b{display:block;font-weight:600}.sc small{color:var(--sub)}.sc:active{transform:scale(.98)}
.tools{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.tool{border:1px solid var(--border);background:rgba(255,255,255,.05);color:var(--text);border-radius:16px;padding:16px 4px;font-size:13px;display:flex;flex-direction:column;align-items:center;gap:6px}
.tool i{all:unset;font-size:26px}.tool:active{transform:scale(.94)}
#nav{display:flex;justify-content:space-around;margin:6px 12px 8px;padding:8px;border-radius:26px;background:rgba(20,26,66,.8);border:1px solid var(--border);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
#nav button{flex:1;border:0;background:transparent;color:var(--sub);font-size:12px;display:flex;flex-direction:column;align-items:center;gap:2px;padding:6px 0;border-radius:18px;transition:all .25s}
#nav button span{font-size:22px}
#nav button.on{color:#fff;background:rgba(124,77,255,.35);box-shadow:0 0 14px rgba(124,77,255,.5)}
#v-me .hb{display:flex;width:100%;justify-content:space-between;margin:0;padding:14px;border-radius:14px;background:rgba(255,255,255,.07);font-size:16px}
#v-me .hb.live{background:#7c4dff;color:#fff}
#bar{background:transparent}
#t,#mic,#cam{background:rgba(24,32,78,.7)}
#splash{position:fixed;inset:0;z-index:100;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden;transition:opacity .6s ease,transform .6s ease}
#splash.out{opacity:0;transform:scale(1.04);pointer-events:none}
.sp-glow{position:absolute;width:140vmax;height:140vmax;left:50%;top:45%;margin:-70vmax 0 0 -70vmax;background:radial-gradient(closest-side,rgba(124,77,255,.38),transparent 70%);opacity:0;animation:spGlow 1.6s ease .1s forwards}
.sp-in{position:relative;display:flex;flex-direction:column;align-items:center;gap:24px;width:84vw;max-width:520px}
.sp-lw{position:relative;width:100%}
.sp-logo{width:100%;aspect-ratio:3.54/1;-webkit-mask:url(/logo.png) center/contain no-repeat;mask:url(/logo.png) center/contain no-repeat;background:linear-gradient(100deg,#fff 0%,#fff 38%,#cdbcff 62%,#7fd6ff 100%);background-size:240% 100%;clip-path:inset(0 100% 0 0);animation:spWipe 1.1s cubic-bezier(.65,0,.2,1) .35s forwards,spShift 2s ease .9s forwards}
.sp-sweep{position:absolute;top:-8%;bottom:-8%;left:0;width:3px;background:linear-gradient(#0000,#d8ccff,#0000);box-shadow:0 0 22px 6px rgba(150,120,255,.85);opacity:0;animation:spSweep 1.1s cubic-bezier(.65,0,.2,1) .35s forwards}
.sp-tag{font:800 15px Inter,system-ui,sans-serif;color:#a98dff;letter-spacing:14px;margin-right:-14px;opacity:0;animation:spTag 1s ease 1.3s forwards}
.sp-ml{font:700 22px 'Noto Sans Malayalam',system-ui,sans-serif;color:#fff;opacity:0;transform:translateY(12px);animation:spUp .7s ease 1.9s forwards}
.sp-bars{display:flex;gap:6px;align-items:center;height:40px;opacity:0;animation:spUp .6s ease 1.5s forwards}
.sp-bars i{display:block;width:5px;height:8px;border-radius:3px;background:#7c4dff;animation:spWave 1s ease-in-out infinite}
.sp-bars i:nth-child(odd){background:#7fd6ff}
.sp-bars i:nth-child(2){animation-delay:.1s}.sp-bars i:nth-child(3){animation-delay:.2s}.sp-bars i:nth-child(4){animation-delay:.3s}.sp-bars i:nth-child(5){animation-delay:.4s}.sp-bars i:nth-child(6){animation-delay:.5s}.sp-bars i:nth-child(7){animation-delay:.6s}
.sp-skip{position:absolute;bottom:28px;color:#666;font:600 13px Inter,system-ui,sans-serif;letter-spacing:2px}
@keyframes spGlow{to{opacity:1}}
@keyframes spWipe{to{clip-path:inset(0 0 0 0)}}
@keyframes spShift{to{background-position:100% 0}}
@keyframes spSweep{0%{left:0;opacity:1}88%{opacity:1}100%{left:100%;opacity:0}}
@keyframes spTag{from{opacity:0;letter-spacing:3px}to{opacity:1;letter-spacing:14px}}
@keyframes spUp{to{opacity:1;transform:none}}
@keyframes spWave{0%,100%{height:8px}50%{height:34px}}
#lock{position:fixed;inset:0;z-index:90;display:flex;align-items:center;justify-content:center;background:radial-gradient(70% 45% at 50% 38%,rgba(124,77,255,.3),transparent 70%),#000}
#lock[hidden]{display:none}
.lk-in{width:84vw;max-width:380px;display:flex;flex-direction:column;align-items:center;gap:16px}
.lk-logo{width:78%;aspect-ratio:3.54/1;-webkit-mask:url(/logo.png) center/contain no-repeat;mask:url(/logo.png) center/contain no-repeat;background:linear-gradient(100deg,#fff,#cdbcff 60%,#7fd6ff);margin-bottom:10px}
.lk-t{font:600 16px 'Noto Sans Malayalam',Inter,system-ui,sans-serif;color:#cfd6ff}
#lpin{width:100%;padding:16px;border-radius:16px;border:1px solid rgba(130,150,255,.4);background:rgba(255,255,255,.06);color:#fff;font-size:26px;letter-spacing:10px;text-align:center;outline:none}
#lpin:focus{border-color:#7c4dff}
#lerr{min-height:20px;color:#ff8a8a;font-size:14px}
#lbtn{width:100%;padding:15px;border:0;border-radius:999px;background:#7c4dff;color:#fff;font-size:18px;font-weight:700}
#lbtn:disabled{opacity:.6}
.lk-r{color:#8f9bd1;font-size:14px;display:flex;align-items:center;gap:8px}
.shake{animation:lkShake .4s}
@keyframes lkShake{0%,100%{transform:translateX(0)}25%{transform:translateX(-8px)}75%{transform:translateX(8px)}}
#demo{position:fixed;top:10px;right:12px;z-index:80;background:#ffd51a;color:#0b2a96;font:800 14px Inter,system-ui,sans-serif;border-radius:999px;padding:7px 14px;box-shadow:0 4px 14px rgba(0,0,0,.4)}
#demo[hidden]{display:none}
#ended{position:fixed;inset:0;z-index:95;display:flex;align-items:center;justify-content:center;text-align:center;padding:30px;background:radial-gradient(70% 45% at 50% 38%,rgba(124,77,255,.3),transparent 70%),#000;color:#fff}
#ended[hidden]{display:none}
#ended .lk-logo{margin:0 auto 22px}
#ended h2{font:700 26px 'Noto Sans Malayalam',Inter,system-ui,sans-serif;margin:0 0 10px}
#ended p{color:#aab2e6;font:500 17px 'Noto Sans Malayalam',Inter,system-ui,sans-serif;margin:0}
</style></head><body>
<div id="splash"><div class="sp-glow"></div>
<div class="sp-in"><div class="sp-lw"><div class="sp-logo"></div><div class="sp-sweep"></div></div>
<div class="sp-tag">AI ASSISTANT</div><div class="sp-ml">ഇനി സംസാരിച്ചാൽ മതി.</div>
<div class="sp-bars"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
<div class="sp-skip">TAP TO SKIP</div></div>
<script>
(function(){
  var sp=document.getElementById('splash');if(!sp)return;
  var HAS=__HAS_INTRO__;
  var reduce=window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(sessionStorage.getItem('splashDone')||reduce){sp.remove();return}
  sessionStorage.setItem('splashDone','1');
  function end(){sp.classList.add('out');setTimeout(function(){sp.remove()},650)}
  sp.onclick=end;
  if(HAS){
    var v=document.createElement('video');v.src='/intro.mp4';v.muted=true;v.autoplay=true;v.playsInline=true;
    v.style.cssText='position:relative;width:100%;height:100%;object-fit:cover';
    v.onended=end;v.onerror=function(){v.remove();setTimeout(end,3400)};
    sp.querySelector('.sp-in').replaceWith(v);v.play().catch(function(){});setTimeout(end,30000);
  }else setTimeout(end,3600);
})();
</script>
<div id="demo" hidden>⏱ <span id="dtime">10:00</span> demo</div>
<div id="ended" hidden><div style="max-width:380px"><div class="lk-logo"></div><h2>Demo കഴിഞ്ഞു. നന്ദി!</h2><p>കൂടുതൽ അറിയാൻ __DEVNAME__-നെ ബന്ധപ്പെടുക.</p></div></div>
<div id="lock" hidden><div class="lk-in"><div class="lk-logo"></div>
<div class="lk-t">PIN നൽകി അൺലോക്ക് ചെയ്യുക</div>
<input id="lpin" type="password" inputmode="numeric" autocomplete="off" placeholder="PIN">
<div id="lerr"></div><button id="lbtn">Unlock</button>
<label class="lk-r"><input type="checkbox" id="lrem"> Remember on this phone</label></div></div>
<div id="app">
<section id="v-home" class="view">
<div class="card"><img src="/logo.png" alt="SUHAS" style="height:42px;display:block;margin-bottom:14px"><div class="hello">Hello, <span>__NAME__</span></div><div class="sub">Enthaanu innu cheyyendath?</div></div>
<div class="hero"><button id="orb">🎤</button><div id="ostate">Tap to speak</div><div class="tap">Orb tap cheythu samsarikkuka</div>
<div class="bub">Hi __NAME__! Njan ready aanu. Enthu venam?</div></div>
<div class="grid2">
<div class="card"><div class="ct">Weather</div><div style="display:flex;justify-content:space-between;align-items:center"><div class="big" id="wt">--</div><div style="font-size:34px" id="wi"></div></div><div class="days" id="wd"></div></div>
<div class="card" style="text-align:center"><div class="ct">AI Insights</div><div id="ring"><div id="score">0</div></div><div class="sub" id="sdone" style="font-size:13px">tasks</div></div>
</div>
<div class="card"><div class="ct">Today's tasks</div><div id="tl"><div class="row"><span>Loading...</span></div></div></div>
<div class="card"><div class="ct">Smart shortcuts</div>
<button class="sc" data-ask="Innathe tasks nokki oru plan undaakku"><span style="font-size:22px">📅</span><span><b>Plan my day</b><small>Schedule and prioritise</small></span></button>
<button class="sc" data-pre="Ee meeting notes summarize cheyyu: "><span style="font-size:22px">📝</span><span><b>Meeting summary</b><small>Summarize your meeting</small></span></button>
<button class="sc" data-pre="Oru professional email draft cheyyu: "><span style="font-size:22px">✉️</span><span><b>Email assistant</b><small>Draft professional email</small></span></button>
<button class="sc" data-pre="Ee topic simple aayi explain cheyyu: "><span style="font-size:22px">🎓</span><span><b>Study helper</b><small>Explain any topic</small></span></button></div>
</section>
<section id="v-tools" class="view" hidden><div class="card"><div class="ct">AI Tools</div><div class="tools">
<button class="tool" data-pre="Ithu summarize cheyyu: "><i>📄</i>Summarize</button>
<button class="tool" data-pre="Ithu translate cheyyu: "><i>🌐</i>Translate</button>
<button class="tool" data-pre="Ithu nannayi ezhuthi tharu: "><i>✏️</i>Write</button>
<button class="tool" data-pre="Ithu analyze cheyyu: "><i>🔍</i>Analyze</button>
<button class="tool" id="tcam"><i>📷</i>Photo</button>
<button class="tool" data-pre="Ee code explain cheythu fix cheyyu: "><i>💻</i>Code</button></div></div><div class="card"><div class="ct">Voiceover (natural voice)</div>
<textarea id="vot" rows="6" placeholder="Oro line-um oru voice clip aakum. Script ivide paste cheyyuka..." style="width:100%;box-sizing:border-box;padding:12px;border-radius:14px;border:1px solid var(--border);background:rgba(255,255,255,.06);color:var(--text);font-size:16px;font-family:inherit"></textarea>
<button class="sc" id="vmake" style="margin-top:10px;justify-content:center"><b>Make voice clips</b></button>
<div id="vclips"></div></div>
</section>
<section id="v-me" class="view" hidden>
<div class="card"><div class="ct">Voice</div><button class="hb" id="vc">🔊 Voice settings</button></div>
<div class="card"><div class="ct">Speech language</div><button class="hb" id="lang">ML</button></div>
<div class="card"><div class="ct">Hands-free</div><button class="hb" id="live">Live: OFF</button></div>
<div class="card"><div class="ct">Chat history</div><button class="hb" id="hopen">📜 Chats</button></div>
__PING_NOTE__
<div class="card"><div class="ct">Security</div><button class="hb" id="lockbtn">🔒 Lock app now</button></div>
</section>
<section id="v-chat" class="view" hidden>
<div id="chead" style="display:flex;align-items:center;gap:10px;padding:10px 14px"><button id="hopen2" class="hb" style="margin:0">☰ Chats</button><span id="ctitle" style="flex:1;text-align:center;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">New chat</span><button id="newchat" class="hb" style="margin:0">＋</button></div>
<div id="st"><span id="wave"><b></b><b></b><b></b><b></b></span><span id="stt"></span></div>
<div id="log"></div>
<div id="prev" hidden><img id="pimg" alt=""><button id="px">✕</button></div>
<div id="bar">
<button id="cam">📷</button>
<input type="file" id="file" accept="image/*" hidden>
<button id="mic">🎤</button>
<input id="t" placeholder="Ask me anything...">
<button id="go">➤</button>
</div>
</section>
</div>
<nav id="nav"><button data-v="home" class="on"><span>🏠</span>Home</button><button data-v="chat"><span>💬</span>Chat</button><button data-v="tools"><span>🧰</span>Tools</button><button data-v="me"><span>👤</span>Profile</button></nav>
<div id="hp" hidden><div class="vcard" style="max-height:88vh;display:flex;flex-direction:column;gap:10px">
<div class="vh"><span>Chats</span><button id="hclose">✕</button></div>
<button class="sc" id="hnew" style="justify-content:center;margin:0"><b>＋ New chat</b></button>
<input id="hq" placeholder="Search chats..." style="padding:12px 16px;border-radius:999px;border:1px solid var(--border);background:rgba(255,255,255,.06);color:var(--text);font-size:16px;outline:none">
<div id="hlist" style="overflow-y:auto;flex:1;min-height:200px"></div>
</div></div>
<div id="vp" hidden><div class="vcard">
<div class="vh"><span>Voice settings</span><button id="vclose">✕</button></div>
<div class="vl">Engine</div>
<div><button class="pill" data-e="natural">Natural</button><button class="pill" data-e="phone">Phone voice</button></div>
<div class="vl">Style</div>
<div><button class="pill" data-s="sweet">Sweet</button><button class="pill" data-s="female">Female</button><button class="pill" data-s="male">Male</button><button class="pill" data-s="child">Child</button></div>
<div class="vl">Voice</div>
<select id="vsel"></select>
<div class="vl">Speed <span id="vrv"></span></div>
<input type="range" id="vrate" min="0.7" max="1.4" step="0.05">
<button id="vtest">▶ Test voice</button>
<div class="vn">Natural: internet vazhi real male/female Malayalam voice (Male = Midhun, others = Sobhana). Phone voice: phone-ile voices, Voice list athinu mathram. Natural work aayillengil thaanE phone voice-ilekku maarum.</div>
</div></div>
<script>
const log=document.getElementById('log'),t=document.getElementById('t'),st=document.getElementById('st');
const mic=document.getElementById('mic'),liveBtn=document.getElementById('live'),langBtn=document.getElementById('lang');
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!localStorage.getItem('pinv2')){localStorage.removeItem('pin');localStorage.setItem('pinv2','1')}
let pin=sessionStorage.getItem('pin')||localStorage.getItem('pin')||'';
let live=localStorage.getItem('live')==='1';
let lang=localStorage.getItem('lang')||'ml-IN';
let pendingImg=null;
let busy=false,listening=false,starting=false,unlocked=false,rec=null,wl=null;

function status(s){
  const o=document.getElementById('ostate');if(o)o.textContent=s||'Tap to speak';
  document.getElementById('stt').textContent=s;
  const k=/Listening/.test(s)?'listening':/Thinking/.test(s)?'thinking':/Speaking/.test(s)?'speaking':'';
  if(k)document.body.dataset.state=k;else document.body.removeAttribute('data-state');
}
function add(txt,cls){const d=document.createElement('div');d.className='m '+cls;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
function addActions(bubble,list){
  (list||[]).forEach(a=>{
    const l=document.createElement('a');l.className='act';
    if(a.type==='call'){l.href='tel:'+a.number;l.textContent='\u{1F4DE} Call '+(a.name||a.number)}
    else if(a.type==='alarm'){
      const hh=String(a.hour).padStart(2,'0'),mm=String(a.minute).padStart(2,'0');
      l.href='intent:#Intent;action=android.intent.action.SET_ALARM;i.android.intent.extra.alarm.HOUR='+a.hour+';i.android.intent.extra.alarm.MINUTES='+a.minute+';S.android.intent.extra.alarm.MESSAGE='+encodeURIComponent(a.label||'')+';end';
      l.textContent='\u23F0 Set alarm '+hh+':'+mm+(a.label?' ('+a.label+')':'');
    } else if(a.type==='link'){
      l.href=a.url;l.textContent=a.label;
      if(a.url.indexOf('http')===0){l.target='_blank';l.rel='noopener'}
    } else return;
    const box=document.createElement('div');box.appendChild(l);bubble.appendChild(box);
  });
  log.scrollTop=log.scrollHeight;
}
function updateUI(){
  liveBtn.textContent='Live: '+(live?'ON':'OFF');
  liveBtn.classList.toggle('live',live);
  langBtn.textContent=lang==='ml-IN'?'ML':'EN';
}
async function wake(){try{if(live&&'wakeLock' in navigator)wl=await navigator.wakeLock.request('screen')}catch(e){}}

function speakPhone(s,fin){
  try{
    speechSynthesis.cancel();
    const u=new SpeechSynthesisUtterance(s);
    u.lang=/[\u0D00-\u0D7F]/.test(s)?'ml-IN':'en-IN';
    const pv=pickVoice(u.lang);if(pv)u.voice=pv;
    const st=STYLE[vStyle]||STYLE.sweet;u.pitch=st.pitch;u.rate=Math.min(2,st.rate*vRate);
    u.onend=fin;u.onerror=fin;
    status('Speaking...');
    speechSynthesis.speak(u);
    setTimeout(fin,Math.min(90000,s.length*130+4000));
  }catch(e){fin()}
}

function startListen(){
  if(!SR||busy||listening||starting)return;
  starting=true;
  rec=new SR();rec.lang=lang;rec.interimResults=false;rec.continuous=false;
  rec.onstart=()=>{starting=false;listening=true;mic.classList.add('on');status('Listening...'+(unlocked?'':' (tap screen once to enable voice replies)'))};
  rec.onresult=e=>{listening=false;send(e.results[0][0].transcript)};
  rec.onerror=e=>{
    starting=false;
    if(e.error==='not-allowed'||e.error==='service-not-allowed'){
      live=false;localStorage.setItem('live','0');updateUI();
      status('Mic blocked. Chrome menu > Site settings > Microphone > Allow.');
    }
  };
  rec.onend=()=>{starting=false;listening=false;mic.classList.remove('on');if(live&&!busy)setTimeout(startListen,400)};
  try{rec.start()}catch(e){starting=false}
}
function stopListen(){try{rec&&rec.abort()}catch(e){}listening=false;mic.classList.remove('on')}
function resume(){if(live){setTimeout(startListen,250)}else{status('')}}

async function send(txt){
  txt=(txt||t.value).trim();if(!txt&&!pendingImg)return;
  if(!txt)txt='Ee photo-yil enthaanu? Vishadeekarikku.';
  t.value='';
  const imgToSend=pendingImg;pendingImg=null;prev.hidden=true;file.value='';
  busy=true;stopListen();try{curAudio&&curAudio.pause()}catch(e){}
  const mb=add(txt,'me');
  if(imgToSend){const im=document.createElement('img');im.src=imgToSend;im.className='thumb';mb.insertBefore(im,mb.firstChild)}
  if(!pin){showLock();return}
  const w=add('','bot');w.innerHTML='<i></i><i></i><i></i>';status('Thinking...');
  let reply=null;
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json','x-pin':pin},body:JSON.stringify({message:txt,image:imgToSend,conv_id:curConv})});
    if(r.status===401){localStorage.removeItem('pin');sessionStorage.removeItem('pin');pin='';w.textContent='Wrong PIN.';busy=false;status('');showLock();return}
    const j=await r.json();w.textContent=j.reply;reply=j.reply;addActions(w,j.actions);loadSummary();if(j.conv_id&&!curConv){curConv=j.conv_id;setTitle(txt.slice(0,40))}
  }catch(e){w.textContent='Error: '+e}
  if(reply){speak(reply,()=>{busy=false;resume()})}else{busy=false;resume()}
}

document.addEventListener('pointerdown',e=>{
  if(live&&!busy&&!listening&&!e.target.closest('button,input'))startListen();
  if(unlocked)return;unlocked=true;
  try{speechSynthesis.speak(new SpeechSynthesisUtterance(''))}catch(e){}
  if(firstRun){firstRun=false;localStorage.setItem('introDone','1');setTimeout(()=>speak(INTRO,()=>status('')),400)}
  else if(lateNow){lateNow=false;setTimeout(()=>speak(LATE,()=>status('')),400)}
},{once:false});
const file=document.getElementById('file'),prev=document.getElementById('prev'),pimg=document.getElementById('pimg');
document.getElementById('cam').onclick=()=>file.click();
document.getElementById('px').onclick=()=>{pendingImg=null;prev.hidden=true;file.value='';status('')};
file.onchange=()=>{
  const f=file.files[0];if(!f)return;
  const img=new Image(),url=URL.createObjectURL(f);
  img.onload=()=>{
    const sc=Math.min(1,1024/Math.max(img.width,img.height));
    const c=document.createElement('canvas');c.width=Math.round(img.width*sc);c.height=Math.round(img.height*sc);
    c.getContext('2d').drawImage(img,0,0,c.width,c.height);
    pendingImg=c.toDataURL('image/jpeg',0.82);
    pimg.src=pendingImg;prev.hidden=false;URL.revokeObjectURL(url);
    status('Photo ready. Type or say your question.');
  };
  img.onerror=()=>status('Could not read the photo');
  img.src=url;
};
document.getElementById('go').onclick=()=>send();
t.onkeydown=e=>{if(e.key==='Enter')send()};
mic.onclick=()=>{if(listening){stopListen()}else{startListen()}};
liveBtn.onclick=()=>{
  live=!live;localStorage.setItem('live',live?'1':'0');updateUI();
  if(live){wake();startListen()}else{stopListen();status('')}
};
langBtn.onclick=()=>{
  lang=lang==='ml-IN'?'en-IN':'ml-IN';localStorage.setItem('lang',lang);updateUI();
  if(live&&!busy){stopListen();setTimeout(startListen,300)}
};
document.addEventListener('visibilitychange',()=>{
  if(document.visibilityState==='visible'&&live){wake();if(!busy)startListen()}
});
if(!SR){mic.style.display='none';liveBtn.style.display='none';status('Voice input not supported in this browser. Use Chrome.')}
setInterval(()=>{if(live&&!busy&&!listening&&document.visibilityState==='visible')startListen()},3000);
window.addEventListener('pageshow',()=>{if(live){wake();if(!busy)startListen()}});
window.addEventListener('focus',()=>{if(live&&!busy)startListen()});
let vEngine=localStorage.getItem('veng')||'natural',curAudio=null;
function chunkText(s){
  const parts=s.match(/[^.!?\u0964\n]+[.!?\u0964]*\s*/g)||[s];
  const out=[];let cur='';
  for(const p of parts){cur+=p;if(out.length===0||cur.length>=80){out.push(cur.trim());cur=''}}
  if(cur.trim())out.push(cur.trim());
  return out.filter(Boolean).length?out.filter(Boolean):[s];
}
async function ttsBlob(text){
  const r=await fetch('/tts',{method:'POST',headers:{'Content-Type':'application/json','x-pin':pin},body:JSON.stringify({text:text,style:vStyle,speed:vRate})});
  if(!r.ok)throw new Error('tts '+r.status);
  return URL.createObjectURL(await r.blob());
}
async function speakNatural(s,fin){
  status('Speaking...');
  const chunks=chunkText(s);
  let next=ttsBlob(chunks[0]);
  for(let i=0;i<chunks.length;i++){
    let url;
    try{url=await next}catch(e){
      if(i===0)throw e;
      speakPhone(chunks.slice(i).join(' '),fin);return;
    }
    next=i+1<chunks.length?ttsBlob(chunks[i+1]):null;
    if(next)next.catch(()=>{});
    await new Promise((res,rej)=>{
      try{curAudio&&curAudio.pause()}catch(e){}
      const a=new Audio(url);curAudio=a;
      a.onended=()=>{URL.revokeObjectURL(url);res()};
      a.onerror=()=>res();
      a.play().catch(rej);
    });
  }
  fin();
}
function speak(s,done){
  let fired=false;const fin=()=>{if(!fired){fired=true;done()}};
  if(vEngine==='natural'){
    speakNatural(s,fin).catch(e=>{console.log(e);try{curAudio&&curAudio.pause()}catch(_){}speakPhone(s,fin)});
  }else speakPhone(s,fin);
}
const STYLE={sweet:{pitch:1.22,rate:0.9},female:{pitch:1.1,rate:1.0},male:{pitch:0.7,rate:0.95},child:{pitch:1.75,rate:1.08}};
let vStyle=localStorage.getItem('vstyle')||'sweet',vName=localStorage.getItem('vname')||'',vRate=parseFloat(localStorage.getItem('vrate')||'1')||1;
const nl=v=>v.lang.toLowerCase().replace('_','-');
function pickVoice(lang){
  const all=speechSynthesis.getVoices(),p=lang.slice(0,2).toLowerCase();
  const sel=all.find(v=>v.name===vName);
  if(sel&&nl(sel).startsWith(p))return sel;
  const m=all.filter(v=>nl(v).startsWith(p));
  return m.find(v=>nl(v)===lang.toLowerCase())||m[0]||null;
}
const vp=document.getElementById('vp'),vsel=document.getElementById('vsel'),vrate=document.getElementById('vrate'),vrv=document.getElementById('vrv');
function fillVoices(){
  const all=speechSynthesis.getVoices().filter(v=>/^(ml|en)/.test(nl(v)));
  vsel.innerHTML='<option value="">Auto</option>'+all.map(v=>'<option value="'+v.name+'">'+v.name+' \u00B7 '+v.lang+'</option>').join('');
  vsel.value=all.some(v=>v.name===vName)?vName:'';
}
function markStyle(){document.querySelectorAll('.pill').forEach(b=>b.classList.toggle('on',b.dataset.s?b.dataset.s===vStyle:b.dataset.e===vEngine))}
try{speechSynthesis.onvoiceschanged=fillVoices}catch(e){}
document.getElementById('vc').onclick=()=>{fillVoices();markStyle();vrate.value=vRate;vrv.textContent=vRate.toFixed(2)+'x';vp.hidden=false};
document.getElementById('vclose').onclick=()=>{vp.hidden=true;try{speechSynthesis.cancel();curAudio&&curAudio.pause()}catch(e){}};
vp.onclick=e=>{if(e.target===vp)vp.hidden=true};
document.querySelectorAll('.pill[data-e]').forEach(b=>b.onclick=()=>{vEngine=b.dataset.e;localStorage.setItem('veng',vEngine);markStyle();testVoice()});
document.querySelectorAll('.pill[data-s]').forEach(b=>b.onclick=()=>{vStyle=b.dataset.s;localStorage.setItem('vstyle',vStyle);markStyle();testVoice()});
vsel.onchange=()=>{vName=vsel.value;localStorage.setItem('vname',vName);testVoice()};
vrate.oninput=()=>{vRate=parseFloat(vrate.value);vrv.textContent=vRate.toFixed(2)+'x';localStorage.setItem('vrate',vRate)};
function testVoice(){speak(lang==='ml-IN'?'\u0D39\u0D32\u0D4B, \u0D1E\u0D3E\u0D7B \u0D28\u0D3F\u0D19\u0D4D\u0D19\u0D33\u0D41\u0D1F\u0D46 \u0D05\u0D38\u0D3F\u0D38\u0D4D\u0D31\u0D4D\u0D31\u0D28\u0D4D\u0D31\u0D4D \u0D06\u0D23\u0D4D.':'Hello, I am your assistant.',()=>status(''))}
document.getElementById('vtest').onclick=testVoice;
const $=id=>document.getElementById(id);
function show(n){['home','chat','tools','me'].forEach(v=>{$('v-'+v).hidden=v!==n});document.querySelectorAll('#nav button').forEach(b=>b.classList.toggle('on',b.dataset.v===n));if(n==='chat')log.scrollTop=log.scrollHeight}
document.querySelectorAll('#nav button').forEach(b=>b.onclick=()=>show(b.dataset.v));
function ask(q){show('chat');t.value=q;send()}
function prefill(q){show('chat');t.value=q;t.focus()}
document.querySelectorAll('[data-ask]').forEach(b=>b.onclick=()=>ask(b.dataset.ask));
document.querySelectorAll('[data-pre]').forEach(b=>b.onclick=()=>prefill(b.dataset.pre));
$('orb').onclick=()=>{show('chat');mic.click()};
$('tcam').onclick=()=>{show('chat');$('cam').click()};
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function loadSummary(){
  if(!pin){showLock();return}
  if(!pin)return;
  try{
    const r=await fetch('/summary',{headers:{'x-pin':pin}});
    if(r.status===401){localStorage.removeItem('pin');pin='';return}
    const d=await r.json();
    $('tl').innerHTML=d.tasks.length?d.tasks.map(x=>'<div class="row"><span>'+esc(x.title)+'</span><small>'+esc(x.due||'anytime')+'</small></div>').join(''):'<div class="row"><span>No pending tasks \u{1F389}</span></div>';
    const tot=d.done+d.pending,sc=tot?Math.round(d.done*100/tot):0;
    $('score').textContent=sc;$('ring').style.setProperty('--p',sc);$('sdone').textContent=d.done+' done \u00B7 '+d.pending+' pending';
  }catch(e){}
}
const WX=c=>c===0?'\u2600\uFE0F':c<=3?'\u26C5':c<=48?'\u{1F32B}\uFE0F':c<=67?'\u{1F327}\uFE0F':c<=77?'\u2744\uFE0F':c<=82?'\u{1F326}\uFE0F':'\u26C8\uFE0F';
function loadWeather(){
  if(!navigator.geolocation)return;
  navigator.geolocation.getCurrentPosition(async p=>{
    try{
      const u='https://api.open-meteo.com/v1/forecast?latitude='+p.coords.latitude+'&longitude='+p.coords.longitude+'&current=temperature_2m,weather_code&daily=temperature_2m_max,weather_code&forecast_days=5&timezone=auto';
      const d=await (await fetch(u)).json();
      $('wt').textContent=Math.round(d.current.temperature_2m)+'\u00B0';$('wi').textContent=WX(d.current.weather_code);
      $('wd').innerHTML=d.daily.time.map((dt,i)=>'<div><small>'+new Date(dt).toLocaleDateString('en',{weekday:'short'})+'</small><div>'+WX(d.daily.weather_code[i])+'</div><small>'+Math.round(d.daily.temperature_2m_max[i])+'\u00B0</small></div>').join('');
    }catch(e){}
  },()=>{},{timeout:8000});
}
$('vmake').onclick=async()=>{
  const lines=$('vot').value.split('\n').map(x=>x.trim()).filter(Boolean);
  const box=$('vclips');box.innerHTML='';
  if(!pin){showLock();return}
  for(let i=0;i<lines.length;i++){
    const row=document.createElement('div');row.className='row';
    row.innerHTML='<span>'+(i+1)+'. '+esc(lines[i].slice(0,38))+'</span><small>making...</small>';
    box.appendChild(row);
    try{
      const url=await ttsBlob(lines[i]);
      row.lastChild.remove();
      const p=document.createElement('button');p.textContent='\u25B6';p.style.cssText='border:0;border-radius:10px;padding:6px 12px;background:#7c4dff;color:#fff';
      p.onclick=()=>{new Audio(url).play()};
      const d=document.createElement('a');d.textContent='\u2B07';d.href=url;d.download='voiceover-'+(i+1)+'.mp3';d.style.cssText='border-radius:10px;padding:6px 12px;background:#7c4dff;color:#fff;text-decoration:none';
      const w=document.createElement('span');w.append(p,' ',d);row.appendChild(w);
    }catch(e){row.lastChild.textContent='failed'}
  }
};
var curConv=null;
function setTitle(t){$('ctitle').textContent=t||'New chat'}
function newChat(){curConv=null;log.innerHTML='';setTitle('New chat');add('Hi __NAME__! \u{1F44B} Enthaanu ariyendath?','bot');$('hp').hidden=true;show('chat')}
async function openChat(id,title){
  const r=await fetch('/chats/'+id,{headers:{'x-pin':pin}});if(!r.ok)return;
  const d=await r.json();log.innerHTML='';
  d.messages.forEach(m=>add(m.content,m.role==='user'?'me':'bot'));
  curConv=id;setTitle(title);$('hp').hidden=true;show('chat');
}
async function loadChats(){
  if(!pin){showLock();return}
  const q=$('hq').value.trim();
  const r=await fetch('/chats'+(q?'?q='+encodeURIComponent(q):''),{headers:{'x-pin':pin}});
  if(!r.ok){if(r.status===401){localStorage.removeItem('pin');pin=''}return}
  const d=await r.json(),L=$('hlist');L.innerHTML='';
  if(!d.chats.length)L.innerHTML='<div class="row"><span>No chats yet.</span></div>';
  d.chats.forEach(c=>{
    const wrap=document.createElement('div');wrap.style.cssText='border-bottom:1px solid var(--border)';
    const row=document.createElement('div');row.style.cssText='display:flex;align-items:center;gap:8px;padding:12px 4px';
    const t=document.createElement('div');t.style.cssText='flex:1;min-width:0;cursor:pointer';
    const tt=document.createElement('div');tt.style.cssText='font-size:17px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';tt.textContent=c.title||'Chat';
    const dd=document.createElement('small');dd.style.color='var(--sub)';dd.textContent=(c.updated||'').slice(0,16);
    t.append(tt,dd);t.onclick=()=>openChat(c.id,c.title);
    const m=document.createElement('button');m.textContent='\u22EE';m.style.cssText='border:0;background:transparent;color:var(--text);font-size:26px;padding:0 10px';
    const acts=document.createElement('div');acts.style.cssText='display:none;gap:10px;padding:0 4px 12px';
    const rn=document.createElement('button');rn.textContent='Rename';
    const dl=document.createElement('button');dl.textContent='Delete';
    [rn,dl].forEach(b=>b.style.cssText='border:1px solid var(--border);background:rgba(255,255,255,.08);color:var(--text);border-radius:999px;padding:8px 18px;font-size:15px');
    dl.style.color='#ff8a8a';
    m.onclick=()=>{acts.style.display=acts.style.display==='none'?'flex':'none'};
    rn.onclick=async()=>{const n=prompt('New name',c.title||'');if(!n)return;
      await fetch('/chats/'+c.id+'/rename',{method:'POST',headers:{'Content-Type':'application/json','x-pin':pin},body:JSON.stringify({title:n})});
      if(curConv===c.id)setTitle(n);loadChats()};
    dl.onclick=async()=>{if(!confirm('Delete this chat?'))return;
      await fetch('/chats/'+c.id,{method:'DELETE',headers:{'x-pin':pin}});
      if(curConv===c.id)newChat();loadChats()};
    acts.append(rn,dl);row.append(t,m);wrap.append(row,acts);L.appendChild(wrap);
  });
}
$('hopen').onclick=()=>{$('hp').hidden=false;$('hq').value='';loadChats()};
$('hopen2').onclick=()=>$('hopen').click();
$('newchat').onclick=newChat;$('hnew').onclick=newChat;
$('hclose').onclick=()=>{$('hp').hidden=true};
$('hp').onclick=e=>{if(e.target===$('hp'))$('hp').hidden=true};
var hT=null;$('hq').oninput=()=>{clearTimeout(hT);hT=setTimeout(loadChats,350)};
function pingNow(){if(document.visibilityState==='visible'&&pin)fetch('/ping',{method:'POST',headers:{'x-pin':pin}}).catch(()=>{})}
setInterval(pingNow,60000);document.addEventListener('visibilitychange',pingNow);setTimeout(pingNow,4000);
function showLock(){$('lock').hidden=false;$('lpin').value='';$('lerr').textContent='';setTimeout(()=>$('lpin').focus(),80)}
async function tryUnlock(){
  const v=$('lpin').value.trim();if(!v)return;
  $('lbtn').disabled=true;
  try{
    const r=await fetch('/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pin:v})});
    if(r.ok){
      pin=v;sessionStorage.setItem('pin',v);
      if($('lrem').checked)localStorage.setItem('pin',v);else localStorage.removeItem('pin');
      $('lock').hidden=true;loadSummary();pollDemo();
    }else{
      $('lerr').textContent=r.status===429?'Too many tries. Wait 5 minutes.':r.status===403?'Demo ippo start cheythittilla. Presenter-inodu start cheyyaan para.':'Wrong PIN';
      $('lpin').value='';const b=document.querySelector('.lk-in');b.classList.remove('shake');void b.offsetWidth;b.classList.add('shake');
    }
  }catch(e){$('lerr').textContent='Connection error'}
  $('lbtn').disabled=false;
}
$('lbtn').onclick=tryUnlock;$('lpin').onkeydown=e=>{if(e.key==='Enter')tryUnlock()};
$('lockbtn').onclick=()=>{sessionStorage.removeItem('pin');localStorage.removeItem('pin');pin='';showLock()};
var demoEnd=0;
function showEnded(){$('ended').hidden=false;$('demo').hidden=true;pin='';sessionStorage.removeItem('pin');localStorage.removeItem('pin');try{speechSynthesis.cancel();stopListen()}catch(e){}}
(function(){var f=window.fetch.bind(window);window.fetch=async function(){var r=await f.apply(null,arguments);
  if(r.status===403&&String(arguments[0]).indexOf('/auth')<0&&String(arguments[0]).indexOf('/admin')<0)showEnded();return r}})();
async function pollDemo(){
  if(!pin)return;
  try{const r=await fetch('/demo/status',{headers:{'x-pin':pin}});if(!r.ok)return;const d=await r.json();
    if(d.demo){demoEnd=Date.now()+d.remaining*1000;$('demo').hidden=false;if(d.remaining<=0)showEnded()}}catch(e){}
}
setInterval(function(){if(!demoEnd)return;var s=Math.max(0,Math.round((demoEnd-Date.now())/1000));
  $('dtime').textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0');if(s<=0){demoEnd=0;showEnded()}},1000);
setInterval(pollDemo,10000);
(function w(){if(document.getElementById('splash'))setTimeout(w,300);else{if(pin){loadSummary();pollDemo()}else showLock()}})();loadWeather();
var INTRO='ഹായ്, ഞാൻ സുഹാസ്! നിങ്ങളുടെ സ്വന്തം AI അസിസ്റ്റന്റ്. ഞാൻ എന്താണ് ചെയ്യേണ്ടത്?';
var firstRun=!localStorage.getItem('introDone');
if(firstRun){add(INTRO,'bot');var hb=document.querySelector('.bub');if(hb)hb.textContent=INTRO}
else add('Hi __NAME__! \u{1F44B} Enthaanu ariyendath?','bot');
var LATE='ഉറങ്ങുന്നില്ലേ? ഇത്ര രാത്രി ആയില്ലേ. നിങ്ങൾക്ക് എന്തെങ്കിലും പറ്റിയാൽ പിന്നെ ഞാൻ ഒറ്റയ്ക്കാകില്ലേ. വേഗം ഉറങ്ങിക്കോ, ഞാൻ നാളെയും ഇവിടെ ഉണ്ടാകും.';
var lateNow=false;
(function(){
  const h=new Date().getHours(),night=h>=23||h<4;
  const key=new Date(Date.now()-4*3600*1000).toDateString();
  if(night&&!firstRun&&localStorage.getItem('lateDay')!==key){
    localStorage.setItem('lateDay',key);lateNow=true;add(LATE,'bot');
    const hb=document.querySelector('.bub');if(hb)hb.textContent=LATE;
  }
})();
updateUI();
if(live){wake();startListen()}
</script></body></html>
"""
 
