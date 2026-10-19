import os, re, time, base64, threading, sqlite3
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
try:
    import edge_tts  # optional: natural Malayalam/English neural voices
except Exception:
    edge_tts = None
from google import genai
from google.genai import types

MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]  # tried in order; each has its own daily quota
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

def recent_messages(limit=20):
    with conn() as c:
        rows = c.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs

def save_msg(role, content):
    with conn() as c:
        c.execute("INSERT INTO messages(role, content) VALUES(?,?)", (role, content))

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

def save_memory(fact: str) -> dict:
    """Remember a lasting fact or preference about the user."""
    with conn() as c:
        c.execute("INSERT INTO memories(fact) VALUES(?)", (fact,))
    return {"ok": True}

# ---------------- AI ----------------
def system_prompt():
    with conn() as c:
        mem = "\n".join(f"- {r['fact']}" for r in c.execute("SELECT fact FROM memories")) or "Nothing yet."
    return (f"You are {USER_NAME}'s personal AI assistant, like a smart personal secretary. "
            "Reply in the same language he writes in. If he writes or speaks Malayalam, including Manglish (Malayalam typed in English letters), always reply in proper Malayalam script, never in Malayalam written in English letters, because replies are read aloud by a voice. Write numbers, times and English words naturally so they are easy to hear. "
            "Keep replies short and practical, in a warm, friendly, polite tone. Write plain text only: never use asterisks, bold, italics, headings or markdown symbols; for lists use simple numbered lines. When he asks what to do today, call "
            "get_today_tasks and give a prioritised plan. To remind/remember, use add_task or save_memory. For reminders at a specific date and time use add_reminder (convert words like 'naale 5 manikku' using the current date and time above; at_time is 24-hour) and tell him to tap the button and press Save for the notification. For alarms use set_alarm; to phone someone use make_call with the number from what you know about him (ask for the number if unknown, and offer to remember it with save_memory). These tools only show a button; tell him to tap it. IMPORTANT: if he wants to call someone and you have (or he just gave) a number, you MUST call make_call in that same turn, even if you also save the number with save_memory. If he gives only a number after you asked whom to call, call make_call right away. For WhatsApp, SMS, email, maps directions, YouTube or Google search requests use open_link (use the number from what you know about him; ask if unknown). PHOTOS: if he sends a photo, look at it carefully and help: explain what it is, answer his doubts, read or translate any text, in his language. SCHEDULES: if a photo or message contains a schedule, timetable, meeting list, invitation or deadlines, read every item carefully. For each item call add_task with its due date (YYYY-MM-DD; assume the current year if missing), and for items that have a specific time also call add_reminder. Then list briefly what you added. If a date, time or text is unclear, ask him instead of guessing. KNOWLEDGE: when he asks a factual doubt or question, first call search_notes (his own saved notes). If nothing relevant and the answer needs current or up-to-date information (news, prices, rates, rules, recent events), call web_search and say the answer came from the web. For stable general knowledge answer directly. When he gives you longer information or a document to keep, use save_note; for short personal facts use save_memory. Never write that a button will appear unless you actually called set_alarm, make_call, open_link or add_reminder in this turn. \n"
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

def chat(text, image=None):
    _local.actions = []
    history = recent_messages()
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in history]
    parts = [types.Part(text=text)]
    if image:
        try:
            raw = base64.b64decode(image.split(",", 1)[1])
            if len(raw) > 6_000_000:
                return "Photo valuthaanu, chinna photo ayakkuka.", []
            parts.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
        except Exception as e:
            print(f"[image] {e}")
            return "Photo vaayikkaan pattiyilla, onnu koodi ayakkuka.", []
    contents.append(types.Content(role="user", parts=parts))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt(),
        tools=[add_task, get_today_tasks, complete_task, save_memory, set_alarm, make_call, open_link, add_reminder,
               save_note, search_notes, web_search],
    )
    for model in MODELS:
        for attempt in range(1):
            try:
                t0 = time.time()
                r = client.models.generate_content(model=model, contents=contents, config=config)
                print(f"[{model}] ok in {time.time()-t0:.1f}s")
                out = clean_reply(r.text or "Sorry, onnum paranjilla.")
                save_msg("user", ("[Photo attached] " if image else "") + text)
                save_msg("assistant", out)
                return out, list(_local.actions)
            except Exception as e:
                print(f"[{model} attempt {attempt+1}] {e}")
                time.sleep(0.3)
    return "Ippo Google AI-yil thirakku koodunnu. Oru minute kazhinju veendum ayakkuka.", []

# ---------------- API ----------------
class ChatIn(BaseModel):
    message: str
    image: str | None = None  # data URL (jpeg) of an attached photo

@app.post("/chat")
def chat_api(body: ChatIn, x_pin: str = Header(default="")):
    if not PIN or x_pin != PIN:
        raise HTTPException(401, "Wrong PIN")
    reply, actions = chat(body.message, body.image)
    return {"reply": reply, "actions": actions}

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

class TTSIn(BaseModel):
    text: str
    style: str = "sweet"
    speed: float = 1.0

@app.post("/tts")
async def tts_api(body: TTSIn, x_pin: str = Header(default="")):
    if not PIN or x_pin != PIN:
        raise HTTPException(401, "Wrong PIN")
    if edge_tts is None:
        raise HTTPException(503, "edge-tts not installed")
    text = re.sub(r"[*_#`~>|]", "", body.text)[:1500].strip()
    if not text:
        raise HTTPException(400, "empty")
    lang = "ml" if re.search(r"[\u0D00-\u0D7F]", text) else "en"
    gender, rate, pitch = tts_params(body.style, body.speed)
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
    return Response(content=audio, media_type="audio/mpeg")

@app.get("/summary")
def summary(x_pin: str = Header(default="")):
    if not PIN or x_pin != PIN:
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
    return PAGE.replace("__NAME__", USER_NAME)

# ---------------- PHONE CHAT PAGE ----------------
PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#7c4dff">
<title>My Assistant</title>
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
#vp{position:fixed;inset:0;background:rgba(0,0,0,.45);display:flex;align-items:flex-end;z-index:10}
#vp[hidden]{display:none}
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
</style></head><body>
<div id="app">
<section id="v-home" class="view">
<div class="card"><div class="hello">Hello, <span>__NAME__</span></div><div class="sub">Enthaanu innu cheyyendath?</div></div>
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
<button class="tool" data-pre="Ee code explain cheythu fix cheyyu: "><i>💻</i>Code</button></div></div></section>
<section id="v-me" class="view" hidden>
<div class="card"><div class="ct">Voice</div><button class="hb" id="vc">🔊 Voice settings</button></div>
<div class="card"><div class="ct">Speech language</div><button class="hb" id="lang">ML</button></div>
<div class="card"><div class="ct">Hands-free</div><button class="hb" id="live">Live: OFF</button></div>
</section>
<section id="v-chat" class="view" hidden>
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
let pin=localStorage.getItem('pin')||'';
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
function resume(){if(live){setTimeout(startListen,400)}else{status('')}}

async function send(txt){
  txt=(txt||t.value).trim();if(!txt&&!pendingImg)return;
  if(!txt)txt='Ee photo-yil enthaanu? Vishadeekarikku.';
  t.value='';
  const imgToSend=pendingImg;pendingImg=null;prev.hidden=true;file.value='';
  busy=true;stopListen();try{curAudio&&curAudio.pause()}catch(e){}
  const mb=add(txt,'me');
  if(imgToSend){const im=document.createElement('img');im.src=imgToSend;im.className='thumb';mb.insertBefore(im,mb.firstChild)}
  if(!pin){pin=prompt('Enter your PIN')||'';localStorage.setItem('pin',pin)}
  const w=add('','bot');w.innerHTML='<i></i><i></i><i></i>';status('Thinking...');
  let reply=null;
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json','x-pin':pin},body:JSON.stringify({message:txt,image:imgToSend})});
    if(r.status===401){localStorage.removeItem('pin');pin='';w.textContent='Wrong PIN. Send again to retry.';busy=false;status('');return}
    const j=await r.json();w.textContent=j.reply;reply=j.reply;addActions(w,j.actions);loadSummary();
  }catch(e){w.textContent='Error: '+e}
  if(reply){speak(reply,()=>{busy=false;resume()})}else{busy=false;resume()}
}

document.addEventListener('pointerdown',e=>{
  if(live&&!busy&&!listening&&!e.target.closest('button,input'))startListen();
  if(unlocked)return;unlocked=true;
  try{speechSynthesis.speak(new SpeechSynthesisUtterance(''))}catch(e){}
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
async function speakNatural(s,fin){
  status('Speaking...');
  const r=await fetch('/tts',{method:'POST',headers:{'Content-Type':'application/json','x-pin':pin},body:JSON.stringify({text:s,style:vStyle,speed:vRate})});
  if(!r.ok)throw new Error('tts '+r.status);
  const url=URL.createObjectURL(await r.blob());
  try{curAudio&&curAudio.pause()}catch(e){}
  const a=new Audio(url);curAudio=a;
  a.onended=()=>{URL.revokeObjectURL(url);fin()};
  a.onerror=()=>fin();
  await a.play();
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
  if(!pin){pin=prompt('Enter your PIN')||'';localStorage.setItem('pin',pin)}
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
loadSummary();loadWeather();
add('Hi __NAME__! \u{1F44B} Enthaanu ariyendath?','bot');
updateUI();
if(live){wake();startListen()}
</script></body></html>
"""
