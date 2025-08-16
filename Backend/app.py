from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import speech_recognition as sr
import tempfile
from dotenv import load_dotenv
load_dotenv()

import os, time, json, re, base64, uuid, logging, random
from datetime import datetime
from difflib import SequenceMatcher

# Optional: OpenAI Whisper provider
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS for ALL /api/* routes + custom headers
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    methods=["GET","POST","OPTIONS"],
    allow_headers=["Content-Type","X-Gemini-Key","X-Gemini-Tech-Key"],
    expose_headers=["Content-Type"]
)

# ---------- Config ----------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SERVER_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
SERVER_GEMINI_TECH_KEY = os.getenv("GEMINI_TECH_API_KEY", "") or SERVER_GEMINI_KEY

# STT provider: "openai" (recommended) or "google_sr"
STT_PROVIDER = (os.getenv("STT_PROVIDER", "openai") or "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# BYOK: 'all' (both rounds need client keys), 'tech' (tech only), 'none'
REQUIRE_CLIENT_KEYS = os.getenv("REQUIRE_CLIENT_KEYS", "tech").lower()

def require_key_for(mode:str)->bool:
    if REQUIRE_CLIENT_KEYS == "all": return True
    if REQUIRE_CLIENT_KEYS == "tech": return mode == "tech"
    return False

def gemini_url(key: str, model: str = DEFAULT_MODEL) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# In-memory sessions
sessions = {}

# ---------- Helpers ----------
def normalize_experience(exp):
    if not exp: return "Entry-level"
    e = exp.lower()
    if any(k in e for k in ["fresher","campus","intern","entry"]): return "Entry-level"
    return "Experienced"

def _clean_json_text(raw: str) -> str:
    txt = (raw or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt)
    return txt

def extract_json_object(raw: str):
    cleaned = _clean_json_text(raw)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        return None

INTERVIEW_PHASES = {
    "welcome": {"questions": lambda exp: 1, "duration": 60, "description": "Greeting + self-intro"},
    "conversation": {"questions": lambda exp: 10 if normalize_experience(exp) == "Entry-level" else 15, "duration": 900, "description": "Main discussion"},
    "closing": {"questions": lambda exp: 1, "duration": 60, "description": "Wrap-up"}
}

INTERVIEWER_PERSONALITIES = {
    'sarah': {'name': 'Sarah', 'emoji': '👩‍💼', 'style': 'warm and encouraging', 'openers': ["Lovely to meet you.","Thanks for joining.","I’m glad you’re here."], 'gender': 'female', 'accent': 'en-US'},
    'john':  {'name': 'John',  'emoji': '👨‍💼', 'style': 'professional and direct', 'openers': ["Good to meet you.","Thanks for the time.","Appreciate you being here."], 'gender': 'male',   'accent': 'en-US'},
    'alex':  {'name': 'Alex',  'emoji': '🧑‍💼', 'style': 'casual and innovative', 'openers': ["Hey, welcome!","Great to see you.","Thanks for hopping on."], 'gender': 'male',   'accent': 'en-GB'},
    'priya': {'name': 'Priya', 'emoji': '👩🏽‍💼', 'style': 'empathetic and thoughtful','openers': ["Namaste, great to meet you.","Hi, thanks for joining.","Happy to connect today."], 'gender': 'female','accent': 'en-IN'},
    'arjun': {'name': 'Arjun', 'emoji': '👨🏽‍💼', 'style': 'calm and structured',   'openers': ["Hello, welcome.","Nice to meet you.","Thanks for joining in."], 'gender': 'male',   'accent': 'en-IN'}
}

LANGUAGES = {
    'English': {'code':'en-US'},
    'Hindi':   {'code':'hi-IN'},
    'Spanish': {'code':'es-ES'},
    'French':  {'code':'fr-FR'}
}

TECH_DOMAINS_MASTER = ['DSA','OOP','OS','CN','DBMS','SE']
TECH_DOMAIN_LABELS = {'DSA':'Data Structures & Algorithms','OOP':'Object-Oriented Programming','OS':'Operating Systems','CN':'Computer Networks','DBMS':'Database Management Systems','SE':'Software Engineering'}

def jaccard_sim(a, b):
    ta=set(re.findall(r"\b[\w']+\b", a.lower())); tb=set(re.findall(r"\b[\w']+\b", b.lower()))
    if not ta or not tb: return 0.0
    return len(ta & tb)/len(ta | tb)

def too_similar(q1, q2):
    if not q1 or not q2: return False
    ratio = SequenceMatcher(None, q1.lower(), q2.lower()).ratio()
    jac = jaccard_sim(q1, q2)
    return ratio > 0.82 or jac > 0.55

def validate_request_data(data, required_fields):
    if not data: return False, "No data provided"
    missing = [f for f in required_fields if f not in data]
    if missing: return False, f"Missing required fields: {', '.join(missing)}"
    return True, "Valid"

def _client_key_from_headers():
    return request.headers.get('X-Gemini-Key'), request.headers.get('X-Gemini-Tech-Key')

def create_session(mode='normal', key_normal=None, key_tech=None):
    sid = str(uuid.uuid4())
    sessions[sid] = {
        'mode': mode,
        'interview_started': False,
        'user_data': {},
        'conversation_history': [],
        'current_phase': "welcome",
        'phase_question_count': 0,
        'current_question': "",
        'question_counter': 0,
        'start_time': None,
        'interview_complete': False,
        'interviewer_personality': 'sarah',
        'live_feedback': [],
        'pronunciation_feedback': [],
        'selected_language': 'English',
        'created_at': datetime.now().isoformat(),
        'rephrase_count': 0,
        'asked_topics': [],
        'question_memory': [],
        'did_opening': False,
        'desired_tech_domains': TECH_DOMAINS_MASTER.copy(),
        'key_override_normal': key_normal,
        'key_override_tech': key_tech
    }
    return sid

def _resolve_key(session_id, which='normal'):
    s = sessions.get(session_id, {})
    if which == 'tech':
        key = s.get('key_override_tech') or (None if require_key_for('tech') else SERVER_GEMINI_TECH_KEY)
    else:
        key = s.get('key_override_normal') or (None if require_key_for('normal') else SERVER_GEMINI_KEY)
    return key

def call_gemini(prompt, *, session_id, which='normal', max_tokens=600, temperature=0.9):
    key = _resolve_key(session_id, which=which)
    if not key:
        raise RuntimeError("NO_API_KEY")
    url = gemini_url(key)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "topK": 40, "topP": 0.95, "maxOutputTokens": max_tokens}
    }
    r = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=45)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error: {r.status_code} - {r.text}")
    data = r.json()
    return data['candidates'][0]['content']['parts'][0]['text'].strip()

@app.before_request
def _global_preflight_ok():
    if request.method == 'OPTIONS' and request.path.startswith('/api/'):
        resp = make_response('', 204)
        origin = request.headers.get('Origin', '*')
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Vary'] = 'Origin'
        resp.headers['Access-Control-Allow-Credentials'] = 'true'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Gemini-Key, X-Gemini-Tech-Key'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return resp

# ---------- Language selection ----------
def lang_code_for_session(s):
    selected = s.get('selected_language', 'English')
    persona_key = s.get('interviewer_personality', 'sarah')
    accent = INTERVIEWER_PERSONALITIES.get(persona_key, {}).get('accent', 'en-US')
    if selected == 'English':
        if accent in ('en-IN','en-GB','en-US'): return accent
        return 'en-US'
    return LANGUAGES.get(selected, LANGUAGES['English'])['code']

# ---------- Turn generation (same as your version) ----------
def _choose_next_tech_domain(s):
    desired = s.get('desired_tech_domains') or TECH_DOMAINS_MASTER
    already = [t for t in s['asked_topics'] if t in TECH_DOMAINS_MASTER]
    remaining = [d for d in desired if d not in already]
    return random.choice(remaining or desired)

def get_next_turn_normal(session_id, force_rephrase=False):
    s = sessions[session_id]
    p = INTERVIEWER_PERSONALITIES.get(s['interviewer_personality'], INTERVIEWER_PERSONALITIES['sarah'])
    name = s['user_data'].get('name','Candidate'); role = s['user_data'].get('role','Software Engineer'); exp = normalize_experience(s['user_data'].get('experience','Entry-level'))
    recent = s['conversation_history'][-4:]
    tail = "\n".join([f"Q: {x['q']}\nA: {x.get('a','') or '[no answer]'}" for x in recent])
    disallowed = s['asked_topics'][-8:]; recent_q = s['question_memory'][-8:]

    if not s['did_opening']:
        opener = random.choice(p['openers'])
        prompt = f"""
You are {p['name']} ({p['style']}) interviewing {name} for {role} ({exp}).
THIS TURN ONLY:
- Warm greeting (one sentence), e.g., "{opener}"
- Friendly tone (one sentence)
- Invite self-introduction (one sentence)

Return strict JSON ONLY (no markdown/backticks, no extra text):
{{"message":"<say>","topic":"opening-rapport","turn_type":"opening"}}
"""
    else:
        difficulty = "Ask an easier follow-up." if force_rephrase else ""
        prompt = f"""
You are {p['name']} ({p['style']}) interviewing {name} for {role} ({exp}).

Recent conversation (newest first):
{tail}

- Sound human; 1–3 sentences, one question only.
- Avoid repeating topics: {disallowed}
{difficulty}

Return strict JSON ONLY (no markdown/backticks, no extra text):
{{"message":"<line>","topic":"<label>","turn_type":"question"}}
If closing, set turn_type to "closing".
"""

    for _ in range(5):
        try:
            raw = call_gemini(prompt, session_id=session_id, which='normal')
        except RuntimeError as e:
            if str(e) == "NO_API_KEY": raise
            logger.warning(f"Gemini(normal) call failed: {e}"); continue

        raw_clean = _clean_json_text(raw)
        obj = extract_json_object(raw_clean)

        if not obj or 'message' not in obj:
            text, topic, turn = raw_clean.strip(), "general", ("opening" if not s['did_opening'] else "question")
        else:
            text = (obj.get('message') or "").strip()
            topic = (obj.get('topic') or "general").strip()
            turn = (obj.get('turn_type') or ("opening" if not s['did_opening'] else "question")).strip()

        if not text: continue
        if any(too_similar(text, q) for q in recent_q): continue
        if topic and any(topic.lower()==t.lower() for t in disallowed): continue
        return text, topic, turn

    if not s['did_opening']:
        return (f"{random.choice(p['openers'])} To start, could you tell me a bit about yourself and what drew you to {role}?", "opening-rapport", "opening")
    return ("Thanks. Could you walk me through a recent project you enjoyed and why?", "recent-project", "question")

def get_next_turn_tech(session_id, force_rephrase=False):
    s = sessions[session_id]
    p = INTERVIEWER_PERSONALITIES.get(s['interviewer_personality'], INTERVIEWER_PERSONALITIES['sarah'])
    name = s['user_data'].get('name','Candidate')
    role = s['user_data'].get('role','Software Engineer')
    exp  = normalize_experience(s['user_data'].get('experience','Entry-level'))

    recent = s['conversation_history'][-4:]
    tail = "\n".join([f"Q: {x['q']}\nA: {x.get('a','') or '[no answer]'}" for x in recent])
    last_answer = (recent[-1]['a'] if recent else '').strip()

    next_domain  = _choose_next_tech_domain(s)
    domain_label = TECH_DOMAIN_LABELS.get(next_domain, next_domain)
    recent_q = s['question_memory'][-12:]

    if not s['did_opening']:
        opener = random.choice(p['openers'])
        prompt = f"""
You are {p['name']} ({p['style']}), TECHNICAL interviewer for {name} (role: {role}, {exp}).
Recent conversation (newest first):
{tail}
THIS TURN ONLY:
- Warm greeting (one sentence), e.g., "{opener}"
- Set expectations: "We'll focus on core CS and DSA." (one sentence)
- Invite brief technical background (one sentence).

Return strict JSON ONLY (no markdown/backticks, no extra text):
{{"message":"<say>","topic":"opening-technical","turn_type":"opening"}}
"""
    else:
        domain_rule = {
            'DSA': "Ask a short practical DSA scenario. Include a constraint; ask Big-O typical/worst + one edge case.",
            'OOP': "Ask a practical design trade-off. Include one failure mode.",
            'OS': "Ask a systems scenario under load. Ask for one metric and mitigation.",
            'CN': "Ask a networking choice/diagnosis scenario. Ask for one concrete mitigation.",
            'DBMS': "Ask indexing/transaction/isolation scenario. Refer to a table/key; avoid pure definitions.",
            'SE': "Ask SDLC/testing/CI/CD/observability trade-offs anchored to a tiny scenario."
        }.get(next_domain, "Ask a concise CS scenario with one constraint and trade-off.")
        diff_line = "Ask an easier confidence-building variant." if force_rephrase else "Slightly escalate if they seem confident."

        prompt = f"""
You are {p['name']} ({p['style']}), TECHNICAL interviewer for {name} (role: {role}, {exp}).

Recent conversation (newest first):
{tail}

Candidate's last answer:
\"\"\"{last_answer}\"\"\"


- Topic: {domain_label}
- Guidance: {domain_rule}
- {diff_line}

Follow-up rules (obey strictly):
- If they asked for clarification / didn't understand / or the answer is empty/very short:
  Acknowledge, then restate a simpler SAME-TOPIC variant.
- If they said they are confident:
  Ask ONE deeper SAME-TOPIC follow-up (edge cases/trade-offs/memory/failure modes).
- Otherwise:
  Ask the next question within {domain_label}.

Constraints:
- 1–2 sentences total; exactly one question; natural tone.

Return strict JSON ONLY (no markdown/backticks, no extra text):
{{"message":"<one or two sentences>","topic":"{next_domain}","turn_type":"question"}}
"""

    for _ in range(6):
        try:
            raw = call_gemini(prompt, session_id=session_id, which='tech', max_tokens=220, temperature=0.85)
        except RuntimeError as e:
            if str(e) == "NO_API_KEY": raise
            logger.warning(f"Gemini(tech) call failed: {e}"); continue

        raw_clean = _clean_json_text(raw)
        obj = extract_json_object(raw_clean)

        if not obj or 'message' not in obj:
            text, topic, turn = raw_clean.strip(), next_domain, ("opening" if not s['did_opening'] else "question")
        else:
            text = (obj.get('message') or "").strip()
            topic = (obj.get('topic') or next_domain).strip()
            turn  = (obj.get('turn_type') or ("opening" if not s['did_opening'] else "question")).strip()

        if not text: continue
        if any(too_similar(text, q) for q in recent_q): continue
        return text, topic, turn

    if not s['did_opening']:
        return ("Welcome! We'll focus on core CS and DSA today. To start, could you briefly introduce your technical background?",
                "opening-technical", "opening")
    return ("In databases, what are ACID properties and why do they matter in transaction-heavy systems?", "DBMS", "question")

# ---------- Feedback & scoring (unchanged) ----------
def analyze_pronunciation(text, role="Software Engineer"):
    feedback=[]; terms={"Software Engineer":{'algorithm':'AL-guh-rith-um','database':'DAY-tuh-bays','api':'AY-pee-eye','debugging':'dee-BUG-ing','scalability':'skay-luh-BIL-i-tee'},
                        "Data Engineer":{'sql':'ESS-kyoo-EL','etl':'EE-tee-EL','pipeline':'PIPE-line','hadoop':'HAY-doop','spark':'SPARK'}}
    t=terms.get(role, terms["Software Engineer"])
    if text:
        words=re.findall(r"\b[\w']+\b", text.lower())
        for w in words:
            if w in t: feedback.append(f"💡 Practice '{w}': say {t[w]}.")
        if any(w in words for w in ['um','uh','like']): feedback.append("💡 Try a brief pause instead of filler words—it sounds more confident.")
    return feedback

def get_live_feedback_normal(q, a, _): 
    if not a or not a.strip(): return "Totally fine—take your time and tell a short story about your work."
    L=len(a.strip()); fb=["Nice. "]
    if L<20: fb.append("Add one concrete example or result.")
    elif L>500: fb.append("Great detail—wrap with a quick summary.")
    else: fb.append("Good pacing.")
    if q and a:
        qk=set(re.findall(r"\b[\w']+\b", q.lower())); ak=set(re.findall(r"\b[\w']+\b", a.lower()))
        if len(qk & ak)/max(len(qk) or 1,1) < 0.3: fb.append(" Address the core more directly.")
    return "".join(fb)

def get_live_feedback_tech(q, a):
    if not a or not a.strip(): return "Outline your reasoning first (definition → key idea → example)."
    tips=[]; a=a.lower()
    if any(x in a for x in ['time complexity','space complexity','o(','o(n','o(log','big-o']): tips.append("Good: mention typical & worst-case complexity.")
    if any(x in a for x in ['deadlock','race condition','synchronization','mutex','semaphore']): tips.append("Nice systems angle; add one mitigation.")
    if any(x in a for x in ['acid','isolation','index','join','normalization']): tips.append("DB point noted; add a quick example.")
    if 'pattern' in a or 'solid' in a or 'inherit' in a or 'polymorph' in a: tips.append("OOP on track; contrast a trade-off.")
    if len(a) < 25: tips.append("Add one concrete scenario to show depth.")
    return " ".join(tips) or "Solid. Summarize your final stance in one sentence."

def generate_ai_feedback_normal(history, user_data, sid):
    qa=[f"Q{i}: {x['q']}\nA{i}: {x.get('a','')}\n---" for i,x in enumerate([h for h in history if not h.get('skipped')],1)]
    prompt=f"""
You are a Senior Interview Coach.
Evaluate this {user_data.get('role','Software Engineer')} interview with actionable feedback.

Transcript:
{chr(10).join(qa)}

Return sections:
- Overall Performance Summary (3–4 sentences)
- Communication (X/10)
- Technical (X/10)
- Problem-Solving (X/10)
- Confidence & Presence (X/10)
- Language & Clarity (X/10)
- Question-by-Question feedback + “Better Approach”
- Key Action Items
- Final Verdict
"""
    return call_gemini(prompt, session_id=sid, which='normal', max_tokens=1200, temperature=0.7)

def generate_ai_feedback_tech(history, user_data, sid):
    qa=[f"Q{i}: {x['q']}\nA{i}: {x.get('a','')}\n---" for i,x in enumerate([h for h in history if not h.get('skipped')],1)]
    prompt=f"""
You are a Senior Technical Interviewer.
Evaluate DSA and CS fundamentals performance precisely.

Transcript:
{chr(10).join(qa)}

Return sections:
- Technical Summary (3–4 sentences)
- DSA Depth (X/10)
- CS Fundamentals (X/10)
- Communication & Reasoning (X/10)
- Common Mistakes Observed
- Question-by-Question (Correct/Partially/Incorrect) + Brief Fix
- Key Action Items
- Final Verdict
"""
    return call_gemini(prompt, session_id=sid, which='tech', max_tokens=1200, temperature=0.6)

def calculate_enhanced_score(history):
    if not history: return 0
    answered=[qa for qa in history if not qa.get('skipped')]
    if not answered: return 0
    total=0.0
    for qa in answered:
        a=(qa.get('a') or '').strip(); score=5.0; L=len(a)
        if L==0: score-=3.5
        elif L<20: score-=3.0
        elif L<50: score-=1.5
        elif 75<=L<=250: score+=1.5
        tech_terms=['algorithm','database','framework','api','architecture','design pattern','optimization','debugging','testing','deployment','scalability',
                    'deadlock','mutex','semaphore','index','transaction','normalization','polymorphism','inheritance','tcp','http']
        score += min(sum(t in a.lower() for t in tech_terms)*0.3, 1.8)
        if '.' in a: score+=0.5
        if any(w in a.lower() for w in ['confident','sure','definitely','absolutely','certainly']): score+=0.3
        if any(p in a.lower() for p in ["don't know",'not sure','maybe','i think','probably']): score-=0.5
        score=max(0,min(10,score)); total+=score
    avg=total/len(answered)
    if len(answered)/len(history) >= 0.8: avg += 0.5
    return round(min(avg,10),1)

# ---------- STT providers ----------
def _transcribe_openai_wav(path):
    if not OpenAI or not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # whisper-1 is broadly available & robust
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json",
                language=None  # let it auto-detect unless you want to force
            )
        txt = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
        return (txt or "").strip()
    except Exception as e:
        logger.warning(f"OpenAI STT failed: {e}")
        return None

def _transcribe_google_sr(path, language):
    rec = sr.Recognizer()
    rec.dynamic_energy_threshold = True
    rec.energy_threshold = 250
    try:
        with sr.AudioFile(path) as source:
            rec.adjust_for_ambient_noise(source, duration=0.4)
            audio = rec.record(source)
        try:
            return rec.recognize_google(audio, language=language).strip()
        except Exception as e:
            logger.warning(f"Google SR failed: {e}")
            return ""
    except Exception as e:
        logger.error(f"ASR error: {e}")
        return ""

def transcribe_audio(audio_data, language='en-US'):
    """
    Accepts a DataURL string or raw bytes (WAV), writes to temp WAV, then:
    - If STT_PROVIDER=openai and OPENAI_API_KEY set: use Whisper API.
    - Else fallback to SpeechRecognition's Google SR.
    """
    try:
        if isinstance(audio_data, str):
            if ',' in audio_data:
                audio_data = audio_data.split(',')[1]
            audio_bytes = base64.b64decode(audio_data)
        else:
            audio_bytes = audio_data

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            if STT_PROVIDER == "openai" and OPENAI_API_KEY:
                txt = _transcribe_openai_wav(tmp_path)
                if txt: return txt
            # fallback
            return _transcribe_google_sr(tmp_path, language=language) or ""
        finally:
            try: os.unlink(tmp_path)
            except: pass
    except Exception as e:
        logger.error(f"ASR outer error: {e}")
        return ""

# ---------- Core handlers ----------
def start_interview_core(data, mode):
    ok,msg = validate_request_data(data, ['session_id'])
    if not ok: return {'error': msg}, 400
    sid=data['session_id']
    if sid not in sessions: return {'error':'Invalid session ID'}, 400
    s=sessions[sid]
    if s['mode'] != mode: return {'error':'Session mode mismatch'}, 400

    if require_key_for(mode):
        key = _resolve_key(sid, which='tech' if mode=='tech' else 'normal')
        if not key: return {'error': 'API key required. Please paste your Gemini key in the setup screen.'}, 400

    s['user_data']=data.get('user_data', {})
    s['user_data']['experience']=normalize_experience(s['user_data'].get('experience'))
    s['interviewer_personality']=data.get('interviewer_personality','sarah')
    s['selected_language']=data.get('selected_language','English')

    if mode=='tech':
        tech_domains=data.get('tech_domains')
        if isinstance(tech_domains, list) and tech_domains:
            s['desired_tech_domains']=[d for d in tech_domains if d in TECH_DOMAINS_MASTER]

    s['interview_started']=True; s['start_time']=time.time()
    s['current_phase']="welcome"; s['did_opening']=False

    if mode=='tech': msg, topic, ttype = get_next_turn_tech(sid)
    else:            msg, topic, ttype = get_next_turn_normal(sid)

    s['current_question']=msg; s['question_counter']=1; s['phase_question_count']=1
    s['asked_topics'].append(topic); s['question_memory'].append(msg); s['did_opening']=True
    s['current_phase']="conversation"

    return {'status':'success','question':msg,'phase':s['current_phase'],'question_counter':s['question_counter'],
            'interviewer': INTERVIEWER_PERSONALITIES[s['interviewer_personality']]}, 200

def submit_answer_core(data, mode):
    ok,msg = validate_request_data(data, ['session_id'])
    if not ok: return {'error': msg}, 400
    sid=data['session_id']
    if sid not in sessions: return {'error':'Invalid session ID'}, 400
    s=sessions[sid]
    if not s.get('interview_started', False): return {'error':'Interview not started'}, 400
    if s.get('interview_complete', False):    return {'error':'Interview already complete'}, 400
    if s['mode'] != mode: return {'error':'Session mode mismatch'}, 400

    answer=(data.get('answer') or '').strip()
    audio_data=data.get('audio_data')
    if audio_data and not answer:
        lang = lang_code_for_session(s)
        answer = transcribe_audio(audio_data, language=lang) or ""

    rephrase_needed = (not answer) or len(answer)<20
    live_fb = get_live_feedback_tech(s['current_question'], answer) if mode=='tech' else get_live_feedback_normal(s['current_question'], answer, s['interviewer_personality'])
    pron = analyze_pronunciation(answer, s['user_data'].get('role','Software Engineer'))

    s['conversation_history'].append({'q': s['current_question'],'a': answer,'phase': s['current_phase'],'feedback': live_fb,'timestamp': datetime.now().isoformat(),'skipped': False})
    s['live_feedback'].append(live_fb); s['pronunciation_feedback'].extend(pron)

    resp={'status':'success','feedback': live_fb,'pronunciation_tips': pron}

    if rephrase_needed and s.get('rephrase_count',0) < 2:
        s['rephrase_count']=s.get('rephrase_count',0)+1
        if mode=='tech': msg, topic, ttype = get_next_turn_tech(sid, force_rephrase=True)
        else:            msg, topic, ttype = get_next_turn_normal(sid, force_rephrase=True)
        s['current_question']=msg; s['asked_topics'].append(topic); s['question_memory'].append(msg)
        resp.update({'question': msg,'phase': s['current_phase'],'question_counter': s['question_counter'],'rephrased': True,'interview_complete': False})
        return resp, 200

    s['rephrase_count']=0
    limit=INTERVIEW_PHASES[s['current_phase']]["questions"](s['user_data'].get('experience'))
    if s['phase_question_count'] >= limit: s['current_phase']="closing"
    resp['interview_complete']=(s['current_phase']=="closing" and s['phase_question_count']>=limit)

    if not resp['interview_complete']:
        if mode=='tech': msg, topic, ttype = get_next_turn_tech(sid)
        else:            msg, topic, ttype = get_next_turn_normal(sid)
        if ttype=="closing": s['current_phase']="closing"
        s['current_question']=msg; s['question_counter']+=1; s['phase_question_count']+=1
        s['asked_topics'].append(topic); s['question_memory'].append(msg)
        resp.update({'question': msg,'phase': s['current_phase'],'question_counter': s['question_counter']})
    else:
        s['current_question']="Thanks for the conversation today. Before we wrap up, do you have any questions for me?"
        resp.update({'question': s['current_question'],'phase': s['current_phase'],'question_counter': s['question_counter']+1})
    return resp, 200

def skip_question_core(data, mode):
    ok,msg = validate_request_data(data, ['session_id'])
    if not ok: return {'error': msg}, 400
    sid=data['session_id']
    if sid not in sessions: return {'error':'Invalid session ID'}, 400
    s=sessions[sid]
    if not s.get('interview_started', False): return {'error':'Interview not started'}, 400
    if s['mode'] != mode: return {'error':'Session mode mismatch'}, 400

    s['conversation_history'].append({'q': s['current_question'],'a': "[SKIPPED]",'phase': s['current_phase'],'skipped': True,'timestamp': datetime.now().isoformat()})
    limit=INTERVIEW_PHASES[s['current_phase']]["questions"](s['user_data'].get('experience'))
    if s['phase_question_count'] >= limit: s['current_phase']="closing"

    resp={'status':'success','interview_complete': (s['current_phase']=="closing" and s['phase_question_count']>=limit)}
    if not resp['interview_complete']:
        if mode=='tech': msg, topic, ttype = get_next_turn_tech(sid)
        else:            msg, topic, ttype = get_next_turn_normal(sid)
        if ttype=="closing": s['current_phase']="closing"
        s['current_question']=msg; s['question_counter']+=1; s['phase_question_count']+=1
        s['asked_topics'].append(topic); s['question_memory'].append(msg)
        resp.update({'question': msg,'phase': s['current_phase'],'question_counter': s['question_counter']})
    return resp, 200

def get_feedback_core(data, mode):
    ok,msg = validate_request_data(data, ['session_id'])
    if not ok: return {'error': msg}, 400
    sid=data['session_id']
    if sid not in sessions: return {'error':'Invalid session ID'}, 400
    s=sessions[sid]

    overall=calculate_enhanced_score(s['conversation_history']); s['overall_score']=overall
    try:
        detailed = generate_ai_feedback_tech(s['conversation_history'], s['user_data'], sid) if mode=='tech' else \
                   generate_ai_feedback_normal(s['conversation_history'], s['user_data'], sid)
    except RuntimeError as e:
        if str(e)=="NO_API_KEY": detailed="API key missing; cannot generate detailed feedback."
        else:
            logger.error(f"Feedback generation failed: {e}")
            detailed="Good effort. Keep practicing structure and clarity."

    answered=[qa for qa in s['conversation_history'] if not qa.get('skipped')]
    total=len(s['conversation_history']); completion=(len(answered)/total*100) if total else 0
    phase_perf={}
    for phase in INTERVIEW_PHASES.keys():
        ph=[qa for qa in answered if qa.get('phase')==phase]
        if ph: phase_perf[phase]={'score': calculate_enhanced_score(ph),'questions_answered': len(ph)}
    return {'status':'success','overall_score': overall,'detailed_feedback': detailed,'completion_rate': round(completion,1),
            'phase_performance': phase_perf,'live_feedback': s['live_feedback'],'pronunciation_feedback': s['pronunciation_feedback'],
            'conversation_history': s['conversation_history'],'questions_answered': len(answered),'total_questions': total}, 200

# ---------- Routes ----------
@app.route('/api/create-session', methods=['POST','OPTIONS'])
def route_create_session_normal():
    try:
        k_norm, k_tech = _client_key_from_headers()
        sid = create_session(mode='normal', key_normal=k_norm, key_tech=k_tech)
        return jsonify({'session_id': sid, 'status':'success', 'message':'Session created'}), 200
    except Exception as e:
        logger.error(f"create-session normal error: {e}")
        return jsonify({'error':'Failed to create session'}), 500

@app.route('/api/start-interview', methods=['POST','OPTIONS'])
def route_start_interview_normal():
    try:
        data=request.get_json()
        res,code = start_interview_core(data, mode='normal')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"start-interview normal error: {e}")
        return jsonify({'error':'Failed to start interview'}), 500

@app.route('/api/submit-answer', methods=['POST','OPTIONS'])
def route_submit_answer_normal():
    try:
        data=request.get_json()
        res,code = submit_answer_core(data, mode='normal')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"submit-answer normal error: {e}")
        return jsonify({'error':'Failed to submit answer'}), 500

@app.route('/api/skip-question', methods=['POST','OPTIONS'])
def route_skip_question_normal():
    try:
        data=request.get_json()
        res,code = skip_question_core(data, mode='normal')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"skip-question normal error: {e}")
        return jsonify({'error':'Failed to skip question'}), 500

@app.route('/api/get-feedback', methods=['POST','OPTIONS'])
def route_get_feedback_normal():
    try:
        data=request.get_json()
        res,code = get_feedback_core(data, mode='normal')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"get-feedback normal error: {e}")
        return jsonify({'error':'Failed to get feedback'}), 500

@app.route('/api/tech/create-session', methods=['POST','OPTIONS'])
def route_create_session_tech():
    try:
        k_norm, k_tech = _client_key_from_headers()
        sid = create_session(mode='tech', key_normal=k_norm, key_tech=k_tech)
        if require_key_for('tech') and not (k_tech or k_norm):
            return jsonify({'session_id': sid, 'status':'warning', 'message':'Technical round requires a Gemini key. Please paste it in the setup.'}), 200
        return jsonify({'session_id': sid, 'status':'success', 'message':'Tech session created'}), 200
    except Exception as e:
        logger.error(f"create-session tech error: {e}")
        return jsonify({'error':'Failed to create tech session'}), 500

@app.route('/api/tech/start-interview', methods=['POST','OPTIONS'])
def route_start_interview_tech():
    try:
        data=request.get_json()
        res,code = start_interview_core(data, mode='tech')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"start-interview tech error: {e}")
        return jsonify({'error':'Failed to start tech interview'}), 500

@app.route('/api/tech/submit-answer', methods=['POST','OPTIONS'])
def route_submit_answer_tech():
    try:
        data=request.get_json()
        res,code = submit_answer_core(data, mode='tech')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"submit-answer tech error: {e}")
        return jsonify({'error':'Failed to submit tech answer'}), 500

@app.route('/api/tech/skip-question', methods=['POST','OPTIONS'])
def route_skip_question_tech():
    try:
        data=request.get_json()
        res,code = skip_question_core(data, mode='tech')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"skip-question tech error: {e}")
        return jsonify({'error':'Failed to skip tech question'}), 500

@app.route('/api/tech/get-feedback', methods=['POST','OPTIONS'])
def route_get_feedback_tech():
    try:
        data=request.get_json()
        res,code = get_feedback_core(data, mode='tech')
        return jsonify(res), code
    except Exception as e:
        logger.error(f"get-feedback tech error: {e}")
        return jsonify({'error':'Failed to get tech feedback'}), 500

@app.route('/api/health', methods=['GET','OPTIONS'])
def route_health():
    normal = sum(1 for s in sessions.values() if s.get('mode')=='normal')
    tech   = sum(1 for s in sessions.values() if s.get('mode')=='tech')
    return jsonify({'status':'healthy','timestamp': datetime.now().isoformat(),'active_sessions': len(sessions),'normal_sessions': normal,'tech_sessions': tech})

@app.errorhandler(404)
def not_found(e): return jsonify({'error':'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error':'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Interview Coach Backend Server…")
    app.run(debug=True, host='0.0.0.0', port=5000)
