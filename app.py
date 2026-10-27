import base64
import hashlib
import io
import wave
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import math
from collections import Counter

import networkx as nx
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from groq import Groq
except ImportError:
    Groq = None

import threading
import uuid
from datetime import datetime, timedelta, timezone

# ----------------------- Indian Standard Time -----------------------
# Cloud servers (Streamlit Cloud included) run on UTC, so every time in the app is converted to IST explicitly.
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:                       # no time-zone database (e.g. some Windows setups without tzdata)
    IST = timezone(timedelta(hours=5, minutes=30), "IST")


@st.cache_resource(show_spinner=False)
def cad_store() -> dict:
    """One store for the whole server: every session (caller or dispatcher) sees the same incidents and units.
    It lives in memory, so it resets when the app restarts (a real deployment would use a database)."""
    return {"incidents": {}, "used_units": set(), "prefs": {}, "seq": 0, "lock": threading.Lock()}


CC_DEFAULTS = {
    "theme": "Dark", "density": "Comfortable", "clock": "12-hour", "refresh": 3, "zoom": 5, "tiles": "Dark",
    "layers": ["Incidents"], "timeline": "Detailed", "sound": False, "large_text": False, "high_contrast": False,
    "reduce_motion": False, "approval": "auto", "profile": "Duty Dispatcher", "role": "Dispatch Officer",
    "sections": ["Overview", "Queue", "Map", "CAD decision", "Agencies", "Timeline"],
    "cards": ["Active incidents", "P1 critical", "P2 urgent", "Units available", "Units dispatched", "Awaiting assignment",
              "Average response"],
    "columns": ["ID", "Incident", "Priority", "Location", "Received", "Status", "Assigned"],
    "agencies": ["Police", "Fire & Rescue", "Ambulance (EMS)", "Disaster Response", "Gas Emergency", "Electricity Board",
                 "Women Helpline", "Child Helpline"],
}


def cc_prefs() -> dict:
    """This user's dashboard preferences (saved defaults per profile live in the shared store)."""
    if "cc_prefs" not in st.session_state:
        saved = cad_store()["prefs"].get(CC_DEFAULTS["profile"], {})
        st.session_state.cc_prefs = {**CC_DEFAULTS, **saved}
    return st.session_state.cc_prefs


def ist(ts: float | None = None) -> datetime:
    """A Unix timestamp (or now) as an IST datetime."""
    return datetime.fromtimestamp(time.time() if ts is None else ts, tz=IST)


def _ampm(d: datetime) -> str:
    """12-hour clock with leading zero and AM/PM, independent of the server's language settings: 08:45:32 PM."""
    return f"{(d.hour % 12) or 12:02d}:{d.minute:02d}:{d.second:02d} {'AM' if d.hour < 12 else 'PM'}"


def ist_time(ts: float | None = None) -> str:
    try:
        if st.session_state.get("cc_prefs", {}).get("clock") == "24-hour":
            return ist(ts).strftime("%H:%M:%S")
    except Exception:
        pass
    return _ist_time12(ts)


def _ist_time12(ts: float | None = None) -> str:
    """Time of day in IST: 08:45:32 PM."""
    return _ampm(ist(ts))


def ist_date(ts: float | None = None) -> str:
    """Indian date order, day first: 26-10-2026."""
    return ist(ts).strftime("%d-%m-%Y")


def ist_stamp(ts: float | None = None) -> str:
    """Full stamp: 26-10-2026 08:45:32 PM IST."""
    return f"{ist_date(ts)} {ist_time(ts)} IST"


def ist_human(ts: float | None = None) -> str:
    """Same format everywhere: 26-10-2026 08:45:32 PM IST."""
    return ist_stamp(ts)


try:                                    # interactive India map for the Command Center
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium, st_folium = None, None

try:                                    # observability: traces, feedback and evaluations in LangSmith
    import langsmith
    from langsmith import Client as LangSmithClient, traceable
except ImportError:                     # the app works the same without it
    langsmith, LangSmithClient = None, None

    def traceable(*dargs, **dkwargs):
        """No-op stand-in for langsmith.traceable (accepts and ignores langsmith_extra)."""
        def deco(fn):
            def wrapper(*a, langsmith_extra=None, **kw):
                return fn(*a, **kw)
            wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
            return wrapper
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return deco(dargs[0])
        return deco

try:                                    # free speech-to-text used when no Groq key is set
    import speech_recognition as sr
except ImportError:
    sr = None

st.set_page_config(
    page_title="Smart City Dynamic Dispatch Grid",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🚨 Smart City Dynamic Dispatch Grid")

st.caption("Synthetic emergency-call triage, incident prioritization, and vehicle dispatch demo · "
           f"🇮🇳 all times in IST (UTC+05:30) · {ist_human()}")
st.warning(
    "Prototype only: this uses synthetic data and a simplified grid. "
    "Do not use it to make real emergency-response decisions."
)

# ----------------------- City and fleet -----------------------
GRID_SIZE = 20
LANDMARKS = {
    "Central Hospital": (3, 3),
    "City Hospital": (3, 3),
    "Railway Station": (15, 4),
    "Main Railway Station": (15, 4),
    "River Bridge": (10, 10),
    "Main Bridge": (10, 10),
    "Central Market": (6, 14),
    "City Market": (6, 14),
    "North School": (4, 17),
    "South School": (16, 15),
    "Fire Station": (2, 10),
    "Police Station": (17, 8),
    "Bus Terminal": (12, 3),
    "Petrol Pump": (8, 6),
    "Old Temple": (5, 5),
    "Shopping Mall": (13, 13),
    "River Bank": (10, 8),
    "Industrial Area": (18, 17),
    "Stadium": (11, 16),
    "Airport Road": (18, 3),
}

VEHICLES = [
    {"Vehicle ID": "AMB-01", "Type": "Ambulance", "Station": (2, 3)},
    {"Vehicle ID": "AMB-02", "Type": "Ambulance", "Station": (4, 12)},
    {"Vehicle ID": "AMB-03", "Type": "Ambulance", "Station": (14, 4)},
    {"Vehicle ID": "AMB-04", "Type": "Ambulance", "Station": (16, 15)},
    {"Vehicle ID": "AMB-05", "Type": "Ambulance", "Station": (8, 8)},
    {"Vehicle ID": "FIRE-01", "Type": "Fire Truck", "Station": (2, 10)},
    {"Vehicle ID": "FIRE-02", "Type": "Fire Truck", "Station": (17, 8)},
    {"Vehicle ID": "FIRE-03", "Type": "Fire Truck", "Station": (11, 16)},
    {"Vehicle ID": "RES-01", "Type": "Rescue Team", "Station": (10, 10)},
    {"Vehicle ID": "RES-02", "Type": "Rescue Team", "Station": (18, 17)},
    {"Vehicle ID": "POL-01", "Type": "Police Car", "Station": (6, 6)},
    {"Vehicle ID": "POL-02", "Type": "Police Car", "Station": (14, 12)},
    {"Vehicle ID": "POL-03", "Type": "Police Car", "Station": (3, 16)},
    {"Vehicle ID": "POL-04", "Type": "Police Car", "Station": (17, 4)},
]

DEMO_LANDMARKS = set(LANDMARKS)      # the demo map's own places (callers can name any other place)


def make_city_graph(seed: int = 7) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
    for u, v in graph.edges:
        graph.edges[u, v]["minutes"] = rng.randint(1, 3)
        graph.edges[u, v]["closed"] = False
    # River-like barrier with three crossing points, matching the demo's project concept.
    # Remove horizontal crossings between x=9 and x=10 except at bridge rows.
    bridge_rows = {3, 10, 16}
    for y in range(GRID_SIZE):
        if y not in bridge_rows:
            a, b = (9, y), (10, y)
            if graph.has_edge(a, b):
                graph.remove_edge(a, b)
    return graph

def add_closures(graph: nx.Graph, closure_rows: list[int], closure_cols: list[int]) -> nx.Graph:
    graph = graph.copy()
    # Close selected horizontal/vertical street segments, not entire junctions.
    for y in closure_rows:
        for x in range(GRID_SIZE - 1):
            if graph.has_edge((x, y), (x + 1, y)):
                graph.remove_edge((x, y), (x + 1, y))
    for x in closure_cols:
        for y in range(GRID_SIZE - 1):
            if graph.has_edge((x, y), (x, y + 1)):
                graph.remove_edge((x, y), (x, y + 1))
    return graph

def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI response did not contain a JSON object.")
    return json.loads(text[start:end + 1])

# Severity guide from the project plan (section 3.2). When a call fits two levels, the higher one wins.
SEVERITY_GUIDE = {
    5: "Life in danger now",
    4: "Serious harm or spreading danger",
    3: "Injury, stable",
    2: "Damage, nobody hurt",
    1: "Information only",
}
SEVERITY_RULES = [   # (level, phrases, reason shown to the dispatcher)
    (5, ["trapped", "not breathing", "unconscious", "not moving", "heart attack", "drowning", "buried"], "someone's life is in danger now"),
    (4, ["spreading", "getting worse", "explosion", "severe bleeding", "heavy bleeding", "bleeding badly", "lot of blood",
         "people inside", "flames", "serious", "critical"], "serious harm or a danger that is spreading"),
    (3, ["injured", "hurt", "broken", "bleeding", "crash", "accident", "fell"], "someone is injured but stable"),
    (2, ["crack", "damage", " zznoharmzz "], "damage, nobody hurt"),
    (1, ["road blocked", "power cut", "power is gone", "blocked"], "information only"),
]
TYPE_RULES = [       # first match wins; the order mirrors how dispatch would respond
    ("Fire", ["fire", "smoke", "flames", "burning", "explosion", "gas leak", "gas smell"]),
    ("Rescue", ["trapped", "collapse", "building fell", "came down", "rubble", "flood", "drowning", "water rising", "crack", "wall"]),
    ("Medical", ["unconscious", "not breathing", "heart attack", "bleeding", "injured", "accident", "crash", "hurt",
                 "fell", "fainted", "broken"]),
]
DEFAULT_SEVERITY = {"Fire": 3, "Rescue": 4, "Medical": 3}
NEGATIONS = ["no one is hurt", "no one hurt", "nobody is hurt", "nobody hurt", "nobody is injured", "no injuries",
             "no one is injured", "not injured", "not hurt"]
PRANK_PHRASES = ["prank", "fake call", "nothing happened", "test call", "no emergency", "just kidding", "wrong number"]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "several": 3, "many": 5, "a few": 3, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
                "kai log": 3, "bahut log": 5}
HINDI_NUMBERS = {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
                 "कई": 3, "बहुत सारे": 5, "बहुत लोग": 5}
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def count_people(t: str) -> int | None:
    """How many people are hurt or need help: digits (also Hindi digits), English and Hindi number words."""
    t = re.sub(r"\b(no one|anyone|someone|everyone)\b", " ", t.translate(DEVANAGARI_DIGITS), flags=re.I)
    nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", t)]
    nums += [v for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", t, flags=re.I)]
    padded = " " + re.sub(r"[।,.!?]", " ", t) + " "
    nums += [v for w, v in HINDI_NUMBERS.items() if f" {w} " in padded]
    nums = [n for n in nums if 0 < n < 200]
    return max(nums) if nums else None


def rule_based_triage(call_text: str) -> dict[str, Any]:
    """Fallback triage if Groq is not configured or its request fails.

    Uses the plan's severity guide and explains every level it picks. Confidence is not
    measured by rules, so it is reported as None instead of a made-up number.
    """
    t = " " + call_text.lower() + " "
    for neg in NEGATIONS:                                     # "nobody hurt" must not count as "hurt"
        t = t.replace(neg, " zznoharmzz ")
    location = match_location(call_text)
    if any(w in t for w in PRANK_PHRASES):
        return {"is_emergency": False, "incident_type": "Noise / prank", "severity": 1, "location": location,
                "summary": "Looks like a prank or non-emergency call; no vehicle is needed.",
                "severity_reason": "the caller says there is no real emergency", "people": None, "confidence": None}
    kind = next((k for k, words in TYPE_RULES if any(w in t for w in words)), None)
    keyword_type = kind is not None
    kind = kind or "Medical"
    hits = [(lvl, reason, next(w for w in words if w in t)) for lvl, words, reason in SEVERITY_RULES if any(w in t for w in words)]
    if hits:
        severity, reason, word = max(hits)                       # torn between levels: the higher one wins
        reason = f"the caller said “{word.strip() if 'zz' not in word else 'nobody hurt'}”"
    else:
        severity, reason = DEFAULT_SEVERITY[kind], "no details on harm, so the usual level for this type is used"
    people = count_people(t)
    if people and people >= 3 and severity == 4 and kind != "Fire":
        severity, reason = 5, reason + "; several people involved"
    where = location or "an unknown location"
    summary = f"{kind} reported near {where}" + (f", about {people} people involved" if people else "") + "."
    return {"is_emergency": True, "incident_type": kind, "severity": severity, "location": location, "type_from_keywords": keyword_type,
            "summary": summary, "severity_reason": reason, "people": people, "confidence": None}


def match_location(text: str) -> str | None:
    lower = text.lower()
    for name in sorted(LANDMARKS, key=len, reverse=True):
        if name.lower() in lower:
            return name
    aliases = {
        "hospital": "Central Hospital",
        "railway": "Railway Station",
        "bridge": "River Bridge",
        "market": "Central Market",
        "school": "North School",
        "mall": "Shopping Mall",
        "stadium": "Stadium",
        "airport": "Airport Road",
        "temple": "Old Temple",
        "petrol": "Petrol Pump",
        "industrial": "Industrial Area",
        "bus stand": "Bus Terminal",
        "bus adda": "Bus Terminal",
        "thana": "Police Station",
        "mandir": "Old Temple",
        "अस्पताल": "Central Hospital", "हॉस्पिटल": "Central Hospital", "रेलवे": "Railway Station", "स्टेशन": "Railway Station",
        "पुल": "River Bridge", "बाजार": "Central Market", "बाज़ार": "Central Market", "मार्केट": "Central Market",
        "स्कूल": "North School", "मॉल": "Shopping Mall", "स्टेडियम": "Stadium", "मंदिर": "Old Temple",
        "पेट्रोल पंप": "Petrol Pump", "बस अड्डा": "Bus Terminal", "बस अड्डे": "Bus Terminal", "बस स्टैंड": "Bus Terminal", "थाना": "Police Station",
        "एयरपोर्ट": "Airport Road", "फैक्ट्री": "Industrial Area", "नदी": "River Bank",
    }
    for phrase, landmark in aliases.items():
        if phrase in lower:
            return landmark
    return None          # never invent a map point: the dispatcher calls back instead (plan, section 3.4)

GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"     # follows the JSON format reliably
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]


@traceable(run_type="llm", name="Groq triage", process_inputs=lambda i: {"call_text": i.get("call_text"), "model": i.get("model")})
def groq_triage(call_text: str, api_key: str, model: str) -> dict[str, Any]:
    if Groq is None:
        raise RuntimeError("Groq package is not installed. Install requirements.txt.")
    client = Groq(api_key=api_key)
    prompt = f"""
You are a cautious triage assistant for a SYNTHETIC emergency dispatch demonstration.
Extract a single call into valid JSON only with keys:
is_emergency (boolean), incident_type (one of Medical, Fire, Rescue, Noise / prank),
severity (integer 1-5, where 5 is most urgent), location (one exact string from allowed locations),
summary (short string), confidence (number 0-1).
Allowed locations: {list(LANDMARKS.keys())}.
Use the best matching location from the list. If no location is clear, use null.
Never invent a location or any other fact.
Severity guide (when torn between two levels, choose the higher one):
5 = life in danger now (trapped, not breathing, unconscious); 4 = serious harm or spreading danger (fire spreading,
heavy bleeding); 3 = injury, stable; 2 = damage, nobody hurt; 1 = information only.
Also return severity_reason (one short phrase quoting what in the call decides the level) and people (integer or null). Prank/test/non-emergency calls should have
is_emergency=false and incident_type="Noise / prank". Severity must be 1-5.
Call text: {call_text!r}
"""
    messages = [
        {"role": "system", "content": "Return only a valid JSON object. This is a demo, not real emergency advice."},
        {"role": "user", "content": prompt},
    ]

    def ask(m: str, json_mode: bool) -> str:
        extra = {}
        if any(tag in m.lower() for tag in ("gpt-oss", "qwen3", "deepseek-r1", "reasoning")):
            # Reasoning models "think" before answering. With a small token limit the thinking uses the whole
            # budget, the answer comes back empty and Groq rejects it ("json_validate_failed"). Keep thinking
            # short and give room for the answer.
            extra["reasoning_effort"] = "low"
            limit = 2048
        else:
            limit = 500
        if json_mode:
            extra["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(model=m, messages=messages, temperature=0.1,
                                              max_completion_tokens=limit, **extra)
        return resp.choices[0].message.content or ""

    # 1) JSON mode  2) same model without JSON mode (we parse it ourselves)  3) a reliable fallback model
    attempts = [(model, True), (model, False)]
    if model != GROQ_FALLBACK_MODEL:
        attempts.append((GROQ_FALLBACK_MODEL, True))
    last_error = None
    for m, json_mode in attempts:
        try:
            content = ask(m, json_mode)
            data = parse_json_response(content)
            data["_model_used"] = m
            break
        except Exception as exc:                # empty or invalid JSON, or an API error: try the next way
            last_error = exc
            if "invalid_api_key" in str(exc) or "401" in str(exc):
                raise RuntimeError("the Groq API key is not valid") from exc
            if "model_not_found" in str(exc) or "does not exist" in str(exc):
                continue
    else:
        raise RuntimeError(f"Groq could not return a usable answer ({str(last_error)[:160]})")
    data["severity"] = max(1, min(5, int(data.get("severity", 3))))
    if data.get("location") not in LANDMARKS:
        data["location"] = match_location(call_text)      # None if the call names no known place
        data["confidence"] = min(float(data.get("confidence") or 0.5), 0.4)
    data["is_emergency"] = bool(data.get("is_emergency", True))
    data["incident_type"] = str(data.get("incident_type", "Medical"))
    data["summary"] = str(data.get("summary", call_text[:180]))
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence") or 0.5)))
    data["severity_reason"] = str(data.get("severity_reason") or SEVERITY_GUIDE[data["severity"]].lower())
    try:
        data["people"] = int(data["people"]) if data.get("people") is not None else None
    except (TypeError, ValueError):
        data["people"] = None
    return data

def route_minutes(graph: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> float:
    # Weighted shortest path based on simulated travel minutes.
    return float(nx.shortest_path_length(graph, start, end, weight="minutes"))

def route_nodes(graph: nx.Graph, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    return nx.shortest_path(graph, start, end, weight="minutes")

def choose_vehicle(incident_type: str, location: str, graph: nx.Graph, available: list[dict]) -> tuple[dict | None, float | None]:
    required = {"Medical": "Ambulance", "Fire": "Fire Truck", "Rescue": "Rescue Team", "Police": "Police Car"}.get(incident_type)
    candidates = [v for v in available if v["Type"] == required]
    if not candidates:
        # Demo fallback: allow any vehicle if a matching type is unavailable.
        candidates = available
    if not candidates:
        return None, None
    if location not in LANDMARKS:
        return None, None                                   # no location yet: wait for the call-back
    target = LANDMARKS[location]
    scored = []
    for v in candidates:
        try:
            minutes = route_minutes(graph, v["Station"], target)
            scored.append((minutes, v))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    if not scored:
        return None, None
    minutes, vehicle = min(scored, key=lambda item: item[0])
    return vehicle, minutes

SEV_COLOR = {5: "#D62839", 4: "#E46A1E", 3: "#D9A21B", 2: "#86AE3A", 1: "#8A97A5"}


def show_triage_card(rec: dict[str, Any]) -> None:
    """Dispatcher-friendly view of one triaged call; the raw JSON stays available for developers."""
    sev = int(rec.get("severity", 3))
    color = SEV_COLOR.get(sev, "#8A97A5")
    with st.container(border=True):
        top = st.columns([3, 2])
        top[0].markdown(f"### {rec['Call ID']}: {rec['incident_type']}")
        if str(rec.get("Source", "")).startswith("Voice"):
            top[0].caption(f"🎙️ Reported by voice, written by {rec.get('Voice engine') or 'speech recognition'}"
                           + (", then edited by hand" if "edited" in rec["Source"] else ""))
        method = rec.get("triage_method", "")
        top[1].markdown(
            f"<div style='text-align:right;margin-top:10px'><span style='background:{'#1F8A5B' if method == 'Groq AI' else '#1C5D99' if 'ML' in method else '#5B6B7A'};"
            f"color:#fff;border-radius:999px;padding:3px 10px;font-size:0.8rem;font-weight:600'>{method}</span></div>",
            unsafe_allow_html=True)
        if not rec.get("is_emergency"):
            st.info("Not an emergency: no vehicle will be sent. " + rec.get("summary", ""))
        c = st.columns(4)
        c[0].markdown(
            f"<div style='font-size:0.85rem;opacity:.75'>Severity</div>"
            f"<div style='display:inline-block;background:{color};color:#fff;border-radius:10px;padding:2px 12px;"
            f"font-size:1.6rem;font-weight:800'>{sev}</div> <span style='font-size:0.85rem'>of 5</span>",
            unsafe_allow_html=True)
        def fact(col, label, value, muted=False):
            col.markdown(f"<div style='font-size:0.85rem;opacity:.75'>{label}</div>"
                         f"<div style='font-size:1.25rem;font-weight:700;{'opacity:.6' if muted else ''}'>{value}</div>",
                         unsafe_allow_html=True)
        fact(c[1], "Location", rec.get("location") or "Unknown", muted=not rec.get("location"))
        fact(c[2], "People", rec.get("people") or "Not stated", muted=not rec.get("people"))
        conf = rec.get("confidence")
        fact(c[3], "Confidence" + (" (model)" if "ML" in method else " (AI estimate)" if isinstance(conf, (int, float)) else ""),
             f"{conf:.0%}" if isinstance(conf, (int, float)) else "Not measured",
             muted=not isinstance(conf, (int, float)))
        st.markdown(f"**{SEVERITY_GUIDE.get(sev, '')}.** Why: {rec.get('severity_reason', 'not given')}.")
        st.markdown(f"**Summary:** {rec.get('summary', '')}")
        if rec.get("is_emergency") and not rec.get("location"):
            st.warning("No known landmark in this call. Call the reporter back to get the location; "
                       "it will not be dispatched or drawn on the map until then.")
        for note in rec.get("model_notes", []):
            st.caption(f"🤖 {note[0].upper() + note[1:]}.")
        if rec.get("needs_review") and rec.get("is_emergency"):
            st.warning(f"🧑‍💼 Please check this call: "
                       + (f"the rules say {rec['incident_type']} but the model says {rec['model_type']}." if not rec.get("model_agrees")
                          else f"the model is only {rec['confidence']:.0%} sure."))
        if method == "Rule-based fallback":
            st.caption("Rule-based triage: add a Groq API key in the sidebar for AI reading of messy or unclear calls.")
        if rec.get("Recording"):
            st.markdown("🎧 **Caller’s recording**")
            st.audio(rec["Recording"], format="audio/wav")
        with st.expander("Raw data (for developers)"):
            st.json({k: v for k, v in rec.items() if k not in ("Call text", "Recording")})
            if "model_type" in rec:
                st.caption("Model details: see MODEL_CARD.md for how it was trained and tested.")

# ----------------------- Learned triage model (exported by notebook Part 8) -----------------------
MODEL_FILE = Path(__file__).parent / "triage_model.json"


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    for part in ("type_model", "severity_model"):                      # pre-compute arrays once
        mm = m[part]
        mm["_W"], mm["_b"] = [list(r) for r in mm["coef"]], list(mm["intercept"])
    return m


def _ngram_counts(text: str, vocab: dict, lo: int, hi: int, stats: dict | None = None) -> Counter:
    """Exactly scikit-learn's char_wb analyzer (lowercase, words padded with spaces)."""
    counts, total = Counter(), 0
    for w in re.sub(r"\s\s+", " ", text.lower()).split():
        w = f" {w} "
        for n in range(lo, hi + 1):
            off, grams = 0, [w[0:n]]
            while off + n < len(w):
                off += 1
                grams.append(w[off:off + n])
            total += len(grams)
            for g in grams:
                j = vocab.get(g)
                if j is not None:
                    counts[j] += 1
            if off == 0:
                break
    if stats is not None:
        stats["familiarity"] = sum(counts.values()) / total if total else 0.0
    return counts


def model_proba(part: dict, text: str, stats: dict | None = None) -> dict[str, float]:
    """Sublinear TF-IDF -> L2 norm -> logistic regression -> temperature-scaled softmax."""
    counts = _ngram_counts(text, part["vocab"], *part["ngram_range"], stats=stats)
    x = {j: (1 + math.log(c)) * part["idf"][j] for j, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in x.values())) or 1.0
    z = [(sum(row[j] * v for j, v in x.items()) / norm + b) / part["temperature"] for row, b in zip(part["_W"], part["_b"])]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    tot = sum(e)
    return {c: v / tot for c, v in zip(part["classes"], e)}


def apply_model(triage: dict, call_text: str, model: dict) -> dict:
    """Safety hybrid from notebook Part 8: rules keep the type; the model rescues calls the rules could not type,
    can raise severity (never lower it), gives a calibrated confidence, and flags disagreement for a human."""
    info: dict = {}
    tp = model_proba(model["type_model"], call_text, stats=info)
    # Calibration only holds for wording like the training data. For unfamiliar wording, shrink confidence
    # towards "no idea" in proportion to how unfamiliar it is, and never claim more than 99%.
    fam = model.get("familiarity", {})
    ref = fam.get("p05", 0.0) or 0.0
    trust = 1.0 if not ref else min(1.0, info.get("familiarity", 0.0) / ref)
    k = len(tp)
    tp = {c: min(0.99, trust * p + (1 - trust) / k) for c, p in tp.items()}
    app_p: dict[str, float] = {}
    for cls, p in tp.items():
        app_p[model["app_type_map"][cls]] = app_p.get(model["app_type_map"][cls], 0.0) + p
    m_type = max(app_p, key=app_p.get)
    thr = model.get("review_threshold", 0.7)
    out = dict(triage)
    notes = []
    confident = app_p[m_type] >= thr
    if out.get("is_emergency") and m_type != "Noise / prank" and m_type != out["incident_type"] and confident and (
            model.get("type_leader") == "model" or not out.get("type_from_keywords")):
        # the leader (chosen on development data in Part 8) decides; the model never dismisses a call on its own
        notes.append(f"type from the model; the rules said {out['incident_type']}"
                     + ("" if out.get("type_from_keywords") else " only as a default"))
        out["incident_type"] = m_type
    if out.get("is_emergency"):
        sp = model_proba(model["severity_model"], call_text)
        m_sev = int(max(sp, key=sp.get))
        if m_sev > int(out["severity"]):
            notes.append(f"model raised severity from {out['severity']} to {m_sev}")
            out["severity_reason"] = f"{out.get('severity_reason', '')}; the model rates calls like this at {m_sev} ({sp[str(m_sev)]:.0%} sure)".lstrip("; ")
            out["severity"] = m_sev
    final = out["incident_type"] if out.get("is_emergency") else "Noise / prank"
    out["confidence"] = round(app_p.get(final, 0.0), 3)
    out["model_type"] = m_type
    out["model_agrees"] = (m_type == final)
    out["familiarity"] = round(info.get("familiarity", 0.0), 3)
    out["needs_review"] = (not out["model_agrees"]) or out["confidence"] < thr
    if trust < 1.0:
        notes.append(f"unfamiliar wording ({out['familiarity']:.0%} of its patterns seen in training), so confidence was lowered")
    out["model_notes"] = notes
    out["triage_method"] = "Rules + ML model"
    return out


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
def cached_groq_triage(call_text: str, api_key: str, model: str) -> dict:
    """Same text + same model = same answer: re-triaging or batch duplicates cost no extra API calls."""
    return groq_triage(call_text, api_key, model)


@traceable(run_type="chain", name="Triage agent")
def run_triage(call_text: str) -> tuple[dict, list[tuple[str, str]]]:
    """Groq AI if switched on and configured, otherwise rules (+ the learned model). Returns (triage, messages)."""
    msgs, triage = [], None
    if use_groq and groq_key:
        try:
            triage = dict(cached_groq_triage(call_text.strip(), groq_key, groq_model))
            used = triage.pop("_model_used", groq_model)
            triage["triage_method"] = "Groq AI"
            if used != groq_model:
                msgs.append(("info", f"{groq_model} did not return a usable answer, so {used} was used for this call."))
        except Exception as exc:
            msgs.append(("warning", f"Groq triage failed ({str(exc)[:200]}), so keyword rules were used for this call."))
    elif use_groq and not groq_key:
        msgs.append(("info", "Groq AI is switched on but no API key is set, so keyword rules were used."))
    if triage is None:
        triage = rule_based_triage(call_text)
        triage["triage_method"] = "Rule-based fallback"
        if use_model and triage_model is not None:
            triage = apply_model(triage, call_text, triage_model)
    return triage, msgs


# ----------------------- Triage helpers: duplicates, records, one-click dispatch -----------------------
COMPATIBLE_TYPES = {"Fire": {"Fire", "Medical"}, "Rescue": {"Rescue", "Medical"}, "Medical": {"Medical", "Fire", "Rescue"}}


def _words(t: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", t.lower())) - {"the", "and", "near", "there", "are", "is", "please", "help", "caller"}


def find_duplicate(rec: dict, calls: list[dict]) -> tuple[str | None, float]:
    """Plan section 3.5: same place + compatible type + similar words = probably the same incident.
    Lean towards keeping calls separate (section 3.7): without a shared landmark, never join."""
    if not rec.get("is_emergency") or not rec.get("location"):
        return None, 0.0
    best, best_score = None, 0.0
    a = _words(rec["Call text"])
    for c in calls:
        if c is rec or not c.get("is_emergency") or c.get("location") != rec["location"] or c.get("duplicate_of"):
            continue
        if rec["incident_type"] not in COMPATIBLE_TYPES.get(c.get("incident_type"), {c.get("incident_type")}):
            continue
        b = _words(c["Call text"])
        text_sim = len(a & b) / len(a | b) if a | b else 0.0
        score = 0.5 + 0.2 * (rec["incident_type"] == c["incident_type"]) + 0.3 * text_sim
        if score >= 0.6 and score > best_score:
            best, best_score = c["Call ID"], score
    return best, round(best_score, 2)


def add_call(text: str, triage: dict, source: str, engine: str | None = None, audio: bytes | None = None) -> dict:
    rec = {"Call ID": f"CALL-{len(st.session_state.calls) + 1:03d}", "Call text": text.strip(), "Source": source,
           "Voice engine": engine, "Recording": audio, "Triaged at": ist_time(), **triage}
    dup, score = find_duplicate(rec, st.session_state.calls)
    if dup:
        rec["duplicate_of"], rec["duplicate_score"] = dup, score
        original = next(c for c in st.session_state.calls if c["Call ID"] == dup)
        original["callers"] = original.get("callers", 1) + 1
        if int(rec["severity"]) > int(original["severity"]):                 # a later caller can raise urgency
            original["severity"] = int(rec["severity"])
    st.session_state.calls.append(rec)
    return rec


# ----------------------- Multi-agent routing: the right problem goes to the right team -----------------------
# Like ERSS-112 in India: one number, one call-taker, then the call is routed to every department that
# has to act. Each step is a small "agent" with one job and a CONSTRAINED output: it can only choose from fixed lists,
# so it can never invent a department, a severity or a place.
DEPARTMENTS = {   # name -> icon, Indian helpline, vehicle type it sends (None = a ticket and a call-back), US equivalent
    "Police":            {"icon": "🚓", "line": "100",  "fleet": "Police Car",  "code": "POL", "agency": "Police ERV (PCR van)", "lines": "112 / 100"},
    "Fire & Rescue":     {"icon": "🚒", "line": "101",  "fleet": "Fire Truck",  "code": "FIR", "agency": "Fire control room", "lines": "101"},
    "Ambulance (EMS)":   {"icon": "🚑", "line": "108",  "fleet": "Ambulance",   "code": "EMS", "agency": "108 ambulance service", "lines": "108 / 102"},
    "Disaster Response": {"icon": "🛟", "line": "1078", "fleet": "Rescue Team", "code": "DRF", "agency": "SDRF / NDRF", "lines": "1078"},
    "Gas Emergency":     {"icon": "🛢️", "line": "1906", "fleet": None,          "code": "GAS", "agency": "Gas distribution company", "lines": "1906"},
    "Electricity Board": {"icon": "⚡", "line": "1912", "fleet": None,          "code": "ELE", "agency": "Electricity board (DISCOM)", "lines": "1912"},
    "Women Helpline":    {"icon": "👩", "line": "1091", "fleet": None,          "code": "WOM", "agency": "Women helpline", "lines": "181 / 1091"},
    "Child Helpline":    {"icon": "🧒", "line": "1098", "fleet": None,          "code": "CHD", "agency": "Childline", "lines": "1098"},
}
PROBLEMS = {      # the ONLY problem categories the router may choose; each one says which teams must act
    "fire":          {"label": "Fire", "kw": ["fire", "smoke", "flames", "burning", "blaze", "aag", "dhuan"], "teams": ["Fire & Rescue"]},
    "gas_leak":      {"label": "Gas leak", "kw": ["gas leak", "gas smell", "smell of gas", "cylinder", "lpg"], "teams": ["Fire & Rescue", "Gas Emergency"]},
    "medical":       {"label": "Medical emergency", "kw": ["unconscious", "not breathing", "heart attack", "chest pain", "fainted", "seizure",
                                                          "stroke", "bleeding", "injured", "hurt", "labour pain", "snake bite", "poison",
                                                          "overdose", "behosh", "fell", "not moving", "collapsed", "burns", "faint"],
                      "teams": ["Ambulance (EMS)"]},
    "road_accident": {"label": "Road accident", "kw": ["accident", "crash", "collided", "collision", "hit by", "overturned", "run over",
                                                       "skidded"], "teams": ["Ambulance (EMS)", "Police"]},
    "collapse":      {"label": "Building collapse", "kw": ["building collapse", "collapsed building", "building collapsed", "house collapsed",
                                                           "wall collapsed", "roof collapsed", "bridge collapsed", "building fell", "wall fell",
                                                           "roof fell", "rubble", "debris", "came down"], "teams": ["Disaster Response", "Fire & Rescue"]},
    "flood":         {"label": "Flood / drowning", "kw": ["flood", "water rising", "water is rising", "rising water", "drowning", "waterlogging",
                                                          "washed away", "stuck on the roof", "stuck on his roof", "baadh"],
                      "teams": ["Disaster Response"]},
    "crime":         {"label": "Crime / violence", "kw": ["robbery", "robbed", "robbing", "theft", "stolen", "thief", "burglary", "break in", "broke into",
                                                          "snatch", "gun", "pistol", "knife", "knives", "shot", "shooting", "stabbed", "fight",
                                                          "beating", "beaten", "beat up", "attack", "assault", "threat", "murder", "chor", "loot"],
                      "teams": ["Police"]},
    "women_safety":  {"label": "Women safety", "kw": ["harass", "stalking", "stalker", "molest", "eve teasing", "domestic violence",
                                                      "husband beat", "beats me", "dowry", "women safety", "woman being", "girl being"],
                      "teams": ["Police", "Women Helpline"]},
    "child":         {"label": "Child in danger", "kw": ["child missing", "missing child", "lost child", "child is missing", "child abuse",
                                                        "child labour", "kid missing", "son is missing", "daughter is missing", "child in danger"],
                      "teams": ["Police", "Child Helpline"]},
    "electric":      {"label": "Electric hazard", "kw": ["live wire", "electric wire", "wire fallen", "wire has fallen", "power line",
                                                         "transformer", "electric shock", "electrocuted", "sparking", "current laga"],
                      "teams": ["Electricity Board"]},
    "trafficking":   {"label": "Trafficking", "kw": ["traffick", "being sold", "selling children", "selling girls", "forced labour",
                                                     "forced labor", "bonded labour", "bonded labor", "taskari", "तस्करी", "बेचा जा रहा",
                                                     "बेची जा रही", "बंधुआ"], "teams": ["Police"]},
    "kidnapping":    {"label": "Kidnapping / abduction", "kw": ["kidnap", "abduct", "forced into a car", "pulled into a van", "pulled into a car", "into a van", "into the van", "into a car", "into the car",
                                                                "taken away by force", "utha le gaye", "utha liya", "agwa", "अपहरण", "उठा ले गए",
                                                                "उठा लिया"], "teams": ["Police"]},
    "missing_person": {"label": "Missing person", "kw": ["missing", "not come home", "not returned home", "has not returned", "can't find",
                                                        "cannot find", "laapata", "lapata", "gayab", "लापता", "गायब", "नहीं मिल रहा",
                                                        "नहीं मिल रही"], "teams": ["Police"]},
    "sexual_violence": {"label": "Sexual violence", "kw": ["rape", "raped", "sexual assault", "sexually assault", "sexual abuse", "balatkar",
                                                          "बलात्कार", "यौन"], "teams": ["Police", "Women Helpline", "Ambulance (EMS)"]},
    "suspicious":    {"label": "Suspicious activity / public threat", "kw": ["suspicious", "unattended bag", "abandoned bag", "bomb",
                                                                            "explosive", "shots fired", "gunshot", "firing", "riot", "mob",
                                                                            "stampede", "terror", "sandigdh", "laawaris", "लावारिस",
                                                                            "संदिग्ध", "बम", "दंगा", "भगदड़"], "teams": ["Police"]},
    "other":         {"label": "Other / Unclassified emergency", "kw": [], "teams": ["Police"]},
}
SUBSUMES = {"child": ["missing_person"], "kidnapping": ["crime"], "trafficking": ["crime", "missing_person"],
            "sexual_violence": ["crime", "women_safety"]}       # a more specific type replaces a general one
HINDI_PROBLEM_WORDS = {   # Devanagari words, as the browser writes Hindi speech
    "fire": ["आग", "धुआं", "धुआँ", "जल रहा", "जल रही"], "gas_leak": ["गैस", "सिलेंडर"],
    "medical": ["बेहोश", "सांस", "साँस", "खून", "घायल", "चोट", "दिल का दौरा", "गिर गए", "गिर गया"],
    "road_accident": ["एक्सीडेंट", "दुर्घटना", "टक्कर"], "collapse": ["मकान गिर", "इमारत गिर", "दीवार गिर", "मलबा"],
    "flood": ["बाढ़", "डूब", "पानी भर"], "crime": ["चोर", "चोरी", "लूट", "मारपीट", "चाकू", "बंदूक", "हमला"],
    "women_safety": ["छेड़", "परेशान कर", "घरेलू हिंसा", "पति मार"], "child": ["बच्चा गायब", "बच्ची गायब", "बच्चा लापता", "बच्चा खो"],
    "electric": ["तार गिर", "करंट", "बिजली का तार", "ट्रांसफार्मर"],
}
for _pid, _hi_words in HINDI_PROBLEM_WORDS.items():
    PROBLEMS[_pid]["kw"] = PROBLEMS[_pid]["kw"] + _hi_words
INJURY_WORDS = ["घायल", "खून", "बेहोश", "फंस", "फँस", "injured", "hurt", "bleeding", "unconscious", "not breathing", "trapped", "wounded", "burn", "electric shock",
                "electrocuted", "stabbed", "shot", "not moving", "drowning"]
CALLBACK_MIN = {"Gas Emergency": 15, "Electricity Board": 20, "Women Helpline": 5, "Child Helpline": 5}


def rule_problems(text: str) -> list[tuple[str, str]]:
    """Every problem category whose keywords appear, with the word that decided it."""
    t = " " + text.lower() + " "
    for neg in NEGATIONS:                                  # "nobody hurt" does not make it a medical call
        t = t.replace(neg, " ")
    found = []
    for pid, p in PROBLEMS.items():
        hit = next((k for k in p["kw"] if k in t), None)
        if hit:
            found.append((pid, hit))
    ids = {pid for pid, _ in found}
    drop = {d for pid in ids for d in SUBSUMES.get(pid, [])}
    return [(pid, hit) for pid, hit in found if pid not in drop]


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
@traceable(run_type="llm", name="Router LLM (constrained)", process_inputs=lambda i: {"text": i.get("text"), "model": i.get("model")})
def llm_problems(text: str, api_key: str, model: str) -> list[str]:
    """Constrained LLM routing: the model may only pick ids from PROBLEMS; anything else is thrown away."""
    prompt = ("You route emergency calls to departments. Choose ALL problem ids that apply, ONLY from this list:\n"
              + "\n".join(f"- {pid}: {p['label']}" for pid, p in PROBLEMS.items())
              + '\nReply with JSON only: {"problems": ["id", ...]}. Use an empty list if it is not an emergency.\n'
              f"Call: {text!r}")
    client = Groq(api_key=api_key)
    for m in (model, GROQ_FALLBACK_MODEL):
        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in m else {}
            r = client.chat.completions.create(model=m, temperature=0, max_completion_tokens=1024 if extra else 200,
                                               response_format={"type": "json_object"},
                                               messages=[{"role": "user", "content": prompt}], **extra)
            ids = parse_json_response(r.choices[0].message.content or "").get("problems", [])
            return [i for i in ids if i in PROBLEMS]          # the constraint: unknown ids are dropped
        except Exception:
            continue
    return []


@traceable(run_type="chain", name="Router agent")
def route_problem(text: str, triage: dict, hint: str | None = None) -> dict:
    """Router agent: problem categories -> departments (lead + support), each with a reason."""
    rules = rule_problems(text)
    method, picked = "rules", [p for p, _ in rules]
    if use_groq and groq_key and Groq is not None:
        llm = llm_problems(text.strip(), groq_key, groq_model)
        if llm:
            method, picked = "Groq AI (constrained to the fixed list)", list(dict.fromkeys(llm + picked))
    picked.sort(key=lambda pid: pid == "medical")             # a specific problem (crime, fire…) leads; medical supports
    if hint and hint in picked:
        picked.remove(hint)
    if hint:
        picked.insert(0, hint)
    why = dict(rules)
    teams, order = {}, []
    for pid in picked:
        for d in PROBLEMS[pid]["teams"]:
            if d not in teams:
                teams[d] = {"dept": d, "role": "lead" if not teams else "support",
                            "why": f"{PROBLEMS[pid]['label'].lower()}" + (f" (caller said “{why[pid]}”)" if pid in why else "")}
                order.append(d)
    t = " " + text.lower() + " "
    for neg in NEGATIONS:                                  # "nobody hurt" is not an injury
        t = t.replace(neg, " ")
    injured = any(w in t for w in INJURY_WORDS) or int(triage.get("severity", 1)) >= 5
    if injured and "Ambulance (EMS)" not in teams and (picked or triage.get("is_emergency")):
        teams["Ambulance (EMS)"] = {"dept": "Ambulance (EMS)", "role": "support", "why": "someone may be injured"}
        order.append("Ambulance (EMS)")
    review = []
    if not picked:
        if triage.get("is_emergency"):
            fallback = {"Fire": "fire", "Rescue": "collapse", "Medical": "medical", "Police": "crime"}.get(triage.get("incident_type"))
            if fallback:
                for d in PROBLEMS[fallback]["teams"]:
                    if d not in teams:
                        teams[d] = {"dept": d, "role": "lead" if not teams else "support", "why": f"triage said {triage['incident_type'].lower()}"}
                        order.append(d)
            else:
                teams["Police"] = {"dept": "Police", "role": "lead", "why": "problem unclear: the police control room takes it"}
                order.append("Police")
            review.append("the problem did not match any known category")
    if not triage.get("location"):
        review.append("no location yet: call the reporter back")
    if triage.get("needs_review"):
        review.append("triage is unsure about this call")
    return {"problems": picked, "labels": [PROBLEMS[p]["label"] for p in picked], "teams": [teams[d] for d in order],
            "method": method, "injured": injured, "review": review}


@traceable(run_type="tool", name="Department agents",
           process_outputs=lambda o: {"assignments": [{k: v for k, v in a.items() if k != "route"} for a in (o or [])]})
def dispatch_departments(teams: list[dict], location: str | None, ref: str) -> list[dict]:
    """Department agents: each sends its nearest free unit, or opens a ticket and calls back."""
    graph, out = None, []
    for tm in teams:
        info = DEPARTMENTS[tm["dept"]]
        a = {**tm, "icon": info["icon"], "line": info["line"], "unit": None, "eta": None, "route": None, "ticket": None}
        if info["fleet"]:
            if not location:
                a["status"] = "waiting for the location"
            else:
                graph = graph or add_closures(make_city_graph(), closure_rows, closure_cols)
                best = None
                for v in VEHICLES:
                    if v["Type"] != info["fleet"] or v["Vehicle ID"] in st.session_state.used_vehicle_ids:
                        continue
                    try:
                        m = route_minutes(graph, v["Station"], LANDMARKS[location])
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    if best is None or m < best[1]:
                        best = (v, m)
                if best:
                    v, m = best
                    st.session_state.used_vehicle_ids.add(v["Vehicle ID"])
                    a.update(unit=v["Vehicle ID"], eta=max(1.0, m), route=route_nodes(graph, v["Station"], LANDMARKS[location]),
                             status="on the way")
                else:
                    a["status"] = "queued: no free unit"
        else:
            st.session_state.ticket_no = st.session_state.get("ticket_no", 0) + 1
            a.update(ticket=f"{info['code']}-{st.session_state.ticket_no:03d}", eta=CALLBACK_MIN.get(tm["dept"], 10),
                     status=f"notified, call-back within {CALLBACK_MIN.get(tm['dept'], 10)} min")
        a["ref"], a["created"] = ref, ist_time()
        out.append(a)
    return out


@traceable(run_type="chain", name="Dispatch pipeline",
           process_inputs=lambda i: {"text": i.get("text"), "ref": i.get("ref"), "source": i.get("source"), "hint": i.get("hint")},
           process_outputs=lambda o: {"problems": o["route"]["labels"], "teams": [t["dept"] for t in o["route"]["teams"]],
                                      "review": o["route"]["review"]})
def run_agents(text: str, triage: dict, ref: str, source: str, voice: dict | None = None, hint: str | None = None,
               dispatch: bool = True) -> dict:
    """The whole pipeline, recorded step by step so anyone can see who decided what."""
    steps = []
    def step(icon, agent, did, ms):
        steps.append({"icon": icon, "agent": agent, "did": did, "ms": round(ms, 1)})
    t0 = time.perf_counter()
    if voice and voice.get("text"):
        step("🎙️", "Communication Officer agent (intake)", f"Voice recorded ({voice.get('duration') or 0:.1f} s) and written by {voice.get('engine')}: “{text[:120]}”", 0)
    else:
        step("🎧", "Communication Officer agent (intake)", f"{source} report received: “{text[:120]}”", 0)
    step("🧭", "Triage agent (constrained)",
         f"{triage.get('incident_type')}, severity {triage.get('severity')} of 5, location {triage.get('location') or 'unknown'}, "
         f"people {triage.get('people') or 'not stated'}. Decided by {triage.get('triage_method', 'rules')}. "
         "Allowed outputs only: 4 types, severity 1-5, a known landmark or 'unknown'.", 0)
    t1 = time.perf_counter()
    route = route_problem(text, triage, hint)
    step("🔀", "Router agent (constrained)",
         ("Problem: " + (", ".join(route["labels"]) or "unclear") + ". Teams: "
          + ", ".join(f"{DEPARTMENTS[t_['dept']]['icon']} {t_['dept']} ({t_['role']}: {t_['why']})" for t_ in route["teams"])
          + f". Method: {route['method']}."), (time.perf_counter() - t1) * 1000)
    assignments = []
    if dispatch:
        t2 = time.perf_counter()
        assignments = dispatch_departments(route["teams"], triage.get("location"), ref)
        for a in assignments:
            did = (f"Sent {a['unit']}, about {a['eta']:.0f} min away" if a.get("unit")
                   else f"Ticket {a['ticket']}: {a['status']}" if a.get("ticket") else a["status"].capitalize())
            step(a["icon"], f"{a['dept']} agent", did, (time.perf_counter() - t2) * 1000 / max(1, len(assignments)))
        st.session_state.setdefault("dept_jobs", []).extend(
            {**{k: v for k, v in a.items() if k != "route"}, "problem": ", ".join(route["labels"]) or "unclear",
             "location": triage.get("location"), "severity": int(triage.get("severity", 1))} for a in assignments)
    step("🧑‍💼", "Supervisor (human in the loop)",
         ("Check needed: " + "; ".join(route["review"])) if route["review"] else "No check needed: every team has its job.", 0)
    run = {"ref": ref, "text": text, "route": route, "assignments": assignments, "steps": steps,
           "total_ms": round((time.perf_counter() - t0) * 1000, 1), "time": ist_time()}
    st.session_state.setdefault("agent_runs", []).append(run)
    return run


def teams_html(assignments: list[dict]) -> str:
    chips = []
    for a in assignments:
        detail = (f"{a['unit']} · {a['eta']:.0f} min" if a.get("unit") else a.get("ticket") or a.get("status", ""))
        chips.append(f"<span style='display:inline-block;border:1px solid rgba(128,128,128,.35);border-radius:999px;padding:4px 10px;"
                     f"margin:2px 4px 2px 0;font-size:.85rem'>{a['icon']} <b>{a['dept']}</b>"
                     f"{' (lead)' if a.get('role') == 'lead' else ''} <span style='opacity:.75'>{detail}</span></span>")
    return "".join(chips)


AGENT_DIAGRAM = """
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#5B6B7A", fillcolor="#F6F8FA"];
  edge [color="#5B6B7A", fontname="Helvetica", fontsize=9];
  caller [label="📞 Citizen\\nspeaks or types", fillcolor="#FBE3E6", color="#D62839"];
  intake [label="🎙️ Intake agent\\nrecord voice\\nspeech → text"];
  triage [label="🧭 Triage agent\\n(constrained)\\ntype · severity 1-5\\nlocation · people"];
  router [label="🔀 Router agent\\n(constrained)\\nproblem → departments"];
  sup [label="🧑‍💼 Supervisor\\nhuman in the loop", fillcolor="#FFF4D6", color="#C98A0C"];
  subgraph cluster_d { label="Department agents"; color="#CBD4DC"; fontname="Helvetica"; fontsize=10;
    pol [label="🚓 Police 100"]; fire [label="🚒 Fire & Rescue 101"]; ems [label="🚑 Ambulance 108"]; drf [label="🛟 Disaster 1078"];
    gas [label="🛢️ Gas 1906"]; ele [label="⚡ Electricity 1912"]; wom [label="👩 Women 1091"]; chd [label="🧒 Child 1098"]; }
  track [label="📍 Tracking & feedback\\nto the citizen", fillcolor="#DDF2E7", color="#1F8A5B"];
  caller -> intake -> triage -> router;
  router -> pol; router -> fire; router -> ems; router -> drf; router -> gas; router -> ele; router -> wom; router -> chd;
  {pol fire ems drf} -> track [label="units"]; {gas ele wom chd} -> track [label="tickets"];
  triage -> sup [style=dashed, label="unsure"]; router -> sup [style=dashed, label="unclear"];
}
"""


# ----------------------- Emergency call: a spoken conversation with a call-taker agent -----------------------
# Like an ERSS-112 call-taker: ask what happened and where, send help as soon as both are known, keep asking the
# questions of that problem's protocol (breathing? trapped? weapon?), raise urgency when the answers demand it,
# alert extra departments when new problems appear, and give life-saving instructions while help is on the way.
CALL_Q = {   # slot -> (English, Hindi)
    "problem": ("112, what is the location of your emergency? Then tell me exactly what happened.",
                "112, आपकी आपात स्थिति की जगह क्या है? फिर ठीक-ठीक बताइए क्या हुआ।"),
    "problem_again": ("Please tell me in your own words what is happening. Is anyone hurt or in danger right now?",
                      "कृपया अपने शब्दों में बताइए क्या हो रहा है। क्या अभी कोई घायल है या खतरे में है?"),
    "location": ("Where are you? Tell me the nearest landmark, like a hospital, station, market, temple or mall.",
                 "आप कहाँ हैं? पास की कोई जगह बताइए, जैसे अस्पताल, स्टेशन, बाज़ार, मंदिर या मॉल।"),
    "location_again": ("I could not find that place. Which landmark is closest? You can also pick it from the list.",
                       "मुझे वह जगह नहीं मिली। सबसे पास कौन सी जगह है? आप सूची से भी चुन सकते हैं।"),
    "breathing": ("Is the injured person awake and breathing?", "क्या घायल व्यक्ति होश में है और साँस ले रहा है?"),
    "bleeding": ("Is anyone bleeding heavily?", "क्या किसी का बहुत खून बह रहा है?"),
    "trapped": ("Is anyone trapped or unable to get out?", "क्या कोई फँसा हुआ है या बाहर नहीं निकल पा रहा?"),
    "spreading": ("Is the fire spreading, and is anyone still inside?", "क्या आग फैल रही है, और क्या कोई अभी अंदर है?"),
    "caller_safe": ("Are you in a safe place right now?", "क्या आप अभी सुरक्षित जगह पर हैं?"),
    "weapon": ("Does anyone have a weapon, like a knife or a gun?", "क्या किसी के पास हथियार है, जैसे चाकू या बंदूक?"),
    "description": ("Can you describe the person or vehicle: clothes, direction they went, vehicle number?",
                    "क्या आप व्यक्ति या गाड़ी के बारे में बता सकते हैं: कपड़े, किस तरफ गए, गाड़ी नंबर?"),
    "child_desc": ("How old is the child, what are they wearing, and where were they last seen?",
                   "बच्चे की उम्र क्या है, उसने क्या पहना है, और आखिरी बार कहाँ देखा गया?"),
    "touching_wire": ("Is anyone touching the wire or hurt by it?", "क्या कोई तार को छू रहा है या उससे घायल है?"),
    "people": ("How many people are hurt or need help?", "कितने लोग घायल हैं या मदद चाहिए?"),
    "name": ("What is your name?", "आपका नाम क्या है?"),
    "phone": ("What number can the team call you back on, if the line drops?", "अगर कॉल कट जाए तो टीम आपको किस नंबर पर वापस कॉल करे?"),
}
PROTOCOL = {   # problem -> the questions a call-taker asks next, in order
    "medical": ["breathing", "bleeding", "people"], "road_accident": ["breathing", "bleeding", "trapped", "people"],
    "collapse": ["trapped", "breathing", "people"], "flood": ["trapped", "people"], "fire": ["spreading", "trapped", "people"],
    "gas_leak": ["caller_safe", "people"], "crime": ["caller_safe", "weapon", "description", "people"],
    "women_safety": ["caller_safe", "weapon", "description"], "child": ["child_desc", "description"], "electric": ["touching_wire", "people"],
    "trafficking": ["caller_safe", "description", "people"], "kidnapping": ["description", "people"],
    "missing_person": ["description", "people"], "sexual_violence": ["caller_safe", "bleeding", "description"],
    "suspicious": ["caller_safe", "description"], "other": ["caller_safe", "people"],
}
YES_NO = ["breathing", "bleeding", "trapped", "spreading", "caller_safe", "weapon", "touching_wire"]
YES_WORDS = ["yes", "yeah", "yep", "haan", "han", "haa", "ji", "sure", "correct", "right", "हाँ", "हां", "जी", "हा"]
NO_WORDS = ["no", "nope", "nahi", "nahin", "not", "isn't", "isnt", "doesn't", "can't", "नहीं", "नही", "ना"]
UNSURE = ["don't know", "dont know", "not sure", "no idea", "pata nahi", "पता नहीं", "मालूम नहीं"]
SAY = {   # fixed phrases (English, Hindi)
    "sending": ("Dispatch confirmed: {teams}. {unit} is about {eta} minutes away. Stay on the line.",
                "डिस्पैच की पुष्टि हो गई: {teams}। {unit} लगभग {eta} मिनट में पहुँचेगी। लाइन पर बने रहिए।"),
    "extra": ("I have also alerted {teams}.", "मैंने {teams} को भी सूचित कर दिया है।"),
    "upgraded": ("This is now marked life-threatening and the team has been told.", "इसे अब जानलेवा माना गया है और टीम को बता दिया गया है।"),
    "queued": ("All suitable units are busy; you are first in the queue.", "सभी गाड़ियाँ व्यस्त हैं; आप कतार में सबसे आगे हैं।"),
    "done": ("Help is on the way. Tell me if anything changes, or press End call.",
             "मदद रास्ते में है। अगर कुछ बदले तो बताइए, या कॉल खत्म करें दबाइए।"),
    "not_heard": ("Sorry, I did not catch that. Please say it again, or type it.", "माफ़ कीजिए, मैं सुन नहीं पाया। कृपया फिर से बोलिए या लिखिए।"),
    "ended": ("Thank you. Your reference is {ref}. Help is on the way.", "धन्यवाद। आपका संदर्भ {ref} है। मदद रास्ते में है।"),
}
INSTRUCT = {   # life-saving instructions triggered by an answer (English, Hindi)
    "not_breathing": ("If you are trained, start CPR now: push hard and fast in the centre of the chest. If not, stay with them; "
                      "the ambulance crew is coming.", "अगर आपको आता है तो अभी CPR शुरू करें: छाती के बीच में तेज़ और ज़ोर से दबाएँ। "
                      "नहीं तो उनके साथ रहें; एम्बुलेंस आ रही है।"),
    "bleeding": ("Press firmly on the wound with a clean cloth and keep pressing.", "घाव पर साफ़ कपड़े से ज़ोर से दबाएँ और दबाए रखें।"),
    "gas": ("Do not switch anything on or off and no flames. Leave the building and wait outside.",
            "कोई भी स्विच चालू या बंद न करें, कोई आग नहीं। इमारत से बाहर निकलें और बाहर इंतज़ार करें।"),
    "fire": ("Get everyone out, stay low under smoke, and do not go back inside.", "सबको बाहर निकालें, धुएँ में नीचे झुककर चलें, और वापस अंदर न जाएँ।"),
    "unsafe": ("Move somewhere safe and lock the door if you can. Do not confront anyone.",
               "किसी सुरक्षित जगह पर जाएँ और हो सके तो दरवाज़ा बंद कर लें। किसी से उलझें नहीं।"),
    "wire": ("Do not touch the wire or anyone touching it. Keep everyone at least 10 metres away.",
             "तार या उसे छूते किसी व्यक्ति को न छुएँ। सबको कम से कम 10 मीटर दूर रखें।"),
    "trapped": ("Do not go in yourself. Keep talking to them so rescuers can find them.",
                "खुद अंदर न जाएँ। उनसे बात करते रहें ताकि बचाव दल उन्हें ढूँढ सके।"),
    "flood": ("Move to higher ground and do not walk or drive through moving water.", "ऊँची जगह पर जाएँ और बहते पानी में न चलें, न गाड़ी चलाएँ।"),
}
PROBLEM_HI = {"fire": "आग", "gas_leak": "गैस लीक", "medical": "मेडिकल इमरजेंसी", "road_accident": "सड़क दुर्घटना",
              "collapse": "इमारत गिरना", "flood": "बाढ़", "crime": "अपराध", "women_safety": "महिला सुरक्षा",
              "child": "बच्चा खतरे में", "electric": "बिजली का खतरा", "trafficking": "तस्करी", "kidnapping": "अपहरण",
              "missing_person": "लापता व्यक्ति", "sexual_violence": "यौन हिंसा", "suspicious": "संदिग्ध गतिविधि / खतरा",
              "other": "अन्य आपात स्थिति"}
TYPE_OF = {"fire": "Fire", "gas_leak": "Fire", "collapse": "Rescue", "flood": "Rescue", "electric": "Rescue", "medical": "Medical",
           "road_accident": "Medical", "crime": "Police", "women_safety": "Police", "child": "Police", "trafficking": "Police",
           "kidnapping": "Police", "missing_person": "Police", "sexual_violence": "Police", "suspicious": "Police", "other": "Police"}


def _l(pair: tuple[str, str], hindi: bool) -> str:
    return pair[1] if hindi else pair[0]


LOC_CUES = r"(?:near(?:by)?|at|in|on|to|opposite|opp\.?|behind|beside|next to|outside|inside|close to|in front of|" \
           r"(?:the\s+)?(?:location|address|place) is|i am at|i'm at|we are at|i am near|we are near)"
LOC_STOP = {"and", "but", "where", "there", "please", "help", "someone", "somebody", "people", "person", "my", "his", "her", "he", "she",
            "they", "it", "is", "are", "was", "were", "has", "have", "had", "because", "so", "with", "who", "which", "that", "fire",
            "smoke", "accident", "injured", "trapped", "bleeding", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "hai", "hain", "ho", "raha", "rahi", "lagi", "aag", "hua", "hui", "a", "an", "come", "send", "quickly",
            "है", "हैं", "में", "आग", "लगी", "हुआ", "जल्दी", "go", "come", "call", "know", "tell", "talk", "be", "do", "see",
            "take", "get", "me", "us", "you", "them", "him", "to", "of", "want", "need"}
HI_POST = r"(?:ke paas|ke pass|ke saamne|ke samne|ke peeche|ke piche|ke andar|mein|me|par|pe|के पास|के सामने|के पीछे|के अंदर|में|पर)"
NOT_A_PLACE = {"mere", "mera", "meri", "hamare", "hamara", "yahan", "wahan", "ghar", "मेरे", "मेरा", "हमारे", "यहाँ", "यहां", "वहाँ",
               "घर", "kitchen", "room", "home", "house", "my", "our", "this", "here", "there"}


def extract_location_text(text: str, pending: str | None = None) -> str | None:
    """The caller's location exactly as said (no forced match to a list). Returns None when no place was given."""
    t = " ".join(text.split())

    def tidy(words: list[str]) -> str | None:
        words = [w for w in words if w]
        while words and words[0].lower() in ("the", "a", "an"):
            words = words[1:]
        place = " ".join(words).strip(" ,.;:-")
        if not place or all(w.lower().strip(",.") in NOT_A_PLACE | LOC_STOP for w in place.split()):
            return None
        return place
    # English cue: "near India Gate, New Delhi", "at Sector 62 Noida", "my address is …"
    m = re.search(r"\b" + LOC_CUES + r"\s+(.+)", t, flags=re.I)
    if m:
        out, toks = [], m.group(1).split()
        for k, tok in enumerate(toks):
            bare = tok.strip(",.!?;:").lower()
            if bare in LOC_STOP:
                if out:
                    break
                continue                                  # skip leading filler ("to talk to …")
            out.append(tok.rstrip(".!?;:"))
            if tok[-1:] in ".!?;" or len(out) >= 8:
                break
            if tok.endswith(",") and k + 1 < len(toks) and toks[k + 1].strip(",.").lower() in LOC_STOP:
                break
        place = tidy(out)
        if place:
            return place
    # Hindi / Hinglish: "India Gate ke paas", "मंदिर के पास", "Sector 18 mein"
    m = re.search(r"((?:[^\s,।.]+\s+){0,3}[^\s,।.]+)\s+" + HI_POST + r"(?=\s|$|[,।.])", t, flags=re.I)
    if m:
        words = [w for w in m.group(1).split() if w.lower() not in LOC_STOP]
        place = tidy(words[-4:])
        if place:
            return place
    # answering "where are you?": the whole short answer is the place
    if pending in ("location", "location_again") and yes_no(t) is None:
        # keep only the parts that are not about the problem: "aag lagi hai. Red Fort. Mandir" -> "Red Fort, Mandir"
        parts = [p.strip(" ,") for p in re.split(r"[.!?।]", t) if p.strip(" ,") and not rule_problems(p)]
        if not parts:
            return None                                   # the answer was about the problem, not a place
        t2 = ", ".join(parts)
        words = re.sub(r"^(?:it is|it's|i am|i'm|we are|the location is|my location is|location|address|पता|जगह)\s+", "", t2, flags=re.I)
        words = words.strip(" .,!?").split()
        if any(w.lower().strip(",.") in LOC_STOP for w in words):
            return None                                   # "a strange smell and people are coughing" is not a place
        if 1 <= len(words) <= 10:
            return tidy(words)
    return None


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
def llm_location(text: str, api_key: str, model: str) -> str | None:
    """Optional AI help: copy the place name exactly as written. Anything not found word-for-word in the text is rejected."""
    try:
        r = Groq(api_key=api_key).chat.completions.create(
            model=model, temperature=0, max_completion_tokens=120, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": "Copy the location of the emergency EXACTLY as written in this text (same words, "
                       'same spelling), or null if no place is given. Reply as JSON {"location": "..."}. Text: ' + repr(text)}])
        loc = (parse_json_response(r.choices[0].message.content or "").get("location") or "").strip()
        return loc if loc and loc.lower() in text.lower() else None        # never invent a place
    except Exception:
        return None


def place_on_demo_map(place: str) -> str | None:
    """For the simulation only: put the caller's place on the demo grid. A known landmark named in full keeps its spot;
    anything else gets a stable simulated spot. The caller's words are never replaced."""
    low = place.lower()
    known = next((n for n in sorted(DEMO_LANDMARKS, key=len, reverse=True) if n.lower() == low or n.lower() in low), None)
    if place not in LANDMARKS:
        if known:
            LANDMARKS[place] = LANDMARKS[known]
        else:
            h = int(hashlib.md5(low.encode("utf-8")).hexdigest(), 16)
            LANDMARKS[place] = (h % GRID_SIZE, (h // GRID_SIZE) % GRID_SIZE)
    return known


def ensure_place(call: dict | None) -> None:
    """Re-register the caller's place on the demo map (needed after every rerun)."""
    if call and call.get("slots", {}).get("location") and call["slots"].get("location_xy"):
        LANDMARKS.setdefault(call["slots"]["location"], tuple(call["slots"]["location_xy"]))


def question_text(call: dict, q: str) -> str:
    """The words for question q (the call-back question confirms the number the system already shows)."""
    sl, hindi = call["slots"], call["hindi"]
    if q == "phone" and sl.get("phone_src") == "ANI" and sl.get("phone"):
        num = f"{sl['phone'][:5]} {sl['phone'][5:]}"
        return (f"मेरी स्क्रीन पर आपका नंबर {num} दिख रहा है। क्या टीम आपको इसी नंबर पर वापस कॉल कर सकती है?" if hindi else
                f"My screen shows your number as {num}. Can the team call you back on this number?")
    if q == "phone" and "phone" in call["asked"]:
        return "किस नंबर पर टीम आपको कॉल करे?" if hindi else "Which number should the team call you on?"
    if q == "location_pick":
        opts = "; ".join(f"{k + 1}: {o['name'][:70]}" for k, o in enumerate(sl.get("geo_options", [])))
        return (f"मुझे नक्शे पर कुछ जगहें मिलीं: {opts}। कौन सी सही है? नंबर बताइए।" if hindi else
                f"I found a few places on the map: {opts}. Which one is it? Say the number.")
    if q == "location_landmark":
        return (f"मुझे “{sl.get('location')}” नक्शे पर नहीं मिला। पास की कोई जानी-पहचानी जगह, मोहल्ला, ज़िला या राज्य बताइए।" if hindi else
                f"I could not find “{sl.get('location')}” on the map. Please tell me a nearby landmark, locality, district or state.")
    if q == "location_confirm":
        if sl.get("geo_state") == "one" and sl.get("geo"):
            return (f"मुझे नक्शे पर “{sl['geo']['name'][:90]}” मिला। क्या आपात स्थिति यहीं है?" if hindi else
                    f"I found “{sl['geo']['name'][:90]}” on the map. Is that where the emergency is?")
        return (f"मैंने जगह “{sl.get('location')}” लिखी है। क्या यह सही है?" if hindi else
                f"I have the location as “{sl.get('location')}”. Is that correct?")
    if q == "location" and call.get("loc_rejected"):
        return ("कृपया जगह फिर से, धीरे-धीरे बताइए: इलाका, सड़क या पास की कोई जानी-पहचानी जगह, और शहर।" if hindi else
                "Please tell me the location again, slowly: the area, road or a well-known place nearby, and the city.")
    return _l(CALL_Q[q], hindi)


def yes_no(text: str) -> bool | None:
    t = " " + re.sub(r"[^\w\u0900-\u097F' ]", " ", text.lower()) + " "
    if any(u in t for u in UNSURE):
        return None
    no = any(f" {w} " in t for w in NO_WORDS)
    yes = any(f" {w} " in t for w in YES_WORDS)
    return False if no else True if yes else None


def new_call(hindi: bool) -> dict:
    store = cad_store()
    with store["lock"]:
        store["seq"] += 1
        st.session_state.call_no = store["seq"]
    _call = {"ref": f"112-{st.session_state.call_no:04d}", "hindi": hindi, "turns": [], "slots": {}, "asked": [], "pending": "problem",
            "misses": {}, "dispatched": False, "assignments": [], "severity": 3, "record": None, "ended": False,
            "started": time.time(), "said_instructions": [], "events": [("📞", "Call started", time.time())]}
    _call["created"] = time.time()
    store["incidents"][_call["ref"]] = _call        # visible to the Command Center
    return _call


EVENT_TYPES = {"📞": "Call received", "🔎": "Incident classified", "📍": "Location reported", "✅": "Location verified",
               "🗺️": "Location check", "🗂️": "Incident created", "🚨": "Unit dispatched", "➕": "Unit dispatched",
               "⬆️": "Priority changed", "🔔": "Escalated to dispatcher", "🎧": "Crew contact", "🤝": "Crew contact",
               "🏥": "Patient transported", "📝": "Agency response completed", "🔚": "Incident closed", "📣": "Status update",
               "🔦": "Status update", "⏳": "Status update", "ℹ️": "Information", "🙋": "Caller details", "↩️": "Unit released",
               "🧑‍💼": "Dispatcher action", "🛑": "Unit cancelled", "📡": "Unit recommended", "👍": "Dispatcher approved"}
EVENT_STATUS = {"Unit dispatched": "Dispatched", "Unit arrived": "On scene", "Patient transported": "Transporting",
                "Agency response completed": "Completed", "Incident closed": "Closed", "Unit cancelled": "Cancelled",
                "Unit recommended": "Recommended", "Dispatcher approved": "Approved", "Location verified": "Verified",
                "Unit released": "Completed", "Escalated to dispatcher": "Attention"}


def add_event(call: dict, icon: str, text: str, etype: str | None = None, actor: str | None = None,
              unit: str | None = None, details: str | None = None) -> None:
    """Append-only audit trail: nothing is ever overwritten or removed when a status changes."""
    now = time.time()
    call.setdefault("events", []).append((icon, text, now))           # simple list (transcript, older views)
    if etype is None:
        low = text.lower()
        etype = ("Unit arrived" if " arrived" in low else "Unit released" if "released" in low else
                 "Patient transported" if "transporting" in low or "at central hospital" in low else
                 "Agency response completed" if "atr" in text else EVENT_TYPES.get(icon, "Update"))
    if unit is None:
        m = re.match(r"([A-Z]{3}-\d{2})\b", text)
        unit = m.group(1) if m else None
    if actor is None:
        actor = (unit if unit and etype in ("Unit arrived", "Patient transported", "Agency response completed", "Unit released")
                 else "CAD" if etype in ("Unit dispatched", "Unit recommended", "Priority changed", "Incident created")
                 else "Call-taker AI" if etype in ("Incident classified", "Location reported", "Location check", "Caller details",
                                                   "Escalated to dispatcher", "Status update", "Information")
                 else "Caller" if etype == "Location verified" else "System")
    audit = call.setdefault("audit", [])
    audit.append({"id": f"{call.get('ref', 'CFS')}-E{len(audit) + 1:03d}", "ts": now, "icon": icon, "type": etype,
                  "actor": actor, "unit": unit, "text": text, "status": EVENT_STATUS.get(etype, ""), "details": details})


def agent_say(call: dict, text: str) -> None:
    call["turns"].append({"role": "agent", "text": text, "t": ist_time()})
    call["last_prompt_at"] = time.time()


def crew_say(call: dict, a_: dict, text: str) -> None:
    """A crew member speaks in the call (three-way call: caller, call-taker, crew)."""
    call["turns"].append({"role": "crew", "who": f"{a_['icon']} {a_['unit']} crew", "text": text, "t": ist_time()})
    call["last_prompt_at"] = time.time()


SUSPECT_WORDS = ["i think", "i suspect", "suspect", "maybe", "might be", "may be", "seems", "looks like", "probably", "not sure",
                 "i doubt", "lagta hai", "shayad", "shak", "शायद", "लगता है", "शक"]
HYPO_WORDS = ["what if", "suppose", "for example", "hypothetically", "agar koi", "अगर कोई", "मान लो"]
PAST_WORDS = ["yesterday", "last night", "last week", "last month", "days ago", "weeks ago", "years ago", "kal raat", "pichhle", "कल रात",
              "पिछले", "साल पहले", "दिन पहले"]
NOW_WORDS = ["now", "right now", "happening", "is being", "are being", "currently", "abhi", "अभी", "हो रहा", "हो रही"]
CHILD_WORDS = ["child", "children", "kid", "kids", "boy", "girl", "baby", "son", "daughter", "minor", "bachcha", "bachchi", "bacche",
               "bachchon", "बच्च", "लड़का", "लड़की"]
WOMAN_WORDS = ["woman", "women", "lady", "wife", "girl", "mahila", "aurat", "महिला", "औरत", "लड़की"]


def understand(text: str) -> dict:
    """Interpret the caller's own sentence (the transcript itself is never changed): is it a report, a suspicion,
    a question or a hypothetical; is it happening now; who is at risk."""
    t = " " + text.lower() + " "
    question = text.strip().endswith("?") and bool(re.match(r"\s*(what|how|can|could|where|who|when|is it|should|kya|क्या|कैसे)\b",
                                                           text.strip().lower()))
    hypo = any(w in t for w in HYPO_WORDS)
    suspected = any(w in t for w in SUSPECT_WORDS)
    past = any(w in t for w in PAST_WORDS) and not any(w in t for w in NOW_WORDS)
    subject = ("child" if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in CHILD_WORDS if w.isascii())
               or any(w in t for w in CHILD_WORDS if not w.isascii()) else
               "woman" if any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in WOMAN_WORDS if w.isascii())
               or any(w in t for w in WOMAN_WORDS if not w.isascii()) else None)
    kind = "hypothetical" if hypo else "question" if question else "suspected" if suspected else "reported"
    return {"certainty": kind, "time": "past" if past else "now or unknown", "subject": subject}


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
def llm_classify(text: str, api_key: str, model: str) -> dict:
    """Optional AI reading of the whole sentence. Free-text label, but the operational category must come from the list."""
    cats = ", ".join(PROBLEMS)
    prompt = ("Classify this emergency call. Reply with a JSON object only, with these keys: "
              "label (a short descriptive incident label in English), "
              f"category (exactly one of: {cats}), certainty (one of: reported, suspected, question, hypothetical), "
              "time (one of: now, past, unknown), subject (one of: child, woman, person, none), immediate_danger (true or false). "
              "Never invent details. Text: " + repr(text))
    try:
        r = Groq(api_key=api_key).chat.completions.create(model=model, temperature=0, max_completion_tokens=200,
                                                          response_format={"type": "json_object"},
                                                          messages=[{"role": "user", "content": prompt}])
        d = parse_json_response(r.choices[0].message.content or "")
        out = {"label": str(d.get("label") or "")[:80]}
        out["category"] = d.get("category") if d.get("category") in PROBLEMS else None
        out["certainty"] = d.get("certainty") if d.get("certainty") in ("reported", "suspected", "question", "hypothetical") else None
        out["subject"] = d.get("subject") if d.get("subject") in ("child", "woman") else None
        out["immediate_danger"] = bool(d.get("immediate_danger"))
        return out
    except Exception:
        return {}


def incident_label(call: dict) -> str:
    """A descriptive label in plain words, e.g. "Suspected child trafficking". Never states an allegation as proven."""
    sl = call["slots"]
    probs = sl.get("problems", [])
    if call.get("ai_label") and probs:
        base = call["ai_label"]
    elif probs:
        base = PROBLEMS[probs[0]]["label"]
        if call.get("subject") == "child" and probs[0] in ("trafficking", "kidnapping", "sexual_violence"):
            base = "Child " + base.lower()
    else:
        return "Not yet clear"
    if (call.get("certainty") == "suspected" or probs[0] == "trafficking") and not base.lower().startswith("suspected"):
        base = "Suspected " + base[0].lower() + base[1:]
    elif probs[0] in ("kidnapping", "sexual_violence", "crime") and not base.lower().startswith(("reported", "suspected")):
        base = "Reported " + base[0].lower() + base[1:]
    if call.get("certainty") in ("question", "hypothetical"):
        base += " (" + call["certainty"] + ")"
    if call.get("time") == "past":
        base += " (reported as past)"
    return base


def escalation_reasons(call: dict) -> list[str]:
    """When an authorised human dispatcher must be alerted (AI classification is never proof)."""
    sl, probs, r = call["slots"], call["slots"].get("problems", []), []
    if set(probs) & {"trafficking", "kidnapping", "sexual_violence"} or (call.get("subject") == "child" and probs):
        r.append("safeguarding: child or serious-harm report")
    if "other" in probs:
        r.append("unclassified emergency")
    if call.get("certainty") == "suspected":
        r.append("suspected, not confirmed")
    if sl.get("location") and sl.get("geo_state") in ("none", "unavailable") and sl.get("location_ok"):
        r.append("location confirmed by the caller but not found on the map")
    if "suspicious" in probs:
        r.append("possible public threat")
    return r


# ---- geographic verification (OpenStreetMap). A match is only used after the caller confirms it.
GEO_URL = "https://nominatim.openstreetmap.org/search"


@st.cache_data(show_spinner=False, max_entries=1000, ttl=86400)
def geocode_india(place: str) -> list[dict] | None:
    """Up to 3 matches in India. Returns None when the service cannot be reached (not the same as "no match")."""
    mock = os.getenv("SAHAYAK_GEO_MOCK")                 # tests only: a JSON file {query: [matches]}
    if mock:
        try:
            return json.loads(Path(mock).read_text(encoding="utf-8")).get(place.lower(), [])
        except Exception:
            return None
    import urllib.parse
    import urllib.request
    try:
        q = urllib.parse.urlencode({"q": place, "format": "jsonv2", "countrycodes": "in", "limit": 3, "addressdetails": 0})
        req = urllib.request.Request(f"{GEO_URL}?{q}", headers={"User-Agent": "Sahayak112-training-demo/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        return [{"name": r_["display_name"], "lat": float(r_["lat"]), "lon": float(r_["lon"])} for r_ in rows[:3]]
    except Exception:
        return None


def verify_place(call: dict) -> None:
    """Look the caller's place up on the map. The caller's own words stay the record either way."""
    sl = call["slots"]
    found = geocode_india(sl["location"]) if sl.get("location") else None
    if found is None:
        sl["geo_state"], sl["geo_options"] = "unavailable", []
    elif not found:
        sl["geo_state"], sl["geo_options"] = "none", []
    else:
        sl["geo_state"], sl["geo_options"] = ("one" if len(found) == 1 else "many"), found
        if len(found) == 1:
            sl["geo"] = found[0]
    add_event(call, "🗺️", {"unavailable": "Map service not reachable: location must be confirmed by the caller",
                          "none": "Place not found on the map: asking for a nearby landmark",
                          "one": "Place found on the map: awaiting caller confirmation",
                          "many": f"{len(found or [])} possible places on the map: asking the caller to choose"}[sl["geo_state"]])


def clean_heard(text: str, call: dict) -> str:
    """Remove the call-taker's own words picked up by the microphone (echo) and stuttered repeats."""
    t = " ".join(text.split())
    for turn in call["turns"][-8:]:
        if turn["role"] == "caller":
            continue
        words = re.findall(r"[\w']+", turn["text"].lower())
        for n in (8, 6, 5, 4, 3):                             # remove any 3+ word run of the call-taker's sentences
            for i in range(0, max(0, len(words) - n + 1)):
                frag = r"\b" + r"\W+".join(map(re.escape, words[i:i + n])) + r"\b"
                t = re.sub(frag, " ", t, flags=re.I)
    words = t.split()
    for n in (4, 3, 2, 1):                                    # "yellow cloth yellow cloth yellow cloth" -> "yellow cloth"
        i = 0
        while i + 2 * n <= len(words):
            if [w.lower().strip(",.!?") for w in words[i:i + n]] == [w.lower().strip(",.!?") for w in words[i + n:i + 2 * n]]:
                del words[i + n:i + 2 * n]
            else:
                i += 1
    return " ".join(words).strip(" ,.")


def caller_text(call: dict) -> str:
    return ". ".join(t_["text"] for t_ in call["turns"] if t_["role"] == "caller")


def extract_into(call: dict, text: str) -> list[str]:
    """Fill every slot this answer tells us about; returns what changed (used for the acknowledgement)."""
    sl, got = call["slots"], []
    t = " " + text.lower() + " "
    # the answer to the question that was asked
    pend = call.get("pending")
    if pend in YES_NO:
        yn = yes_no(text)
        if yn is not None:
            sl[pend] = yn; got.append(pend)
    elif pend == "phone":
        digits = re.sub(r"\D", "", text.translate(DEVANAGARI_DIGITS))
        if len(digits) >= 10:
            sl["phone"], sl["phone_src"], sl["phone_ok"] = digits[-10:], "caller", True; got.append("phone")
        elif sl.get("phone_src") == "ANI" and yes_no(text) is True:
            sl["phone_ok"] = True; got.append("phone")           # "yes, this number is fine"
        elif sl.get("phone_src") == "ANI" and yes_no(text) is False:
            sl["phone"], sl["phone_src"] = None, None           # ask for another number next
        elif yes_no(text) is False or any(u in t for u in UNSURE):
            sl["phone"] = "not given"; got.append("phone")
    elif pend in ("description", "child_desc", "name") and len(text.strip()) > 1 and yes_no(text) is not False \
            and (pend != "name" or looks_like_name(text)):
        val = text.strip()
        if pend == "name":
            m_ = re.search(r"(?:my name is|name is|i am|i'm|this is|mera naam|मेरा नाम)\s+([^\d,.!?।]+)", val, flags=re.I)
            words = (m_.group(1) if m_ else val).split()
            words = [w for w in words if w.lower() not in ("है", "hai", "hoon", "हूँ", "हूं", "ji", "जी", "sir")]
            val = " ".join(words[:3]).strip().title() or val.strip()
        sl[pend] = val; got.append(pend)
    # anything else the caller volunteered
    probs = [p for p, _ in rule_problems(text)]
    if probs:
        before = set(sl.get("problems", []))
        sl["problems"] = list(dict.fromkeys(sl.get("problems", []) + probs))
        if set(sl["problems"]) - before:
            got.append("problems")
    if pend == "location_pick":
        low = " " + text.lower() + " "
        pick = next((k for k, words in enumerate([["1", "one", "first", "pehla", "पहला", "एक"], ["2", "two", "second", "doosra", "दूसरा", "दो"],
                                                  ["3", "three", "third", "teesra", "तीसरा", "तीन"]])
                     if any(f" {w} " in low or low.strip().startswith(w) for w in words)), None)
        opts = sl.get("geo_options", [])
        if pick is not None and pick < len(opts):
            sl["geo"], sl["location_ok"] = opts[pick], True
            got.append("location_ok")
            return got
        if yes_no(text) is False or "none" in low or "koi nahi" in low:
            sl["geo_state"] = "none"                      # none of these: ask for a nearby landmark
            return got
    if pend == "location_landmark" and text.strip() and yes_no(text) is None:
        extra = extract_location_text(text, "location") or text.strip()
        place = f"{extra}, {sl.get('location')}" if sl.get("location") and sl["location"].lower() not in extra.lower() else extra
        sl["location"], sl["location_ok"] = place, False
        sl["location_map"] = place_on_demo_map(place)
        sl["location_xy"] = LANDMARKS[place]
        call["geo_retry"] = call.get("geo_retry", 0) + 1
        call["asked"] = [q for q in call["asked"] if q not in ("location_confirm", "location_pick")]
        verify_place(call)
        got.append("location")
        return got
    if pend == "location_confirm":
        yn = yes_no(text)
        if yn is True:
            sl["location_ok"] = True; got.append("location_ok")
        elif yn is False:
            call["loc_rejected"] = True
            sl["location"], sl["location_ok"] = None, False
            call["asked"] = [q for q in call["asked"] if q not in ("location", "location_again", "location_confirm")]
    # the first answer to "what is the location?" may be just a place name
    pend_loc = "location" if pend == "problem" else pend
    if pend == "location_confirm" and yes_no(text) is False:
        rest = re.sub(r"^\s*(no|nope|nahi|nahin|नहीं|ना)[\s,.]*", "", text, flags=re.I)   # "no, it is Sector 18 Noida"
        text_for_loc, pend_loc = (rest, "location") if rest.strip() else (text, None)
    else:
        text_for_loc = text
    place = extract_location_text(text_for_loc, pend_loc)
    if not place and use_groq and groq_key and Groq is not None and (pend in ("location", "location_again") or not sl.get("location")):
        place = llm_location(text.strip(), groq_key, groq_model)
    if not place and not sl.get("location"):        # a demo-map place named in full (a hint, never a replacement)
        place = next((n for n in sorted(DEMO_LANDMARKS, key=len, reverse=True) if n.lower() in text.lower()), None)
    if place and place != sl.get("location") and not (pend == "location_confirm" and yes_no(text) is True):
        sl["location"], sl["location_ok"] = place, False
        sl["location_map"] = place_on_demo_map(place)
        sl["location_xy"] = LANDMARKS[place]
        sl.pop("geo", None)
        verify_place(call)
        call["asked"] = [q for q in call["asked"] if q != "location_confirm"]
        got.append("location")
    people_ctx = re.search(r"(\d+|\b(one|two|three|four|five|six|seven|eight|nine|ten|several|many|kai|bahut)\b|"
                           r"एक|दो|तीन|चार|पांच|पाँच|छह|सात|आठ|नौ|दस|कई|बहुत)\s*(people|persons?|men|women|kids|children|injured|hurt|"
                           r"trapped|inside|log|लोग|आदमी|व्यक्ति|घायल|बच्चे)", t)
    n = count_people(t) if (call.get("pending") == "people" or people_ctx) and not re.search(
        r"\b(minutes?|mins?|km|floor|gate|number|no\.|मिनट|नंबर|गेट|मंज़िल)\b", t) or people_ctx else None
    if n and n != sl.get("people"):
        sl["people"], sl["people_src"] = n, "caller"; got.append("people")
    elif not sl.get("people") and re.search(r"\b(my (father|mother|son|daughter|wife|husband|friend|brother|sister|grandfather|"
                                             r"grandmother)|a (man|woman|boy|girl|child|person)|someone|somebody)\b|"
                                             r"(मेरे पिता|मेरी माँ|मेरा बेटा|मेरी बेटी|एक आदमी|एक औरत|कोई)", t):
        sl["people"], sl["people_src"] = 1, "inferred"
    flags = {"breathing": (["not breathing", "stopped breathing", "unconscious", "not moving", "behosh", "बेहोश", "सांस नहीं", "साँस नहीं"], False),
             "trapped": (["trapped", "stuck inside", "can't get out", "फंस", "फँस"], True),
             "bleeding": (["bleeding", "lot of blood", "खून"], True),
             "spreading": (["spreading", "getting bigger", "फैल"], True),
             "weapon": (["knife", "gun", "pistol", "weapon", "चाकू", "बंदूक"], True)}
    for slot, (words, val) in flags.items():
        if any(w in t for w in words) and sl.get(slot) != val:
            sl[slot] = val
            if slot not in got:
                got.append(slot)
    return got


def call_triage(call: dict) -> dict:
    """Triage the whole conversation, then let the answers raise the severity (never lower it)."""
    text = caller_text(call)
    tri, _ = run_triage(text) if text else ({"severity": 3}, [])
    sl = call["slots"]
    sev = int(tri.get("severity", 3))
    if sl.get("breathing") is False or sl.get("trapped") is True:
        sev = 5
    elif any(sl.get(k) is True for k in ("bleeding", "spreading", "weapon", "touching_wire")) or sl.get("caller_safe") is False:
        sev = max(sev, 4)
    if (sl.get("people") or 0) >= 3 and sev >= 4:
        sev = 5
    if set(sl.get("problems", [])) & {"trafficking", "kidnapping", "sexual_violence"}:
        sev = max(sev, 4)                                 # serious harm to a person: never shown as "injury, stable"
    probs = sl.get("problems", [])
    tri.update(is_emergency=True, severity=sev, location=sl.get("location"), people=sl.get("people"),
               incident_type=TYPE_OF.get(probs[0], tri.get("incident_type", "Medical")) if probs else tri.get("incident_type", "Medical"),
               severity_reason=", ".join(k.replace("_", " ") + (" yes" if v is True else " no") for k, v in sl.items()
                                         if k in YES_NO and v is not None) or tri.get("severity_reason", ""))
    return tri


def _needs(call: dict, slot: str) -> bool:
    c, sl, asked = cfg(), call["slots"], call["asked"]
    if slot == "name":
        return c["name_when"] != "off" and not sl.get("name") and asked.count("name") < 2
    if slot == "phone":
        return (c["ask_phone"] and not sl.get("phone_ok") and sl.get("phone") != "not given" and asked.count("phone") < 2
                and bool(sl.get("name") or "name" in asked or c["name_when"] == "off"))
    return False


def next_question(call: dict) -> str | None:
    sl, asked = call["slots"], call["asked"]
    c = cfg()
    units = [a_ for a_ in call.get("assignments", []) if a_.get("unit")]
    if c["arrival"] == "crew" and any(a_.get("arrived") for a_ in units):
        if not call.get("crew_contact"):
            return "crew_locate"
        for slot in ("name", "phone"):
            if _needs(call, slot) and slot not in asked:
                return slot
        # the crew is with the caller, but the record still needs the key safety facts (at most two questions)
        late = [q for p in sl.get("problems", []) for q in PROTOCOL.get(p, [])
                if q in YES_NO + ["people"] and q not in asked and sl.get(q) is None and not (q == "people" and sl.get("people"))]
        if late and call.get("late_asked", 0) < 2:
            call["late_asked"] = call.get("late_asked", 0) + 1
            return late[0]
        return "anything_else"
    if units and all(a_.get("arrived") for a_ in units):
        # the crew is there: collect who to call back (if still missing), then hand over
        for slot in ("name", "phone"):
            if _needs(call, slot) and slot not in asked:
                return slot
        return "handover_confirm"
    if call["dispatched"] and c["name_when"] == "early":
        for slot in ("name", "phone"):
            if _needs(call, slot) and slot not in asked:
                return slot
    if sl.get("location") and not sl.get("location_ok") and sl.get("problems"):
        if sl.get("geo_state") == "many" and call["asked"].count("location_pick") < 2:
            return "location_pick"
        if sl.get("geo_state") == "none" and call.get("geo_retry", 0) < 1 and "location_landmark" not in asked:
            return "location_landmark"
        if "location_confirm" not in asked:
            return "location_confirm"
    if not sl.get("problems"):
        return "problem_again" if "problem_again" not in asked and call["turns"] and any(t_["role"] == "caller" for t_ in call["turns"]) else (
            None if "problem_again" in asked else "problem")
    if not sl.get("location"):
        return "location" if "location" not in asked else ("location_again" if "location_again" not in asked else None)
    if not sl.get("location_ok") and sl.get("geo_state") == "many" and call["asked"].count("location_pick") < 2:
        return "location_pick"
    if not sl.get("location_ok") and sl.get("geo_state") == "none" and call.get("geo_retry", 0) < 1 \
            and "location_landmark" not in asked:
        return "location_landmark"
    if not sl.get("location_ok") and "location_confirm" not in asked:
        return "location_confirm"
    for p in sl["problems"]:
        for q in PROTOCOL.get(p, []):
            if q not in asked and sl.get(q) is None and not (q == "people" and sl.get("people")):
                return q
    for slot in ("name", "phone"):
        if _needs(call, slot) and slot not in asked:
            return slot
    return None


ETA_WORDS = ["how long", "how much time", "when will", "where is the", "eta", "are they coming", "kitni der", "kab aayega",
             "kab ayega", "kab aayegi", "कितनी देर", "कब आएगी", "कब आएगा", "कब तक", "कहाँ है", "कहां है"]
CALL_DEFAULTS = {
    "name_when": "early",       # early: right after help is sent | end: after the protocol questions | off
    "ask_phone": True,          # ask for a call-back number after the name
    "arrival": "crew",          # crew: the crew joins the call and finds the caller | confirm: ask "can you see them?" |
                                # stay: never close by itself | auto: close when all arrive
    "announce": True,           # tell the caller in the call when each unit arrives
    "wait": 30,                 # seconds of silence before the call-taker checks on the caller (never hangs up on them)
    "speed": 6.0,               # real seconds per simulated minute of travel (a 7-minute trip = 42 s of talking)
}


DIAL_SERVICES = {   # number -> (service, problem hint used only if the call stays unclear)
    "112": ("Emergency 112", None), "100": ("Police 100", "crime"), "101": ("Fire 101", "fire"), "102": ("Ambulance 102", "medical"),
    "108": ("Ambulance 108", "medical"), "1078": ("Disaster 1078", "flood"), "1091": ("Women helpline 1091", "women_safety"),
    "181": ("Women helpline 181", "women_safety"), "1098": ("Child helpline 1098", "child"),
}
DIAL_GREETING = ("{svc}. You are connected to the 112 emergency response centre. What is the location of your emergency? "
                 "Then tell me exactly what happened.",
                 "{svc}। आप 112 आपातकालीन सेवा से जुड़ गए हैं। आपात स्थिति की जगह क्या है? फिर ठीक-ठीक बताइए क्या हुआ।")
DIALER_CSS = """
<style>
.st-key-phone{max-width:380px;margin:4px auto 0;background:linear-gradient(180deg,#18212b,#0e141a);border:1px solid rgba(128,128,128,.35);
  border-radius:40px;padding:20px 24px 24px;box-shadow:0 18px 40px -18px rgba(0,0,0,.6)}
.st-key-phone, .st-key-phone p, .st-key-phone label{color:#E6ECF2}
[class*="st-key-dial_k_"] button{border-radius:999px !important;height:64px !important;font-size:1.55rem !important;font-weight:600 !important;
  background:rgba(255,255,255,.08) !important;border:1px solid rgba(255,255,255,.12) !important;color:#fff !important}
[class*="st-key-dial_k_"] button:hover{background:rgba(255,255,255,.16) !important}
[class*="st-key-dial_k_"] button p{font-size:1.6rem !important;font-weight:500 !important}
.st-key-dial_call button p{font-size:1.5rem !important}
.st-key-dial_back button p, .st-key-dial_clear button p{font-size:1.35rem !important}
.st-key-dial_call button{background:#1F9D5B !important;border-color:#1F9D5B !important;color:#fff !important;border-radius:999px !important;
  height:64px !important;font-size:1.15rem !important;font-weight:700 !important}
.st-key-dial_back button, .st-key-dial_clear button{border-radius:999px !important;height:64px !important;background:transparent !important;
  border:0 !important;color:#cfd8e3 !important;font-size:1.2rem !important}
.st-key-ring_cancel button, .st-key-end_call button{background:#D62839 !important;border-color:#D62839 !important;color:#fff !important;
  border-radius:999px !important;font-weight:700 !important}
.dial-screen{text-align:center;padding:6px 0 12px}
.dial-screen .num{font-size:2.6rem;font-weight:300;letter-spacing:.12em;color:#fff;min-height:3.2rem;font-variant-numeric:tabular-nums}
.dial-screen .svc{font-size:.9rem;color:#9fb0c2;min-height:1.3rem}
.ring{display:grid;place-items:center;padding:26px 0 18px}
.ring .bub{width:110px;height:110px;border-radius:50%;background:#1C5D99;display:grid;place-items:center;font-size:2.6rem;position:relative}
.ring .bub:before,.ring .bub:after{content:"";position:absolute;inset:0;border-radius:50%;border:2px solid #4f8fd6;animation:rp 1.6s ease-out infinite}
.ring .bub:after{animation-delay:.8s}
@keyframes rp{from{transform:scale(1);opacity:.7}to{transform:scale(1.9);opacity:0}}
.ring .t1{margin-top:22px;font-size:1.5rem;color:#fff;font-weight:600}.ring .t2{color:#9fb0c2}
.callbar{display:flex;align-items:center;gap:12px;background:linear-gradient(90deg,#0f3d2a,#14532d);color:#fff;border-radius:16px;
  padding:10px 16px;margin-bottom:8px}
.callbar .dot{width:10px;height:10px;border-radius:50%;background:#4ade80;animation:cb 1.2s infinite}
@keyframes cb{50%{opacity:.25}}
.callbar b{font-size:1.05rem}.callbar span{opacity:.85;font-size:.9rem}
@media (prefers-reduced-motion:reduce){.ring .bub:before,.ring .bub:after,.callbar .dot{animation:none}}
</style>
"""


def cfg() -> dict:
    """The call settings chosen in "⚙️ Call settings" (defaults above)."""
    return dict(CALL_DEFAULTS)                     # fixed, recommended behaviour (no settings panel)
CALL_Q.update({
    "crew_locate": ("Please tell the crew exactly where you are: a gate, shop, floor, or what you are wearing.",
                    "कृपया टीम को ठीक-ठीक बताइए आप कहाँ हैं: गेट, दुकान, मंज़िल, या आपने क्या पहना है।"),
    "anything_else": ("The crew is with you now. Is there anything else I can help with?",
                      "टीम अब आपके पास है। क्या मैं और किसी चीज़ में मदद कर सकता हूँ?"),
})
CREW_SAY = {
    "hello": ("Hello{name}, this is the {dept} crew from {unit}. We have reached {place}. Where exactly are you? "
              "Tell us a gate, shop, floor, or what you are wearing.",
              "नमस्ते{name}, मैं {unit} से {dept} टीम बोल रहा हूँ। हम {place} पहुँच गए हैं। आप ठीक-ठीक कहाँ हैं? "
              "गेट, दुकान, मंज़िल, या आपने क्या पहना है, बताइए।"),
    "found": ("Thank you{name}. We can see you now. Stay where you are, we are coming to you.",
              "धन्यवाद{name}। हम आपको देख पा रहे हैं। वहीं रहिए, हम आपके पास आ रहे हैं।"),
    "search": ("No problem. Keep your phone on, and wave or shout if you can. We are searching near {place}.",
               "कोई बात नहीं। फ़ोन चालू रखिए, और हो सके तो हाथ हिलाइए या आवाज़ दीजिए। हम {place} के पास ढूँढ रहे हैं।"),
    "joined": ("{unit} is here too and joining the {lead} crew.", "{unit} भी पहुँच गई है और {lead} टीम के साथ जुड़ रही है।"),
}
DONE_WORDS = ["no", "nothing", "that's all", "thats all", "thank you", "thanks", "bas", "nahi", "nahin", "dhanyavad", "shukriya",
              "नहीं", "बस", "धन्यवाद", "शुक्रिया", "कुछ नहीं"]
SAY.update({
    "still_there": ("Are you still on the line? Say anything, or tap the microphone, so I know you are okay.",
                    "क्या आप अभी लाइन पर हैं? कुछ भी बोलिए, ताकि मुझे पता चले आप ठीक हैं।"),
    "kept_open": ("I will keep this line open. The crew is at {place} and is looking for you.",
                  "मैं यह लाइन खुली रखूँगा। टीम {place} पर है और आपको ढूँढ रही है।"),
})


def _name_bit(call: dict) -> str:
    n = call["slots"].get("name")
    return f" {n}" if n else ""


def looks_like_name(text: str) -> bool:
    """A reply to "what is your name?" that is really a name (not "he stopped breathing")."""
    if re.search(r"(?:my name is|name is|i am|i'm|this is|mera naam|मेरा नाम)", text, flags=re.I):
        return True
    words = [w for w in re.sub(r"[,.!?।]", " ", text.lower()).split() if w not in ("है", "hai", "hoon", "हूँ", "हूं", "ji", "जी", "sir")]
    not_name = {"he", "she", "they", "it", "is", "are", "has", "have", "was", "the", "not", "please", "help", "breathing", "bleeding",
                "trapped", "stuck", "fire", "smoke", "hurt", "injured", "dying", "dead", "coming", "where", "वह", "वो", "हैं", "नहीं",
                "मदद", "आग", "खून", "फँसा", "फंसा", "साँस", "सांस", "कहाँ"}
    return (1 <= len(words) <= 4 and not re.search(r"\d", text) and not rule_problems(text) and not set(words) & not_name
            and yes_no(text) is None and not any(w in text.lower() for w in ETA_WORDS))


SEE_TEAM_WORDS = ["they are here", "they're here", "with me", "i can see", "arrived", "reached", "aa gaye", "aa gayi", "pahunch",
                  "आ गए", "आ गई", "आ गयी", "पहुँच", "पहुंच", "दिख रहे", "दिख रही"]
SAY.update({
    "before_handover": ("The team is with you now. Before I hand you over:", "टीम अब आपके पास है। आपको सौंपने से पहले:"),
    "handover_q": ("All teams have reached {place}. Can you see them? Say yes when they are with you.",
                   "सभी टीमें {place} पहुँच गई हैं। क्या आप उन्हें देख पा रहे हैं? जब वे आपके पास हों तो हाँ बोलिए।"),
    "not_found": ("Stay where you are and keep your phone on. I have told the team to look for you near {place}. "
                  "Tell me when you can see them.",
                  "जहाँ हैं वहीं रहिए और फ़ोन चालू रखिए। मैंने टीम को {place} के पास आपको ढूँढने के लिए कहा है। जब वे दिखें तो बताइए।"),
    "yes_no_again": ("Please answer yes or no:", "कृपया हाँ या नहीं में बताइए:"),
    "eta_one": ("{unit} ({dept}) is about {mins} minutes away.", "{unit} ({dept}) लगभग {mins} मिनट दूर है।"),
    "eta_here": ("{unit} ({dept}) has already reached you.", "{unit} ({dept}) आपके पास पहुँच चुकी है।"),
    "eta_none": ("Help is being arranged; I will tell you as soon as a unit is on its way.",
                 "मदद की व्यवस्था हो रही है; गाड़ी निकलते ही मैं आपको बताऊँगा।"),
    "arrived": ("{icon} {unit} from {dept} has reached {place}. Please go to them or wave so they can see you, and follow their instructions.",
                "{icon} {dept} की {unit} {place} पहुँच गई है। कृपया उनके पास जाएँ या हाथ हिलाएँ ताकि वे आपको देख सकें, और उनके निर्देश मानें।"),
    "handover": ("All teams are with you now, so I am handing you over to them. Your reference is {ref}. Stay safe.",
                 "सभी टीमें अब आपके पास हैं, इसलिए मैं आपको उन्हें सौंप रहा हूँ। आपका संदर्भ {ref} है। सुरक्षित रहें।"),
})


def unit_left(a_: dict, now: float | None = None) -> float:
    """Simulated minutes until this unit arrives (0 when it is there). A unit not yet dispatched has not moved."""
    if not a_.get("sent_at"):
        return float(a_.get("eta") or 0.0)
    now = now or time.time()
    return max(0.0, a_["eta"] - (now - a_.get("sent_at", now)) / a_.get("speed", sim_seconds))


def eta_reply(call: dict) -> str:
    hindi, units = call["hindi"], [a_ for a_ in call["assignments"] if a_.get("unit")]
    if not units:
        return _l(SAY["eta_none"], hindi)
    lines = []
    for a_ in sorted(units, key=unit_left):
        left = unit_left(a_)
        lines.append((_l(SAY["eta_here"], hindi) if left <= 0 else _l(SAY["eta_one"], hindi)).format(
            unit=a_["unit"], dept=a_["dept"], mins=max(1, math.ceil(left))))
    return " ".join(lines)


# ======================= CAD backend: how India's ERSS-112 handles a call (simplified) =======================
# ERSS (Emergency Response Support System) works through a state Emergency Response Centre (ERC):
#  1) the COMMUNICATION OFFICER (call-taker) answers 112 and creates a Call for Service (CFS) with event type and priority;
#  2) the DISPATCH OFFICER assigns the nearest police Emergency Response Vehicle (ERV, tracked by GPS) and, when needed,
#     conferences the call to the fire control room (101) and the ambulance service (108) or a helpline (181/1091/1098/1906/1912);
#  3) each unit reports its status on its mobile data terminal: assigned, en route, reached, then the job is closed with an
#     Action Taken Report (ATR). Priorities here are a simplified P1-P4 scale, not an official ERSS code list.
LEVELS = {   # internal code -> priority, meaning, response, colour
    "E": ("P1 · CRITICAL", "Life at immediate risk", "Nearest units, emergency driving", "#7C3AED"),
    "D": ("P1 · CRITICAL", "Life-threatening", "Emergency driving", "#D62839"),
    "C": ("P2 · URGENT", "Serious: help needed fast", "Emergency driving", "#E46A1E"),
    "B": ("P3 · NORMAL", "Urgent, not life-threatening", "Normal driving, priority", "#D9A21B"),
    "A": ("P3 · NORMAL", "Non-urgent", "Normal driving", "#86AE3A"),
    "O": ("P4 · ADVICE", "No vehicle needed: helpline call-back", "Call-back", "#8A97A5"),
}
AGENCY = {   # who a department's job is passed to in the ERSS model
    "Police": "ERV (police)", "Fire & Rescue": "101 Fire control", "Ambulance (EMS)": "108 Ambulance",
    "Disaster Response": "SDRF / NDRF", "Gas Emergency": "1906 Gas", "Electricity Board": "1912 Power",
    "Women Helpline": "181 / 1091", "Child Helpline": "1098 Childline",
}
LEVEL_ORDER = "OABCDE"


MIN_LEVEL = {"fire": "C", "collapse": "D", "road_accident": "C", "gas_leak": "C", "flood": "C", "women_safety": "C", "child": "C",
             "crime": "B", "electric": "B", "medical": "A", "trafficking": "C", "kidnapping": "D", "missing_person": "B",
             "sexual_violence": "C", "suspicious": "C", "other": "B"}     # e.g. every building fire is at least P2


def response_level(call: dict) -> str:
    lvl = _response_level(call)
    if "other" in call["slots"].get("problems", []) and (call["slots"].get("people") or 0) >= 3 and LEVEL_ORDER.index(lvl) < LEVEL_ORDER.index("C"):
        lvl = "C"                                         # an unclassified event affecting several people is at least urgent
    for p in call["slots"].get("problems", []):
        if lvl != "O" and LEVEL_ORDER.index(MIN_LEVEL.get(p, "A")) > LEVEL_ORDER.index(lvl):
            lvl = MIN_LEVEL[p]
    return lvl


def _response_level(call: dict) -> str:
    sl, sev = call["slots"], int(call["severity"])
    probs = sl.get("problems", [])
    if sl.get("breathing") is False:
        return "E"
    if sl.get("trapped") is True or sl.get("weapon") is True or sev >= 5:
        return "D"
    if any(sl.get(k) is True for k in ("bleeding", "spreading", "touching_wire")) or sl.get("caller_safe") is False or sev == 4:
        return "C"
    if probs and all(DEPARTMENTS[d]["fleet"] is None for p in probs for d in PROBLEMS[p]["teams"]):
        return "O"
    return "B" if sev == 3 else "A"


def run_card(call: dict, level: str) -> list[dict]:
    """Call type + response level -> the units to send, each with the reason (repeated departments = more units)."""
    sl = call["slots"]
    probs = sl.get("problems", []) or ([call["dial_hint"]] if call.get("dial_hint") else [])
    hi = LEVEL_ORDER.index(level)
    injured = sl.get("bleeding") is True or sl.get("breathing") is False or sl.get("trapped") is True or "medical" in probs
    many = (sl.get("people") or 0) >= 3
    card = []

    def add(dept, why):
        if not any(c["dept"] == dept and c["why"] == why for c in card):
            card.append({"dept": dept, "why": why})
    for p in probs:
        if p == "fire":
            add("Fire & Rescue", "fire attack")
            if hi >= LEVEL_ORDER.index("D"):
                add("Fire & Rescue", "second engine: working fire / rescue")
            if hi >= LEVEL_ORDER.index("C") or injured:
                add("Ambulance (EMS)", "108 on standby for injuries")
            if hi >= LEVEL_ORDER.index("D"):
                add("Police", "ERV: crowd and traffic control")
        elif p == "gas_leak":
            add("Fire & Rescue", "gas leak: ventilate, remove ignition sources")
            add("Gas Emergency", "shut off the gas supply")
            if hi >= LEVEL_ORDER.index("C"):
                add("Police", "evacuation and cordon")
        elif p == "medical":
            add("Ambulance (EMS)", "108: patient care and transport")
            if level == "E":
                add("Police", "nearest ERV as first responder until 108 arrives")
            if many and hi >= LEVEL_ORDER.index("D"):
                add("Ambulance (EMS)", "second ambulance: several patients")
        elif p == "road_accident":
            add("Ambulance (EMS)", "patient care and transport")
            add("Police", "traffic control and accident report (MLC)")
            if sl.get("trapped") is True:
                add("Fire & Rescue", "extrication: free trapped people")
            if many:
                add("Ambulance (EMS)", "second ambulance: several patients")
        elif p == "collapse":
            add("Disaster Response", "search and rescue")
            add("Fire & Rescue", "rescue support")
            add("Ambulance (EMS)", "patient care")
            if hi >= LEVEL_ORDER.index("D"):
                add("Police", "cordon and crowd control")
        elif p == "flood":
            add("Disaster Response", "water rescue")
            if injured or hi >= LEVEL_ORDER.index("C"):
                add("Ambulance (EMS)", "patient care")
        elif p == "crime":
            add("Police", "respond to the crime")
            if sl.get("weapon") is True or hi >= LEVEL_ORDER.index("D"):
                add("Police", "backup unit: weapon or violence reported")
            if injured:
                add("Ambulance (EMS)", "staged nearby for the injured")
        elif p == "women_safety":
            add("Police", "protect the caller")
            add("Women Helpline", "counselling and follow-up")
            if injured:
                add("Ambulance (EMS)", "patient care")
        elif p == "child":
            add("Police", "search for the child")
            add("Child Helpline", "child protection follow-up")
        elif p == "trafficking":
            add("Police", "urgent safeguarding check: verify and protect")
            if hi >= LEVEL_ORDER.index("D"):
                add("Police", "backup unit")
            if call.get("subject") == "child":
                add("Child Helpline", "child protection (Childline 1098)")
            if call.get("subject") == "woman":
                add("Women Helpline", "support for the woman")
        elif p == "kidnapping":
            add("Police", "immediate search: vehicle, direction, description")
            add("Police", "second unit to block the likely route")
            if call.get("subject") == "child":
                add("Child Helpline", "child protection (Childline 1098)")
        elif p == "missing_person":
            add("Police", "missing person report and local search")
        elif p == "sexual_violence":
            add("Police", "protect the survivor and secure evidence")
            add("Ambulance (EMS)", "medical care for the survivor")
            add("Women Helpline", "support (181 / One Stop Centre)")
        elif p == "suspicious":
            add("Police", "check the threat and secure the area")
            if any(w in caller_text(call).lower() for w in ("bomb", "explosive", "बम")):
                add("Fire & Rescue", "standby: possible explosive (bomb squad via police)")
                add("Ambulance (EMS)", "standby for casualties")
        elif p == "other":
            add("Police", "ERV to assess an unclassified emergency")
        elif p == "electric":
            add("Electricity Board", "isolate the power line")
            if hi >= LEVEL_ORDER.index("C"):
                add("Police", "ERV: cordon off the area")
            if sl.get("touching_wire") is True or injured:
                add("Ambulance (EMS)", "electric shock patient")
    if not card and sl.get("location"):
        add("Police", "unclear problem: police check")
    return card


def radio(call: dict, a_: dict, level: str) -> None:
    """The dispatcher's radio call to a unit (kept as the radio log)."""
    lv = LEVELS[level]
    probs = ", ".join(PROBLEMS[p]["label"].lower() for p in call["slots"].get("problems", [])) or "check the caller"
    agency = AGENCY.get(a_["dept"], a_["dept"])
    if a_.get("unit") and a_["dept"] == "Police":
        txt = f"CFS {call['ref']} assigned to {a_['unit']} ({agency}): {probs} at {call['slots'].get('location')}, {lv[0]}. {a_['why'].capitalize()}."
    elif a_.get("unit"):
        txt = (f"Call conferenced to {agency}: {a_['unit']} assigned to {call['slots'].get('location')}, {probs}, {lv[0]}. "
               f"{a_['why'].capitalize()}.")
    else:
        txt = f"Passed to {agency}: ticket {a_.get('ticket') or ''}, {a_['why']}."
    call.setdefault("radio", []).append((time.time(), a_["icon"], txt))


def approve_unit(call: dict, a_: dict, actor: str) -> None:
    """Dispatcher approval turns a recommendation into a dispatch (the only place a unit starts moving)."""
    if a_.get("approval") == "approved" or a_.get("cancelled"):
        return
    now = time.time()
    a_["approval"], a_["approved_by"], a_["approved_at"] = "approved", actor, now
    if a_.get("unit"):
        a_["sent_at"] = now
    add_event(call, "👍", f"{a_.get('unit') or a_.get('ticket') or a_['dept']} approved by {actor}", etype="Dispatcher approved",
              actor=actor, unit=a_.get("unit") or a_.get("ticket"))
    radio(call, a_, call.get("level") or "B")
    add_event(call, a_["icon"], (f"{a_['unit']} dispatched ({AGENCY.get(a_['dept'], a_['dept'])}): {a_.get('why', '')}" if a_.get("unit")
                                 else f"{a_['dept']} notified, ticket {a_.get('ticket') or ''}"),
              etype="Unit dispatched", actor=actor, unit=a_.get("unit") or a_.get("ticket"))


def cancel_unit(call: dict, a_: dict, actor: str, reason: str) -> None:
    if a_.get("cancelled"):
        return
    a_["cancelled"], a_["approval"], a_["cancel_reason"] = True, "cancelled", reason
    if a_.get("unit"):
        cad_store()["used_units"].discard(a_["unit"])
    add_event(call, "🛑", f"{a_.get('unit') or a_['dept']} cancelled by {actor}: {reason}", etype="Unit cancelled",
              actor=actor, unit=a_.get("unit"))


def unit_state(a_: dict) -> tuple[str, str]:
    """Display status of a unit row and its colour key."""
    if a_.get("cancelled"):
        return "Cancelled", "muted"
    if a_.get("approval") == "awaiting":
        return "Awaiting approval", "amber"
    if not a_.get("unit") and not a_.get("ticket"):
        return "Recommended", "gray"
    if a_.get("released") or a_.get("atr") or a_.get("at_hospital"):
        return "Completed", "done"
    if not a_.get("unit"):
        return "Dispatched", "blue"
    if a_.get("transport_at"):
        return "Transporting", "orange"
    if a_.get("arrived"):
        return "On scene", "green"
    if a_.get("sent_at") and time.time() - a_["sent_at"] < 3:
        return "Dispatched", "blue"
    return ("En route", "blue") if a_.get("sent_at") else ("Recommended", "gray")


def cad_dispatch(call: dict, hindi: bool) -> list[str]:
    """Send whatever the run card needs and is not yet assigned. Returns what to tell the caller."""
    sl, level = call["slots"], response_level(call)
    old = call.get("level")
    call["level"] = level
    card = run_card(call, level)
    call["run_card"] = card
    have = [(a_["dept"], a_["why"]) for a_ in call["assignments"]]
    need = [c for c in card if (c["dept"], c["why"]) not in have]
    out = []
    if old and LEVEL_ORDER.index(level) > LEVEL_ORDER.index(old):
        add_event(call, "⬆️", f"Response level raised: {LEVELS[old][0]} → {LEVELS[level][0]}")
    if not need:
        return out
    teams = [{"dept": c["dept"], "role": "lead" if not call["assignments"] and k == 0 else "support", "why": c["why"]}
             for k, c in enumerate(need)]
    sent = dispatch_departments(teams, sl.get("location"), call["ref"])
    first = not call["assignments"]
    if first:
        add_event(call, "🗂️", f"Incident created: CFS {call['ref']}, {incident_label(call)}, {LEVELS[level][0]}", etype="Incident created")
    mode = cad_store().get("approval", "auto")             # a control-room policy, the same for every session
    for a_ in sent:
        a_["speed"], a_["recommended_at"] = cfg()["speed"], time.time()
        add_event(call, "📡", f"{a_.get('unit') or a_.get('ticket') or a_['dept']} recommended ({a_['dept']}): {a_.get('why', '')}",
                  etype="Unit recommended", unit=a_.get("unit") or a_.get("ticket"))
        if mode == "auto":
            approve_unit(call, a_, "Auto-approval (DEMO setting)")
            a_["announced"] = True                       # the caller is told in this same reply
        else:
            a_["approval"] = "awaiting"
    call["assignments"] += sent
    st.session_state.setdefault("dept_jobs", []).extend(
        {**{k: v for k, v in a_.items() if k != "route"}, "problem": ", ".join(PROBLEMS[p]["label"] for p in sl.get("problems", [])) or "unclear",
         "location": sl.get("location"), "severity": call["severity"]} for a_ in sent)
    names = ", ".join(dict.fromkeys(a_["dept"] for a_ in sent))
    if mode != "auto":
        out.append(("आपकी रिपोर्ट ड्यूटी डिस्पैचर को भेज दी गई है, जो नज़दीकी टीम तय कर रहे हैं। टीम रवाना होते ही मैं आपको बताऊँगा।"
                    if hindi else "Your report is with the duty dispatcher, who is assigning the nearest units now. "
                                  "I will tell you the moment they are dispatched.") if first else
                   ("मैंने डिस्पैचर से {t} भी जोड़ने को कहा है।" if hindi else "I have asked the dispatcher to add {t} as well.").format(t=names))
        return out
    units = sorted([a_ for a_ in sent if a_.get("unit")], key=lambda a_: a_["eta"])
    if first and units:
        out.append(_l(SAY["sending"], hindi).format(teams=names, unit=units[0]["unit"], eta=max(1, round(units[0]["eta"]))))
    elif first:
        out.append(_l(SAY["extra"], hindi).format(teams=names))
    else:
        out.append(_l(SAY["extra"], hindi).format(teams=names))
    return out


def check_arrivals(call: dict) -> bool:
    """Connect the call with the teams. The first crew to arrive joins the call and finds the caller; later units join
    them. The call-taker never hangs up on a silent caller (unless "Hand over automatically" is chosen): it checks on them."""
    ensure_place(call)
    changed, now, c = False, time.time(), cfg()
    fresh = [a_ for a_ in call.get("assignments", []) if a_.get("approval") == "approved" and not a_.get("announced")]
    if fresh:
        for a_ in fresh:
            a_["announced"] = True
        if not call.get("ended"):
            mov = sorted([a_ for a_ in fresh if a_.get("unit")], key=lambda a_: a_["eta"])
            names = ", ".join(dict.fromkeys(a_["dept"] for a_ in fresh))
            agent_say(call, (f"डिस्पैच की पुष्टि हो गई: {names}।" + (f" {mov[0]['unit']} लगभग {max(1, round(mov[0]['eta']))} मिनट में पहुँचेगी।" if mov else ""))
                      if call["hindi"] else
                      (f"Dispatch confirmed: {names}." + (f" {mov[0]['unit']} is about {max(1, round(mov[0]['eta']))} minutes away." if mov else "")))
        changed = True
    units = [a_ for a_ in call.get("assignments", []) if a_.get("unit") and not a_.get("cancelled")]
    for a_ in units:
        if a_.get("sent_at") and not a_.get("arrived") and unit_left(a_, now) <= 0:
            a_["arrived"], changed = True, True
            add_event(call, a_["icon"], f"{a_['unit']} ({a_['dept']}) arrived")
            if call["ended"]:
                continue
            place = call["slots"].get("location") or "you"
            if c["arrival"] == "crew" and not call.get("crew_unit"):
                # the first crew on scene joins the call (like a real crew phoning the caller)
                call["crew_unit"] = a_["unit"]
                call["resume_q"] = call.get("pending")
                if c["announce"]:
                    agent_say(call, _l(SAY["arrived"], call["hindi"]).format(icon=a_["icon"], unit=a_["unit"], dept=a_["dept"], place=place))
                crew_say(call, a_, _l(CREW_SAY["hello"], call["hindi"]).format(name=_name_bit(call), dept=a_["dept"], unit=a_["unit"], place=place))
                call["pending"] = "crew_locate"
                add_event(call, "🎧", f"{a_['unit']} crew joined the call")
            elif c["arrival"] == "crew" and call.get("crew_unit"):
                lead = next((u for u in units if u["unit"] == call["crew_unit"]), a_)
                if call.get("crew_contact"):
                    a_["contact"] = True
                if c["announce"]:
                    agent_say(call, _l(CREW_SAY["joined"], call["hindi"]).format(unit=f"{a_['icon']} {a_['unit']}", lead=lead["dept"]))
            elif c["announce"]:
                agent_say(call, _l(SAY["arrived"], call["hindi"]).format(icon=a_["icon"], unit=a_["unit"], dept=a_["dept"], place=place))
    hospital = "Central Hospital"
    for a_ in units:
        if a_.get("arrived") and not a_.get("arrived_at"):
            a_["arrived_at"] = now
        sl_ = call["slots"]
        has_patient = (sl_.get("breathing") is False or sl_.get("bleeding") is True or sl_.get("trapped") is True
                       or "medical" in sl_.get("problems", []) or sl_.get("touching_wire") is True
                       or bool(sl_.get("people") and sl_.get("people_src") == "caller"))
        on_scene_min = (now - a_["arrived_at"]) / a_.get("speed", sim_seconds) if a_.get("arrived_at") else 0
        if (a_["dept"] == "Ambulance (EMS)" and a_.get("arrived_at") and not has_patient and not a_.get("released")
                and a_.get("contact") and on_scene_min >= 6 and sl_.get("bleeding") is not True
                and all(u.get("arrived") for u in units if u is not a_ and u["dept"] != "Ambulance (EMS)")):
            # standby ambulance with nobody to treat: released back to service (never "transporting" an empty stretcher)
            a_["released"] = True
            st.session_state.used_vehicle_ids.discard(a_["unit"])
            add_event(call, a_["icon"], f"{a_['unit']} released: no patient needed transport")
            call.setdefault("radio", []).append((now, a_["icon"], f"{a_['unit']}: no patient at scene, back in service."))
            changed = True
        if a_["dept"] != "Ambulance (EMS)" and a_.get("contact") and not a_.get("atr") and on_scene_min >= 10:
            a_["atr"] = True                                   # the job is closed with an Action Taken Report
            add_event(call, "📝", f"{a_['unit']}: job closed, Action Taken Report (ATR) submitted")
            call.setdefault("radio", []).append((now, "📝", f"{a_['unit']}: CFS {call['ref']} closed, ATR submitted."))
            changed = True
        if a_["dept"] == "Ambulance (EMS)" and has_patient and a_.get("arrived_at") and call["slots"].get("location") != hospital:
            if a_.get("contact") and not a_.get("transport_at") and on_scene_min >= 4:
                try:
                    a_["transport_eta"] = max(1.0, route_minutes(make_city_graph(), LANDMARKS[call["slots"]["location"]], LANDMARKS[hospital]))
                except Exception:
                    a_["transport_eta"] = 6.0
                a_["transport_at"] = now
                add_event(call, "🏥", f"{a_['unit']} transporting the patient to {hospital}")
                if not call["ended"]:                          # keep the caller in the loop
                    agent_say(call, (f"{a_['unit']} मरीज़ को {hospital} ले जा रही है, लगभग {a_['transport_eta']:.0f} मिनट।"
                                     if call["hindi"] else
                                     f"{a_['unit']} is taking the patient to {hospital}, about {a_['transport_eta']:.0f} minutes away."))
                call.setdefault("radio", []).append((now, a_["icon"], f"{a_['unit']}: transporting one patient to {hospital}, "
                                                                     f"about {a_['transport_eta']:.0f} minutes."))
                changed = True
            if a_.get("transport_at") and not a_.get("at_hospital") and                     (now - a_["transport_at"]) / a_.get("speed", sim_seconds) >= a_["transport_eta"]:
                a_["at_hospital"] = True
                add_event(call, "🏥", f"{a_['unit']} at {hospital}: patient handed to the hospital")
                call.setdefault("radio", []).append((now, a_["icon"], f"{a_['unit']}: at {hospital}, patient handed over."))
                changed = True
    all_there = bool(units) and all(a_.get("arrived") or a_.get("released") for a_ in units)
    if call.get("handover_when_all") and all_there and not call["ended"]:
        hand_over(call, "Last unit reached the scene: call handed over")
        return True
    if c["arrival"] != "crew" and all_there and not call["ended"] and not call.get("handover_asked"):
        call["handover_asked"] = now
        add_event(call, "📣", "All teams on scene")
        missing = [slot for slot in ("name", "phone") if _needs(call, slot)]
        if c["arrival"] == "auto" and not missing:
            hand_over(call, "All teams on scene: call handed over automatically")
        elif missing and call.get("pending") not in missing:
            call["pending"] = missing[0]; call["asked"].append(missing[0])
            agent_say(call, _l(SAY["before_handover"], call["hindi"]) + " " + question_text(call, missing[0]))
        elif not missing:
            call["pending"] = "handover_confirm"
            agent_say(call, _l(SAY["handover_q"], call["hindi"]).format(place=call["slots"].get("location") or "you"))
        changed = True
    # a silent caller: check on them; only "Hand over automatically" may close the call by itself
    arrived_any = any(a_.get("arrived") for a_ in units)
    if arrived_any and not call["ended"] and call["turns"] and call["turns"][-1]["role"] != "caller":
        quiet = now - max(call.get("last_prompt_at", 0), call.get("last_caller_at", 0))
        if quiet > c["wait"]:
            if c["arrival"] == "auto" and all_there:
                hand_over(call, "No reply after the teams arrived: call handed over automatically")
                changed = True
            elif call.get("still_checks", 0) < 2:
                call["still_checks"] = call.get("still_checks", 0) + 1
                agent_say(call, _l(SAY["still_there"], call["hindi"]))
                add_event(call, "🔔", "Caller silent: checking they are still on the line")
                changed = True
            elif not call.get("kept_open"):
                call["kept_open"] = True
                agent_say(call, _l(SAY["kept_open"], call["hindi"]).format(place=call["slots"].get("location") or "you"))
                add_event(call, "📶", "Caller still silent: line kept open while the crew searches")
                changed = True
    return changed


def hand_over(call: dict, why: str = "Caller confirmed the crew is with them: call handed over") -> None:
    agent_say(call, _l(SAY["handover"], call["hindi"]).format(ref=call["ref"]))
    call["ended"], call["ended_by"], call["ended_at"], call["pending"] = True, "handover", time.time(), None
    add_event(call, "🤝", why)


@traceable(run_type="chain", name="Call-taker turn",
           process_inputs=lambda i: {"caller": i.get("text"), "ref": i["call"]["ref"], "asked": i["call"].get("pending"),
                                     "language": "hi" if i["call"].get("hindi") else "en"})
def caller_says(call: dict, text: str) -> dict:
    """One turn of the conversation: understand, dispatch or update, instruct, ask the next question."""
    hindi = call["hindi"]
    ensure_place(call)
    call["turns"].append({"role": "caller", "text": text.strip(), "t": ist_time()})
    call["last_caller_at"] = time.time()
    call["still_checks"], call["kept_open"] = 0, False          # the caller spoke: they are on the line
    if call.get("pending") == "crew_locate":
        crew = next((a_ for a_ in call["assignments"] if a_.get("unit") == call.get("crew_unit")), None)
        yn = yes_no(text)
        if crew and (yn is False or any(u in text.lower() for u in UNSURE)) and len(text.split()) <= 4:
            crew_say(call, crew, _l(CREW_SAY["search"], hindi).format(place=call["slots"].get("location") or "you"))
            add_event(call, "🔦", "Caller could not describe the spot: crew searching")
            return {"reply": call["turns"][-1]["text"], "understood": ["crew_search"], "severity": call["severity"]}
        if crew:
            # the caller described where they are: the crew finds them
            call["crew_contact"] = True
            for a_ in call["assignments"]:
                if a_.get("arrived"):
                    a_["contact"] = True
            crew_say(call, crew, _l(CREW_SAY["found"], hindi).format(name=_name_bit(call)))
            add_event(call, "🤝", f"{crew['unit']} crew reached the caller: {text.strip()[:60]}")
            extract_into(call, text)                     # the description may also carry new facts
            q = next_question(call)
            call["pending"] = q
            if q:
                call["asked"].append(q)
                agent_say(call, question_text(call, q))
            return {"reply": call["turns"][-1]["text"], "understood": ["crew_contact"], "severity": call["severity"]}
    if call.get("pending") == "anything_else":
        low = " " + re.sub(r"[^\w\u0900-\u097F' ]", " ", text.lower()) + " "
        if any(f" {w} " in low for w in DONE_WORDS) and not rule_problems(text):
            coming = [a_ for a_ in call["assignments"] if a_.get("unit") and not a_.get("arrived") and not a_.get("released")]
            if coming:
                # some units are still on the way: stay on the line until they arrive, then hand over
                nxt = min(coming, key=unit_left)
                call["handover_when_all"], call["pending"] = True, None
                agent_say(call, (f"ठीक है। {nxt['dept']} की {nxt['unit']} अभी लगभग {max(1, math.ceil(unit_left(nxt)))} मिनट दूर है। "
                                 "उनके पहुँचने तक मैं लाइन पर हूँ।") if hindi else
                          (f"Okay. {nxt['unit']} from {nxt['dept']} is still about {max(1, math.ceil(unit_left(nxt)))} minutes away. "
                           "I will stay on the line until they reach you."))
                add_event(call, "⏳", "Caller needs nothing more: waiting for " + ", ".join(a_["unit"] for a_ in coming))
                return {"reply": call["turns"][-1]["text"], "understood": ["done_waiting"], "severity": call["severity"]}
            if not call.get("dispatched"):
                agent_say(call, "112 पर कॉल करने के लिए धन्यवाद। सुरक्षित रहें।" if hindi else "Thank you for calling 112. Stay safe.")
                call["ended"], call["ended_by"], call["ended_at"], call["pending"] = True, "caller", time.time(), None
                add_event(call, "🔚", "Call closed: no emergency to dispatch")
                return {"reply": call["turns"][-1]["text"], "understood": ["closed"], "severity": call["severity"]}
            hand_over(call, "Caller needed nothing more: call handed over to the crew")
            return {"reply": call["turns"][-1]["text"], "understood": ["done"], "severity": call["severity"]}
        call["pending"] = None                       # new information: handle it as a normal turn, then ask again
    if call.get("pending") == "handover_confirm":
        low = text.lower()
        yn = yes_no(text)                            # check "no" first: "नहीं दिख रहे" is a no, not a yes
        if yn is not False and (yn is True or any(w in low for w in SEE_TEAM_WORDS)):
            hand_over(call)
            return {"reply": call["turns"][-1]["text"], "understood": ["team_contact"], "severity": call["severity"]}
        if yn is False:
            agent_say(call, _l(SAY["not_found"], hindi).format(place=call["slots"].get("location") or "you"))
            add_event(call, "🔦", "Caller cannot see the team: crew told to search nearby")
            call["handover_asked"] = time.time()
            return {"reply": call["turns"][-1]["text"], "understood": ["team_not_found"], "severity": call["severity"]}
        call["pending"] = None                       # something else: handle it as a normal turn below
    if any(w in text.lower() for w in ETA_WORDS) and call["dispatched"]:
        # "how long will they take?": answer from the live position of the units, then repeat the open question
        reply = eta_reply(call)
        if call.get("pending"):
            reply += " " + _l(CALL_Q[call["pending"]], hindi)
        agent_say(call, reply)
        return {"reply": reply, "understood": ["eta_question"], "severity": call["severity"]}
    u = understand(text)
    if u["certainty"] != "reported" or not call.get("certainty"):
        call["certainty"] = u["certainty"] if not (call.get("certainty") == "suspected" and u["certainty"] == "reported") else "suspected"
    if u["time"] == "past" or not call.get("time"):
        call["time"] = u["time"]
    if u["subject"] and not call.get("subject"):
        call["subject"] = u["subject"]
    if not call.get("original"):
        call["original"] = text.strip()                       # the caller's first words, kept as said
    if call.get("pending") == "happening_now":
        yn = yes_no(text)
        if yn is True or any(w in (" " + text.lower() + " ") for w in NOW_WORDS):
            call["certainty"] = "reported"                    # it is a real, current emergency
            add_event(call, "📣", "Caller confirmed the emergency is happening now")
        elif yn is False:
            add_event(call, "ℹ️", "Information request, not an emergency report: no dispatch")
            agent_say(call, ("ठीक है, कोई टीम नहीं भेजी जा रही। अगर कभी ऐसा हो, तो तुरंत 112 पर कॉल करें और जगह बताएँ। "
                             "क्या कुछ और है जिसमें मैं मदद कर सकता हूँ?") if hindi else
                      ("Okay, no team is being sent. If this ever happens, call 112 straight away and tell us where. "
                       "Is there anything else I can help with?"))
            call["pending"] = "anything_else"
            return {"reply": call["turns"][-1]["text"], "understood": ["not_an_emergency"], "severity": call["severity"]}
    if call.get("certainty") in ("question", "hypothetical") and not call.get("explained_q"):
        call["explained_q"] = True
        agent_say(call, ("मैं आपकी मदद के लिए हूँ। क्या यह अभी कहीं हो रहा है?" if hindi else
                         "I am here to help. Is this happening somewhere right now?"))
        call["pending"] = "happening_now"
        call["original"] = call.get("original") or text.strip()
        return {"reply": call["turns"][-1]["text"], "understood": ["question"], "severity": call["severity"]}
    got = extract_into(call, text)
    sl = call["slots"]
    if use_groq and groq_key and Groq is not None and (not sl.get("problems") or not call.get("ai_label")):
        ai = llm_classify(caller_text(call)[:600], groq_key, groq_model)
        if ai.get("category") and ai["category"] not in sl.get("problems", []):
            sl["problems"] = list(dict.fromkeys(sl.get("problems", []) + [ai["category"]])); got.append("problems")
        call["ai_label"] = ai.get("label") or call.get("ai_label")
        if ai.get("certainty") in ("suspected", "question", "hypothetical"):
            call["certainty"] = ai["certainty"]
        call["subject"] = call.get("subject") or ai.get("subject")
    # a description that matches no known type is still a report: keep it as "Other / Unclassified"
    caller_turns_ = sum(1 for t_ in call["turns"] if t_["role"] == "caller")
    if not sl.get("problems") and "problem_again" in call["asked"] and call.get("pending") == "problem_again" \
            and len(text.split()) >= 3 and call.get("certainty") not in ("question", "hypothetical"):
        sl["problems"] = ["other"]; got.append("problems")
    parts = []
    pend0 = call.get("pending")
    if (pend0 in YES_NO and pend0 not in got) or (not got and pend0 not in ("description", "child_desc", "name")):
        call["misses"][pend0] = call["misses"].get(pend0, 0) + 1
    if "problems" in got:
        add_event(call, "🔎", "Problem: " + ", ".join(PROBLEMS[p]["label"] for p in sl.get("problems", [])))
    if "location" in got:
        add_event(call, "📍", f"Location heard: “{sl.get('location')}”" + (f" (demo map: {sl['location_map']})" if sl.get("location_map") else
                                                                          " (placed on the demo map for simulation)"))
    if "location_ok" in got:
        add_event(call, "✅", f"Location confirmed by the caller: “{sl.get('location')}”")
    if "name" in got:
        add_event(call, "🙋", f"Caller: {sl.get('name')}")
    if "phone" in got:
        add_event(call, "📱", "Call-back number: " + (sl["phone"] if sl["phone"] == "not given" else "••••••" + sl["phone"][-4:]))
    if "problems" in got or "location" in got:
        if sl.get("problems") == ["other"]:
            what = "आपकी रिपोर्ट दर्ज है" if hindi else "I have noted your report"
        else:
            what = (", ".join(PROBLEM_HI.get(p, PROBLEMS[p]["label"]) for p in sl.get("problems", [])) if hindi
                    else incident_label(call)[0].lower() + incident_label(call)[1:]) if sl.get("problems") else ("आपात स्थिति" if hindi else "an emergency")
        where = sl.get("location")
        parts.append((f"ठीक है, {what}" + (f", {where} के पास।" if where else "।")) if hindi else
                     (f"Okay, {what}" + (f" near {where}." if where else ".")))
    if "people" in got and "problems" not in got:
        parts.append(f"ठीक है, {sl['people']} लोग।" if hindi else f"Okay, {sl['people']} {'person' if sl['people'] == 1 else 'people'}.")
    if "name" in got:
        parts.append(f"धन्यवाद, {sl['name']}।" if hindi else f"Thank you, {sl['name']}.")
    tri = call_triage(call)
    old_sev, call["severity"] = call["severity"], int(tri["severity"])
    # dispatch as soon as we know what and where (like a real call-taker, before all questions are done)
    caller_turns = sum(1 for t_ in call["turns"] if t_["role"] == "caller")
    unclear = not sl.get("problems") and sl.get("location") and caller_turns >= 3      # still unclear: police control room takes it
    if (sl.get("problems") or unclear) and sl.get("location") and sl.get("location_ok") and not call["dispatched"] \
            and call.get("certainty") not in ("question", "hypothetical"):
        rec = add_call(caller_text(call), tri, "Emergency call")
        call["record"] = rec["Call ID"]
        # agents classify and route (traced); the CAD run card decides the actual units
        run_ = run_agents(caller_text(call), tri, call["ref"], "Emergency call",
                          hint=(sl.get("problems") or [call.get("dial_hint")])[0], dispatch=False)
        parts += cad_dispatch(call, hindi)
        call["dispatched"] = True
        reasons = escalation_reasons(call)
        if reasons:
            call["escalated"] = reasons
            add_event(call, "🔔", "Duty dispatcher alerted: " + "; ".join(reasons))
            parts.append("इस कॉल की जानकारी ड्यूटी डिस्पैचर को भी दे दी गई है।" if hindi else
                         "A duty dispatcher has also been alerted to this call.")
        run_["assignments"] = call["assignments"]          # same list: later units appear in the trace too
        run_["call"] = call
        rec["assignments"], rec["teams"] = call["assignments"], [a_["dept"] for a_ in call["assignments"]]
        first_unit = next((a_ for a_ in sorted(call["assignments"], key=lambda a_: a_.get("eta") or 1e9) if a_.get("unit")), None)
        if first_unit:
            rec["Help vehicle"] = first_unit["unit"]
    elif call["dispatched"]:
        # new answers can raise the response level or reveal new problems: the run card adds the missing units
        parts += cad_dispatch(call, hindi)
        if call["severity"] == 5 and old_sev < 5:
            parts.append(_l(SAY["upgraded"], hindi))
        rec = next((c for c in st.session_state.calls if c["Call ID"] == call["record"]), None)
        if rec:
            rec.update(severity=call["severity"], people=sl.get("people"), **{"Call text": caller_text(call)})
            rec["assignments"], rec["teams"] = call["assignments"], [a_["dept"] for a_ in call["assignments"]]
    # life-saving instructions, each given once
    triggers = [("not_breathing", sl.get("breathing") is False), ("bleeding", sl.get("bleeding") is True),
                ("gas", "gas_leak" in sl.get("problems", [])), ("fire", "fire" in sl.get("problems", [])),
                ("unsafe", sl.get("caller_safe") is False), ("wire", "electric" in sl.get("problems", [])),
                ("trapped", sl.get("trapped") is True), ("flood", "flood" in sl.get("problems", []))]
    for key, on in triggers:
        if on and key not in call["said_instructions"]:
            call["said_instructions"].append(key)
            parts.append(_l(INSTRUCT[key], hindi))
    pend = call.get("pending")
    if pend == "location_confirm" and not sl.get("location_ok") and "location" not in got and sl.get("location") \
            and call["misses"].get("location_confirm", 0) < 2 and yes_no(text) is None:
        call["misses"]["location_confirm"] = call["misses"].get("location_confirm", 0) + 1
        agent_say(call, " ".join(p for p in parts if p) + " " + _l(SAY["yes_no_again"], hindi) + " " + question_text(call, "location_confirm"))
        return {"reply": call["turns"][-1]["text"], "understood": got, "severity": call["severity"], "reasked": pend}
    if pend in YES_NO and pend not in got and call["misses"].get(pend, 0) == 1 and not (set(got) & {"problems", "location"}):
        # the answer did not say yes or no: ask the same question once more, plainly
        call["misses"][pend] += 1
        agent_say(call, " ".join(p for p in parts if p) + " " + _l(SAY["yes_no_again"], hindi) + " " + _l(CALL_Q[pend], hindi))
        return {"reply": call["turns"][-1]["text"], "understood": got, "severity": call["severity"], "reasked": pend}
    q = next_question(call)
    if q == "handover_confirm" and cfg()["arrival"] == "auto":
        if parts:
            agent_say(call, " ".join(p for p in parts if p))
        hand_over(call, "Caller details collected and all teams on scene: call handed over automatically")
        return {"reply": call["turns"][-1]["text"], "understood": got, "severity": call["severity"]}
    if q == "handover_confirm":
        call["handover_asked"], call["pending"] = time.time(), "handover_confirm"
        parts.append(_l(SAY["handover_q"], hindi).format(place=sl.get("location") or "you"))
        agent_say(call, " ".join(p for p in parts if p))
        return {"reply": call["turns"][-1]["text"], "understood": got, "severity": call["severity"]}
    if q is None and not got and not call["dispatched"]:
        parts.append(_l(SAY["not_heard"], hindi))
        q = call.get("pending")
    if q:
        call["asked"].append(q)
        call["pending"] = q
        parts.append(question_text(call, q))
    else:
        call["pending"] = None
        parts.append(_l(SAY["done"], hindi))
    agent_say(call, " ".join(p for p in parts if p))
    return {"reply": call["turns"][-1]["text"], "understood": got, "severity": call["severity"],
            "problems": sl.get("problems", []), "location": sl.get("location"),
            "teams": [a_["dept"] for a_ in call["assignments"]], "next_question": call.get("pending")}


CARD_CSS = """
<style>
.ic{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:14px 16px;margin-bottom:12px;background:rgba(128,128,128,.05)}
.ic-h{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.ic-h b{font-size:1.05rem}.ic-h .ref{opacity:.65;font-size:.85rem}
.pill{margin-left:auto;font-size:.78rem;font-weight:700;border-radius:999px;padding:3px 10px;white-space:nowrap}
.pill.live{background:rgba(214,40,57,.15);color:#E5484D}.pill.live:before{content:"● ";animation:icb 1s infinite}
.pill.done{background:rgba(31,138,91,.15);color:#3BB57E}
@keyframes icb{50%{opacity:.2}}
.sev{display:flex;align-items:center;gap:10px;margin:2px 0 12px}
.sev .n{font-size:1.6rem;font-weight:800;color:#fff;border-radius:10px;padding:0 12px;line-height:2.1rem}
.sev .bars{display:flex;gap:3px}.sev .bars i{width:22px;height:8px;border-radius:3px;background:rgba(128,128,128,.25)}
.sev .lbl{font-size:.88rem;opacity:.85}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}
.f .k{font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;opacity:.6}
.f .v{font-weight:650;margin-top:2px}.f .v.muted{opacity:.45;font-weight:500}
.chip{display:inline-block;border-radius:999px;padding:2px 9px;margin:2px 4px 0 0;font-size:.8rem;font-weight:650;background:rgba(128,128,128,.15)}
.chip.red{background:rgba(214,40,57,.16);color:#E5484D}.chip.green{background:rgba(31,138,91,.16);color:#3BB57E}
.prog{margin-top:12px}.prog .row{display:flex;justify-content:space-between;font-size:.78rem;opacity:.75;margin-bottom:4px}
.bar{height:7px;border-radius:4px;background:rgba(128,128,128,.22);overflow:hidden}.bar i{display:block;height:100%;border-radius:4px}
.steps{display:flex;gap:4px;margin-top:6px}.steps span{flex:1;font-size:.68rem;text-align:center;padding-top:5px;border-top:3px solid rgba(128,128,128,.25);opacity:.6}
.steps span.on{border-top-color:#3BB57E;opacity:1;font-weight:700}
.unit{padding:10px 0;border-top:1px solid rgba(128,128,128,.18)}.unit:first-of-type{border-top:0}
.unit .top{display:flex;align-items:center;gap:8px}.unit .top .eta{margin-left:auto;font-weight:800;font-variant-numeric:tabular-nums}
.tl{list-style:none;margin:0;padding:0}.tl li{display:grid;grid-template-columns:88px 22px 1fr;gap:4px;font-size:.84rem;padding:3px 0}
.tl time{opacity:.55;font-variant-numeric:tabular-nums}
.facts{margin-top:8px;display:grid;gap:2px}
.rr{display:grid;grid-template-columns:130px 1fr;gap:8px;font-size:.82rem;padding:3px 0;border-top:1px solid rgba(128,128,128,.12)}
.rr span:first-child{opacity:.6}
.fr{display:grid;grid-template-columns:22px 104px 1fr auto;gap:6px;align-items:center;font-size:.84rem;padding:4px 0;
  border-top:1px solid rgba(128,128,128,.14)}.fr:first-child{border-top:0}
.fi{width:18px;height:18px;border-radius:50%;display:grid;place-items:center;font-size:.7rem;font-weight:800;background:rgba(128,128,128,.2)}
.fi.ok{background:rgba(31,138,91,.25);color:#3BB57E}.fi.ask{background:rgba(201,138,12,.25);color:#E9B03A;animation:icb 1.2s infinite}
.fl{opacity:.7}.fv{font-weight:650}.fs{font-size:.72rem;opacity:.55;text-align:right}
.sys{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:12px;padding-top:10px;border-top:1px dashed rgba(128,128,128,.3);
  font-size:.76rem;opacity:.8}
.erss{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:.76rem;opacity:.8;margin:2px 0 10px}
.cad{list-style:none;margin:6px 0 0;padding:0}.cad li{display:grid;grid-template-columns:24px 1fr auto;gap:6px;font-size:.86rem;
  padding:5px 0;border-top:1px solid rgba(128,128,128,.18)}.cad li:first-child{border-top:0}
</style>
"""


def incident_card_html(c_: dict) -> str:
    """The dispatcher's live view of a call: facts, completeness, teams and timeline."""
    sl, now = c_["slots"], time.time()
    sev = int(c_["severity"])
    if c_["ended"]:
        c_.setdefault("ended_at", now)
    secs = int((c_["ended_at"] if c_["ended"] else now) - c_["started"])
    status = (f"<span class='pill done'>Handed over · {secs // 60}:{secs % 60:02d}</span>" if c_.get("ended_by") == "handover" else
              f"<span class='pill done'>Ended · {secs // 60}:{secs % 60:02d}</span>" if c_["ended"] else
              f"<span class='pill live'>On call · {secs // 60}:{secs % 60:02d}</span>")
    bars = "".join(f"<i style='background:{SEV_COLOR.get(sev)}'></i>" if k < sev else "<i></i>" for k in range(5))
    probs = "".join(f"<span class='chip'>{PROBLEMS[p]['label']}</span>" for p in sl.get("problems", [])) or "<span class='v muted'>listening…</span>"
    danger = [lbl for key, val, lbl in [("breathing", False, "not breathing"), ("trapped", True, "trapped"), ("bleeding", True, "heavy bleeding"),
                                        ("spreading", True, "spreading"), ("weapon", True, "weapon"), ("caller_safe", False, "caller not safe"),
                                        ("touching_wire", True, "touching wire")] if sl.get(key) is val]
    safe = [lbl for key, val, lbl in [("breathing", True, "breathing"), ("bleeding", False, "no heavy bleeding"), ("trapped", False, "nobody trapped"),
                                      ("caller_safe", True, "caller safe"), ("weapon", False, "no weapon")] if sl.get(key) is val]
    signs = "".join(f"<span class='chip red'>{d}</span>" for d in danger) + "".join(f"<span class='chip green'>{g}</span>" for g in safe)
    signs = signs or "<span class='v muted'>none reported</span>"
    # information completeness: each fact with its live value and where it came from (caller / system / inferred)
    checked = any(sl.get(k) is not None for k in YES_NO)
    pend = c_.get("pending")
    def fact(label, ok, value, src="", asking=False):
        return {"label": label, "ok": ok, "value": value, "src": src, "asking": asking}
    ph = sl.get("phone")
    ph_txt = (f"{ph[:5]} {ph[5:]}" if ph and ph != "not given" else "not given" if ph == "not given" else "")
    answered = [k.replace("_", " ") for k in YES_NO if sl.get(k) is not None]
    facts = [
        fact("What", bool(sl.get("problems")), ", ".join(PROBLEMS[p]["label"] for p in sl.get("problems", [])), "caller",
             pend in ("problem", "problem_again")),
        fact("Where", bool(sl.get("location_ok")), (sl.get("location") or "") + ("" if sl.get("location_ok") or not sl.get("location") else " (to confirm)"),
             "caller · confirmed" if sl.get("location_ok") else "caller", pend in ("location", "location_again", "location_confirm")),
        fact("How many", bool(sl.get("people")), str(sl.get("people") or ""), sl.get("people_src", "caller"), pend == "people"),
        fact("Danger checked", checked, ", ".join(danger + safe) or ", ".join(answered), "caller", pend in YES_NO),
        fact("Caller", bool(sl.get("name")), sl.get("name") or "", "caller", pend == "name"),
        fact("Call-back no.", bool(sl.get("phone_ok")), ph_txt,
             ("system (ANI) · confirmed" if sl.get("phone_ok") else "system (ANI) · to confirm") if sl.get("phone_src") == "ANI"
             else "caller" if ph and ph != "not given" else "", pend == "phone"),
    ]
    pct = round(100 * sum(f_["ok"] for f_ in facts) / len(facts))
    fact_rows = "".join(
        f"<div class='fr'><span class='fi {'ok' if f_['ok'] else 'ask' if f_['asking'] else ''}'>{'✓' if f_['ok'] else '…' if f_['asking'] else '○'}</span>"
        f"<span class='fl'>{f_['label']}</span>"
        f"<span class='fv'>{f_['value'] or ('asking now…' if f_['asking'] else 'not yet')}</span>"
        f"<span class='fs'>{f_['src'] if f_['value'] else ''}</span></div>" for f_ in facts)
    missing = [n for n, ok in [("location confirmation", sl.get("location_ok")), ("what happened", sl.get("problems")),
                               ("number of people", sl.get("people")), ("danger check", checked)] if not ok]
    nxt = ("Ask: " + question_text(c_, pend) if pend and pend in CALL_Q or pend in ("location_confirm", "location_pick", "location_landmark")
           else "Units responding; keep the caller updated" if c_.get("dispatched") else "Collect location and what happened")
    geo_txt = ({"one": "found on map", "many": "several map matches", "none": "not found on map",
                "unavailable": "map service unavailable"}.get(sl.get("geo_state"), "not checked")
               + (" · caller confirmed" if sl.get("location_ok") else " · not yet confirmed"))
    rec_rows = [("Original statement", f"“{c_.get('original', '')}”"), ("Incident", incident_label(c_)),
                ("Operational category", ", ".join(PROBLEMS[p]["label"] for p in sl.get("problems", [])) or "not yet clear"),
                ("Priority", (LEVELS[c_["level"]][0] + (" · urgent safeguarding assessment"
                              if set(sl.get("problems", [])) & {"trafficking", "kidnapping", "sexual_violence", "child"} else ""))
                             if c_.get("level") else "set when the location is confirmed"),
                ("Certainty", {"reported": "Reported by caller", "suspected": "Suspected, not confirmed", "question": "Question, not a report",
                               "hypothetical": "Hypothetical, not a report"}.get(c_.get("certainty"), "—")),
                ("Time", "Reported as past" if c_.get("time") == "past" else "Now / not stated"),
                ("At risk", {"child": "A child", "woman": "A woman"}.get(c_.get("subject"), "—")),
                ("Reported location", (sl.get("location") or "not given") + f" ({geo_txt})"),
                ("Map match", sl["geo"]["name"][:90] if sl.get("geo") else "—"),
                ("Missing", ", ".join(missing) or "nothing critical"), ("Next step", nxt),
                ("Dispatcher", "Alerted: " + "; ".join(c_["escalated"]) if c_.get("escalated") else "Not needed so far")]
    record = ("<div class='rec'><div class='k' style='font-size:.74rem;text-transform:uppercase;opacity:.6;margin:12px 0 4px'>"
              "🗂️ Incident record</div>" + "".join(f"<div class='rr'><span>{k}</span><span>{v}</span></div>" for k, v in rec_rows)
              + "<div style='font-size:.72rem;opacity:.6;margin-top:6px'>AI classification is not proof that an allegation is true.</div></div>")
    sysbox = record + (f"<div class='sys'><span>📟 <b>System</b></span><span>Caller ID (ANI): {c_.get('ani', 'withheld')}</span>"
              f"<span>Received: {ist_stamp(c_['started'])}</span>"
              f"<span>Dialled: {c_.get('dialed', '112')}</span><span>Line: mobile</span>"
              f"<span>Language: {'Hindi' if c_.get('hindi') else 'English'}</span>"
              f"<span>Location (ALI): {'from caller' if sl.get('location') else 'not available: ask the caller'}</span>"
              + (f"<span>Demo map: {sl['location_map'] or 'simulated spot'}</span>" if sl.get("location") else "") + "</div>")
    desc = (sl.get("child_desc") or "") + " " + (sl.get("description") or "")
    html = [CARD_CSS, "<div class='ic'>",
            f"<div class='ic-h'><span style='font-size:1.3rem'>📋</span><div><b>What the call-taker knows</b><br>"
            f"<span class='ref'>Incident {c_['ref']}</span></div>{status}</div>",
            f"<div class='sev'><span class='n' style='background:{SEV_COLOR.get(sev)}'>{sev}</span><div><div class='bars'>{bars}</div>"
            f"<div class='lbl'>{SEVERITY_GUIDE.get(sev, '')}</div></div></div>",
            "<div class='grid'>",
            f"<div class='f'><div class='k'>🔥 Problem</div><div class='v'>{probs}</div></div>",
            f"<div class='f'><div class='k'>📍 Location</div><div class='v{'' if sl.get('location') else ' muted'}'>{sl.get('location') or 'not yet'}</div></div>",
            f"<div class='f'><div class='k'>👥 People</div><div class='v{'' if sl.get('people') else ' muted'}'>{sl.get('people') or 'not stated'}</div></div>",
            f"<div class='f'><div class='k'>🙋 Caller</div><div class='v{'' if sl.get('name') else ' muted'}'>{sl.get('name') or 'not given'}"
            + (f"<div style='font-weight:500;opacity:.8;font-size:.85rem'>📱 {sl['phone'][:5]} {sl['phone'][5:]}"
               f"{' ✓' if sl.get('phone_ok') else ''}</div>" if sl.get("phone") and sl["phone"] != "not given" else "")
            + "</div></div>",
            f"<div class='f' style='grid-column:1/-1'><div class='k'>⚠️ Danger signs</div><div class='v'>{signs}</div></div>",
            (f"<div class='f' style='grid-column:1/-1'><div class='k'>📝 Description</div><div class='v'>{desc.strip()}</div></div>" if desc.strip() else ""),
            "</div>",
            f"<div class='prog'><div class='row'><span>Information collected (live)</span><span>{pct}%</span></div>"
            f"<div class='bar'><i style='width:{pct}%;background:{'#3BB57E' if pct >= 80 else '#C98A0C' if pct >= 40 else '#E5484D'}'></i></div>"
            f"<div class='facts'>{fact_rows}</div></div>{sysbox}</div>"]
    # teams
    html.append("<div class='ic'><div class='ic-h'><span style='font-size:1.3rem'>🚨</span><b>Teams</b></div>")
    if not c_["assignments"]:
        html.append("<div class='v muted'>Help is sent as soon as the call-taker knows what happened and where.</div>")
    for a_ in c_["assignments"]:
        if a_.get("approval") == "awaiting" or a_.get("cancelled"):
            st_ = "awaiting dispatcher approval" if a_.get("approval") == "awaiting" else "cancelled"
            html.append(f"<div class='unit'><div class='top'>{a_['icon']} <b>{a_['dept']}</b> <span style='opacity:.7'>"
                        f"{a_.get('unit') or a_.get('ticket') or ''}</span><span class='eta' style='font-weight:600;opacity:.8'>{st_}</span></div></div>")
            continue
        if a_.get("unit"):
            left = unit_left(a_, now)
            frac = 1 - left / a_["eta"] if a_["eta"] else 1
            stage = 3 if a_.get("contact") else 2 if left <= 0 else 1 if frac > 0.03 else 0
            eta = ("↩️ released" if a_.get("released") else "📝 ATR closed" if a_.get("atr") else
                   "🏥 at hospital" if a_.get("at_hospital") else "🚑 transporting" if a_.get("transport_at") else
                   "🤝 with caller" if a_.get("contact") else "✅ on scene" if left <= 0 else f"{max(1, math.ceil(left))} min")
            html.append(f"<div class='unit'><div class='top'>{a_['icon']} <b>{a_['dept']}</b>{' <span class=chip>lead</span>' if a_.get('role') == 'lead' else ''}"
                        f" <span style='opacity:.7'>{a_['unit']}</span><span class='eta'>{eta}</span></div>"
                        f"<div style='font-size:.78rem;opacity:.7;margin-top:2px'>{a_.get('why', '')}</div>"
                        f"<div class='bar' style='margin-top:6px'><i style='width:{100 * min(1, frac):.0f}%;background:{'#3BB57E' if left <= 0 else '#1C5D99'}'></i></div>"
                        "<div class='steps'>" + "".join(f"<span class='{'on' if k <= stage else ''}'>{n}</span>"
                                                       for k, n in enumerate(["Dispatched", "On the way", "On scene", "With caller"])) + "</div></div>")
        else:
            html.append(f"<div class='unit'><div class='top'>{a_['icon']} <b>{a_['dept']}</b> <span style='opacity:.7'>{a_.get('ticket') or ''}</span>"
                        f"<span class='eta' style='font-weight:600;opacity:.8'>{a_['status']}</span></div></div>")
    html.append("</div>")
    # CAD decision: response level, run card and the radio log
    if c_.get("level"):
        lv = LEVELS[c_["level"]]
        rows = []
        for cd in c_.get("run_card", []):
            a_ = next((x for x in c_["assignments"] if x["dept"] == cd["dept"] and x["why"] == cd["why"]), None)
            what = ((a_.get("unit") or a_.get("ticket") or a_["status"])
                    + (" · released" if a_.get("released") else " · ATR" if a_.get("atr") else " · at hospital" if a_.get("at_hospital")
                       else " · transporting" if a_.get("transport_at") else " · reached" if a_.get("arrived") else " · en route"
                       if a_.get("unit") else "")) if a_ else "pending"
            rows.append(f"<li><span>{DEPARTMENTS[cd['dept']]['icon']}</span><span><b>{cd['dept']}</b> · {cd['why']}</span>"
                        f"<span style='opacity:.75'>{what}</span></li>")
        radio_rows = "".join(f"<li><time>{ist_time(t_)}</time><span>{ic}</span><span>{tx}</span></li>"
                             for t_, ic, tx in c_.get("radio", [])[-6:])
        etype = ", ".join(PROBLEMS[p]["label"] for p in c_["slots"].get("problems", [])) or "Unclear (police check)"
        agencies = " · ".join(dict.fromkeys(AGENCY.get(a_["dept"], a_["dept"]) for a_ in c_["assignments"]))
        html.append("<div class='ic'><div class='ic-h'><span style='font-size:1.3rem'>🧠</span><div><b>ERSS-112 CAD decision</b><br>"
                    f"<span class='ref'>Call for Service {c_['ref']} · event type: {etype}</span></div></div>"
                    "<div class='erss'><span>📞 Communication Officer: CFS created</span><span>🗺️ Dispatch Officer: nearest units by GPS</span>"
                    f"<span>🔀 Agencies: {agencies or '…'}</span></div>"
                    f"<div class='sev'><span class='n' style='background:{lv[3]};font-size:1.05rem;padding:0 10px'>{lv[0]}</span>"
                    f"<div><div style='font-weight:650'>{lv[1]}</div><div class='lbl'>Response: {lv[2]}</div></div></div>"
                    "<ul class='cad'>" + "".join(rows) + "</ul>"
                    + (f"<div class='k' style='font-size:.74rem;text-transform:uppercase;opacity:.6;margin-top:10px'>📟 ERV / agency messages</div>"
                       f"<ul class='tl'>{radio_rows}</ul>" if radio_rows else "") + "</div>")
    # timeline
    ev = c_.get("events", [])
    if ev:
        html.append("<div class='ic'><div class='ic-h'><span style='font-size:1.3rem'>🕒</span><b>Timeline</b></div><ul class='tl'>"
                    + "".join(f"<li><time>{ist_time(t_)}</time><span>{ic}</span><span>{tx}</span></li>"
                              for ic, tx, t_ in ev[-9:]) + "</ul></div>")
    return "".join(html)


def call_transcript(call: dict) -> str:
    lines = [f"Emergency call {call['ref']}", f"Received: {ist_stamp(call['started'])} (Indian Standard Time, UTC+05:30)"]
    for t_ in call["turns"]:
        who = {"agent": "Call-taker", "caller": "Caller"}.get(t_["role"], t_.get("who", "Crew"))
        lines.append(f"[{t_['t']}] {who}: {t_['text']}")
        if t_.get("raw") and t_["raw"].strip() != t_["text"].strip():
            lines.append(f"            (speech-to-text heard: {t_['raw']})")
    sl = call["slots"]
    lines += ["", "Summary:", f"  Problem: {', '.join(PROBLEMS[p]['label'] for p in sl.get('problems', [])) or 'unclear'}",
              f"  Location: {sl.get('location') or 'unknown'}", f"  Severity: {call['severity']} of 5", f"  People: {sl.get('people') or 'not stated'}",
              "  Teams: " + (", ".join(f"{a_['dept']} ({a_.get('unit') or a_.get('ticket') or a_.get('status')})" for a_ in call["assignments"]) or "none"),
              "", "Timeline:"] + [f"  {ist_time(t_)} {ic} {tx}" for ic, tx, t_ in call.get("events", [])]
    return "\n".join(lines)


# ----------------------- Speech to text (works even when browser voice typing does not) -----------------------
VOICE_LANGS = {"English (India)": "en-IN", "हिन्दी (Hindi)": "hi-IN"}
WHISPER_HINT = ("Emergency call in India, in Indian English, Hindi or Hinglish. Write exactly what is said, including any "
                "place name, even if unfamiliar: cities such as New Delhi, Ghaziabad, Noida, Gurugram, Bengaluru, Hyderabad, "
                "Mumbai, Kolkata, Chennai, Thiruvananthapuram; places such as India Gate, sector, chowk, mandir, masjid, "
                "gurudwara, thana, railway station, bus stand, hospital.")


def wav_stats(wav: bytes) -> dict:
    """Duration and loudness of a WAV recording, to tell 'silent microphone' apart from 'not understood'."""
    import numpy as np
    try:
        with wave.open(io.BytesIO(wav)) as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if width != 2:
            return {"duration": n / rate if rate else 0.0, "peak": 1.0, "rms": 1.0, "ok": True}
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        x /= 32768.0
        return {"duration": len(x) / rate, "peak": float(np.abs(x).max()) if len(x) else 0.0,
                "rms": float(np.sqrt(np.mean(x * x))) if len(x) else 0.0, "ok": True}
    except Exception:
        return {"duration": 0.0, "peak": 1.0, "rms": 1.0, "ok": False}


def clean_wav(wav: bytes) -> bytes:
    """Mono, 16-bit, and turned up if it was recorded quietly: speech services understand it much better."""
    import numpy as np
    try:
        with wave.open(io.BytesIO(wav)) as w:
            ch, width, rate, raw = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())
        if width != 2:
            return wav
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        peak = float(np.abs(x).max()) if len(x) else 0.0
        if 0 < peak < 20000:
            x *= min(8.0, 26000.0 / peak)                    # gentle gain, at most 8x
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())
        return out.getvalue()
    except Exception:
        return wav


def transcribe_audio(wav: bytes, lang: str, api_key: str, whisper_model: str) -> tuple[str, str]:
    """Turn a WAV recording into text. Groq Whisper first (if a key is set), then free Google speech.

    Returns (text, engine name). Raises RuntimeError with a readable reason when nothing worked.
    """
    problems = []
    wav = clean_wav(wav)
    if api_key and Groq is not None:
        try:
            result = Groq(api_key=api_key).audio.transcriptions.create(
                file=("report.wav", wav), model=whisper_model, language=lang[:2], prompt=WHISPER_HINT, temperature=0.0)
            text = (getattr(result, "text", "") or "").strip()
            if text:
                return text, "Groq Whisper"
            problems.append("Groq Whisper heard no words")
        except Exception as exc:
            problems.append(f"Groq Whisper failed ({str(exc)[:120]})")
    if sr is not None:
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(wav)) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio, language=lang).strip(), "Google speech"
        except sr.UnknownValueError:
            problems.append("no words could be recognised: speak closer to the microphone and a little slower")
        except Exception as exc:
            problems.append(f"the free speech service failed ({str(exc)[:120]})")
    else:
        problems.append("the SpeechRecognition package is not installed (add it to requirements.txt)")
    raise RuntimeError("; ".join(problems))


# ---- Live voice typing: runs in the browser page itself (not an iframe), so the microphone and the browser's
# ---- own speech recognition work. It also records the voice as a WAV file, so it can always be played back and,
# ---- if live typing is unavailable, turned into text on the server.
VT_HTML = """
<div class="vt">
  <div class="vt-row">
    <button class="vt-btn" type="button"><span class="vt-dot"></span><span class="vt-lbl">Start speaking</span></button>
    <div class="vt-meter" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <span class="vt-time"></span>
  </div>
  <div class="vt-status" aria-live="polite">Tap <b>Start speaking</b>, allow the microphone, say what happened and where, then tap <b>Stop</b>.</div>
  <div class="vt-live"></div>
</div>
"""
VT_CSS = """
.vt{font-family:inherit;color:var(--st-text-color,inherit)}
.vt-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.vt-btn{display:inline-flex;align-items:center;gap:8px;border:0;border-radius:999px;padding:10px 18px;font-size:15px;font-weight:700;
  cursor:pointer;background:#D62839;color:#fff;box-shadow:0 6px 16px -6px rgba(214,40,57,.6)}
.vt-btn:disabled{opacity:.6;cursor:wait}
.vt-btn.on{background:#15202B}
.vt-dot{width:10px;height:10px;border-radius:50%;background:#fff}
.vt-btn.on .vt-dot{background:#D62839;animation:vtp 1s infinite}
@keyframes vtp{50%{opacity:.25}}
.vt-meter{display:flex;gap:3px;align-items:flex-end;height:26px}
.vt-meter i{width:5px;height:4px;border-radius:3px;background:#D62839;transition:height .08s linear}
.vt-time{font-variant-numeric:tabular-nums;opacity:.7;font-size:13px}
.vt-status{margin-top:8px;font-size:13.5px;opacity:.85}
.vt-status.err{color:#E5484D;opacity:1;font-weight:600}
.vt-live{margin-top:8px;min-height:22px;font-size:15px;line-height:1.45;padding:8px 10px;border-radius:10px;
  background:rgba(128,128,128,.12);white-space:pre-wrap}
.vt-live:empty{display:none}
@media (prefers-reduced-motion:reduce){.vt-btn.on .vt-dot{animation:none}}
"""
VT_JS = r"""
export default function (component) {
  const { data, setTriggerValue, setStateValue, parentElement } = component;
  const q = (s) => parentElement.querySelector(s);
  const btn = q('.vt-btn'), lbl = q('.vt-lbl'), status = q('.vt-status'), live = q('.vt-live'), clock = q('.vt-time');
  const bars = [...parentElement.querySelectorAll('.vt-meter i')];
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const lang = (data && data.lang) || 'en-IN';
  const MAX_MS = (data && data.auto) ? 30000 : 90000;     // one answer in a conversation is short
  let s = null;
  const say = (html, err) => { status.innerHTML = html; status.classList.toggle('err', !!err); };
  const report = (extra) => { try { setStateValue('env', Object.assign(env, extra || {})); } catch (e) {} };
  const ua = navigator.userAgent || '';
  const env = {
    live_typing: !!SR, microphone_api: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    secure: !!window.isSecureContext, in_frame: window.self !== window.top,
    browser: navigator.brave ? 'Brave' : /Edg\//.test(ua) ? 'Edge' : /OPR\//.test(ua) ? 'Opera' : /Firefox\//.test(ua) ? 'Firefox'
             : /Chrome\//.test(ua) ? 'Chrome' : /Safari\//.test(ua) ? 'Safari' : 'Other',
    permission: 'unknown', last_error: null, last_peak: null,
  };
  try {
    navigator.permissions.query({ name: 'microphone' }).then((p) => { report({ permission: p.state }); p.onchange = () => report({ permission: p.state }); })
      .catch(() => report());
  } catch (e) { report(); }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    say('This browser cannot use the microphone on this page. Use the recorder below.', true); btn.disabled = true; return;
  }
  if (!SR) say('Live typing needs Chrome or Edge. Your voice will still be recorded and turned into text after you stop.');
  btn.onclick = () => (s ? stop() : start());
  if (data && data.auto) {      // conversation mode: when the caller pauses after speaking, finish the answer by itself
    const watch = setInterval(() => {
      if (!s || s.stopped) return;
      const words = (s.finals + (s.interim || '')).trim();
      if (words && s.lastWords !== words) { s.lastWords = words; s.lastChange = Date.now(); }
      if (words && s.lastChange && Date.now() - s.lastChange > 1800) stop();
    }, 300);
    parentElement.__watch && clearInterval(parentElement.__watch); parentElement.__watch = watch;
  }
  // Conversation mode: read the call-taker's question aloud once, then (hands-free) start listening
  const sayText = data && data.say, sayId = data && data.say_id;
  if (sayText && sayId != null && parentElement.__said !== sayId) {
    parentElement.__said = sayId;
    // a new question: stop any listening first, so the microphone never records the call-taker's own voice
    if (s) { s.stopped = true; s.discard = true; try { s.rec && s.rec.abort(); } catch (e) {} cleanup(s); s = null;
             btn.classList.remove('on'); btn.disabled = false; lbl.textContent = 'Start speaking'; }
    const listenNext = () => { if (data.auto && !s) setTimeout(() => { if (!s) start(); }, 350); };
    if ('speechSynthesis' in window && data.speak !== false) {
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(sayText); u.lang = lang; u.rate = 1;
        window.__ttsUntil = Date.now() + 2500 + sayText.length * 75;          // safety end, in case onend never fires
        const done = () => { window.__ttsUntil = Date.now() + 250; listenNext(); };
        u.onend = done; u.onerror = done;
        speechSynthesis.speak(u);
      } catch (e) { listenNext(); }
    } else { listenNext(); }
  }

  async function start() {
    s = { finals: '', interim: '', peak: 0, loud: 0, stopped: false, started: Date.now(), pcm: [], rate: 48000, err: null };
    const sess = s;
    btn.classList.add('on'); lbl.textContent = 'Stop'; live.textContent = ''; say('Listening… speak now.');
    try {
      sess.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch (e) {
      const blocked = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
      fail(blocked ? 'The microphone is blocked. Click the 🔒 icon left of the address bar, allow <b>Microphone</b>, then reload the page.'
                   : 'No microphone was found, or another app (Zoom, Meet) is using it.');
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    sess.ctx = new AC(); if (sess.ctx.state === 'suspended') { try { await sess.ctx.resume(); } catch (e) {} }
    sess.rate = sess.ctx.sampleRate;
    const src = sess.ctx.createMediaStreamSource(sess.stream);
    const an = sess.ctx.createAnalyser(); an.fftSize = 1024; src.connect(an);
    // capture raw sound for the WAV recording
    const proc = sess.ctx.createScriptProcessor(4096, 1, 1), mute = sess.ctx.createGain(); mute.gain.value = 0;
    proc.onaudioprocess = (e) => { if (!sess.stopped) sess.pcm.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
    src.connect(proc); proc.connect(mute); mute.connect(sess.ctx.destination); sess.proc = proc;
    const buf = new Float32Array(an.fftSize);
    const loop = () => {
      if (s !== sess || !sess.ctx) return;
      an.getFloatTimeDomainData(buf); let sum = 0; for (const x of buf) sum += x * x;
      const rms = Math.sqrt(sum / buf.length); sess.peak = Math.max(sess.peak, rms); if (rms > 0.02) sess.loud += 16;
      const l = Math.min(1, rms * 12);
      bars.forEach((b, k) => { const w = 1 - Math.abs(k - 4) / 5; b.style.height = (4 + 22 * l * w) + 'px'; });
      const sec = Math.floor((Date.now() - sess.started) / 1000); clock.textContent = `0:${String(sec).padStart(2, '0')}`;
      sess.raf = requestAnimationFrame(loop);
    };
    loop();
    if (SR) listen(sess);
    sess.timer = setTimeout(() => { if (s === sess) stop(); }, MAX_MS);
    sess.diag = setInterval(() => {
      if (s !== sess || sess.finals || sess.interim) return;
      const el = Date.now() - sess.started;
      if (el > 5000 && sess.peak < 0.01) say('No sound is reaching the microphone. Check the 🔒 icon → <b>Microphone</b> shows the right device, and that it is not muted.', true);
      else if (el > 7000 && sess.loud > 500 && SR) say('I can hear you but live typing has no words yet. Keep talking: your voice is recorded and will be turned into text when you stop.');
    }, 1500);
  }

  function listen(sess) {
    let r;
    try { r = new SR(); } catch (e) { return; }
    sess.rec = r; r.lang = lang; r.continuous = true; r.interimResults = true; r.maxAlternatives = 1;
    r.onresult = (e) => {
      if (window.__ttsUntil && Date.now() < window.__ttsUntil) return;     // that is the call-taker's own voice
      if (s !== sess) return;
      let fin = '', int = '';
      for (let i = e.resultIndex; i < e.results.length; i++) { const x = e.results[i]; if (x.isFinal) fin += x[0].transcript + ' '; else int += x[0].transcript; }
      sess.finals += fin; sess.interim = int; live.textContent = (sess.finals + int).trim();
      say('Hearing you… tap <b>Stop</b> when you have finished.');
    };
    r.onerror = (e) => {
      if (s !== sess || e.error === 'no-speech' || e.error === 'aborted') return;
      sess.err = e.error; report({ last_error: 'speech recognition: ' + e.error });
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed' || e.error === 'network')
        say('Live typing is not available here, but your voice is still being recorded. Tap <b>Stop</b> and it will be turned into text.');
    };
    r.onend = () => {
      if (s !== sess) { return; }
      const hard = ['not-allowed', 'service-not-allowed', 'network', 'audio-capture'].includes(sess.err);
      if (!sess.stopped && !hard) { setTimeout(() => { if (s === sess && !sess.stopped) listen(sess); }, 120); return; }   // browser paused: keep listening
      sess.recDone = true; maybeFinish(sess);
    };
    try { r.start(); } catch (e) {}
  }

  function stop() {
    if (!s) return;
    const sess = s; sess.stopped = true; btn.disabled = true; lbl.textContent = 'Finishing…';
    if (sess.rec) { try { sess.rec.stop(); } catch (e) { sess.recDone = true; } } else sess.recDone = true;
    setTimeout(() => { sess.recDone = true; maybeFinish(sess); }, 2000);        // never wait forever for the browser
    maybeFinish(sess);
  }

  function maybeFinish(sess) {
    if (sess.discard) return;
    if (!sess.stopped || !sess.recDone || sess.sent) return;
    sess.sent = true;
    const text = (sess.finals + (sess.interim || '')).replace(/\s+/g, ' ').trim();
    const wav = encodeWav(sess.pcm, sess.rate);
    cleanup(sess);
    report({ last_peak: Math.round(sess.peak * 1000) / 1000, last_error: sess.err ? 'speech recognition: ' + sess.err : (text ? null : env.last_error) });
    setTriggerValue('done', { id: Date.now(), text, audio: wav, peak: sess.peak, loud: sess.loud, error: sess.err, lang });
    btn.disabled = false; btn.classList.remove('on'); lbl.textContent = 'Start speaking'; clock.textContent = '';
    if (s === sess) s = null;
    if (text) say('Done. Your words are in the box below. You can listen to the recording too.');
    else if (sess.peak < 0.01) say('No sound was recorded. Check your microphone and try again.', true);
    else say('Recorded. Turning your voice into text…');
  }

  function fail(msg) { report({ last_error: msg.replace(/<[^>]+>/g, '') }); const sess = s; if (sess) cleanup(sess); s = null; btn.disabled = false; btn.classList.remove('on'); lbl.textContent = 'Start speaking'; say(msg, true); }

  function cleanup(sess) {
    cancelAnimationFrame(sess.raf); clearTimeout(sess.timer); clearInterval(sess.diag);
    try { sess.proc && (sess.proc.onaudioprocess = null); } catch (e) {}
    try { sess.stream && sess.stream.getTracks().forEach((t) => t.stop()); } catch (e) {}
    try { sess.ctx && sess.ctx.close(); } catch (e) {}
    sess.ctx = null; bars.forEach((b) => (b.style.height = '4px'));
  }

  // 16 kHz mono 16-bit WAV, averaged down from the microphone rate (a simple anti-alias filter)
  function encodeWav(chunks, rate) {
    let n = 0; for (const c of chunks) n += c.length;
    if (!n) return null;
    const all = new Float32Array(n); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
    const target = 16000, ratio = rate / target, len = Math.floor(n / ratio), pcm = new Int16Array(len);
    for (let i = 0; i < len; i++) {
      const a = Math.floor(i * ratio), b = Math.min(n, Math.floor((i + 1) * ratio)); let sum = 0;
      for (let j = a; j < b; j++) sum += all[j];
      const v = Math.max(-1, Math.min(1, sum / Math.max(1, b - a))); pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    const buf = new ArrayBuffer(44 + pcm.length * 2), dv = new DataView(buf);
    const ws = (p, t) => { for (let i = 0; i < t.length; i++) dv.setUint8(p + i, t.charCodeAt(i)); };
    ws(0, 'RIFF'); dv.setUint32(4, 36 + pcm.length * 2, true); ws(8, 'WAVE'); ws(12, 'fmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, target, true);
    dv.setUint32(28, target * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); ws(36, 'data');
    dv.setUint32(40, pcm.length * 2, true); new Int16Array(buf, 44).set(pcm);
    let bin = ''; const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(bin);
  }

  return () => { if (s) { s.stopped = true; try { s.rec && s.rec.stop(); } catch (e) {} cleanup(s); s = null; } };
}
"""
try:
    VOICE_TYPING = st.components.v2.component("sahayak_voice_typing", html=VT_HTML, css=VT_CSS, js=VT_JS)
except Exception:                         # Streamlit older than 1.51: only the recorder below is offered
    VOICE_TYPING = None


def _transcribe_into(key: str, audio: bytes, lang: str, api_key: str, whisper_model: str) -> None:
    """Server-side speech-to-text for a recording; the recording is kept even when this fails."""
    stats = wav_stats(audio)
    res = {"text": None, "lang": lang, "engine": None, "new": False, "audio": audio, "mime": "audio/wav",
           "duration": stats["duration"], "error": None}
    if stats["ok"] and stats["duration"] < 0.6:
        res["error"] = "the recording is too short. Hold on and speak for a few seconds."
    elif stats["ok"] and stats["peak"] < 0.01:
        res["error"] = "the recording is silent. Check that the right microphone is selected and not muted."
    else:
        with st.spinner("Turning your voice into text…"):
            try:
                res["text"], res["engine"] = transcribe_audio(audio, lang, api_key, whisper_model)
                res["new"] = True
            except RuntimeError as exc:
                res["error"] = str(exc)
    st.session_state[f"{key}_result"] = res


def voice_check(key: str, env: dict | None, api_key: str, whisper_model: str) -> None:
    """A plain-language checklist that says exactly why speaking does not work, and how to fix it."""
    import importlib.metadata as md
    with st.popover("🔧 Speaking not working? Check here"):
        rows = []
        def row(ok, what, fix=""):
            rows.append(("✅" if ok is True else "❌" if ok is False else "⚪", what, "" if ok is True else fix))
        ver = md.version("streamlit")
        row(VOICE_TYPING is not None, f"Streamlit {ver}: live voice typing " + ("available" if VOICE_TYPING else "needs 1.51 or newer"),
            "Put `streamlit>=1.51` in requirements.txt, push it to GitHub, then Manage app → Reboot.")
        row(sr is not None, "Free speech-to-text package (SpeechRecognition) " + ("installed" if sr else "missing"),
            "Add `SpeechRecognition>=3.10` to requirements.txt and reboot the app.")
        row(bool(api_key), "Groq key for Whisper speech-to-text " + ("set" if api_key else "not set"),
            "Optional but most reliable: add GROQ_API_KEY in Manage app → Settings → Secrets.")
        if env:
            b = env.get("browser", "?")
            row(env.get("secure"), "Secure page (https)", "Open the app with https://, or on your own computer at http://localhost.")
            row(env.get("microphone_api"), "Browser can use the microphone", "Use Chrome or Edge.")
            perm = env.get("permission", "unknown")
            row(True if perm == "granted" else False if perm == "denied" else None, f"Microphone permission: {perm}",
                "Click the 🔒 icon left of the address bar → Site settings → Microphone → Allow, then reload the page.")
            row(env.get("live_typing") and b not in ("Brave",), f"{b}: live voice typing " + ("supported" if env.get("live_typing") else "not supported"),
                "Live typing works in Chrome and Edge. Brave and Firefox block it: use the ⏺️ Record tab instead.")
            if env.get("last_peak") is not None:
                row(env["last_peak"] >= 0.01, f"Last recording loudness: {env['last_peak']:.3f}",
                    "Silence was recorded: choose the right microphone (🔒 icon → Microphone), unmute it, close Zoom/Meet, speak closer.")
            if env.get("last_error"):
                err = env["last_error"]
                fix = ("Your browser could not reach its speech service (company network, VPN, firewall or Brave). Use the ⏺️ Record tab."
                       if "network" in err else
                       "The microphone is blocked for this site: 🔒 icon → Microphone → Allow, then reload."
                       if "not-allowed" in err else
                       "The browser's speech service could not open the microphone: close other apps or tabs using it (Zoom, Meet), "
                       "check the right microphone is selected, or use the ⏺️ Record tab."
                       if "audio-capture" in err else
                       "Use the ⏺️ Record tab, which does not depend on the browser's speech service.")
                row(False, f"Last voice error: {err}", fix)
        else:
            row(None, "Browser checks appear after the voice panel has loaded once.")
        for icon, what, fix in rows:
            st.markdown(f"{icon} {what}" + (f"  \n<span style='opacity:.75;font-size:.9em'>→ {fix}</span>" if fix else ""), unsafe_allow_html=True)
        if api_key and st.button("Test the Groq Whisper connection", key=f"{key}_wtest"):
            import numpy as np
            t = np.linspace(0, 1, 16000, endpoint=False)
            tone = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(tone)
            try:
                Groq(api_key=api_key).audio.transcriptions.create(file=("test.wav", buf.getvalue()), model=whisper_model)
                st.success(f"Groq Whisper ({whisper_model}) is reachable and the key works.")
            except Exception as exc:
                st.error(f"Groq Whisper failed: {str(exc)[:200]}")


def voice_recorder(key: str, label: str, api_key: str, whisper_model: str) -> dict | None:
    """Speak a report: live voice typing in the browser (words appear while you speak) plus a WAV recording
    you can listen to. If live typing is not available, the recording is turned into text on the server.
    The recording is always kept and playable, even when turning it into text fails."""
    lang_name = st.radio("Language you will speak", list(VOICE_LANGS), horizontal=True, key=f"{key}_lang")
    lang = VOICE_LANGS[lang_name]
    env = None
    if VOICE_TYPING is not None:
        # With a Groq key, recording + Whisper is the most reliable way, so it comes first.
        names = ["🎙️ Live voice typing", "⏺️ Record, then write" + (" (Groq Whisper)" if api_key else "")]
        tabs_ = st.tabs(names[::-1] if api_key else names)
        live_tab, rec_tab = (tabs_[1], tabs_[0]) if api_key else (tabs_[0], tabs_[1])
        with live_tab:
            res = VOICE_TYPING(key=f"{key}_live_{lang}", data={"lang": lang}, on_done_change=lambda: None,
                               on_env_change=lambda: None)
            env = getattr(res, "env", None)
        done = getattr(res, "done", None)
        if done and done.get("id") != st.session_state.get(f"{key}_live_id"):
            st.session_state[f"{key}_live_id"] = done["id"]
            audio = base64.b64decode(done["audio"]) if done.get("audio") else None
            if done.get("text"):
                st.session_state[f"{key}_result"] = {"text": done["text"], "lang": lang, "engine": "live voice typing",
                                                     "new": True, "audio": audio, "mime": "audio/wav",
                                                     "duration": wav_stats(audio)["duration"] if audio else 0, "error": None}
            elif audio:
                _transcribe_into(key, audio, lang, api_key, whisper_model)
            else:
                st.session_state[f"{key}_result"] = {"text": None, "audio": None, "lang": lang, "new": False,
                                                     "error": "nothing was recorded. Check the microphone and try again."}
        with rec_tab:
            st.caption("Press the microphone, speak, press stop. Your recording is turned into text on the server"
                       + (" by Groq Whisper." if api_key else " (add a Groq key for Whisper, which is more reliable)."))
            audio_in = st.audio_input(label, key=f"{key}_audio")
    else:
        audio_in = st.audio_input(label, key=f"{key}_audio")
    if audio_in is not None:
        data = audio_in.getvalue()
        digest = hashlib.sha1(data).hexdigest()
        if st.session_state.get(f"{key}_hash") != digest:      # a new recording: transcribe it once
            st.session_state[f"{key}_hash"] = digest
            _transcribe_into(key, data, lang, api_key, whisper_model)

    voice_check(key, env, api_key, whisper_model)
    res = st.session_state.get(f"{key}_result")
    if res:
        if res.get("text"):
            st.success(f"✍️ Written ({res['engine']}): “{res['text']}”")
        elif res.get("error"):
            st.error(f"Could not turn your voice into text: {res['error']}")
        if res.get("audio"):
            dur = res.get("duration") or 0
            st.markdown(f"🎧 **Listen to what you recorded**" + (f" ({dur:.1f} s)" if dur else ""))
            st.audio(res["audio"], format=res.get("mime", "audio/wav"))
            if not res.get("text") and st.button("Try turning it into text again", key=f"{key}_retry"):
                _transcribe_into(key, res["audio"], lang, api_key, whisper_model)
                st.rerun()
        if not res.get("text"):
            st.caption("You can always type the report instead.")
    return res


# ----------------------- Settings (no configuration panel) -----------------------
def _secret(name: str) -> str:
    """Read a key from Streamlit Secrets or the environment, never from the page."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip().strip('"').strip("'").strip()
        for section in st.secrets.values():                 # also accept [langsmith] / [groq] style sections
            if hasattr(section, "get") and section.get(name):
                return str(section.get(name)).strip().strip('"').strip("'").strip()
    except Exception:
        pass
    return os.getenv(name, "").strip()


def setup_langsmith():
    """Switch tracing on only when a LangSmith key is in Secrets; otherwise everything still works, untraced."""
    global ls_reason
    key = _secret("LANGSMITH_API_KEY") or _secret("LANGCHAIN_API_KEY")
    if langsmith is None:
        ls_reason = ("The **langsmith** package is not installed on the server. Add the line `langsmith>=0.3` to "
                     "**requirements.txt** in your GitHub repository, then reboot the app.")
    elif not key:
        ls_reason = ("No **LANGSMITH_API_KEY** was found in Secrets. Check the spelling (all capitals, underscores), that it is "
                     "saved, and that the app was rebooted after saving.")
    elif not key.startswith("lsv2_"):
        ls_reason = "The LANGSMITH_API_KEY in Secrets does not look like a LangSmith key (it should start with `lsv2_`)."
        key = ""
    if langsmith is None or not key:
        os.environ["LANGSMITH_TRACING"] = "false"
        return None, None
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_TRACING"] = "true"
    project = _secret("LANGSMITH_PROJECT") or "sahayak-112"
    os.environ["LANGSMITH_PROJECT"] = project
    if _secret("LANGSMITH_ENDPOINT"):                      # e.g. https://eu.api.smith.langchain.com
        os.environ["LANGSMITH_ENDPOINT"] = _secret("LANGSMITH_ENDPOINT")
    try:
        langsmith.utils.get_env_var.cache_clear()           # make the library re-read the settings above
    except Exception:
        pass
    return _ls_client(key, project, _secret("LANGSMITH_ENDPOINT")), project


@st.cache_resource(show_spinner=False)
def _ls_client(key: str, project: str, endpoint: str):
    """One LangSmith client for the whole app (not a new one on every click)."""
    try:
        return LangSmithClient()
    except Exception:
        return None


ls_reason = ""
ls_client, ls_project = setup_langsmith()


def ls_extra(tag: str, thread: str) -> dict:
    """Give a trace our own id (to attach feedback later) and group a call's turns as one LangSmith thread."""
    return {"run_id": uuid.uuid4(), "metadata": {"thread_id": thread, "app": "sahayak-112"}, "tags": [tag]}


def send_feedback(run_id: str | None, scores: dict, comment: str = "", correction: dict | None = None) -> str:
    """Store feedback locally (for the accuracy page) and in LangSmith on the run it judges."""
    st.session_state.setdefault("feedback_log", []).append({"run_id": run_id, **scores, "comment": comment,
                                                            "correction": correction, "time": ist_time()})
    if ls_client is None or not run_id:
        return "saved in this session (LangSmith is not connected)"
    try:
        for key, val in scores.items():
            ls_client.create_feedback(run_id, key=key, score=val, comment=comment or None,
                                      correction=correction if key == "routing_correct" else None)
        return "saved in this session and sent to LangSmith"
    except Exception as exc:
        return f"saved in this session; LangSmith said: {str(exc)[:120]}"


# A labelled test set: the answers a trained call-taker would give. Used to measure accuracy on demand.
TEST_CALLS = [
    ("Smoke and flames are coming from a building near the Shopping Mall.", ["fire"], ["Fire & Rescue"], 4, "Shopping Mall"),
    ("Fire at the petrol pump, a man has bad burns on his arms", ["fire"], ["Fire & Rescue", "Ambulance (EMS)"], 4, "Petrol Pump"),
    ("Strong gas smell from a cylinder in a house near the old temple, nobody hurt", ["gas_leak"], ["Fire & Rescue", "Gas Emergency"], 3, "Old Temple"),
    ("A person is unconscious and not breathing near Railway Station.", ["medical"], ["Ambulance (EMS)"], 5, "Railway Station"),
    ("My father fell in the bathroom and his leg is broken, we are near the stadium", ["medical"], ["Ambulance (EMS)"], 3, "Stadium"),
    ("Bus overturned near the river bridge, many people injured", ["road_accident", "medical"], ["Ambulance (EMS)", "Police"], 5, "River Bridge"),
    ("Two bikes collided at the bus stand, one rider is bleeding", ["road_accident", "medical"], ["Ambulance (EMS)", "Police"], 4, "Bus Terminal"),
    ("Several people are trapped after a building collapse in the Industrial Area.", ["collapse"],
     ["Disaster Response", "Fire & Rescue", "Ambulance (EMS)"], 5, "Industrial Area"),
    ("Water is rising fast near the river bank, an old man is stuck on his roof", ["flood"], ["Disaster Response"], 4, "River Bank"),
    ("Two men are robbing a shop near central market, they have knives", ["crime"], ["Police"], 4, "Central Market"),
    ("Someone snatched my chain near the airport road and my friend is bleeding", ["crime", "medical"], ["Police", "Ambulance (EMS)"], 4, "Airport Road"),
    ("A woman is being harassed by two men near the shopping mall", ["women_safety"], ["Police", "Women Helpline"], 4, "Shopping Mall"),
    ("My husband beats me every night, I live near the north school", ["women_safety"], ["Police", "Women Helpline"], 4, "North School"),
    ("My child is missing since evening near the central market", ["child"], ["Police", "Child Helpline"], 4, "Central Market"),
    ("A live wire has fallen on the road near the stadium", ["electric"], ["Electricity Board"], 3, "Stadium"),
    ("A boy got an electric shock from a transformer near the petrol pump", ["electric"], ["Electricity Board", "Ambulance (EMS)"], 4, "Petrol Pump"),
    ("This is a prank call, there is no emergency", [], [], 1, None),
    ("मेरे घर में आग लगी है, मंदिर के पास", ["fire"], ["Fire & Rescue"], 4, "Old Temple"),
    ("बस अड्डे पर एक्सीडेंट हुआ है, दो लोग घायल हैं", ["road_accident", "medical"], ["Ambulance (EMS)", "Police"], 4, "Bus Terminal"),
    ("Someone collapsed in the market and is not moving", ["medical"], ["Ambulance (EMS)"], 5, "Central Market"),
]


@traceable(run_type="chain", name="Routing under test")
def predict_routing(text: str) -> dict:
    """The system's answer for one call, without sending any vehicle (safe to run in a test)."""
    tri, _ = run_triage(text)
    route = route_problem(text, tri)
    return {"problems": sorted(route["problems"]), "departments": sorted(t_["dept"] for t_ in route["teams"]),
            "lead": route["teams"][0]["dept"] if route["teams"] else None, "severity": int(tri.get("severity", 1)),
            "location": tri.get("location")}


def score_one(out: dict, problems, departments, severity, location) -> dict:
    p, e = set(out["departments"]), set(departments)
    prec = len(p & e) / len(p) if p else (1.0 if not e else 0.0)
    rec = len(p & e) / len(e) if e else (1.0 if not p else 0.0)
    return {"problem_exact": float(set(out["problems"]) == set(problems)),
            "departments_exact": float(p == e),
            "department_f1": 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec),
            "lead_correct": float((out["lead"] or None) == (departments[0] if departments else None)),
            "severity_within_1": float(abs(out["severity"] - severity) <= 1),
            "location_correct": float(out["location"] == location)}


# LangSmith evaluators: (outputs, reference_outputs) -> score
def _ref(r):
    return r["problems"], r["departments"], r["severity"], r["location"]


def ev_departments_exact(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "departments_exact", "score": score_one(outputs, *_ref(reference_outputs))["departments_exact"]}


def ev_department_f1(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "department_f1", "score": score_one(outputs, *_ref(reference_outputs))["department_f1"]}


def ev_problem_exact(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "problem_exact", "score": score_one(outputs, *_ref(reference_outputs))["problem_exact"]}


def ev_severity(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "severity_within_1", "score": score_one(outputs, *_ref(reference_outputs))["severity_within_1"]}


def ev_location(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "location_correct", "score": score_one(outputs, *_ref(reference_outputs))["location_correct"]}


def run_langsmith_experiment() -> str:
    """Upload the test set as a LangSmith dataset (once) and run an experiment with the evaluators above."""
    name = "sahayak-112-routing"
    if not ls_client.has_dataset(dataset_name=name):
        ds = ls_client.create_dataset(name, description="Emergency calls with the departments a trained call-taker would send.")
        ls_client.create_examples(
            dataset_id=ds.id,
            inputs=[{"text": t} for t, *_ in TEST_CALLS],
            outputs=[{"problems": p, "departments": d, "severity": sv, "location": loc} for _, p, d, sv, loc in TEST_CALLS])
    from langsmith.evaluation import evaluate as ls_evaluate
    res = ls_evaluate(lambda inputs: predict_routing(inputs["text"]), data=name,
                             evaluators=[ev_departments_exact, ev_department_f1, ev_problem_exact, ev_severity, ev_location],
                             experiment_prefix="routing-" + ("groq" if use_groq else "rules"),
                             metadata={"triage": "groq" if use_groq else "rules", "model": groq_model if use_groq else None},
                             max_concurrency=0, client=ls_client)
    return getattr(res, "experiment_name", "experiment created")


groq_key = _secret("GROQ_API_KEY")          # add it in Manage app -> Settings -> Secrets to switch on Groq AI
use_groq = bool(groq_key)
groq_model = GROQ_FALLBACK_MODEL             # llama-3.3-70b-versatile: the most reliable for triage
whisper_model = "whisper-large-v3-turbo"     # Groq speech-to-text, used only when a key is set
triage_model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else 0.0)
use_model = triage_model is not None         # used automatically when triage_model.json is next to app.py
closure_rows, closure_cols = [], []          # no simulated road closures
sim_seconds = 4.0                            # demo speed: 4 real seconds per simulated minute (a 7-minute trip = 28 s)

if "calls" not in st.session_state:
    st.session_state.calls = []
if "dispatches" not in st.session_state:
    st.session_state.dispatches = []

# ======================= COMMAND CENTER: reusable components (business logic stays in the CAD functions above) =======================
APP_ENV = (_secret("APP_ENV") or "DEMO").upper()
PRIO_STYLE = {   # internal level -> (short label, icon, css colour key)
    "E": ("P1", "▲▲", "red"), "D": ("P1", "▲▲", "red"), "C": ("P2", "▲", "orange"), "B": ("P3", "■", "amber"),
    "A": ("P3", "■", "green"), "O": ("P4", "●", "blue"),
}


def cc_css(p: dict) -> str:
    dark = p["theme"] == "Dark"
    base = 15 if p["large_text"] else 13.5
    pad = "8px 10px" if p["density"] == "Compact" else "12px 14px"
    v = {"bg": "#0f141a" if dark else "#f4f6f9", "panel": "#161d26" if dark else "#ffffff", "border": "#2a3542" if dark else "#d6dde6",
         "text": "#eef2f6" if dark else "#14202c", "muted": "#9aa8b6" if dark else "#5b6b7a"}
    if p["high_contrast"]:
        v.update(border="#8796a8" if dark else "#3a4655", muted="#d0d8e2" if dark else "#2c3844")
    return f"""<style>
:root{{--cc-bg:{v['bg']};--cc-panel:{v['panel']};--cc-border:{v['border']};--cc-text:{v['text']};--cc-muted:{v['muted']};
--red:#e5484d;--orange:#f0883e;--amber:#e2b33c;--green:#3fb87f;--blue:#4c8ff0;--gray:#8b98a7;--done:#5f9f7d;--muted:#6f7c89}}
{'.stApp{background:var(--cc-bg)}' if p['theme'] == 'Light' else ''}
.cc *{{box-sizing:border-box}}
.cc{{color:var(--cc-text);font-size:{base}px}}
.cc .panel{{background:var(--cc-panel);border:1px solid var(--cc-border);border-radius:12px;padding:{pad};margin-bottom:10px}}
.cc .muted{{color:var(--cc-muted)}}
.cc-top{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:var(--cc-panel);border:1px solid var(--cc-border);
  border-radius:12px;padding:10px 14px}}
.cc-top .brand{{font-weight:800;font-size:1.05rem;letter-spacing:.02em}}
.cc-top .env{{font-size:.72rem;font-weight:800;border-radius:6px;padding:2px 8px;border:1px solid var(--amber);color:var(--amber)}}
.cc-top .clock{{font-variant-numeric:tabular-nums;font-weight:700}}
.cc-top .st{{font-size:.8rem;color:var(--cc-muted)}}.cc-top .st b{{color:var(--cc-text)}}
.cc-top .sp{{margin-left:auto}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}}
.badge{{display:inline-flex;align-items:center;gap:4px;font-size:.74rem;font-weight:800;border-radius:6px;padding:2px 7px;
  border:1px solid currentColor;white-space:nowrap}}
.c-red{{color:var(--red)}}.c-orange{{color:var(--orange)}}.c-amber{{color:var(--amber)}}.c-green{{color:var(--green)}}
.c-blue{{color:var(--blue)}}.c-gray{{color:var(--gray)}}.c-done{{color:var(--done)}}.c-muted{{color:var(--muted)}}
.kv{{display:grid;grid-template-columns:150px 1fr;gap:6px 12px;font-size:.88em}}.kv span:nth-child(odd){{color:var(--cc-muted)}}
.cad-h{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}.cad-h .id{{font-weight:800;font-size:1.05em}}
.cad-h .cts{{margin-left:auto;color:var(--cc-muted);font-size:.82em}}
.sec{{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--cc-muted);margin:12px 0 6px;font-weight:700}}
table.ut{{width:100%;border-collapse:collapse;font-size:.84em}}
table.ut th{{text-align:left;color:var(--cc-muted);font-weight:600;font-size:.86em;border-bottom:1px solid var(--cc-border);padding:5px 6px}}
table.ut td{{border-bottom:1px solid var(--cc-border);padding:6px;vertical-align:top}}
.tl2{{position:relative;margin-left:6px}}
.ev{{display:grid;grid-template-columns:{'86px' if p['clock'] == '12-hour' else '64px'} 22px 1fr;gap:8px;padding:{'3px' if p['timeline'] == 'Compact' else '6px'} 0;
  border-bottom:1px solid var(--cc-border);font-size:.86em}}
.ev time{{color:var(--cc-muted);font-variant-numeric:tabular-nums}}.ev .t{{font-weight:700}}.ev .a{{color:var(--cc-muted);font-size:.9em}}
.ev details summary{{cursor:pointer;color:var(--cc-muted);font-size:.85em}}
.ag{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}}
.ag .panel{{margin:0}}.ag .lead{{border-color:var(--blue)}}
{'.cc *{animation:none !important;transition:none !important}' if p['reduce_motion'] else ''}
[class*="st-key-kpi_"] button{{min-height:84px;height:84px;border-radius:12px;text-align:left;justify-content:flex-start;background:var(--cc-panel);
  border:1px solid var(--cc-border)}}
[class*="st-key-kpi_"] button p{{font-size:.82rem;line-height:1.25;white-space:pre-line;text-align:left}}
[class*="st-key-kpi_"] button strong{{font-size:1.6rem;display:block}}
</style>"""


def prio_badge(level: str | None) -> str:
    if not level:
        return "<span class='badge c-gray'>● Pending</span>"
    lab, ic, col = PRIO_STYLE[level]
    return f"<span class='badge c-{col}' title='{LEVELS[level][1]}'>{ic} {lab} · {LEVELS[level][0].split('·')[-1].strip()}</span>"


def status_badge(label: str, col: str) -> str:
    icon = {"red": "⚠", "orange": "➜", "amber": "⏳", "green": "●", "blue": "➜", "gray": "○", "done": "✔", "muted": "✕"}.get(col, "●")
    return f"<span class='badge c-{col}'>{icon} {label}</span>"


def incident_status(c_: dict) -> tuple[str, str]:
    sl, units = c_["slots"], [a_ for a_ in c_.get("assignments", []) if not a_.get("cancelled")]
    if c_.get("certainty") in ("question", "hypothetical") and not c_.get("dispatched"):
        return "Information only", "gray"
    if not sl.get("location") or not sl.get("problems"):
        return "Intake", "gray"
    if not sl.get("location_ok"):
        return "Verifying location", "amber"
    if not units:
        return "Awaiting assignment", "amber"
    states = [unit_state(a_)[0] for a_ in units]
    if "Awaiting approval" in states:
        return "Awaiting approval", "amber"
    if any(x in ("En route", "Dispatched") for x in states):
        return "Units en route", "blue"
    if any(x in ("On scene", "Transporting") for x in states):
        return "On scene", "green"
    if all(x == "Completed" for x in states):
        return "Completed", "done"
    return "Active", "blue"


def esc(x) -> str:
    import html as _h
    return _h.escape(str(x if x is not None else ""))


def all_incidents() -> list[dict]:
    """Every incident on this server, newest first (from real application data only)."""
    return sorted(cad_store()["incidents"].values(), key=lambda c_: c_.get("created", 0), reverse=True)


def kpis(incs: list[dict]) -> dict:
    active = [c_ for c_ in incs if incident_status(c_)[0] not in ("Completed", "Information only")]
    units = [a_ for c_ in incs for a_ in c_.get("assignments", []) if a_.get("unit") and not a_.get("cancelled")]
    moving = [a_ for a_ in units if unit_state(a_)[0] in ("Dispatched", "En route", "On scene", "Transporting")]
    resp = [(a_["arrived_at"] - a_["sent_at"]) / a_.get("speed", sim_seconds) for a_ in units if a_.get("arrived_at") and a_.get("sent_at")]
    return {"Active incidents": len(active),
            "P1 critical": sum(1 for c_ in active if c_.get("level") in ("D", "E")),
            "P2 urgent": sum(1 for c_ in active if c_.get("level") == "C"),
            "Units available": sum(1 for v in VEHICLES if v["Vehicle ID"] not in cad_store()["used_units"]),
            "Units dispatched": len(moving),
            "Awaiting assignment": sum(1 for c_ in active if incident_status(c_)[0] in ("Verifying location", "Awaiting assignment",
                                                                                       "Awaiting approval", "Intake")),
            "Average response": (f"{sum(resp) / len(resp):.1f} min" if resp else "—")}


def describe(c_: dict) -> str:
    """A plain description built only from what the caller said (no invented details)."""
    sl = c_["slots"]
    bits = [incident_label(c_)]
    if sl.get("location"):
        bits.append(f"at {sl['location']}")
    if sl.get("people"):
        bits.append(f"{sl['people']} {'person' if sl['people'] == 1 else 'people'} affected")
    danger = [lbl for key, val, lbl in [("breathing", False, "not breathing"), ("trapped", True, "trapped"), ("bleeding", True, "heavy bleeding"),
                                        ("spreading", True, "spreading"), ("weapon", True, "weapon reported")] if sl.get(key) is val]
    if danger:
        bits.append("danger: " + ", ".join(danger))
    return "; ".join(bits) + "."


def cad_card_html(c_: dict) -> str:
    sl = c_["slots"]
    stt, col = incident_status(c_)
    geo_state = {"one": "Found on map", "many": "Several map matches", "none": "Not found on map",
                 "unavailable": "Map service unavailable"}.get(sl.get("geo_state"), "Not checked")
    verified = ("Verified (caller confirmed" + (" + map match)" if sl.get("geo") else ", not on map)")) if sl.get("location_ok") else "Not verified"
    missing = [n for n, ok in [("location confirmation", sl.get("location_ok")), ("what happened", sl.get("problems")),
                               ("number of people", sl.get("people")), ("caller name", sl.get("name"))] if not ok]
    conf = ("AI (Groq) classification: no numeric confidence" if c_.get("ai_label") else "Keyword rules: confidence not measured")
    rows = []
    for a_ in c_.get("assignments", []):
        lab, ucol = unit_state(a_)
        fleet = DEPARTMENTS[a_["dept"]]["fleet"] or "Helpline ticket"
        rows.append(f"<tr><td>{a_['icon']} {esc(AGENCY.get(a_['dept'], a_['dept']))}</td><td><b>{esc(a_.get('unit') or a_.get('ticket') or '—')}</b></td>"
                    f"<td>{esc(fleet)}</td><td>{esc(a_.get('why', ''))}</td>"
                    f"<td>{ist_time(a_['sent_at']) if a_.get('sent_at') else '—'}</td>"
                    f"<td>{(str(max(1, math.ceil(unit_left(a_)))) + ' min (simulated)') if a_.get('sent_at') and not a_.get('arrived') else '—'}</td>"
                    f"<td>{ist_time(a_['arrived_at']) if a_.get('arrived_at') else '—'}</td>"
                    f"<td>{status_badge(lab, ucol)}</td><td class='muted'>{esc(a_.get('approved_by') or ('pending' if a_.get('approval') == 'awaiting' else '—'))}</td></tr>")
    units_tbl = ("<table class='ut'><tr><th>Agency</th><th>Unit</th><th>Type</th><th>Task / reason</th><th>Dispatched</th><th>ETA</th>"
                 "<th>Arrived</th><th>Status</th><th>Approved by</th></tr>" + "".join(rows) + "</table>") if rows else \
        "<div class='muted'>No units yet: the CAD recommends units once the location is confirmed.</div>"
    return (f"<div class='cc'><div class='panel'><div class='cad-h'><span class='id'>CFS {esc(c_['ref'])}</span>"
            f"<span>{esc(incident_label(c_))}</span>{prio_badge(c_.get('level'))}{status_badge(stt, col)}"
            f"<span class='cts'>Created {ist_stamp(c_.get('created', c_['started']))}</span></div>"
            "<div class='sec'>Incident summary</div><div class='kv'>"
            f"<span>Description</span><span>{esc(describe(c_))}</span>"
            f"<span>Caller's words</span><span>“{esc(c_.get('original', ''))}”</span>"
            f"<span>Location</span><span>{esc(sl.get('location') or 'not given')}{(' → ' + esc(sl['geo']['name'])) if sl.get('geo') else ''}</span>"
            f"<span>Location status</span><span>{verified} · {geo_state}</span>"
            f"<span>Certainty</span><span>{esc({'reported': 'Reported by caller', 'suspected': 'Suspected, not confirmed', 'question': 'Question, not a report', 'hypothetical': 'Hypothetical'}.get(c_.get('certainty'), '—'))}</span>"
            f"<span>Classification</span><span>{conf}</span>"
            f"<span>Needs attention</span><span>{esc(', '.join(missing) or 'nothing critical')}{(' · escalated: ' + esc('; '.join(c_['escalated']))) if c_.get('escalated') else ''}</span>"
            f"<span>Caller</span><span>{esc(sl.get('name') or 'not given')} · {('••••• ' + sl['phone'][-4:]) if sl.get('phone') and sl['phone'] != 'not given' else 'no number'}</span>"
            "</div><div class='sec'>Dispatch recommendation and units</div>" + units_tbl +
            "<div class='muted' style='font-size:.78em;margin-top:6px'>Travel times are from the simulated demo road grid, not live traffic. "
            "AI classification is not proof that an allegation is true.</div></div></div>")


def timeline_html(c_: dict, p: dict) -> str:
    ev = sorted(c_.get("audit", []), key=lambda e: (e["ts"], e["id"]))
    rows = []
    for e in ev:
        det = (f"<details><summary>details</summary><div class='muted'>Event {esc(e['id'])} · {ist_stamp(e['ts'])} · "
               f"{esc(e.get('details') or e['text'])}</div></details>" if p["timeline"] == "Detailed" else "")
        stt = f" {status_badge(e['status'], {'Dispatched': 'blue', 'On scene': 'green', 'Transporting': 'orange', 'Completed': 'done', 'Cancelled': 'muted', 'Recommended': 'gray', 'Approved': 'blue', 'Verified': 'green', 'Closed': 'done', 'Attention': 'red'}.get(e['status'], 'gray'))}" if e.get("status") else ""
        rows.append(f"<div class='ev'><time>{ist_time(e['ts'])}</time><span>{e['icon']}</span><div><span class='t'>{esc(e['type'])}</span>{stt}"
                    f"{(' · <b>' + esc(e['unit']) + '</b>') if e.get('unit') else ''}<div>{esc(e['text'])}</div>"
                    f"<div class='a'>by {esc(e['actor'])}</div>{det}</div></div>")
    return "<div class='cc'><div class='panel tl2'>" + ("".join(rows) or "<div class='muted'>No events yet.</div>") + "</div></div>"


def agencies_html(c_: dict, p: dict) -> str:
    by = {}
    for a_ in c_.get("assignments", []):
        if a_["dept"] in p["agencies"]:
            by.setdefault(a_["dept"], []).append(a_)
    if not by:
        return "<div class='cc'><div class='panel muted'>No agencies involved yet.</div></div>"
    cards = []
    for d, units in by.items():
        lead = any(a_.get("role") == "lead" for a_ in units)
        rows = "".join(f"<div style='margin-top:6px'><b>{esc(a_.get('unit') or a_.get('ticket') or '—')}</b> · {esc(a_.get('why', ''))}<br>"
                       f"{status_badge(*unit_state(a_))}</div>" for a_ in units)
        cards.append(f"<div class='panel {'lead' if lead else ''}'><div><b>{DEPARTMENTS[d]['icon']} {esc(d)}</b> "
                     f"<span class='badge {'c-blue' if lead else 'c-gray'}'>{'Lead agency' if lead else 'Supporting'}</span></div>"
                     f"<div class='muted' style='font-size:.8em'>{esc(DEPARTMENTS[d]['agency'])} · ☎ {esc(DEPARTMENTS[d]['lines'])}</div>{rows}</div>")
    return "<div class='cc'><div class='ag'>" + "".join(cards) + "</div></div>"


def india_map(incs: list[dict], selected: str | None, p: dict, center=None):
    """Interactive India map: verified incident locations only (unit GPS is not available in this demo)."""
    tiles = {"Dark": "CartoDB dark_matter", "Light": "CartoDB positron", "Street": "OpenStreetMap"}
    sel = next((c_ for c_ in incs if c_["ref"] == selected), None)
    if center is None and sel and sel["slots"].get("geo") and sel["slots"].get("location_ok"):
        center = (sel["slots"]["geo"]["lat"], sel["slots"]["geo"]["lon"])
    m = folium.Map(location=center or (22.6, 79.0), zoom_start=13 if center else p["zoom"], tiles=None, control_scale=True,
                   max_bounds=True, min_lat=5, max_lat=38, min_lon=66, max_lon=99)
    for name, t in tiles.items():
        folium.TileLayer(t, name=f"Base map: {name}", show=(name == p["tiles"])).add_to(m)
    fg = folium.FeatureGroup(name="Incidents", show="Incidents" in p["layers"])
    colours = {"red": "#e5484d", "orange": "#f0883e", "amber": "#e2b33c", "green": "#3fb87f", "blue": "#4c8ff0"}
    for c_ in incs:
        g = c_["slots"].get("geo")
        if not (g and c_["slots"].get("location_ok")):
            continue                                    # only verified locations go on the CAD map
        lab, ic, col = PRIO_STYLE.get(c_.get("level") or "B")
        tip = f"{c_['ref']} · {lab} · {incident_label(c_)}"
        folium.CircleMarker((g["lat"], g["lon"]), radius=14 if c_["ref"] == selected else 9, color="#ffffff" if c_["ref"] == selected else colours[col],
                            weight=3 if c_["ref"] == selected else 1.5, fill=True, fill_color=colours[col], fill_opacity=.85,
                            tooltip=tip, popup=folium.Popup(f"<b>{esc(c_['ref'])}</b><br>{esc(incident_label(c_))}<br>{lab} · "
                                                            f"{esc(incident_status(c_)[0])}<br>{esc(g['name'][:90])}", max_width=260)).add_to(fg)
    fg.add_to(m)
    folium.FeatureGroup(name="Response units (needs live GPS / AVL)", show=False).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def beep_wav() -> bytes:
    import numpy as np
    t = np.linspace(0, .35, int(16000 * .35), endpoint=False)
    tone = (0.25 * np.sin(2 * np.pi * 880 * t) * (t < .15) + 0.25 * np.sin(2 * np.pi * 660 * t) * (t >= .2)) * 32767
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(tone.astype("<i2").tobytes())
    return buf.getvalue()


def settings_panel(p: dict) -> None:
    st.markdown("**Dashboard settings**")
    c1, c2 = st.columns(2)
    p["theme"] = c1.radio("Theme", ["Dark", "Light"], index=["Dark", "Light"].index(p["theme"]), horizontal=True, key="cc_theme")
    p["density"] = c2.radio("Layout", ["Comfortable", "Compact"], index=["Comfortable", "Compact"].index(p["density"]), horizontal=True, key="cc_density")
    p["clock"] = c1.radio("Time format", ["12-hour", "24-hour"], index=["12-hour", "24-hour"].index(p["clock"]), horizontal=True, key="cc_clock")
    p["timeline"] = c2.radio("Timeline", ["Detailed", "Compact"], index=["Detailed", "Compact"].index(p["timeline"]), horizontal=True, key="cc_tl")
    p["sections"] = st.multiselect("Sections shown", CC_DEFAULTS["sections"], default=p["sections"], key="cc_sections")
    p["cards"] = st.multiselect("Overview cards", CC_DEFAULTS["cards"], default=p["cards"], key="cc_cards")
    p["columns"] = st.multiselect("Queue columns", CC_DEFAULTS["columns"], default=p["columns"], key="cc_cols")
    p["agencies"] = st.multiselect("Agencies shown", list(DEPARTMENTS), default=p["agencies"], key="cc_ag")
    c3, c4 = st.columns(2)
    p["zoom"] = c3.slider("Default map zoom", 4, 12, int(p["zoom"]), key="cc_zoom")
    p["tiles"] = c4.selectbox("Base map", ["Dark", "Light", "Street"], index=["Dark", "Light", "Street"].index(p["tiles"]), key="cc_tiles")
    p["layers"] = st.multiselect("Map layers", ["Incidents"], default=p["layers"], key="cc_layers")
    p["refresh"] = st.select_slider("Refresh every (seconds)", [2, 3, 5, 10, 30], value=p["refresh"], key="cc_refresh")
    store_ = cad_store()
    store_["approval"] = st.radio("Dispatch approval (control-room policy, applies to every call)", ["auto", "required"],
                                  index=["auto", "required"].index(store_.get("approval", "auto")),
                                  format_func=lambda x: {"auto": "Auto-approve (DEMO only)", "required": "Dispatcher must approve"}[x],
                                  horizontal=True, key="cc_approval")
    c5, c6, c7 = st.columns(3)
    p["sound"] = c5.toggle("Sound on new P1", value=p["sound"], key="cc_sound")
    p["large_text"] = c6.toggle("Larger text", value=p["large_text"], key="cc_large")
    p["high_contrast"] = c7.toggle("High contrast", value=p["high_contrast"], key="cc_hc")
    p["reduce_motion"] = c5.toggle("Reduce motion", value=p["reduce_motion"], key="cc_rm")
    p["profile"] = st.text_input("Dispatcher name", value=p["profile"], key="cc_profile")
    p["role"] = st.selectbox("Role", ["Dispatch Officer", "Communication Officer", "Shift Supervisor"],
                             index=["Dispatch Officer", "Communication Officer", "Shift Supervisor"].index(p["role"]), key="cc_role")
    if st.button("Save as my defaults", key="cc_save"):
        cad_store()["prefs"][p["profile"]] = dict(p)
        st.success("Saved on this server for this dispatcher name (until the app restarts).")
    st.caption("Theme applies to the Command Center panels; Streamlit's own controls follow the app theme (⋮ menu → Settings).")


st.session_state.used_vehicle_ids = cad_store()["used_units"]     # one fleet for every session

tab_cc, tab_call, tab_dept, tab_acc, tab2 = st.tabs(["🖥️ Command Center", "📞 Emergency call", "🏢 Departments & agents",
                                                    "📊 Accuracy & observability", "🚑 Dispatch board"])


with tab_cc:
    P_ = cc_prefs()
    st.markdown(cc_css(P_), unsafe_allow_html=True)
    pin = _secret("DISPATCHER_PIN")
    if pin and not st.session_state.get("cc_auth"):
        st.markdown("#### 🔒 Dispatcher sign-in")
        entered = st.text_input("Dispatcher PIN", type="password", key="cc_pin")
        if st.button("Sign in", key="cc_signin"):
            if entered == pin:
                st.session_state.cc_auth = True
                st.rerun()
            st.error("Wrong PIN.")
    else:
        incs_now = all_incidents()
        awaiting = sum(1 for c_ in incs_now for a_ in c_.get("assignments", []) if a_.get("approval") == "awaiting")
        escal = [c_ for c_ in incs_now if c_.get("escalated") and incident_status(c_)[0] != "Completed"]

        t1, t2, t3 = st.columns([6, 1.1, 1.1])
        with t1:
            @st.fragment(run_every=1.0)
            def top_bar() -> None:
                st.markdown(
                    "<div class='cc'><div class='cc-top'><span class='brand'>ERSS-112 CAD</span>"
                    f"<span class='env'>{esc(APP_ENV)}</span><span class='clock'>🕒 {ist_stamp()}</span>"
                    "<span class='st'><span class='dot' style='background:var(--amber)'></span>Control room link: <b>simulated</b></span>"
                    f"<span class='st'>AI: <b>{'Groq' if use_groq else 'rules'}</b> · Tracing: <b>{'on' if ls_client else 'off'}</b></span>"
                    f"<span class='st sp'>👤 <b>{esc(P_['profile'])}</b> · {esc(P_['role'])}</span></div></div>", unsafe_allow_html=True)
            top_bar()
        with t2.popover(f"🔔 {awaiting + len(escal)}"):
            st.markdown("**Notifications**")
            for c_ in incs_now:
                for a_ in c_.get("assignments", []):
                    if a_.get("approval") == "awaiting":
                        st.markdown(f"⏳ **{c_['ref']}**: {a_.get('unit') or a_['dept']} awaiting approval")
            for c_ in escal:
                st.markdown(f"⚠️ **{c_['ref']}**: {'; '.join(c_['escalated'])}")
            if not awaiting and not escal:
                st.caption("Nothing needs attention.")
        with t3.popover("⚙️ Settings"):
            settings_panel(P_)
        if not pin:
            st.caption(f"{APP_ENV} mode: open access. Set DISPATCHER_PIN in Secrets to restrict this page to dispatchers. "
                       "All incidents come from calls made in this app; nothing is invented.")

        @st.fragment(run_every=float(P_["refresh"]))
        def live_board() -> None:
            p = cc_prefs()
            incs = all_incidents()
            for c_ in incs:                                  # keep every open incident moving, even if its caller has left
                if c_.get("phase") == "connected":
                    check_arrivals(c_)
            if p["sound"]:
                seen = st.session_state.setdefault("cc_seen_p1", set())
                new_p1 = [c_["ref"] for c_ in incs if c_.get("level") in ("D", "E") and c_["ref"] not in seen]
                if new_p1:
                    seen.update(new_p1)
                    st.audio(beep_wav(), format="audio/wav", autoplay=True)
            k = kpis(incs)
            if "Overview" in p["sections"] and p["cards"]:
                cols = st.columns(len(p["cards"]))
                for col, name in zip(cols, p["cards"]):
                    flt = st.session_state.get("cc_filter")
                    if col.button(f"**{k[name]}**\n{name}" + (" ✓" if flt == name else ""), key=f"kpi_{name.replace(' ', '_')}",
                                  width="stretch", help="Filter the incident queue"):
                        st.session_state.cc_filter = None if flt == name else name
                        st.rerun(scope="fragment")
            if "Queue" in p["sections"]:
                f1, f2, f3, f4 = st.columns([2.2, 1.3, 1.3, 1.2])
                q = f1.text_input("Search", placeholder="ID, incident, place, unit…", key="cc_q", label_visibility="collapsed")
                fp = f2.multiselect("Priority", ["P1", "P2", "P3", "P4"], key="cc_fp", placeholder="Priority", label_visibility="collapsed")
                fs = f3.multiselect("Status", ["Intake", "Verifying location", "Awaiting assignment", "Awaiting approval", "Units en route",
                                               "On scene", "Completed", "Information only"], key="cc_fs", placeholder="Status",
                                    label_visibility="collapsed")
                srt = f4.selectbox("Sort", ["Newest", "Priority", "Status"], key="cc_sort", label_visibility="collapsed")
                rows = []
                for c_ in incs:
                    stt = incident_status(c_)[0]
                    lab = PRIO_STYLE[c_["level"]][0] if c_.get("level") else "—"
                    units_ = ", ".join(a_.get("unit") or a_.get("ticket") or a_["dept"] for a_ in c_.get("assignments", []) if not a_.get("cancelled"))
                    rows.append({"ID": c_["ref"], "Incident": incident_label(c_), "Priority": lab,
                                 "Location": (c_["slots"].get("location") or "—") + ("" if c_["slots"].get("location_ok") else " (unverified)"),
                                 "Received": ist_time(c_.get("created", c_["started"])), "Status": stt, "Assigned": units_ or "—",
                                 "_lvl": "EDCBAO".index(c_["level"]) if c_.get("level") else 9, "_t": c_.get("created", 0)})
                flt = st.session_state.get("cc_filter")
                rows = [r for r in rows if (not q or q.lower() in " ".join(str(v) for v in r.values()).lower())
                        and (not fp or r["Priority"] in fp) and (not fs or r["Status"] in fs)
                        and (flt in (None, "Active incidents", "Units available", "Average response")
                             or (flt == "P1 critical" and r["Priority"] == "P1") or (flt == "P2 urgent" and r["Priority"] == "P2")
                             or (flt == "Units dispatched" and r["Status"] in ("Units en route", "On scene"))
                             or (flt == "Awaiting assignment" and r["Status"] in ("Verifying location", "Awaiting assignment", "Awaiting approval", "Intake")))
                        and (flt != "Active incidents" or r["Status"] not in ("Completed", "Information only"))]
                rows.sort(key=lambda r: (r["_lvl"], -r["_t"]) if srt == "Priority" else (r["Status"], -r["_t"]) if srt == "Status" else (-r["_t"],))
                if rows:
                    df_ = pd.DataFrame(rows)[[c for c in p["columns"] if c in rows[0]]]
                    ev_ = st.dataframe(df_, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                       key="cc_queue", height=min(38 * (len(df_) + 1) + 4, 300))
                    sel_rows = ev_.selection.rows if ev_ and hasattr(ev_, "selection") else []
                    if sel_rows:
                        new_sel = df_.iloc[sel_rows[0]]["ID"] if "ID" in df_ else rows[sel_rows[0]]["ID"]
                        if new_sel != st.session_state.get("cc_sel"):
                            st.session_state.cc_sel = new_sel
                            st.rerun(scope="app")                # keep the map and details in step
                    if not st.session_state.get("cc_sel"):
                        st.session_state.cc_sel = rows[0]["ID"]
                else:
                    st.info("No incidents match. Incidents appear here when someone makes an 📞 Emergency call.")

            sel = next((c_ for c_ in incs if c_["ref"] == st.session_state.get("cc_sel")), None)
            if sel is None:
                return
            if "CAD decision" in p["sections"]:
                with st.expander(f"CAD decision · CFS {sel['ref']}", expanded=True):
                    st.markdown(cad_card_html(sel), unsafe_allow_html=True)
                    pend = [a_ for a_ in sel.get("assignments", []) if a_.get("approval") == "awaiting"]
                    act = [a_ for a_ in sel.get("assignments", []) if not a_.get("cancelled") and unit_state(a_)[0] not in ("Completed",)]
                    b1, b2, b3 = st.columns([1.2, 1.6, 1.2])
                    who = f"{p['profile']} ({p['role']})"
                    if pend and b1.button(f"✅ Approve all ({len(pend)})", type="primary", key=f"appr_{sel['ref']}", width="stretch"):
                        for a_ in pend:
                            approve_unit(sel, a_, who)
                        st.rerun(scope="app")
                    if act:
                        pick = b2.selectbox("Unit", [a_.get("unit") or a_.get("ticket") or a_["dept"] for a_ in act],
                                            key=f"pick_{sel['ref']}", label_visibility="collapsed")
                        tgt = next(a_ for a_ in act if (a_.get("unit") or a_.get("ticket") or a_["dept"]) == pick)
                        if tgt.get("approval") == "awaiting" and b3.button("Approve unit", key=f"ap1_{sel['ref']}", width="stretch"):
                            approve_unit(sel, tgt, who)
                            st.rerun(scope="app")
                        elif tgt.get("approval") != "awaiting" and b3.button("Cancel unit", key=f"can_{sel['ref']}", width="stretch"):
                            cancel_unit(sel, tgt, who, "cancelled by the dispatcher")
                            st.rerun(scope="app")
            if "Agencies" in p["sections"]:
                st.markdown("##### Agency coordination")
                st.markdown(agencies_html(sel, p), unsafe_allow_html=True)
            if "Timeline" in p["sections"]:
                st.markdown(f"##### Activity timeline · CFS {sel['ref']}")
                st.markdown(timeline_html(sel, p), unsafe_allow_html=True)

        left_, right_ = st.columns([1.25, 1], gap="medium") if "Map" in P_["sections"] else (st.container(), None)
        with left_:
            live_board()
        if right_ is not None:
            with right_:
                st.markdown("##### India map")
                incs_m = all_incidents()
                if folium is None:
                    st.warning("The map needs the folium and streamlit-folium packages (see requirements.txt).")
                else:
                    sel_ref = st.session_state.get("cc_sel")
                    out_ = st_folium(india_map(incs_m, sel_ref, P_, st.session_state.get("cc_center")), key="cc_map", height=470,
                                     use_container_width=True, returned_objects=["last_object_clicked_tooltip"])
                    tip = (out_ or {}).get("last_object_clicked_tooltip")
                    if tip and tip.split(" · ")[0] != sel_ref and st.session_state.get("cc_last_tip") != tip:
                        st.session_state.cc_last_tip = tip
                        st.session_state.cc_sel, st.session_state.cc_center = tip.split(" · ")[0], None
                        st.rerun()
                    n_unv = sum(1 for c_ in incs_m if c_["slots"].get("location") and not c_["slots"].get("location_ok"))
                    st.caption("Only verified locations are shown. " + (f"{n_unv} incident(s) still need location verification. " if n_unv else "")
                               + "Response units appear only with live GPS / AVL data, which this demo does not have.")
                st.markdown("##### Find a place")
                place_q = st.text_input("Search any place in India", placeholder="e.g. India Gate, New Delhi · Sector 62 Noida · Majestic Bengaluru",
                                        key="cc_place", label_visibility="collapsed")
                if place_q:
                    res = geocode_india(place_q.strip())
                    if res is None:
                        st.error("The map service could not be reached. Try again, or confirm the place with the caller.")
                    elif not res:
                        st.warning("No match in India. Try a nearby landmark, locality, district or state.")
                    else:
                        choice = st.radio("Matches", [r_["name"] for r_ in res], key="cc_place_pick", label_visibility="collapsed")
                        g_ = next(r_ for r_ in res if r_["name"] == choice)
                        m1, m2 = st.columns(2)
                        if m1.button("Show on map", key="cc_show", width="stretch"):
                            st.session_state.cc_center = (g_["lat"], g_["lon"])
                            st.rerun()
                        sel_c = cad_store()["incidents"].get(st.session_state.get("cc_sel") or "")
                        if sel_c and m2.button(f"Set as location of {sel_c['ref']}", key="cc_setloc", width="stretch"):
                            old = sel_c["slots"].get("location")
                            sel_c["slots"].update(geo=g_, geo_state="one", location_ok=True, location=sel_c["slots"].get("location") or place_q.strip())
                            add_event(sel_c, "🧑‍💼", f"Location set by dispatcher: {g_['name']} (caller said: {old or 'nothing'})",
                                      etype="Location verified", actor=f"{P_['profile']} ({P_['role']})")
                            st.session_state.cc_center = None
                            st.rerun()

with tab_call:
    st.markdown(DIALER_CSS, unsafe_allow_html=True)
    call = st.session_state.get("call")
    if call is None or call.get("ended") and st.session_state.get("call_closed_seen"):
        # ---------------- start screen ----------------
        st.markdown(
            "<div style='background:linear-gradient(135deg,#15202B,#1C5D99);color:#fff;border-radius:18px;padding:16px 22px;margin-bottom:8px'>"
            "<div style='font-size:1.5rem;font-weight:800'>📞 Emergency call · 112</div>"
            "<div style='opacity:.9'>Talk to the call-taker as you would on the phone. It asks what happened and where, sends the right teams "
            "as soon as it knows, keeps asking the important questions, and tells you what to do while help comes.</div></div>",
            unsafe_allow_html=True)
        st.caption("Training simulation: no real vehicle is sent. In a real emergency in India, call 112.")
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        lang_pick = c1.radio("Language", ["English", "हिन्दी (Hindi)"], horizontal=True, key="call_lang")
        c2.toggle("Hands-free", value=True, key="call_hands",
                  help="The call-taker reads each question aloud, then the microphone starts listening by itself.")
        if c3.button("📞 Start emergency call", type="primary", width="stretch"):
            call = new_call(lang_pick.startswith("ह"))
            call["dialed"], call["dial_hint"] = "112", None
            call["ani"] = "+91 9" + "".join(str(random.randint(0, 9)) for _ in range(4)) + " " + "".join(str(random.randint(0, 9)) for _ in range(5))
            call["slots"]["phone"] = re.sub(r"\D", "", call["ani"])[-10:]       # the system already knows the number (ANI)
            call["slots"]["phone_src"] = "ANI"
            call["phase"], call["started"] = "connected", time.time()
            call["events"] = []
            add_event(call, "📞", "Call received at the 112 response centre", etype="Call received", actor="Caller")
            agent_say(call, _l(CALL_Q["problem"], call["hindi"]))
            call["asked"].append("problem")
            st.session_state.call, st.session_state.call_closed_seen = call, False
            st.rerun()
        if call and call.get("ended"):
            st.info(f"Last call {call['ref']} has ended. Start a new call when you need to.")
    elif call.get("phase") == "ringing":
        call.setdefault("ani", "+91 9" + "".join(str(random.randint(0, 9)) for _ in range(4)) + " " + "".join(str(random.randint(0, 9)) for _ in range(5)))
        # ---------------- ringing, then connected ----------------
        with st.container(key="phone"):
            st.markdown(f"<div class='ring'><div class='bub'>📞</div><div class='t1'>Calling {call['dialed']}…</div>"
                        f"<div class='t2'>{DIAL_SERVICES[call['dialed']][0]} · Ringing</div></div>", unsafe_allow_html=True)
            if st.button("Cancel", key="ring_cancel", width="stretch"):
                st.session_state.call = None
                st.rerun()

        @st.fragment(run_every=0.5)
        def ringing() -> None:
            c_ = st.session_state.get("call")
            if c_ and c_.get("phase") == "ringing" and time.time() - c_["ring_start"] >= 2.8:
                c_["phase"], c_["started"] = "connected", time.time()
                c_["slots"]["phone"] = re.sub(r"\D", "", c_.get("ani", ""))[-10:] or None   # the system already knows the number
                c_["slots"]["phone_src"] = "ANI"
                c_["events"] = [("📞", f"Call from {c_['dialed']} connected to the 112 response centre", time.time())]
                svc = DIAL_SERVICES[c_["dialed"]][0]
                agent_say(c_, _l(CALL_Q["problem"], c_["hindi"]) if c_["dialed"] == "112" else _l(DIAL_GREETING, c_["hindi"]).format(svc=svc))
                c_["asked"].append("problem")
                st.rerun(scope="app")
        ringing()
    else:
        check_arrivals(call)
        if not call["ended"]:
            bar1, bar2, bar3, bar5, bar4 = st.columns([3.0, 1.0, 1.0, 1.1, 1.0])
            bar5.toggle("✍️ Review", value=False, key="call_review", help="Check and edit what speech-to-text heard before it is sent.")
            bar1.markdown(f"<div class='callbar'><span class='dot'></span><div><b>Connected · {call.get('dialed', '112')} · "
                          f"since {ist_time(call['started'])} IST</b><br>"
                          f"<span>112 Emergency Response Centre · Caller ID (ANI) {call.get('ani', 'withheld')}</span></div></div>",
                          unsafe_allow_html=True)
            bar2.toggle("🎤 Auto-listen", value=True, key="call_hands", help="After each question, the microphone listens by itself.")
            bar3.toggle("🔊 Speaker", value=True, key="call_speaker", help="The call-taker's voice is read aloud.")
            if bar4.button("End call", key="end_call", width="stretch"):
                call["ended"], call["ended_by"], call["ended_at"] = True, "caller", time.time()
                add_event(call, "🔚", "Call ended by the caller")
                agent_say(call, _l(SAY["ended"], call["hindi"]).format(ref=call["ref"]))
                st.rerun()
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            box = st.container(height=430, border=True)
            with box:
                for t_ in call["turns"]:
                    if t_["role"] == "crew":
                        with st.chat_message("assistant", avatar=t_["who"].split()[0]):
                            st.markdown(f"**{t_['who'].split(' ', 1)[-1]}**  \n{t_['text']}")
                        continue
                    with st.chat_message("assistant" if t_["role"] == "agent" else "user",
                                         avatar="🧑‍💼" if t_["role"] == "agent" else "🙋"):
                        st.markdown(t_["text"])
                        if t_.get("raw") and t_["raw"].strip() != t_["text"].strip():
                            st.caption(f"Heard as: “{t_['raw']}”")
                        if t_.get("audio"):
                            st.audio(t_["audio"], format="audio/wav")
            lang = "hi-IN" if call["hindi"] else "en-IN"
            answer = None
            if not call["ended"]:
                last_agent = next((t_ for t_ in reversed(call["turns"]) if t_["role"] != "caller"), None)
                if VOICE_TYPING is not None:
                    res = VOICE_TYPING(key=f"call_live_{lang}",
                                       data={"lang": lang, "say": last_agent["text"] if last_agent else "",
                                             "say_id": f"{call['ref']}-{len(call['turns'])}", "auto": st.session_state.get("call_hands", True),
                                             "speak": st.session_state.get("call_speaker", True)},
                                       on_done_change=lambda: None, on_env_change=lambda: None)
                    done = getattr(res, "done", None)
                    if done and done.get("id") != st.session_state.get("call_live_id"):
                        st.session_state.call_live_id = done["id"]
                        audio = base64.b64decode(done["audio"]) if done.get("audio") else None
                        raw = (done.get("text") or "").strip()
                        text = clean_heard(raw, call)
                        if not text and audio and wav_stats(audio)["peak"] >= 0.01:
                            try:
                                text, _eng = transcribe_audio(audio, lang, groq_key, whisper_model)
                            except RuntimeError:
                                text = ""
                        if text and st.session_state.get("call_review", False):
                            st.session_state.call_draft = {"text": text, "audio": audio, "raw": raw, "id": done["id"]}
                        elif text:
                            answer = (text, audio, raw)
                        else:
                            agent_say(call, _l(SAY["not_heard"], call["hindi"]))
                            st.rerun()
                draft = st.session_state.get("call_draft")
                if draft:
                    with st.container(border=True):
                        st.markdown("**✍️ Check what was heard before sending**")
                        fixed = st.text_area("Recognised words (edit if needed)", value=draft["text"], key=f"draft_{draft['id']}", height=70)
                        if draft["raw"] and draft["raw"] != draft["text"]:
                            st.caption(f"Raw speech-to-text: “{draft['raw']}”")
                        k1, k2 = st.columns(2)
                        if k1.button("Send", type="primary", key=f"draft_send_{draft['id']}", width="stretch"):
                            answer = (fixed.strip(), draft["audio"], draft["raw"])
                            st.session_state.call_draft = None
                        if k2.button("Discard and speak again", key=f"draft_drop_{draft['id']}", width="stretch"):
                            st.session_state.call_draft = None
                            st.rerun()
                typed = st.chat_input("Or type your answer…", key="call_typed")
                if typed:
                    answer = (typed, None, typed)
                if call.get("pending") == "location_again" or (call["slots"].get("problems") and not call["slots"].get("location")
                                                               and "location_again" in call["asked"]):
                    pick = st.selectbox("Optional: pick a known place (you can also just say or type any place)",
                                        [""] + sorted(n for n in LANDMARKS if n in DEMO_LANDMARKS), key=f"call_pick_{len(call['turns'])}")
                    if pick:
                        answer = (pick, None)
                if answer:
                    extra_ = ls_extra("emergency-call", call["ref"])
                    caller_says(call, answer[0], langsmith_extra=extra_)
                    call.setdefault("run_ids", []).append(str(extra_["run_id"]))
                    if call["dispatched"] and not call.get("dispatch_run"):
                        call["dispatch_run"] = str(extra_["run_id"])     # feedback about routing goes on this turn
                    mine = next((t_ for t_ in reversed(call["turns"]) if t_["role"] == "caller"), None)
                    if mine is not None:
                        if answer[1]:
                            mine["audio"] = answer[1]                   # the caller's turn keeps its recording
                        mine["raw"] = answer[2] if len(answer) > 2 else answer[0]   # exactly what speech-to-text produced
                    st.rerun()
                st.download_button("Download transcript", call_transcript(call).encode("utf-8"), f"{call['ref']}.txt", "text/plain")
            else:
                st.success(f"Call {call['ref']} ended. The teams keep working; you can still follow them on the right.")
                d1, d2 = st.columns(2)
                d1.download_button("Download transcript", call_transcript(call).encode("utf-8"), f"{call['ref']}.txt",
                                   "text/plain", width="stretch")
                with st.container(border=True):
                    st.markdown("**Was this call handled correctly?** Your answer measures accuracy"
                                + (" and is sent to LangSmith." if ls_client else "."))
                    if call.get("feedback_done"):
                        st.success("Thanks: " + call["feedback_done"])
                    else:
                        ok = st.feedback("thumbs", key=f"fb_ok_{call['ref']}")
                        sent = [a_["dept"] for a_ in call["assignments"]]
                        should = st.multiselect("Departments that should have been sent", list(DEPARTMENTS), default=list(dict.fromkeys(sent)),
                                                key=f"fb_dep_{call['ref']}")
                        sev_ok = st.select_slider("Correct severity", options=[1, 2, 3, 4, 5], value=call["severity"],
                                                  key=f"fb_sev_{call['ref']}")
                        note = st.text_input("Comment (optional)", key=f"fb_note_{call['ref']}")
                        if st.button("Send feedback", key=f"fb_send_{call['ref']}"):
                            p_, e_ = set(sent), set(should)
                            f1 = (2 * len(p_ & e_) / (len(p_) + len(e_))) if (p_ or e_) else 1.0
                            scores = {"routing_correct": float(ok == 1) if ok is not None else float(p_ == e_),
                                      "department_f1": round(f1, 3),
                                      "severity_error": float(abs(call["severity"] - sev_ok))}
                            call["feedback_done"] = send_feedback(call.get("dispatch_run") or (call.get("run_ids") or [None])[-1],
                                                                  scores, note, {"departments": should, "severity": sev_ok})
                            st.rerun()
                if d2.button("📞 New call", type="primary", width="stretch"):
                    st.session_state.call_closed_seen = True
                    st.session_state.call = None
                    st.rerun()

        with right:
            @st.fragment(run_every=1.0)
            def call_panel() -> None:
                c_ = st.session_state.get("call")
                if not c_:
                    return
                if check_arrivals(c_):
                    st.rerun(scope="app")                       # a unit arrived: show (and say) it in the call
                st.markdown(incident_card_html(c_), unsafe_allow_html=True)
            call_panel()
            geo_ = call["slots"].get("geo")
            if geo_ and call["slots"].get("location_ok"):
                st.markdown(f"##### 🗺️ CAD map · verified location  \n<span style='opacity:.7;font-size:.85rem'>{geo_['name']}</span>",
                            unsafe_allow_html=True)
                st.map(pd.DataFrame([{"lat": geo_["lat"], "lon": geo_["lon"]}]), zoom=14, size=40, color="#D62839")
            elif call["slots"].get("location"):
                st.caption("🗺️ The incident appears on the CAD map once its location is verified.")

DEPT_CSS = """
<style>
.dh{border-radius:16px;margin-bottom:14px;background:rgba(128,128,128,.07);border:1px solid rgba(128,128,128,.25);
  border-top:5px solid #FF9933;border-bottom:5px solid #138808}
.dh>div{padding:14px 20px}
.dh b{font-size:1.45rem}.dh p{margin:4px 0 0;opacity:.8}
.dflow{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:10px;margin:6px 0 4px}
@media (max-width:1100px){.dflow{grid-template-columns:repeat(2,minmax(0,1fr))}}
.dfs{position:relative;border:1px solid rgba(128,128,128,.3);border-radius:14px;padding:12px;background:rgba(128,128,128,.06)}
.dfs .i{font-size:1.5rem}.dfs b{display:block;margin:4px 0 2px;font-size:.95rem}.dfs p{margin:0;font-size:.78rem;opacity:.78;line-height:1.35}
.dfs .tag{display:inline-block;margin-top:6px;font-size:.68rem;font-weight:700;border-radius:999px;padding:2px 8px;background:rgba(28,93,153,.18);color:#5aa2e8}
.dfs.cit{border-color:#D62839;background:rgba(214,40,57,.08)}.dfs.end{border-color:#1F8A5B;background:rgba(31,138,91,.08)}
.dfs.cad{border-color:#E46A1E;background:rgba(228,106,30,.08)}
.sup2{border:1px dashed #C98A0C;border-radius:12px;padding:8px 12px;font-size:.84rem;background:rgba(201,138,12,.08);margin:6px 0 18px}
.trace{position:relative;margin-left:8px;padding-left:22px;border-left:2px solid rgba(128,128,128,.25)}
.ts{position:relative;margin:0 0 10px;border:1px solid rgba(128,128,128,.22);border-radius:12px;padding:10px 12px;background:rgba(128,128,128,.04)}
.ts:before{content:"";position:absolute;left:-30px;top:14px;width:12px;height:12px;border-radius:50%;background:#1C5D99;border:2px solid #0e141a}
.ts.tcad:before{background:#E46A1E}.ts.dept:before{background:#1F8A5B}.ts.sup:before{background:#C98A0C}
.ts .h{display:flex;gap:8px;align-items:center}.ts .h b{font-size:.95rem}.ts .h span.ms{margin-left:auto;font-size:.72rem;opacity:.55}
.ts .d{font-size:.86rem;opacity:.88;margin-top:3px}
.chips2 span{display:inline-block;border-radius:999px;padding:3px 10px;margin:0 6px 6px 0;font-size:.8rem;font-weight:650;background:rgba(128,128,128,.14)}
.boards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
@media (max-width:1100px){.boards{grid-template-columns:repeat(2,minmax(0,1fr))}}
.bd{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:12px 14px;background:rgba(128,128,128,.04)}
.bd .t{display:flex;align-items:center;gap:8px}.bd .t b{font-size:1rem}.bd .t .n{margin-left:auto;font-size:1.5rem;font-weight:800}
.bd .s{font-size:.76rem;opacity:.65;margin:2px 0 8px}
.bd .st{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px}.bd .st span{font-size:.7rem;border-radius:999px;padding:2px 8px;background:rgba(128,128,128,.14)}
.bd .j{font-size:.8rem;border-top:1px solid rgba(128,128,128,.16);padding:5px 0}
.bd .j em{font-style:normal;font-size:.7rem;border-radius:6px;padding:1px 6px;margin-left:4px;background:rgba(28,93,153,.2)}
.bd .j em.ok{background:rgba(31,138,91,.22)}.bd .j em.cl{background:rgba(128,128,128,.22)}
.bd .none{font-size:.8rem;opacity:.5}
</style>
"""


def unit_status(a_: dict) -> tuple[str, str]:
    """A unit's ERSS-style status and a css class."""
    if a_.get("released"):
        return "released", "cl"
    if a_.get("atr"):
        return "ATR closed", "cl"
    if a_.get("at_hospital"):
        return "at hospital", "ok"
    if a_.get("transport_at"):
        return "transporting", "ok"
    if a_.get("contact"):
        return "with caller", "ok"
    if a_.get("arrived"):
        return "reached", "ok"
    if a_.get("unit"):
        return "en route", ""
    return a_.get("status", "ticket"), ""


with tab_dept:
    st.markdown(DEPT_CSS, unsafe_allow_html=True)
    st.markdown("<div class='dh'><div><b>🇮🇳 ERSS-112: how a call reaches the right team</b>"
                "<p>One number for every emergency. Each agent does one job and passes the call on. The triage and router agents are "
                "constrained to fixed lists, the CAD sets the priority and the units, and anything unclear goes to a human supervisor.</p></div></div>",
                unsafe_allow_html=True)
    steps_html = [("cit", "📞", "Citizen", "Dials 112 and speaks in English or Hindi.", "112"),
                  ("", "🎧", "Communication Officer agent", "Answers, records the voice, writes it as text; caller number from the system.", "intake"),
                  ("", "🧭", "Triage agent", "Type, severity 1-5, location, people. Fixed answers only.", "constrained"),
                  ("", "🔀", "Router agent", "Finds every problem and the departments that must act.", "constrained"),
                  ("cad", "🧠", "CAD", "Priority P1-P4 and the run card: which units, and why.", "ERSS-112"),
                  ("", "🗺️", "Dispatch Officer", "Nearest ERV by GPS; conferences 101, 108 and helplines.", "dispatch"),
                  ("end", "📍", "Tracking & ATR", "Live status in the call, hand-over, Action Taken Report.", "citizen sees it")]
    st.markdown("<div class='dflow'>" + "".join(
        f"<div class='dfs {c}'><div class='i'>{i}</div><b>{t}</b><p>{d}</p><span class='tag'>{g}</span></div>" for c, i, t, d, g in steps_html)
        + "</div><div class='sup2'>🧑‍💼 <b>Supervisor (human in the loop)</b> checks any call the agents are unsure about: unknown place, "
          "unclear problem, low confidence. A call is never dismissed by an agent on its own.</div>", unsafe_allow_html=True)

    @st.fragment(run_every=2.0)
    def dept_live() -> None:
        """Trace and boards refresh every 2 seconds, so unit status stays in step with the call."""
        runs = st.session_state.get("agent_runs", [])
        st.markdown("#### 🔎 Latest call, step by step")
        if not runs:
            st.info("Make an **Emergency call**. Every agent's decision appears here, live.")
        else:
            pick = st.selectbox("Call for Service", [r_["ref"] for r_ in runs][::-1], key="agent_pick", label_visibility="collapsed")
            run_ = next(r_ for r_ in runs if r_["ref"] == pick)
            call_ = run_.get("call")
            base = [stp for stp in run_["steps"] if not stp["agent"].startswith("Supervisor")]
            sup = next((stp for stp in run_["steps"] if stp["agent"].startswith("Supervisor")), None)
            html_ = []
            for stp in base:
                html_.append(f"<div class='ts'><div class='h'><span>{stp['icon']}</span><b>{stp['agent']}</b><span class='ms'>{stp['ms']} ms</span></div>"
                             f"<div class='d'>{stp['did']}</div></div>")
            if call_ and call_.get("level"):
                lv = LEVELS[call_["level"]]
                card = "".join(f"<li>{DEPARTMENTS[cd['dept']]['icon']} <b>{cd['dept']}</b>: {cd['why']}</li>" for cd in call_.get("run_card", []))
                html_.append(f"<div class='ts tcad'><div class='h'><span>🧠</span><b>CAD decision (ERSS-112)</b></div>"
                             f"<div class='d'>Priority <b style='color:{lv[3]}'>{lv[0]}</b>: {lv[1]}. Run card:<ul style='margin:4px 0 0 18px'>{card}</ul></div></div>")
            for a_ in (call_["assignments"] if call_ else run_.get("assignments", [])):
                stt, _cls = unit_status(a_)
                who = AGENCY.get(a_["dept"], a_["dept"])
                what = (f"{a_['unit']} sent ({a_.get('why', '')}), about {a_['eta']:.0f} min" if a_.get("unit")
                        else f"Ticket {a_.get('ticket') or ''}: {a_.get('status', '')}")
                html_.append(f"<div class='ts dept'><div class='h'><span>{a_['icon']}</span><b>{a_['dept']} · {who}</b>"
                             f"<span class='ms'>{stt}</span></div><div class='d'>{what}</div></div>")
            if sup:
                html_.append(f"<div class='ts sup'><div class='h'><span>🧑‍💼</span><b>Supervisor</b></div><div class='d'>{sup['did']}</div></div>")
            chips_ = [f"CFS {run_['ref']}"]
            if call_ and call_.get("level"):
                chips_.append(LEVELS[call_["level"]][0])
                chips_ += list(dict.fromkeys(AGENCY.get(a_["dept"], a_["dept"]) for a_ in call_["assignments"]))
            st.markdown("<div class='chips2'>" + "".join(f"<span>{c}</span>" for c in chips_) + "</div>"
                        "<div class='trace'>" + "".join(html_) + "</div>", unsafe_allow_html=True)
            st.caption(f"Agents' processing time: {run_['total_ms']} ms.")

        st.markdown("#### 🏢 Department boards")
        jobs_by = {}
        for r_ in st.session_state.get("agent_runs", []):
            c_ = r_.get("call")
            for a_ in (c_["assignments"] if c_ else r_.get("assignments", [])):
                jobs_by.setdefault(a_["dept"], []).append((r_["ref"], c_, a_))
        cards = []
        for d, info in DEPARTMENTS.items():
            jobs = jobs_by.get(d, [])
            counts = {"en route": 0, "on scene": 0, "closed": 0}
            rows = []
            for ref, c_, a_ in reversed(jobs):
                stt, cls = unit_status(a_)
                counts["closed" if cls == "cl" else "on scene" if cls == "ok" else "en route"] += 1
                place = (c_ or {}).get("slots", {}).get("location") or "location pending"
                prob = ", ".join(PROBLEMS[p]["label"] for p in (c_ or {}).get("slots", {}).get("problems", [])) or "—"
                rows.append(f"<div class='j'><b>{ref}</b> {prob}<br>{place} · {a_.get('unit') or a_.get('ticket') or ''}<em class='{cls}'>{stt}</em></div>")
            cards.append(f"<div class='bd'><div class='t'><span style='font-size:1.3rem'>{info['icon']}</span><b>{d}</b><span class='n'>{len(jobs)}</span></div>"
                         f"<div class='s'>{info['agency']} · ☎ {info['lines']}</div>"
                         + (f"<div class='st'><span>🚦 en route {counts['en route']}</span><span>📍 on scene {counts['on scene']}</span>"
                            f"<span>✔ closed {counts['closed']}</span></div>" if jobs else "")
                         + ("".join(rows[:4]) if rows else "<div class='none'>No jobs yet.</div>") + "</div>")
        st.markdown("<div class='boards'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    dept_live()

    with st.expander("🇮🇳 How ERSS-112 works, step by step"):
        st.markdown("""
| Step | In ERSS-112 | In this app |
|---|---|---|
| One number for every emergency | **112** (100, 101, 108, 1091, 1098 also reach help) | 📞 Emergency call tab |
| Call answered at the state Emergency Response Centre (ERC) | Communication Officer; caller number and location from the network or the 112 India app | 🎧 Communication Officer agent, Caller ID (ANI) |
| Event created | Call for Service (CFS) with event type and priority | 🧭 Triage agent + 🧠 CAD priority P1-P4 |
| Decide who must act | Dispatch Officer picks the nearest police ERV by GPS and conferences the call to 101 fire, 108 ambulance or a helpline | 🔀 Router agent + CAD run card |
| Units respond | Assigned → en route → reached, updates on the mobile data terminal | 🚓 🚒 🚑 🛟 department agents, live status in the call |
| Specialist help | 181 / 1091 women, 1098 child, 1906 gas, 1912 electricity, SDRF / NDRF | 👩 🧒 🛢️ ⚡ tickets and call-backs |
| Job closed | Action Taken Report (ATR) | 📝 ATR in the timeline |
| Unclear calls | Supervisor at the ERC | 🧑‍💼 human in the loop |
""")
        st.caption("A simplified model for training: the priorities and run cards here are a design based on how ERSS-112 is described, "
                   "not the official ERSS code lists.")


with tab_acc:
    st.subheader("Accuracy & observability (LangSmith)")
    if not ls_client and ls_reason:
        st.error("Why it is not connected: " + ls_reason)
    if ls_client:
        if st.button("🔌 Test the LangSmith connection"):
            try:
                list(ls_client.list_projects(limit=1))
                st.success("The key works: LangSmith answered. Make a call, then look in Workspace 1 → Tracing projects → "
                           f"{ls_project}.")
            except Exception as exc:
                msg = str(exc)
                hint = (" The key was rejected: it may be deleted, expired, or for another workspace. Create a new key and update Secrets."
                        if "401" in msg or "403" in msg or "nauthorized" in msg or "orbidden" in msg else
                        " If your LangSmith account is in the EU, add LANGSMITH_ENDPOINT = \"https://eu.api.smith.langchain.com\" to Secrets."
                        if "404" in msg or "onnection" in msg else "")
                st.error(f"LangSmith did not accept the connection: {msg[:200]}.{hint}")
        st.success(f"LangSmith is connected. Every call is traced in project **{ls_project}**: each agent's input, output and time, "
                   "with a call's turns grouped as one thread. Open https://smith.langchain.com to explore them.")
    else:
        st.info("LangSmith is not connected, so traces are not sent; everything below still works in this session. To connect it, add "
                "these to **Manage app → Settings → Secrets** and reboot:\n\n"
                "```toml\nLANGSMITH_API_KEY = \"lsv2_...\"\nLANGSMITH_PROJECT = \"sahayak-112\"\n"
                "# LANGSMITH_ENDPOINT = \"https://eu.api.smith.langchain.com\"   # only for EU accounts\n```")
    st.markdown("#### 1. Live accuracy from feedback")
    fb_ = st.session_state.get("feedback_log", [])
    if not fb_:
        st.caption("After an emergency call ends, answer **Was this call handled correctly?** Each answer is a labelled example "
                   "for measuring accuracy (and is attached to the call's trace in LangSmith).")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calls rated", len(fb_))
        m2.metric("Routed correctly", f"{100 * sum(f['routing_correct'] for f in fb_) / len(fb_):.0f}%")
        m3.metric("Department F1", f"{sum(f['department_f1'] for f in fb_) / len(fb_):.2f}")
        m4.metric("Severity error", f"{sum(f['severity_error'] for f in fb_) / len(fb_):.1f}", help="Average distance from the correct severity (0 is perfect).")
        st.dataframe(pd.DataFrame(fb_).drop(columns=["correction"], errors="ignore"), hide_index=True, width="stretch")

    st.markdown("#### 2. Accuracy test on labelled calls")
    st.caption(f"{len(TEST_CALLS)} calls with the answers a trained call-taker would give (English, Hindi, mixed problems, a prank and "
               "one known trap). No vehicle is sent. Run it after every change to the rules, the model or the prompt.")
    c1, c2 = st.columns(2)
    if c1.button("▶️ Run accuracy test", type="primary"):
        rows = []
        with st.spinner("Testing…"):
            for text, probs, depts, sev, loc in TEST_CALLS:
                out = predict_routing(text, langsmith_extra={"tags": ["accuracy-test"], "metadata": {"app": "sahayak-112"}})
                sc = score_one(out, probs, depts, sev, loc)
                rows.append({"Call": text, "Expected teams": ", ".join(depts) or "none", "Predicted teams": ", ".join(out["departments"]) or "none",
                             "Teams ✓": "✅" if sc["departments_exact"] else "❌", "Lead ✓": "✅" if sc["lead_correct"] else "❌",
                             "Severity (exp/pred)": f"{sev} / {out['severity']}", "Location ✓": "✅" if sc["location_correct"] else "❌",
                             **{f"_{k}": v for k, v in sc.items()}})
        st.session_state.acc_rows = rows
    if ls_client and c2.button("☁️ Run as a LangSmith experiment"):
        with st.spinner("Uploading the dataset and running the experiment…"):
            try:
                st.success(f"Experiment **{run_langsmith_experiment()}** is in LangSmith (Datasets & Experiments → sahayak-112-routing).")
            except Exception as exc:
                st.error(f"LangSmith experiment failed: {str(exc)[:200]}")
    rows = st.session_state.get("acc_rows")
    if rows:
        df_ = pd.DataFrame(rows)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Teams exactly right", f"{100 * df_['_departments_exact'].mean():.0f}%")
        k2.metric("Department F1", f"{df_['_department_f1'].mean():.2f}")
        k3.metric("Lead team right", f"{100 * df_['_lead_correct'].mean():.0f}%")
        k4.metric("Severity within 1", f"{100 * df_['_severity_within_1'].mean():.0f}%")
        k5.metric("Location right", f"{100 * df_['_location_correct'].mean():.0f}%")
        st.dataframe(df_[[c for c in df_.columns if not c.startswith("_")]], hide_index=True, width="stretch",
                     column_config={"Call": st.column_config.TextColumn("Call", width="large")})
        st.caption("Method: " + ("Groq AI with constrained outputs" if use_groq else "keyword rules")
                   + ". ❌ rows are the cases to fix next, exactly like an error analysis.")

with tab2:
    st.subheader("Prioritized incident and dispatch board")
    emergency_calls = [c for c in st.session_state.calls if c.get("is_emergency") and not c.get("duplicate_of")]
    # A simple duplicate grouping by exact normalized location + incident type.
    grouped = {}
    for call in emergency_calls:
        key = (call.get("location"), call.get("incident_type"))
        if key not in grouped:
            grouped[key] = {
                "Incident ID": f"INC-{len(grouped) + 1:03d}",
                "Location": call.get("location"),
                "Incident type": call.get("incident_type"),
                "Severity": int(call.get("severity", 3)),
                "Calls": call.get("callers", 1),
                "Summary": call.get("summary", ""),
                "Call IDs": [call.get("Call ID")],
                "Help vehicle": call.get("Help vehicle"),
            }
        else:
            grouped[key]["Calls"] += call.get("callers", 1)
            grouped[key]["Severity"] = max(grouped[key]["Severity"], int(call.get("severity", 3)))
            grouped[key]["Call IDs"].append(call.get("Call ID"))
            grouped[key]["Help vehicle"] = grouped[key].get("Help vehicle") or call.get("Help vehicle")
    incidents = sorted(grouped.values(), key=lambda i: (-i["Severity"], i["Incident ID"]))
    if not incidents:
        st.info("Triage an emergency call first to populate the dispatch board.")
    else:
        board = pd.DataFrame(incidents)
        board["Location"] = board["Location"].fillna("Unknown: call back")
        st.dataframe(board[["Incident ID", "Location", "Incident type", "Severity", "Calls", "Summary"]], width="stretch")
        if st.button("Dispatch available vehicles", type="primary"):
            graph = add_closures(make_city_graph(), closure_rows, closure_cols)
            available = [v.copy() for v in VEHICLES if v["Vehicle ID"] not in st.session_state.used_vehicle_ids]
            results = []
            for inc in incidents:
                if inc.get("Help vehicle"):
                    results.append({"Incident ID": inc["Incident ID"], "Location": inc["Location"], "Severity": inc["Severity"],
                                    "Vehicle": inc["Help vehicle"], "Estimated travel (simulated min)": None,
                                    "Status": "Handled from Emergency Help (vehicle already sent)"})
                    continue
                assigned, minutes = choose_vehicle(inc["Incident type"], inc["Location"], graph, available)
                if assigned is None:
                    results.append({
                        "Incident ID": inc["Incident ID"], "Location": inc["Location"] or "Unknown",
                        "Severity": inc["Severity"], "Vehicle": "UNASSIGNED",
                        "Estimated travel (simulated min)": None,
                        "Status": "Needs location: call the reporter back" if not inc["Location"] else "No available reachable vehicle",
                    })
                    continue
                # Claim the vehicle exactly once for this dispatch batch.
                st.session_state.used_vehicle_ids.add(assigned["Vehicle ID"])
                available = [v for v in available if v["Vehicle ID"] != assigned["Vehicle ID"]]
                results.append({
                    "Incident ID": inc["Incident ID"], "Location": inc["Location"],
                    "Severity": inc["Severity"], "Vehicle": assigned["Vehicle ID"],
                    "Estimated travel (simulated min)": round(minutes, 1), "Status": "Assigned"
                })
            st.session_state.dispatches = results
            st.success("Dispatch batch completed. Assigned vehicles cannot be reused in this session.")
        if st.session_state.dispatches:
            st.markdown("#### Latest dispatch result")
            st.dataframe(pd.DataFrame(st.session_state.dispatches), width="stretch")
            st.download_button(
                "Download dispatch results (CSV)",
                pd.DataFrame(st.session_state.dispatches).to_csv(index=False).encode("utf-8"),
                file_name="dispatch_results.csv",
                mime="text/csv",
            )
    if st.button("Reset demo state"):
        st.session_state.calls = []
        st.session_state.dispatches = []
        st.session_state.used_vehicle_ids.clear()
        st.rerun()

st.divider()
st.caption("Audit note: this demo displays triage method, confidence, call history, incident grouping, and simulated dispatch estimates. Verify all logic and safety requirements before any real-world use.")
