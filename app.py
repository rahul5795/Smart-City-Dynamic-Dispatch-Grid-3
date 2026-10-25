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

st.caption("Synthetic emergency-call triage, incident prioritization, and vehicle dispatch demo")
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
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "several": 3, "many": 5}


def count_people(t: str) -> int | None:
    nums = [int(n) for n in re.findall(r"\b(\d{1,2})\b", t)]
    nums += [v for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", t)]
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
        "पेट्रोल पंप": "Petrol Pump", "बस अड्डा": "Bus Terminal", "बस स्टैंड": "Bus Terminal", "थाना": "Police Station",
        "एयरपोर्ट": "Airport Road", "फैक्ट्री": "Industrial Area", "नदी": "River Bank",
    }
    for phrase, landmark in aliases.items():
        if phrase in lower:
            return landmark
    return None          # never invent a map point: the dispatcher calls back instead (plan, section 3.4)

GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"     # follows the JSON format reliably
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]


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
           "Voice engine": engine, "Recording": audio, "Triaged at": time.strftime("%H:%M:%S"), **triage}
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
# Like 112 in India or 911 in the USA: one number, one call-taker, then the call is routed to every department that
# has to act. Each step is a small "agent" with one job and a CONSTRAINED output: it can only choose from fixed lists,
# so it can never invent a department, a severity or a place.
DEPARTMENTS = {   # name -> icon, Indian helpline, vehicle type it sends (None = a ticket and a call-back), US equivalent
    "Police":            {"icon": "🚓", "line": "100",  "fleet": "Police Car",  "code": "POL", "us": "Police (911)"},
    "Fire & Rescue":     {"icon": "🚒", "line": "101",  "fleet": "Fire Truck",  "code": "FIR", "us": "Fire Department (911)"},
    "Ambulance (EMS)":   {"icon": "🚑", "line": "108",  "fleet": "Ambulance",   "code": "EMS", "us": "EMS / ambulance (911)"},
    "Disaster Response": {"icon": "🛟", "line": "1078", "fleet": "Rescue Team", "code": "DRF", "us": "Urban Search & Rescue / FEMA"},
    "Gas Emergency":     {"icon": "🛢️", "line": "1906", "fleet": None,          "code": "GAS", "us": "Gas utility emergency line"},
    "Electricity Board": {"icon": "⚡", "line": "1912", "fleet": None,          "code": "ELE", "us": "Power company outage line"},
    "Women Helpline":    {"icon": "👩", "line": "1091", "fleet": None,          "code": "WOM", "us": "Domestic violence hotline"},
    "Child Helpline":    {"icon": "🧒", "line": "1098", "fleet": None,          "code": "CHD", "us": "Child protective services"},
}
PROBLEMS = {      # the ONLY problem categories the router may choose; each one says which teams must act
    "fire":          {"label": "Fire", "kw": ["fire", "smoke", "flames", "burning", "blaze", "aag", "dhuan"], "teams": ["Fire & Rescue"]},
    "gas_leak":      {"label": "Gas leak", "kw": ["gas leak", "gas smell", "smell of gas", "cylinder", "lpg"], "teams": ["Fire & Rescue", "Gas Emergency"]},
    "medical":       {"label": "Medical emergency", "kw": ["unconscious", "not breathing", "heart attack", "chest pain", "fainted", "seizure",
                                                          "stroke", "bleeding", "injured", "hurt", "labour pain", "snake bite", "poison",
                                                          "overdose", "behosh", "fell", "not moving"], "teams": ["Ambulance (EMS)"]},
    "road_accident": {"label": "Road accident", "kw": ["accident", "crash", "collided", "collision", "hit by", "overturned", "run over",
                                                       "skidded"], "teams": ["Ambulance (EMS)", "Police"]},
    "collapse":      {"label": "Building collapse", "kw": ["collapse", "building fell", "wall fell", "roof fell", "rubble", "debris",
                                                           "came down"], "teams": ["Disaster Response", "Fire & Rescue"]},
    "flood":         {"label": "Flood / drowning", "kw": ["flood", "water rising", "drowning", "waterlogging", "washed away", "baadh"],
                      "teams": ["Disaster Response"]},
    "crime":         {"label": "Crime / violence", "kw": ["robbery", "robbed", "theft", "stolen", "thief", "burglary", "break in", "broke into",
                                                          "snatch", "gun", "pistol", "knife", "shot", "shooting", "stabbed", "fight",
                                                          "beating", "attack", "assault", "threat", "kidnap", "murder", "chor", "loot"],
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
}
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
    found = []
    for pid, p in PROBLEMS.items():
        hit = next((k for k in p["kw"] if k in t), None)
        if hit:
            found.append((pid, hit))
    return found


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
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
    t = text.lower()
    injured = any(w in t for w in INJURY_WORDS) or int(triage.get("severity", 1)) >= 4
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
        a["ref"], a["created"] = ref, time.strftime("%H:%M:%S")
        out.append(a)
    return out


def run_agents(text: str, triage: dict, ref: str, source: str, voice: dict | None = None, hint: str | None = None,
               dispatch: bool = True) -> dict:
    """The whole pipeline, recorded step by step so anyone can see who decided what."""
    steps = []
    def step(icon, agent, did, ms):
        steps.append({"icon": icon, "agent": agent, "did": did, "ms": round(ms, 1)})
    t0 = time.perf_counter()
    if voice and voice.get("text"):
        step("🎙️", "Intake agent", f"Voice recorded ({voice.get('duration') or 0:.1f} s) and written by {voice.get('engine')}: “{text[:120]}”", 0)
    else:
        step("⌨️", "Intake agent", f"{source} report received: “{text[:120]}”", 0)
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
           "total_ms": round((time.perf_counter() - t0) * 1000, 1), "time": time.strftime("%H:%M:%S")}
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
# Like a 112 / 911 call-taker: ask what happened and where, send help as soon as both are known, keep asking the
# questions of that problem's protocol (breathing? trapped? weapon?), raise urgency when the answers demand it,
# alert extra departments when new problems appear, and give life-saving instructions while help is on the way.
CALL_Q = {   # slot -> (English, Hindi)
    "problem": ("112, what is your emergency? Tell me what is happening.",
                "112, आपकी आपात स्थिति क्या है? बताइए क्या हो रहा है।"),
    "problem_again": ("I need to understand what happened. Is it a fire, an accident, someone ill or hurt, a crime, a building collapse, "
                      "a flood, a gas leak or a fallen electric wire?",
                      "मुझे समझना है क्या हुआ। आग, दुर्घटना, कोई बीमार या घायल, अपराध, इमारत गिरना, बाढ़, गैस लीक या बिजली का तार?"),
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
    "name": ("What is your name, so the team can call you back on this number?", "आपका नाम क्या है, ताकि टीम आपको इसी नंबर पर कॉल कर सके?"),
}
PROTOCOL = {   # problem -> the questions a call-taker asks next, in order
    "medical": ["breathing", "bleeding", "people"], "road_accident": ["breathing", "bleeding", "trapped", "people"],
    "collapse": ["trapped", "breathing", "people"], "flood": ["trapped", "people"], "fire": ["spreading", "trapped", "people"],
    "gas_leak": ["caller_safe", "people"], "crime": ["caller_safe", "weapon", "description", "people"],
    "women_safety": ["caller_safe", "weapon", "description"], "child": ["child_desc", "description"], "electric": ["touching_wire", "people"],
}
YES_NO = ["breathing", "bleeding", "trapped", "spreading", "caller_safe", "weapon", "touching_wire"]
YES_WORDS = ["yes", "yeah", "yep", "haan", "han", "haa", "ji", "sure", "correct", "right", "हाँ", "हां", "जी", "हा"]
NO_WORDS = ["no", "nope", "nahi", "nahin", "not", "isn't", "isnt", "doesn't", "can't", "नहीं", "नही", "ना"]
UNSURE = ["don't know", "dont know", "not sure", "no idea", "pata nahi", "पता नहीं", "मालूम नहीं"]
SAY = {   # fixed phrases (English, Hindi)
    "sending": ("I am sending {teams} now. {unit} is about {eta} minutes away. Stay on the line.",
                "मैं अभी {teams} भेज रहा हूँ। {unit} लगभग {eta} मिनट में पहुँचेगी। लाइन पर बने रहिए।"),
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
              "child": "बच्चा खतरे में", "electric": "बिजली का खतरा"}
TYPE_OF = {"fire": "Fire", "gas_leak": "Fire", "collapse": "Rescue", "flood": "Rescue", "electric": "Rescue", "medical": "Medical",
           "road_accident": "Medical", "crime": "Police", "women_safety": "Police", "child": "Police"}


def _l(pair: tuple[str, str], hindi: bool) -> str:
    return pair[1] if hindi else pair[0]


def yes_no(text: str) -> bool | None:
    t = " " + re.sub(r"[^\w\u0900-\u097F' ]", " ", text.lower()) + " "
    if any(u in t for u in UNSURE):
        return None
    no = any(f" {w} " in t for w in NO_WORDS)
    yes = any(f" {w} " in t for w in YES_WORDS)
    return False if no else True if yes else None


def new_call(hindi: bool) -> dict:
    st.session_state.call_no = st.session_state.get("call_no", 0) + 1
    return {"ref": f"112-{st.session_state.call_no:03d}", "hindi": hindi, "turns": [], "slots": {}, "asked": [], "pending": "problem",
            "misses": {}, "dispatched": False, "assignments": [], "severity": 3, "record": None, "ended": False,
            "started": time.time(), "said_instructions": []}


def agent_say(call: dict, text: str) -> None:
    call["turns"].append({"role": "agent", "text": text, "t": time.strftime("%H:%M:%S")})


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
    elif pend in ("description", "child_desc", "name") and len(text.strip()) > 1 and yes_no(text) is not False:
        val = text.strip()
        if pend == "name":
            m_ = re.search(r"(?:my name is|name is|i am|i'm|this is|mera naam|मेरा नाम)\s+([^\s,.]+)", val, flags=re.I)
            val = (m_.group(1) if m_ else val.split()[0]).strip().title()
        sl[pend] = val; got.append(pend)
    # anything else the caller volunteered
    probs = [p for p, _ in rule_problems(text)]
    if probs:
        before = set(sl.get("problems", []))
        sl["problems"] = list(dict.fromkeys(sl.get("problems", []) + probs))
        if set(sl["problems"]) - before:
            got.append("problems")
    place = match_location(text)
    if place and place != sl.get("location"):
        sl["location"] = place; got.append("location")
    n = count_people(t)
    if n and n != sl.get("people"):
        sl["people"] = n; got.append("people")
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
    probs = sl.get("problems", [])
    tri.update(is_emergency=True, severity=sev, location=sl.get("location"), people=sl.get("people"),
               incident_type=TYPE_OF.get(probs[0], tri.get("incident_type", "Medical")) if probs else tri.get("incident_type", "Medical"),
               severity_reason=", ".join(k.replace("_", " ") + (" yes" if v is True else " no") for k, v in sl.items()
                                         if k in YES_NO and v is not None) or tri.get("severity_reason", ""))
    return tri


def next_question(call: dict) -> str | None:
    sl, asked = call["slots"], call["asked"]
    if not sl.get("problems"):
        return "problem_again" if "problem_again" not in asked and call["turns"] and any(t_["role"] == "caller" for t_ in call["turns"]) else (
            None if "problem_again" in asked else "problem")
    if not sl.get("location"):
        return "location" if "location" not in asked else ("location_again" if "location_again" not in asked else None)
    for p in sl["problems"]:
        for q in PROTOCOL.get(p, []):
            if q not in asked and sl.get(q) is None and not (q == "people" and sl.get("people")):
                return q
    if "name" not in asked:
        return "name"
    return None


def caller_says(call: dict, text: str) -> None:
    """One turn of the conversation: understand, dispatch or update, instruct, ask the next question."""
    hindi = call["hindi"]
    call["turns"].append({"role": "caller", "text": text.strip(), "t": time.strftime("%H:%M:%S")})
    got = extract_into(call, text)
    sl = call["slots"]
    parts = []
    if not got and call.get("pending") not in ("description", "child_desc", "name"):
        call["misses"][call.get("pending")] = call["misses"].get(call.get("pending"), 0) + 1
    if "problems" in got or "location" in got:
        what = ", ".join((PROBLEM_HI.get(p, PROBLEMS[p]["label"]) if hindi else PROBLEMS[p]["label"].lower())
                         for p in sl.get("problems", [])) or ("आपात स्थिति" if hindi else "an emergency")
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
    if (sl.get("problems") or unclear) and sl.get("location") and not call["dispatched"]:
        rec = add_call(caller_text(call), tri, "Emergency call")
        call["record"] = rec["Call ID"]
        run_ = run_agents(caller_text(call), tri, call["ref"], "Emergency call", hint=(sl.get("problems") or [None])[0])
        for a_ in run_["assignments"]:
            a_["sent_at"] = time.time()
        call["assignments"], call["dispatched"] = run_["assignments"], True
        rec["assignments"], rec["teams"] = run_["assignments"], [a_["dept"] for a_ in run_["assignments"]]
        units = sorted([a_ for a_ in run_["assignments"] if a_.get("unit")], key=lambda a_: a_["eta"])
        teams = ", ".join(a_["dept"] for a_ in run_["assignments"])
        if units:
            rec["Help vehicle"] = units[0]["unit"]
            parts.append(_l(SAY["sending"], hindi).format(teams=teams, unit=units[0]["unit"], eta=max(1, round(units[0]["eta"]))))
        else:
            parts.append((_l(SAY["queued"], hindi) + " ") if any(DEPARTMENTS[a_["dept"]]["fleet"] for a_ in run_["assignments"]) else "")
            parts.append(_l(SAY["extra"], hindi).format(teams=teams))
    elif call["dispatched"]:
        # new problems in the conversation -> alert the extra departments
        have = {a_["dept"] for a_ in call["assignments"]}
        route = route_problem(caller_text(call), tri)
        extra = [t_ for t_ in route["teams"] if t_["dept"] not in have]
        if extra:
            more = dispatch_departments(extra, sl.get("location"), call["ref"])
            for a_ in more:
                a_["sent_at"] = time.time()
            call["assignments"] += more
            st.session_state.setdefault("dept_jobs", []).extend(
                {**{k: v for k, v in a_.items() if k != "route"}, "problem": ", ".join(route["labels"]), "location": sl.get("location"),
                 "severity": call["severity"]} for a_ in more)
            parts.append(_l(SAY["extra"], hindi).format(teams=", ".join(a_["dept"] for a_ in more)))
        if call["severity"] == 5 and old_sev < 5:
            parts.append(_l(SAY["upgraded"], hindi))
        rec = next((c for c in st.session_state.calls if c["Call ID"] == call["record"]), None)
        if rec:
            rec.update(severity=call["severity"], people=sl.get("people"), **{"Call text": caller_text(call)})
    # life-saving instructions, each given once
    triggers = [("not_breathing", sl.get("breathing") is False), ("bleeding", sl.get("bleeding") is True),
                ("gas", "gas_leak" in sl.get("problems", [])), ("fire", "fire" in sl.get("problems", [])),
                ("unsafe", sl.get("caller_safe") is False), ("wire", "electric" in sl.get("problems", [])),
                ("trapped", sl.get("trapped") is True), ("flood", "flood" in sl.get("problems", []))]
    for key, on in triggers:
        if on and key not in call["said_instructions"]:
            call["said_instructions"].append(key)
            parts.append(_l(INSTRUCT[key], hindi))
    q = next_question(call)
    if q is None and not got and not call["dispatched"]:
        parts.append(_l(SAY["not_heard"], hindi))
        q = call.get("pending")
    if q:
        call["asked"].append(q)
        call["pending"] = q
        parts.append(_l(CALL_Q[q], hindi))
    else:
        call["pending"] = None
        parts.append(_l(SAY["done"], hindi))
    agent_say(call, " ".join(p for p in parts if p))


def call_transcript(call: dict) -> str:
    lines = [f"Emergency call {call['ref']} ({time.strftime('%Y-%m-%d', time.localtime(call['started']))})"]
    for t_ in call["turns"]:
        lines.append(f"[{t_['t']}] {'Call-taker' if t_['role'] == 'agent' else 'Caller'}: {t_['text']}")
    sl = call["slots"]
    lines += ["", "Summary:", f"  Problem: {', '.join(PROBLEMS[p]['label'] for p in sl.get('problems', [])) or 'unclear'}",
              f"  Location: {sl.get('location') or 'unknown'}", f"  Severity: {call['severity']} of 5", f"  People: {sl.get('people') or 'not stated'}",
              "  Teams: " + (", ".join(f"{a_['dept']} ({a_.get('unit') or a_.get('ticket') or a_.get('status')})" for a_ in call["assignments"]) or "none")]
    return "\n".join(lines)


# ----------------------- Emergency Help (native citizen page) -----------------------
HELPLINES = [("112", "All emergencies"), ("108", "Ambulance"), ("101", "Fire"), ("100", "Police"),
             ("1091", "Women helpline"), ("1098", "Child helpline"), ("1906", "Gas leak"), ("1912", "Electricity")]
HELP_TYPES = {   # button -> (triage type, words added to the report, problem id for the router)
    "🔥 Fire": ("Fire", "fire", "fire"),
    "🚑 Medical": ("Medical", "medical emergency", "medical"),
    "🚗 Road accident": ("Medical", "road accident, people injured", "road_accident"),
    "🏚️ Building collapse": ("Rescue", "building collapse", "collapse"),
    "🌊 Flood": ("Rescue", "flood water rising", "flood"),
    "💨 Gas leak": ("Fire", "gas leak, strong gas smell", "gas_leak"),
    "🚓 Crime / fight": ("Police", "crime, robbery or fight", "crime"),
    "👩 Women safety": ("Police", "woman being harassed, women safety", "women_safety"),
    "🧒 Child in danger": ("Police", "child missing or in danger", "child"),
    "⚡ Electric hazard": ("Rescue", "live electric wire fallen", "electric"),
}
DANGER_SIGNS = {"Someone is trapped": 5, "Someone is not breathing or unconscious": 5,
                "Heavy bleeding": 4, "Fire or danger is spreading": 4, "Someone is injured": 3}
WHILE_YOU_WAIT = {
    "Fire": ["Get everyone out and stay out. Do not go back inside for belongings.",
             "If there is smoke, stay low, close doors behind you, and do not use lifts.",
             "If gas is leaking: no flames or switches; open doors and windows only if it is safe, then leave."],
    "Medical": ["Stay with the person and keep them still, warm and calm. Do not give food or drink.",
                "If they are not breathing normally and you are trained, start CPR; otherwise follow the 112 operator's instructions.",
                "Press firmly on heavy bleeding with a clean cloth and keep pressing.",
                "After a road accident, switch on hazard lights and do not move injured people unless they are in danger."],
    "Police": ["Move to a safe place and lock the door if you can. Do not follow or confront anyone.",
               "Note what you can safely remember: faces, clothes, vehicle number, direction they went.",
               "Keep your phone with you and pick up when the police or helpline calls back."],
    "Rescue": ["Move away from damaged buildings, walls and electric wires. Do not go inside.",
               "Never touch a fallen wire or anyone still touching one; keep others at least 10 metres away.",
               "If someone is trapped, keep talking to them and tap on pipes or walls so rescuers can find them.",
               "In a flood, move to higher ground and never walk or drive through moving water."],
}
HELP_STAGES = ["Report received", "Help assigned", "On the way", "Arrived", "Completed"]
SCENE_MINUTES = 10


def dispatch_help(rep_: dict) -> None:
    """Send every team the router chose; track the first unit to arrive. Retries teams that were queued."""
    if not rep_.get("location"):
        return
    teams = rep_.get("teams") or []
    queued = [t_ for t_ in teams if t_.get("unit") is None and DEPARTMENTS[t_["dept"]]["fleet"]]
    if queued:
        fresh = dispatch_departments([{k: t_[k] for k in ("dept", "role", "why")} for t_ in queued], rep_["location"], rep_["ref"])
        for old, new in zip(queued, fresh):
            if new.get("unit"):
                old.update(unit=new["unit"], eta=new["eta"], route=new["route"], status="on the way", sent_at=time.time())
    units = [t_ for t_ in teams if t_.get("unit")]
    for t_ in units:
        t_.setdefault("sent_at", time.time())
    if not units:
        rep_["status"] = "queued" if any(DEPARTMENTS[t_["dept"]]["fleet"] for t_ in teams) else "notified"
        return
    first = min(units, key=lambda t_: t_["sent_at"] + t_["eta"] * sim_seconds)
    rep_.update(vehicle=first["unit"], eta=first["eta"], assigned_at=first["sent_at"], route=first["route"], status="assigned")
    for c in st.session_state.calls:
        if c.get("Call ID") == rep_["call_id"]:
            c["Help vehicle"] = first["unit"]
            c["teams"] = [t_["dept"] for t_ in teams]


def help_progress(rep_: dict) -> dict:
    """Where the help is now, from the simulated clock (one simulated minute = `sim_seconds` real seconds)."""
    if not rep_.get("assigned_at"):
        return {"stage": 0, "remaining": None, "frac": 0.0}
    elapsed = (time.time() - rep_["assigned_at"]) / sim_seconds
    if elapsed < rep_["eta"]:
        return {"stage": 2 if elapsed > 0.2 else 1, "remaining": rep_["eta"] - elapsed, "frac": elapsed / rep_["eta"]}
    last = max([rep_["eta"]] + [(t_["sent_at"] - rep_["assigned_at"]) / sim_seconds + t_["eta"]
                                for t_ in rep_.get("teams", []) if t_.get("unit")])
    if elapsed < last + SCENE_MINUTES:
        return {"stage": 3, "remaining": 0.0, "frac": 1.0}
    if not rep_.get("released"):                                           # every unit is free for new jobs
        for t_ in rep_.get("teams", []):
            if t_.get("unit"):
                st.session_state.used_vehicle_ids.discard(t_["unit"])
                t_["status"] = "job completed"
        rep_["released"] = True
    return {"stage": 4, "remaining": 0.0, "frac": 1.0}


def draw_help_map(rep_: dict, frac: float):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_xlim(-0.5, GRID_SIZE - 0.5); ax.set_ylim(-0.5, GRID_SIZE - 0.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for k in range(GRID_SIZE):
        ax.axhline(k, color="#cbd4dc", lw=0.6, zorder=0); ax.axvline(k, color="#cbd4dc", lw=0.6, zorder=0)
    ax.axvspan(9.15, 9.85, color="#c3dcec", zorder=0.5)                   # river between x=9 and x=10
    for y in (3, 10, 16):
        ax.plot([9, 10], [y, y], color="#5b6b7a", lw=3, zorder=1)          # bridges
    for y in closure_rows:
        ax.axhline(y, color="#d62839", lw=2, ls="--", alpha=.6)
    for x in closure_cols:
        ax.axvline(x, color="#d62839", lw=2, ls="--", alpha=.6)
    colors = {"Police": "#1C5D99", "Fire & Rescue": "#D62839", "Ambulance (EMS)": "#1F8A5B", "Disaster Response": "#C98A0C"}
    units = [t_ for t_ in rep_.get("teams", []) if t_.get("route")] or (
        [{"route": rep_["route"], "unit": rep_["vehicle"], "dept": "", "eta": rep_["eta"], "sent_at": rep_.get("assigned_at")}] if rep_.get("route") else [])
    now = time.time()
    for t_ in units:
        route = t_["route"]
        f = 1.0 if frac >= 1.0 and t_.get("unit") == rep_.get("vehicle") else min(1.0, max(0.0, (now - (t_.get("sent_at") or now)) / sim_seconds / max(t_["eta"], 0.1)))
        xs, ys = zip(*route)
        col = colors.get(t_.get("dept"), "#1c5d99")
        ax.plot(xs, ys, color=col, lw=3, alpha=.8, zorder=2)
        pos = route[min(len(route) - 1, int(round(f * (len(route) - 1))))]
        ax.scatter(*pos, s=170, marker="s", color="white", edgecolors=col, linewidths=2.5, zorder=4)
        ax.annotate(t_["unit"], pos, xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight="bold")
    if rep_.get("location"):
        tx, ty = LANDMARKS[rep_["location"]]
        ax.scatter(tx, ty, s=260, color=SEV_COLOR.get(rep_["severity"], "#d62839"), edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(rep_["location"], (tx, ty), xytext=(6, -12), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------- Speech to text (works even when browser voice typing does not) -----------------------
VOICE_LANGS = {"English (India)": "en-IN", "हिन्दी (Hindi)": "hi-IN"}
WHISPER_HINT = ("Emergency call in an Indian city. Words that may appear: fire, smoke, accident, injured, trapped, "
                "building collapse, flood, gas leak, ambulance, hospital, police station, bus stand, railway station, "
                "market, mandir, masjid, gurudwara, chowk, thana.")


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
  const MAX_MS = 90000;
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
    const listenNext = () => { if (data.auto && !s) start(); };
    if ('speechSynthesis' in window && data.speak !== false) {
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(sayText); u.lang = lang; u.rate = 1;
        u.onend = listenNext; u.onerror = listenNext;
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
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, "")


groq_key = _secret("GROQ_API_KEY")          # add it in Manage app -> Settings -> Secrets to switch on Groq AI
use_groq = bool(groq_key)
groq_model = GROQ_FALLBACK_MODEL             # llama-3.3-70b-versatile: the most reliable for triage
whisper_model = "whisper-large-v3-turbo"     # Groq speech-to-text, used only when a key is set
triage_model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else 0.0)
use_model = triage_model is not None         # used automatically when triage_model.json is next to app.py
closure_rows, closure_cols = [], []          # no simulated road closures
sim_seconds = 2.0                            # demo speed: 2 real seconds per simulated minute

if "calls" not in st.session_state:
    st.session_state.calls = []
if "dispatches" not in st.session_state:
    st.session_state.dispatches = []
if "used_vehicle_ids" not in st.session_state:
    st.session_state.used_vehicle_ids = set()
if "help_reports" not in st.session_state:
    st.session_state.help_reports = []

tab_call, tab_help, tab_dept, tab2 = st.tabs(["📞 Emergency call", "🆘 Emergency Help", "🏢 Departments & agents",
                                             "🚑 Dispatch board"])

with tab_call:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#15202B,#1C5D99);color:#fff;border-radius:18px;padding:16px 22px;margin-bottom:8px'>"
        "<div style='font-size:1.5rem;font-weight:800'>📞 Emergency call · 112</div>"
        "<div style='opacity:.9'>Talk to the call-taker as you would on the phone. It asks what happened and where, sends the right teams "
        "as soon as it knows, keeps asking the important questions, and tells you what to do while help comes.</div></div>",
        unsafe_allow_html=True)
    st.caption("Training simulation: no real vehicle is sent. In a real emergency in India, call 112.")
    call = st.session_state.get("call")
    if call is None or call.get("ended") and st.session_state.get("call_closed_seen"):
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        lang_pick = c1.radio("Language", ["English", "हिन्दी (Hindi)"], horizontal=True, key="call_lang")
        hands = c2.toggle("Hands-free", value=True, key="call_hands",
                          help="The call-taker reads each question aloud, then the microphone starts listening by itself.")
        if c3.button("📞 Start emergency call", type="primary", width="stretch"):
            call = new_call(lang_pick.startswith("ह"))
            agent_say(call, _l(CALL_Q["problem"], call["hindi"]))
            call["asked"].append("problem")
            st.session_state.call = call
            st.session_state.call_closed_seen = False
            st.rerun()
        if call and call.get("ended"):
            st.info(f"Last call {call['ref']} has ended. Start a new call when you need to.")
    else:
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            box = st.container(height=430, border=True)
            with box:
                for t_ in call["turns"]:
                    with st.chat_message("assistant" if t_["role"] == "agent" else "user",
                                         avatar="🧑‍💼" if t_["role"] == "agent" else "🙋"):
                        st.markdown(t_["text"])
                        if t_.get("audio"):
                            st.audio(t_["audio"], format="audio/wav")
            lang = "hi-IN" if call["hindi"] else "en-IN"
            answer = None
            if not call["ended"]:
                last_agent = next((t_ for t_ in reversed(call["turns"]) if t_["role"] == "agent"), None)
                if VOICE_TYPING is not None:
                    res = VOICE_TYPING(key=f"call_live_{lang}",
                                       data={"lang": lang, "say": last_agent["text"] if last_agent else "",
                                             "say_id": f"{call['ref']}-{len(call['turns'])}", "auto": st.session_state.get("call_hands", True)},
                                       on_done_change=lambda: None, on_env_change=lambda: None)
                    done = getattr(res, "done", None)
                    if done and done.get("id") != st.session_state.get("call_live_id"):
                        st.session_state.call_live_id = done["id"]
                        audio = base64.b64decode(done["audio"]) if done.get("audio") else None
                        text = (done.get("text") or "").strip()
                        if not text and audio and wav_stats(audio)["peak"] >= 0.01:
                            try:
                                text, _eng = transcribe_audio(audio, lang, groq_key, whisper_model)
                            except RuntimeError:
                                text = ""
                        if text:
                            answer = (text, audio)
                        else:
                            agent_say(call, _l(SAY["not_heard"], call["hindi"]))
                            st.rerun()
                typed = st.chat_input("Or type your answer…", key="call_typed")
                if typed:
                    answer = (typed, None)
                if call.get("pending") == "location_again" or (call["slots"].get("problems") and not call["slots"].get("location")
                                                               and "location_again" in call["asked"]):
                    pick = st.selectbox("Pick the nearest landmark", [""] + sorted(LANDMARKS), key=f"call_pick_{len(call['turns'])}")
                    if pick:
                        answer = (pick, None)
                if answer:
                    caller_says(call, answer[0])
                    if answer[1]:
                        call["turns"][-2]["audio"] = answer[1]           # the caller's turn keeps its recording
                    st.rerun()
                b1, b2 = st.columns(2)
                if b1.button("🔚 End call", width="stretch"):
                    call["ended"] = True
                    agent_say(call, _l(SAY["ended"], call["hindi"]).format(ref=call["ref"]))
                    st.rerun()
                b2.download_button("Download transcript", call_transcript(call).encode("utf-8"), f"{call['ref']}.txt",
                                   "text/plain", width="stretch")
            else:
                st.success(f"Call {call['ref']} ended. The teams keep working; you can still follow them on the right.")
                d1, d2 = st.columns(2)
                d1.download_button("Download transcript", call_transcript(call).encode("utf-8"), f"{call['ref']}.txt",
                                   "text/plain", width="stretch")
                if d2.button("📞 Start a new call", type="primary", width="stretch"):
                    st.session_state.call_closed_seen = True
                    st.session_state.call = None
                    st.rerun()

        with right:
            @st.fragment(run_every=1.0)
            def call_panel() -> None:
                c_ = st.session_state.get("call")
                if not c_:
                    return
                sl = c_["slots"]
                with st.container(border=True):
                    st.markdown(f"#### What the call-taker knows  \n<span style='opacity:.7'>Reference {c_['ref']}</span>", unsafe_allow_html=True)
                    sev = c_["severity"]
                    st.markdown(f"<span style='display:inline-block;background:{SEV_COLOR.get(sev)};color:#fff;border-radius:10px;padding:2px 12px;"
                                f"font-size:1.4rem;font-weight:800'>{sev}</span> of 5 · {SEVERITY_GUIDE.get(sev, '')}", unsafe_allow_html=True)
                    k1, k2 = st.columns(2)
                    k1.markdown(f"**Problem**  \n{', '.join(PROBLEMS[p]['label'] for p in sl.get('problems', [])) or '…listening'}")
                    k2.markdown(f"**Location**  \n{sl.get('location') or '…not yet'}")
                    k3, k4 = st.columns(2)
                    k3.markdown(f"**People**  \n{sl.get('people') or 'not stated'}")
                    signs = [lbl for key, val, lbl in [("breathing", False, "not breathing"), ("trapped", True, "trapped"),
                                                       ("bleeding", True, "heavy bleeding"), ("spreading", True, "spreading"),
                                                       ("weapon", True, "weapon"), ("caller_safe", False, "caller not safe"),
                                                       ("touching_wire", True, "touching wire")] if sl.get(key) is val]
                    k4.markdown(f"**Danger signs**  \n{', '.join(signs) or 'none reported'}")
                    if sl.get("description") or sl.get("child_desc"):
                        st.markdown(f"**Description**  \n{sl.get('child_desc') or ''} {sl.get('description') or ''}")
                    if sl.get("name"):
                        st.markdown(f"**Caller**  \n{sl['name']}")
                with st.container(border=True):
                    st.markdown("#### Teams")
                    if not c_["assignments"]:
                        st.caption("Help is sent as soon as the call-taker knows what happened and where.")
                    now = time.time()
                    for a_ in c_["assignments"]:
                        if a_.get("unit"):
                            left_ = a_["eta"] - (now - a_.get("sent_at", now)) / sim_seconds
                            state = f"{max(1, math.ceil(left_))} min away" if left_ > 0 else "✅ arrived"
                            st.markdown(f"{a_['icon']} **{a_['dept']}**{' (lead)' if a_.get('role') == 'lead' else ''} · {a_['unit']} · {state}")
                            if left_ > 0:
                                st.progress(min(1.0, max(0.0, 1 - left_ / a_["eta"])))
                        else:
                            st.markdown(f"{a_['icon']} **{a_['dept']}** · {a_.get('ticket') or ''} {a_['status']}")
                    if c_["assignments"]:
                        st.caption("Every agent's decision is on the 🏢 Departments & agents tab.")
            call_panel()

with tab_help:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#D62839,#A51D2D);color:#fff;border-radius:18px;padding:18px 22px;margin-bottom:6px'>"
        "<div style='font-size:1.6rem;font-weight:800'>🆘 Need help now?</div>"
        "<div style='opacity:.92'>Tell us what happened and where. We find the fastest vehicle on today’s roads and show you when it will arrive.</div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px'>" + "".join(
        f"<a href='tel:{n}' style='text-decoration:none;border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:6px 12px;"
        f"color:inherit'><b style='color:#D62839'>📞 {n}</b> <span style='opacity:.8'>{label}</span></a>" for n, label in HELPLINES)
        + "</div>", unsafe_allow_html=True)
    st.caption("This page is a training simulation: no real vehicle is sent. In a real emergency in India, call 112.")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 1. What happened?")
            h_kind = st.pills("Type of emergency", list(HELP_TYPES), key="help_kind", label_visibility="collapsed")
            st.markdown("#### 2. Where are you?")
            h_place = st.selectbox("Nearest landmark", ["Detect from my words"] + sorted(LANDMARKS), key="help_place")
            st.markdown("#### 3. Tell us more")
            with st.expander("🎙️ Speak instead of typing", expanded=False):
                h_voice = voice_recorder("help", "Record what is happening", groq_key, whisper_model)
                if h_voice and h_voice.get("new") and h_voice.get("text"):
                    st.session_state.help_text = h_voice["text"]
                    h_voice["new"] = False
            h_text = st.text_area("What is happening? (optional)", key="help_text", height=80,
                                  placeholder="For example: smoke from the second floor, my neighbour is still inside")
            h_signs = st.pills("Danger signs", list(DANGER_SIGNS), selection_mode="multi", key="help_signs")
            h_people = st.number_input("How many people need help?", min_value=0, max_value=200, value=0, step=1, key="help_people",
                                       help="Leave at 0 if you don't know.")
            send = st.button("🚨 Send for help", type="primary")
        if send:
            words = HELP_TYPES[h_kind][1] if h_kind else ""
            place = None if h_place == "Detect from my words" else h_place
            report_text = ". ".join(x for x in [words, (h_text or "").strip(), f"near {place}" if place else "",
                                                ", ".join(h_signs or [])] if x)
            if not report_text:
                st.error("Choose what happened, or say or type a few words, then send.")
            else:
                triage, msgs = run_triage(report_text)
                for kind_, m_ in msgs:
                    getattr(st, kind_)(m_)
                triage["is_emergency"] = True                       # someone pressed "Send for help": never dismiss it
                if h_kind:
                    triage["incident_type"] = HELP_TYPES[h_kind][0]
                elif triage.get("incident_type") not in ("Fire", "Rescue", "Medical", "Police"):
                    triage["incident_type"] = "Medical"
                if h_signs:
                    top = max(DANGER_SIGNS[sg] for sg in h_signs)
                    if top > int(triage.get("severity", 1)):
                        triage["severity"], triage["severity_reason"] = top, f"you said: {', '.join(h_signs).lower()}"
                if place:
                    triage["location"] = place
                if h_people:
                    triage["people"] = int(h_people)
                call_id = f"CALL-{len(st.session_state.calls) + 1:03d}"
                ref = f"HELP-{len(st.session_state.help_reports) + 1:03d}"
                voiced = bool(h_voice) and bool(h_voice.get("text")) and (h_text or "").strip() == h_voice.get("text", "")
                rec_ = add_call(report_text, triage, "Emergency Help", h_voice["engine"] if voiced else None,
                                h_voice["audio"] if voiced else None)
                rec_["help_ref"] = ref
                call_id = rec_["Call ID"]
                rep_ = {"ref": ref, "call_id": call_id, "created": time.time(), "type": triage["incident_type"],
                        "severity": int(triage["severity"]), "location": triage.get("location"), "people": triage.get("people"),
                        "reason": triage.get("severity_reason", ""), "method": triage.get("triage_method", ""),
                        "text": report_text, "status": "received" if triage.get("location") else "needs_location",
                        "vehicle": None, "feedback": None}
                run_ = run_agents(report_text, triage, ref, "Emergency Help", h_voice if voiced else None,
                                  hint=HELP_TYPES[h_kind][2] if h_kind else None)
                rep_["teams"] = run_["assignments"]
                rep_["problem"] = ", ".join(run_["route"]["labels"]) or triage["incident_type"]
                lead = next((t_["dept"] for t_ in rep_["teams"] if t_["role"] == "lead"), None)
                if lead == "Police":
                    rep_["type"] = "Police"
                rec_["teams"] = [t_["dept"] for t_ in rep_["teams"]]
                dispatch_help(rep_)
                st.session_state.help_reports.append(rep_)
                st.session_state.help_track = ref
                st.toast(f"Report {ref} sent to " + ", ".join(t_["dept"] for t_ in rep_["teams"]) + ".", icon="🚨")

    with right:
        reports = st.session_state.help_reports
        if not reports:
            with st.container(border=True):
                st.markdown("#### Your help request")
                st.info("After you send a report, its reference number, the vehicle on its way and the arrival time appear here.")
        else:
            refs = [r_["ref"] for r_ in reports][::-1]                 # newest first
            track_ref = st.selectbox("Track a report", refs, key="help_track")
            rep_ = next(r_ for r_ in reports if r_["ref"] == track_ref)

            if rep_["status"] == "needs_location":
                with st.container(border=True):
                    st.warning(f"**{rep_['ref']}**: we could not find your location. Which landmark is closest?")
                    fix = st.selectbox("Nearest landmark", sorted(LANDMARKS), key=f"fix_{rep_['ref']}")
                    if st.button("Set location and send help", key=f"fixbtn_{rep_['ref']}"):
                        rep_["location"], rep_["status"] = fix, "received"
                        for c in st.session_state.calls:
                            if c.get("Call ID") == rep_["call_id"]:
                                c["location"] = fix
                        dispatch_help(rep_)
                        st.rerun()

            @st.fragment(run_every=1.0)
            def live_tracking(ref_: str) -> None:
                r_ = next(x for x in st.session_state.help_reports if x["ref"] == ref_)
                if r_["status"] == "queued" or any(t_.get("unit") is None and DEPARTMENTS[t_["dept"]]["fleet"]
                                                   for t_ in r_.get("teams", [])):
                    dispatch_help(r_)                               # retry: a unit may have become free
                pr = help_progress(r_)
                with st.container(border=True):
                    head = st.columns([3, 1])
                    title = ["Help is being arranged", "Help assigned", "Help is on the way", "Help has arrived", "Completed"][pr["stage"]]
                    if r_["status"] == "queued":
                        title = "Waiting for a free vehicle"
                    elif r_["status"] == "notified":
                        title = "Your report was passed to the helpline"
                    head[0].markdown(f"### {title}")
                    head[1].markdown(f"<div style='text-align:right;margin-top:12px'><code>{r_['ref']}</code></div>", unsafe_allow_html=True)
                    if pr["stage"] in (1, 2):
                        st.markdown(f"<div style='font-size:3rem;font-weight:800;line-height:1'>{max(1, math.ceil(pr['remaining']))}"
                                    f"<span style='font-size:1rem;font-weight:500;opacity:.75'> min until {r_['vehicle']} arrives</span></div>",
                                    unsafe_allow_html=True)
                        st.progress(min(1.0, pr["frac"]))
                    elif pr["stage"] >= 3:
                        st.success(f"{r_['vehicle']} reached {r_['location']} after {r_['eta']:.0f} simulated minutes.")
                    elif r_["status"] == "queued":
                        st.warning("All suitable vehicles are busy. You are in the queue; the first free one will be sent.")
                    elif r_["status"] == "notified":
                        st.info("No vehicle is needed for this problem. The helpline below will call you back shortly.")
                    steps = st.columns(len(HELP_STAGES))
                    for k_, (col, name) in enumerate(zip(steps, HELP_STAGES)):
                        done = k_ <= pr["stage"] and (k_ == 0 or r_.get("vehicle"))
                        col.markdown(f"<div style='text-align:center;font-size:.8rem'><div style='font-size:1.2rem'>{'🟢' if done else '⚪'}</div>{name}</div>",
                                     unsafe_allow_html=True)
                    if r_.get("teams"):
                        st.markdown("**Teams alerted**  \n" + teams_html(r_["teams"]), unsafe_allow_html=True)
                    f1, f2, f3 = st.columns(3)
                    f1.markdown(f"**Emergency**  \n{r_.get('problem') or r_['type']}")
                    f2.markdown(f"**Severity**  \n<span style='color:{SEV_COLOR.get(r_['severity'])};font-weight:800'>{r_['severity']} of 5</span>",
                                unsafe_allow_html=True)
                    f3.markdown(f"**Location**  \n{r_['location'] or 'not found yet'}")
                    if r_.get("route"):
                        fig_ = draw_help_map(r_, pr["frac"])
                        st.pyplot(fig_, width="stretch")
                        plt.close(fig_)
                if pr["stage"] >= 3 and not r_.get("_shown_done"):
                    r_["_shown_done"] = True
                    st.rerun(scope="app")                           # show the feedback form below

            live_tracking(track_ref)

            with st.container(border=True):
                st.markdown("#### While you wait")
                for tip in WHILE_YOU_WAIT.get(rep_["type"], WHILE_YOU_WAIT["Medical"]):
                    st.markdown(f"- {tip}")
                st.caption("General safety advice only. Always follow the instructions of the emergency operator.")

            if rep_.get("_shown_done") or help_progress(rep_)["stage"] >= 3:
                with st.container(border=True):
                    if rep_.get("feedback"):
                        fb = rep_["feedback"]
                        st.success(f"Thank you. You rated this response {'★' * fb['rating']} ({fb['rating']} of 5).")
                    else:
                        st.markdown("#### How was the response?")
                        stars = st.feedback("stars", key=f"stars_{rep_['ref']}")
                        tags = st.pills("What stood out?", ["Fast response", "Crew was helpful", "Kept me informed",
                                                           "Took too long", "Hard to explain location", "App was hard to use"],
                                        selection_mode="multi", key=f"tags_{rep_['ref']}")
                        comment = st.text_input("Anything else? (optional)", key=f"cmt_{rep_['ref']}")
                        if st.button("Send feedback", key=f"fbbtn_{rep_['ref']}"):
                            if stars is None:
                                st.error("Choose 1 to 5 stars first.")
                            else:
                                rep_["feedback"] = {"rating": stars + 1, "tags": tags or [], "comment": comment}
                                st.rerun()

        if reports:
            with st.expander(f"📊 Help requests this session ({len(reports)})"):
                rated = [r_ for r_ in reports if r_.get("feedback")]
                m1, m2, m3 = st.columns(3)
                m1.metric("Requests", len(reports))
                m2.metric("Average travel time", f"{sum(r_['eta'] for r_ in reports if r_.get('eta')) / max(1, sum(1 for r_ in reports if r_.get('eta'))):.1f} min")
                m3.metric("Average rating", f"{sum(r_['feedback']['rating'] for r_ in rated) / len(rated):.1f} ★" if rated else "–")
                table = pd.DataFrame([{"Reference": r_["ref"], "Emergency": r_["type"], "Severity": r_["severity"],
                                       "Location": r_["location"] or "Unknown", "Vehicle": r_["vehicle"] or "waiting",
                                       "Teams": ", ".join(t_["dept"] for t_ in r_.get("teams", [])),
                                       "Travel (sim min)": round(r_["eta"], 1) if r_.get("eta") else None,
                                       "Rating": r_["feedback"]["rating"] if r_.get("feedback") else None,
                                       "Tags": "; ".join(r_["feedback"]["tags"]) if r_.get("feedback") else "",
                                       "Comment": r_["feedback"]["comment"] if r_.get("feedback") else ""} for r_ in reports])
                st.dataframe(table, width="stretch", hide_index=True)
                st.download_button("Download help requests and feedback (CSV)", table.to_csv(index=False).encode("utf-8"),
                                   file_name="help_requests.csv", mime="text/csv")

with tab_dept:
    st.subheader("How a call reaches the right team")
    st.caption("Like 112 in India or 911 in the USA: one number, then each agent does one job and passes the call on. "
               "The triage and router agents are constrained: they can only choose from fixed lists, so they never invent "
               "a department, a severity or a place. Anything unclear goes to a human supervisor.")
    st.graphviz_chart(AGENT_DIAGRAM)

    runs = st.session_state.get("agent_runs", [])
    st.markdown("#### Latest call, step by step")
    if not runs:
        st.info("Make an **Emergency call** or send a report on **Emergency Help**. Every agent's decision appears here.")
    else:
        pick = st.selectbox("Show the agents' work for", [r_["ref"] for r_ in runs][::-1], key="agent_pick")
        run_ = next(r_ for r_ in runs if r_["ref"] == pick)
        for k_, stp in enumerate(run_["steps"], 1):
            st.markdown(f"<div style='display:grid;grid-template-columns:34px 1fr auto;gap:10px;align-items:start;padding:8px 10px;"
                        f"border:1px solid rgba(128,128,128,.25);border-radius:12px;margin-bottom:6px'>"
                        f"<div style='font-size:1.4rem'>{stp['icon']}</div><div><b>{k_}. {stp['agent']}</b><br>"
                        f"<span style='opacity:.85'>{stp['did']}</span></div><div style='opacity:.6;font-size:.8rem'>{stp['ms']} ms</div></div>",
                        unsafe_allow_html=True)
        st.caption(f"Whole pipeline: {run_['total_ms']} ms.")

    st.markdown("#### Department boards")
    jobs = st.session_state.get("dept_jobs", [])
    names = list(DEPARTMENTS)
    for row_ in (names[:4], names[4:]):
        cols = st.columns(4)
        for col, d in zip(cols, row_):
            info = DEPARTMENTS[d]
            mine = [j for j in jobs if j["dept"] == d]
            with col.container(border=True):
                st.markdown(f"**{info['icon']} {d}**  \n<span style='opacity:.7;font-size:.85rem'>Helpline {info['line']} · "
                            f"USA: {info['us']}</span>", unsafe_allow_html=True)
                st.metric("Jobs", len(mine), label_visibility="collapsed")
                for j in mine[::-1][:4]:
                    what = j.get("unit") or j.get("ticket") or "waiting"
                    st.markdown(f"<div style='font-size:.85rem;border-top:1px solid rgba(128,128,128,.2);padding-top:4px'>"
                                f"<b>{j['ref']}</b> {j['problem']}<br>{j.get('location') or 'location unknown'} · {what} · "
                                f"<span style='opacity:.7'>{j['status']}</span></div>", unsafe_allow_html=True)
                if not mine:
                    st.caption("No jobs yet.")

    with st.expander("🇮🇳 India 112 and 🇺🇸 USA 911: the same idea"):
        st.markdown("""
| Step | India (112 ERSS) | USA (911) | In this app |
|---|---|---|---|
| One emergency number | 112 (also 100, 101, 108) | 911 | Emergency Help page |
| Call-taker answers | ERC call-taker | PSAP call-taker | 🎙️ Intake agent |
| Questions: what, where, how bad | Structured questions | Protocols such as EMD | 🧭 Triage agent (constrained) |
| Decide who must act | Dispatcher selects agencies | CAD system recommends units | 🔀 Router agent (constrained) |
| Agencies send units | Police, fire, ambulance, NDRF | Police, fire, EMS | 🚓 🚒 🚑 🛟 department agents |
| Specialist lines | 1091 women, 1098 child, 1906 gas, 1912 power | Hotlines, utilities, CPS | 👩 🧒 🛢️ ⚡ tickets and call-backs |
| Supervisor checks unclear calls | Shift supervisor | Supervisor / QA | 🧑‍💼 human in the loop |
""")

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
        st.session_state.used_vehicle_ids = set()
        st.rerun()

st.divider()
st.caption("Audit note: this demo displays triage method, confidence, call history, incident grouping, and simulated dispatch estimates. Verify all logic and safety requirements before any real-world use.")x
