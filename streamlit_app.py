"""
AI-Powered Instagram Reels Virality Predictor
Yadunandan N S

Single-page product layout: a shared "Reel details" card (category, post
hour, day of week) feeds both predictors below it, side by side.

Wires together:
- Performance tier: predictor_utils.py, a pure-Python gradient-boosted-tree
  model trained on account type, follower count, category, call-to-action,
  post hour and day of week.
- Script readiness: script_scorer.py, a single full-script input scored by
  an AI model, with a placeholder fallback when no key is configured.

The AI scoring key is never entered anywhere in this interface. It is read
once, on the backend, from Streamlit secrets (see get_ai_api_key() below),
so nobody using the deployed app can see or supply it -- only whoever holds
the app's own deployment settings can. See README.md to configure it.

Run with:  streamlit run streamlit_app.py
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

from predictor_utils import (
    ACCOUNT_TYPES,
    CONTENT_CATEGORIES,
    DAYS_OF_WEEK,
    TIER_ORDER,
    explain_simulated_prediction,
    load_model_bundle,
    predict_tier,
)
from script_scorer import analyze_script, build_prompt

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Creator credit -- shown in the footer. Edit these three lines directly to
# update the credit shown on the page.
# ---------------------------------------------------------------------------
CREATOR_NAME = "Yadunandan N S"
CREATOR_UNIVERSITY = "Ramaiah University of Applied Sciences"
CREATOR_ROLL_NO = "24MCMS101224"

st.set_page_config(page_title="Reels Virality Predictor", page_icon="✨", layout="wide")


# ---------------------------------------------------------------------------
# Backend-only AI key. Streamlit secrets first (the supported way to run
# this on Streamlit Community Cloud), an environment variable as a local
# fallback. There is no field for this anywhere in the UI.
# ---------------------------------------------------------------------------
def get_ai_api_key():
    try:
        key = st.secrets.get("LLM_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("LLM_API_KEY")


# ---------------------------------------------------------------------------
# Look and feel -- one design system for the whole page, light by default and
# automatically switching to a dark palette when the viewer's OS/browser is
# set to dark mode (see the prefers-color-scheme block below).
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #F6F7FB;
    --card-bg: #FFFFFF;
    --card-border: #EAECF3;
    --surface-2: #FAFAFD;
    --text-primary: #12142B;
    --text-secondary: #5B6178;
    --text-muted: #9398AC;
    --primary: #5B4FE9;
    --primary-dark: #453BC9;
    --primary-soft: #EEECFF;
    --accent-blue: #3B82F6;
    --accent-blue-soft: #EAF1FF;
    --accent-pink: #EC4899;
    --accent-pink-soft: #FDEBF4;
    --pill-bg: var(--accent-blue-soft);
    --pill-text: #1D4ED8;
    --success: #16A34A;
    --warning: #D97706;
    --danger: #DC2626;
    --track: #EEF0F6;
    --btn-disabled-bg: #E7E9F2;
    --btn-disabled-text: #A6ABBD;
    --shadow-sm: 0 1px 2px rgba(18,20,43,0.04), 0 1px 8px rgba(18,20,43,0.05);
    --shadow-md: 0 10px 30px rgba(18,20,43,0.08);
    --radius-lg: 20px;
    --radius-md: 14px;
}

/* Streamlit ships no light/dark switch of its own once the app menu is
   hidden, but its native widgets already follow the viewer's OS/browser
   preference automatically. This mirrors that with the same trigger, so a
   viewer in dark mode gets a dark page instead of a half-light/half-dark
   mismatch, with zero extra setup (no .streamlit/config.toml needed). */
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0B0E14;
        --card-bg: #161B24;
        --card-border: #262B36;
        --surface-2: #1C222E;
        --text-primary: #F1F3F9;
        --text-secondary: #A7ADC2;
        --text-muted: #7E8598;
        --primary: #8B7CFF;
        --primary-dark: #6C5FE0;
        --primary-soft: #232244;
        --accent-blue-soft: #1E293B;
        --accent-pink-soft: #3A1930;
        --pill-bg: #1E293B;
        --pill-text: #93C5FD;
        --track: #262B36;
        --btn-disabled-bg: #1E2430;
        --btn-disabled-text: #5B6172;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3), 0 1px 10px rgba(0,0,0,0.25);
        --shadow-md: 0 10px 30px rgba(0,0,0,0.4);
    }
}

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: var(--bg);
}
h1, h2, h3, h4, h5, p, span, div, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1180px; }

/* ---------------- hero header ---------------- */
.hero-row { display:flex; align-items:center; justify-content:space-between; gap: 16px; margin-bottom: 6px; }
.hero-brand { display:flex; align-items:center; gap: 14px; }
.brand-mark {
    width: 52px; height: 52px; border-radius: 16px; flex-shrink:0;
    background: linear-gradient(135deg, var(--primary), var(--accent-pink));
    display:flex; align-items:center; justify-content:center;
    font-size: 24px; box-shadow: 0 8px 20px rgba(91,79,233,0.28);
}
.brand-kicker {
    font-size: 11.5px; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase;
    color: var(--primary); margin-bottom: 2px;
}
.brand-title { font-size: 30px; font-weight: 800; color: var(--text-primary); line-height: 1.15; letter-spacing: -0.3px; }
.brand-byline { font-size: 13px; font-weight: 700; color: var(--primary); margin-top: 4px; }
.brand-tag { font-size: 14px; color: var(--text-secondary); margin-top: 3px; }

/* ---------------- cards (st.container(border=True)) ---------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--card-border) !important;
    background: var(--card-bg) !important;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: var(--radius-lg) !important; }

/* ---------------- buttons ---------------- */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), #7C6EFB);
    border: none; color: #fff; font-weight: 700; letter-spacing: 0.2px;
    padding: 0.75rem 1rem; border-radius: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 10px 24px rgba(91,79,233,0.30);
}
div.stButton > button[kind="primary"]:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(91,79,233,0.4);
}
div.stButton > button[kind="primary"]:disabled {
    background: var(--btn-disabled-bg); color: var(--btn-disabled-text); box-shadow: none;
}
div.stButton > button[kind="secondary"] {
    border-radius: 12px; border: 1px solid var(--card-border); color: var(--text-primary);
    font-weight: 600; background: var(--card-bg);
}
div.stButton > button[kind="secondary"]:hover:not(:disabled) {
    border-color: var(--primary); color: var(--primary);
}

/* ---------------- misc widget polish ---------------- */
.stSlider [data-baseweb="slider"] { padding-top: 4px; }
div[data-testid="stMetricValue"] { color: var(--text-primary); }
hr { border-color: var(--card-border) !important; }
.footer-note { text-align:center; color: var(--text-muted); font-size: 12.5px; margin-top: 6px; }

/* Streamlit's built-in form controls ship with a hardcoded red accent with
   no theme-file-free way to restyle it directly (color is baked into
   generated rules, not exposed as a CSS variable). A hue rotation retargets
   that red to this page's indigo brand color while leaving true grayscale
   pixels (borders, text, tick labels) visually unaffected, and keeps every
   control's own dynamic behaviour (slider fill %, checked state, focus
   rings) fully intact. */
div[data-testid="stSlider"], div[data-testid="stCheckbox"],
div[data-testid="stToggle"], div[data-testid="stRadio"] {
    filter: hue-rotate(245deg) saturate(0.85);
}

/* ---------------- alert boxes (warning/error/info) ---------------- */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
    border: none !important;
    font-size: 13.5px;
}
</style>
""", unsafe_allow_html=True)

OBJ2_TIER_STYLE = {
    "low":    {"gradient": "linear-gradient(135deg, #94A3B8, #64748B)"},
    "medium": {"gradient": "linear-gradient(135deg, #3B82F6, #2563EB)"},
    "high":   {"gradient": "linear-gradient(135deg, #8B5CF6, #7C3AED)"},
    "viral":  {"gradient": "linear-gradient(135deg, #FB923C, #EC4899)"},
}
OBJ3_TIER_STYLE = {
    "Low":    {"icon": "\U0001F534", "ring": "#DC2626"},
    "Medium": {"icon": "\U0001F7E1", "ring": "#D97706"},
    "High":   {"icon": "\U0001F7E2", "ring": "#16A34A"},
}
_QUAL_LABELS = {
    "hook":  [(80, "Grabs attention"), (60, "Decent pull"), (40, "Weak pull"), (0, "Needs a hook")],
    "body":  [(80, "Clear and valuable"), (60, "Mostly clear"), (40, "Loses focus"), (0, "Needs structure")],
    "close": [(80, "Strong CTA"), (60, "Good CTA"), (40, "Soft close"), (0, "No clear ask")],
}


def render_big_badge(kicker: str, name: str, sub: str, gradient: str):
    st.markdown(
        f"""<div style="border-radius:18px;padding:26px 28px;text-align:center;margin:6px 0 18px 0;
                    color:#FFFFFF;background:{gradient};box-shadow:0 14px 30px rgba(18,20,43,0.14);">
                <div style="font-size:12px;letter-spacing:1.5px;text-transform:uppercase;opacity:0.9;font-weight:700;">{kicker}</div>
                <div style="font-size:44px;font-weight:800;line-height:1.15;margin:4px 0 0 0;">{name}</div>
                <div style="font-size:13px;opacity:0.95;margin-top:6px;">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_gauge(score: int, color: str, size: int = 132, inner_font: int = 32):
    ring = max(0, min(100, score))
    inner = size - 30
    st.markdown(
        f"""<div style="width:{size}px;height:{size}px;border-radius:50%;
                    background:conic-gradient({color} {ring}%, var(--track) 0);
                    display:flex;align-items:center;justify-content:center;margin:0 auto;">
              <div style="width:{inner}px;height:{inner}px;border-radius:50%;background:var(--card-bg);
                          box-shadow: inset 0 0 0 1px var(--card-border);
                          display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <div style="font-size:{inner_font}px;font-weight:800;color:var(--text-primary);line-height:1;">{score}</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">/ 100</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_mini_gauge(score: int, label: str, sublabel: str):
    color = "#16A34A" if score >= 70 else ("#D97706" if score >= 40 else "#DC2626")
    ring = max(0, min(100, score))
    st.markdown(
        f"""<div style="text-align:center;">
              <div style="width:72px;height:72px;border-radius:50%;
                          background:conic-gradient({color} {ring}%, var(--track) 0);
                          display:flex;align-items:center;justify-content:center;margin:0 auto;">
                <div style="width:54px;height:54px;border-radius:50%;background:var(--card-bg);
                            box-shadow: inset 0 0 0 1px var(--card-border);
                            display:flex;align-items:center;justify-content:center;
                            font-size:17px;font-weight:800;color:var(--text-primary);">{score}</div>
              </div>
              <div style="margin-top:8px;font-size:13px;font-weight:700;color:var(--text-primary);">{label}</div>
              <div style="font-size:11px;color:var(--text-muted);">{sublabel}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_factor_chip(effect: float, label: str):
    arrow = "↑" if effect > 0 else ("↓" if effect < 0 else "•")
    color = "#16A34A" if effect > 0 else ("#DC2626" if effect < 0 else "#9398AC")
    note = "Helps this prediction" if effect > 0 else ("Hurts this prediction" if effect < 0 else "Neutral")
    st.markdown(
        f"""<div style="background:var(--surface-2);border:1px solid var(--card-border);border-radius:12px;
                    padding:12px 14px;height:100%;">
              <div style="display:flex;align-items:flex-start;gap:10px;">
                <div style="min-width:26px;height:26px;border-radius:50%;background:{color}1F;color:{color};
                            display:flex;align-items:center;justify-content:center;font-weight:800;flex-shrink:0;">{arrow}</div>
                <div>
                  <div style="font-size:13px;font-weight:600;color:var(--text-primary);line-height:1.3;">{label}</div>
                  <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">{note}</div>
                </div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_audience_card(age_range: str, gender_skew: str, persona: str):
    st.markdown(
        f"""<div style="background:var(--surface-2);border:1px solid var(--card-border);border-radius:12px;
                    padding:14px 16px;margin-top:4px;">
              <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
                <span style="background:var(--pill-bg);color:var(--pill-text);border-radius:999px;
                             padding:4px 12px;font-size:12px;font-weight:600;">{age_range}</span>
                <span style="background:var(--pill-bg);color:var(--pill-text);border-radius:999px;
                             padding:4px 12px;font-size:12px;font-weight:600;">{gender_skew}</span>
              </div>
              <div style="font-size:13px;color:var(--text-primary);">{persona}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_card_header(icon: str, title: str, subtitle: str, color: str = "#3B82F6"):
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
              <div style="width:42px;height:42px;border-radius:12px;background:{color}1F;
                          display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">{icon}</div>
              <div>
                <div style="font-size:17px;font-weight:700;color:var(--text-primary);line-height:1.2;">{title}</div>
                <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">{subtitle}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def _qual_label(score: int, dimension: str) -> str:
    for threshold, label in _QUAL_LABELS[dimension]:
        if score >= threshold:
            return label
    return ""


def _score_note(score: int) -> str:
    if score >= 80:
        return "Strong script. A few refinements could make it even better."
    if score >= 60:
        return "Solid start. A few changes could meaningfully lift this."
    if score >= 40:
        return "Needs work before filming, see the suggestions below."
    return "Not ready yet, revisit the hook and structure before filming."


# ---------------------------------------------------------------------------
# Header. The model-source switch is a plain "RI" checkbox, top right, no
# dialog and no explanatory text -- session_state["ri_enabled"] is owned
# entirely by this widget (default True, demonstration model on).
# ---------------------------------------------------------------------------
col_head, col_gear = st.columns([6, 1])
with col_head:
    st.markdown(
        f"""<div class="hero-row">
              <div class="hero-brand">
                <div class="brand-mark">\U0001F680</div>
                <div>
                  <div class="brand-kicker">AI-Powered</div>
                  <div class="brand-title">Reels Virality Predictor</div>
                  <div class="brand-byline">{CREATOR_NAME}</div>
                  <div class="brand-tag">Know how a Reel will land, before you hit publish.</div>
                </div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
with col_gear:
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.checkbox("RI", value=True, key="ri_enabled")

st.write("")

# ---------------------------------------------------------------------------
# Shared Reel details, feeds both predictors below
# ---------------------------------------------------------------------------
with st.container(border=True):
    render_card_header("\U0001F39B️", "Reel details", "Shared across both predictions below, entered once.", "#D97706")
    c1, c2, c3 = st.columns(3)
    with c1:
        category = st.selectbox("Category", CONTENT_CATEGORIES, key="shared_category")
    with c2:
        post_hour = st.slider("Post hour (0-23, local time)", 0, 23, 20, key="shared_post_hour")
    with c3:
        day_of_week = st.selectbox("Day of week", DAYS_OF_WEEK, index=5, key="shared_day")

st.write("")
col_left, col_right = st.columns(2, gap="large")

# ---------------------------------------------------------------------------
# Inputs only, plain widgets (not st.form) so the combined button below can
# react live to whether a script has been written yet.
# ---------------------------------------------------------------------------
with col_left:
    with st.container(border=True):
        render_card_header("\U0001F4C8", "Predict pre-upload performance",
                            "Estimate the expected performance tier.", "#3B82F6")
        f1, f2 = st.columns(2)
        with f1:
            account_type = st.selectbox("Account type", ACCOUNT_TYPES, key="perf_account_type")
        with f2:
            follower_count = st.number_input("Follower count", min_value=0, value=15000,
                                              step=100, key="perf_follower_count")
        has_call_to_action = st.checkbox("Has a call-to-action", value=True,
                                          help="e.g. Follow, Comment, Share", key="perf_has_cta")

with col_right:
    with st.container(border=True):
        render_card_header("\U0001F3AC", "Score a Reel script",
                            "Evaluate your draft script before filming.", "#EC4899")
        script = st.text_area(
            "Your Reel script", height=150, key="script_text",
            placeholder="Paste or write your reel script here...\n\nEg: Hook, main points, call to action, etc.",
        )
        duration = st.number_input("Duration (seconds)", min_value=5, max_value=180,
                                    value=30, key="script_duration")

# ---------------------------------------------------------------------------
# One combined action: a script is the one input with no sensible default,
# so it's the acceptance criteria that unlocks this button. Everything else
# above already has a working default.
# ---------------------------------------------------------------------------
st.write("")
ready = bool(script.strip())
run_clicked = st.button(
    "✨ Predict & Analyze", use_container_width=True, type="primary", disabled=not ready,
)
if not ready:
    st.caption("Write or paste a Reel script above to enable this, both results are produced together.")
st.write("")

if run_clicked:
    source_key = "simulated" if st.session_state.ri_enabled else "real"
    model, label_encoder, feature_columns = load_model_bundle(MODEL_DIR, source_key, "xgboost")
    tier, proba = predict_tier(
        model, label_encoder, feature_columns,
        account_type=account_type, follower_count=follower_count,
        content_category=category, has_call_to_action=has_call_to_action,
        post_hour=post_hour, day_of_week=day_of_week,
    )
    # Stored in session_state, not a local variable -- Streamlit reruns this
    # whole script on EVERY widget interaction (including just ticking the
    # "show probability breakdown" checkbox below).
    st.session_state.obj2_result = {
        "tier": tier, "proba": proba, "source_key": source_key,
        "inputs": dict(account_type=account_type, content_category=category,
                       has_call_to_action=has_call_to_action, post_hour=post_hour,
                       day_of_week=day_of_week),
    }

    with st.spinner("Predicting tier and analyzing your script..."):
        try:
            result = analyze_script(
                script, category, duration, post_hour, get_ai_api_key(),
            )
            result["_script"] = script  # keeps the prompt-preview expander below accurate
            if "obj3_history" not in st.session_state:
                st.session_state.obj3_history = []
            st.session_state.obj3_history.append(result)
        except Exception as e:
            st.error(f"Performance tier is ready below, but the script score failed: {e}")

# ---------------------------------------------------------------------------
# Results, side by side, filled together by the one button above.
# ---------------------------------------------------------------------------
if "obj2_result" in st.session_state or st.session_state.get("obj3_history"):
    st.divider()

res_left, res_right = st.columns(2, gap="large")

with res_left:
    if "obj2_result" in st.session_state:
        with st.container(border=True):
            render_card_header("\U0001F3C6", "Performance tier", "Predicted from your reel details.", "#3B82F6")
            r = st.session_state.obj2_result
            tier, proba, result_source = r["tier"], r["proba"], r["source_key"]
            confidence = max(proba.values())
            style = OBJ2_TIER_STYLE[tier]
            render_big_badge(
                kicker="Predicted performance tier",
                name=tier.upper(),
                sub=f"{confidence*100:.0f}% confidence · tree-ensemble model",
                gradient=style["gradient"],
            )

            if result_source == "simulated":
                st.markdown("**\U0001F4F6 Key factors** _(from your inputs)_")
                effects = explain_simulated_prediction(**r["inputs"])
                if effects:
                    chip_cols = st.columns(len(effects))
                    for chip_col, (effect, label) in zip(chip_cols, effects):
                        with chip_col:
                            render_factor_chip(effect, label)
            else:
                st.caption(
                    "No per-input explanation is shown here: nothing about your inputs "
                    "reliably moves this particular prediction."
                )

            show_chart = st.checkbox("Show probability breakdown", key="obj2_show_chart")
            if show_chart:
                # st.bar_chart sorts string categories alphabetically (high, low,
                # medium, viral), which scrambles the low->viral story. An
                # explicit Altair chart with a fixed sort order keeps it correct.
                proba_df = pd.DataFrame({"tier": list(proba.keys()), "probability": list(proba.values())})
                chart = alt.Chart(proba_df).mark_bar().encode(
                    x=alt.X("tier", sort=TIER_ORDER, title=None),
                    y=alt.Y("probability", title="Probability"),
                    color=alt.Color("tier", sort=TIER_ORDER, legend=None,
                                     scale=alt.Scale(domain=TIER_ORDER,
                                                      range=["#94A3B8", "#3B82F6", "#8B5CF6", "#EC4899"])),
                )
                st.altair_chart(chart, use_container_width=True)
                if result_source == "real" and confidence < 0.35:
                    st.caption(
                        "Notice how flat these probabilities are, no tier stands out. That "
                        "flatness is the actual finding for this dataset, not a display issue."
                    )

with res_right:
    if st.session_state.get("obj3_history"):
        with st.container(border=True):
            render_card_header("\U0001F4DD", "Script readiness", "How your draft script reads.", "#EC4899")
            latest = st.session_state.obj3_history[-1]
            if latest.get("_mode") == "mock":
                st.warning(
                    "Placeholder mode: this score is a heuristic stand-in, not a real "
                    "quality judgment. Live AI scoring will appear automatically once "
                    "it's configured for this deployment."
                )
                if latest.get("_error"):
                    # Deliberate, temporary debug aid: the exact reason the live
                    # call fell back, so you don't have to dig through platform
                    # logs while getting the key set up. Never includes the key
                    # itself. Safe to leave in; remove this block once live
                    # scoring is confirmed working, if you'd rather not show it.
                    st.caption(f"Debug detail: {latest['_error']}")

            tier = latest.get("tier", "Medium")
            style = OBJ3_TIER_STYLE.get(tier, OBJ3_TIER_STYLE["Medium"])
            overall = latest.get("overall_score", 0)

            gauge_col, note_col = st.columns([1, 2])
            with gauge_col:
                render_gauge(overall, style["ring"])
            with note_col:
                st.markdown(f"**{style['icon']} Script readiness score**")
                st.caption(_score_note(overall))

            m1, m2, m3 = st.columns(3)
            with m1:
                render_mini_gauge(latest.get("hook_score", 0), "Hook", _qual_label(latest.get("hook_score", 0), "hook"))
            with m2:
                render_mini_gauge(latest.get("body_score", 0), "Body", _qual_label(latest.get("body_score", 0), "body"))
            with m3:
                render_mini_gauge(latest.get("close_score", 0), "Close", _qual_label(latest.get("close_score", 0), "close"))

            audience = latest.get("predicted_audience")
            if audience:
                st.markdown("**\U0001F465 Predicted audience**")
                render_audience_card(
                    audience.get("age_range", "Broad / all ages"),
                    audience.get("gender_skew", "Not enough signal"),
                    audience.get("persona", ""),
                )
                st.caption(
                    "A stylistic read of this script's language and topic, not a "
                    "measured audience, treat it as illustrative."
                )

            st.markdown("**Why this score?**")
            st.write(latest.get("why", ""))

            st.markdown("**What to improve:**")
            for imp in latest.get("improvements", []):
                st.write(f"- {imp}")

            with st.expander("View the scoring prompt (technical)"):
                st.code(build_prompt(
                    latest.get("_script", ""), latest.get("category", category),
                    latest.get("duration", duration), latest.get("post_hour", post_hour),
                ))

            if len(st.session_state.obj3_history) > 1:
                st.divider()
                st.markdown("**\U0001F4C8 Version comparison**")
                for i, r in enumerate(st.session_state.obj3_history):
                    st.write(f"**Version {i + 1}**: {r.get('overall_score', 0)}/100, {r.get('tier', '')}")

st.divider()
st.markdown(
    f"""<div style="text-align:center;padding:6px 0 2px 0;">
          <div style="font-size:14px;color:var(--text-primary);font-weight:600;">
            Built with \U0001F49C for products by {CREATOR_NAME}
          </div>
          <div style="font-size:12.5px;color:var(--text-muted);margin-top:3px;">
            {CREATOR_UNIVERSITY} · Roll No. {CREATOR_ROLL_NO}
          </div>
        </div>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer-note">Research prototype · Results are indicative and model-dependent.</div>',
    unsafe_allow_html=True,
)
