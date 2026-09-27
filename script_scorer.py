"""
AI-driven Reel script scoring.

Takes one full talking-head Reel script (matching how a creator actually
drafts a Reel) and asks a language model to locate and judge the opening
(hook), middle (body), and close within it as a whole. Scores are 0-100 per
dimension, combined into an overall score and a Low/Medium/High tier.

Categories and reference examples are shared with the performance predictor
(see predictor_utils.CONTENT_CATEGORIES and few_shot_exemplars.json), and
post_hour is the same 0-23 slider used there too, so the user is only ever
asked for either once.

Two modes, chosen automatically:
- LIVE: an API key is available -> a real scoring call, tried against a
  short fixed list of fallback models so one model hiccup doesn't take the
  whole feature down.
- PLACEHOLDER: no key configured, or every live attempt failed -> a
  deterministic, clearly-labelled stand-in, so the product still works
  end to end with nothing to configure.

Security note: this module never reads, stores, or asks for the API key
itself -- it only accepts one as a plain function argument. The key is
fetched once on the backend, from Streamlit secrets (see
streamlit_app.get_ai_api_key()), and there is no field for it anywhere in
the interface. Nobody who opens the running app can see or extract it.
"""

import json
import os
import re

# Anchored to this file's own folder so it works no matter which directory
# the app is launched from (Streamlit Community Cloud, a terminal, an IDE).
EXEMPLARS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "few_shot_exemplars.json")

WEIGHTS = {"hook": 0.40, "body": 0.35, "close": 0.25}
TIER_THRESHOLDS = {"low": 40, "medium": 70}  # <40 Low, 40-69 Medium, 70-100 High

# A short, fixed fallback chain rather than a live "list models" discovery
# call: a fast/economical model first, a stronger one as backup if the first
# is briefly unavailable. Keep this list small -- each entry tried is a real
# scoring call against a quota, not a free retry.
MODEL_CANDIDATES = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]
MAX_MODEL_ATTEMPTS = len(MODEL_CANDIDATES)


def load_exemplars(category: str, path: str = EXEMPLARS_PATH) -> list:
    with open(path) as f:
        data = json.load(f)
    if category not in data:
        valid = [g for g in data if not g.startswith("_")]
        raise ValueError(f"Unknown category {category!r}. Valid: {valid}")
    return data[category]


def _reference_block(category: str) -> str:
    lines = []
    for i, ex in enumerate(load_exemplars(category), 1):
        lines.append(
            f"{i}. Hook: \"{ex['hook']}\" | Body: \"{ex['body']}\" | Close: \"{ex['close']}\"\n"
            f"   (demonstrates: {ex['principle']})"
        )
    return "\n".join(lines)


def build_prompt(script: str, category: str, duration: int, post_hour: int) -> str:
    reference = _reference_block(category)
    posting_time = f"{post_hour:02d}:00 (24-hour, local time)"
    return f"""You are an expert Instagram Reels content strategist. Evaluate the DRAFT \
talking-head Reel script below, written for the "{category}" category, before it is \
filmed or posted.

Identify and judge three parts within the script as a whole:
- Hook: the opening line(s)/action -- does it create curiosity, tension, or an \
immediate reason to keep watching?
- Body: the middle -- does it deliver on the hook's promise with clear structure \
and pacing appropriate for {category} content?
- Close: the ending -- does it land a payoff and give a clear reason to comment, \
save, share, or follow?

Reference examples of STRONG {category} scripts, shown as separate Hook/Body/Close \
lines for calibration only (the script below will likely blend these together \
naturally, that's expected):

{reference}

Now evaluate this DRAFT script:
\"\"\"
{script}
\"\"\"

Planned duration: {duration} seconds. Planned posting time: {posting_time}. Factor in \
whether the duration suits {category}'s typical pacing and whether the posting time is \
a reasonable slot.

Score hook_score, body_score, close_score each 0-100. Compute \
overall_score = round(0.40*hook_score + 0.35*body_score + 0.25*close_score). \
Tier: below 40 is "Low", 40 to 69 is "Medium", 70 and above is "High".

Also estimate the likely audience for this script, based ONLY on its language, \
references, subject matter and tone, not on any real analytics. This is your \
stylistic read of the text, not a measured audience, keep it to one short, honest \
estimate rather than a false-precision breakdown:
- age_range: your best-guess primary age band, e.g. "18-24", "25-34", "35-44", "45+", \
or "Broad / all ages" if the script has no clear age signal.
- gender_skew: one of "Skews female", "Skews male", "Roughly balanced", or \
"Not enough signal".
- persona: one short sentence describing the likely viewer, e.g. "Early-20s, urban, \
into skincare and self-care content."

Respond with ONLY a JSON object, no markdown formatting, no code fences, no preamble, \
in exactly this shape:
{{"hook_score": <0-100 int>, "body_score": <0-100 int>, "close_score": <0-100 int>, \
"overall_score": <0-100 int>, "tier": "<Low|Medium|High>", \
"why": "<1-2 sentence explanation>", "improvements": ["<suggestion 1>", "<suggestion 2>"], \
"predicted_audience": {{"age_range": "<age band>", "gender_skew": "<skew>", \
"persona": "<one sentence>"}}}}"""


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text!r}")
        return json.loads(match.group(0))


def _rank_candidate_models() -> list:
    """Fixed, best-guess-first order. No live discovery call is made against
    the provider -- these are known-good model identifiers, tried in order,
    fast/economical first."""
    return list(MODEL_CANDIDATES)


def _score_live(prompt: str, api_key: str) -> dict:
    from anthropic import Anthropic  # AI SDK, only imported once a key is present

    client = Anthropic(api_key=api_key)
    candidates = _rank_candidate_models()[:MAX_MODEL_ATTEMPTS]

    last_error = None
    for model_name in candidates:
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            parsed = _parse_json_response(text)
            parsed["_mode"] = "live"
            parsed["_model"] = model_name
            return parsed
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"None of the {len(candidates)} configured model(s) returned a usable "
        f"response (tried: {', '.join(candidates)}). Last error: {last_error}"
    )


# Illustrative-only, placeholder-mode stand-in for a category's typical Reels
# audience. Live mode gets its estimate from the model reading the actual
# script text; this dict exists only so placeholder mode has something
# plausible to show with no key configured, it is not derived from any
# measured data.
_MOCK_AUDIENCE_BY_CATEGORY = {
    "Beauty":       ("18-24", "Skews female", "Early-20s viewers into skincare and beauty tips."),
    "Comedy":       ("18-24", "Roughly balanced", "Young adults who enjoy relatable, meme-style humor."),
    "Fashion":      ("18-24", "Skews female", "Style-focused viewers keeping up with current trends."),
    "Fitness":      ("25-34", "Roughly balanced", "Young professionals building a home or gym routine."),
    "Food":         ("25-34", "Roughly balanced", "Home cooks looking for quick, shareable recipes."),
    "Lifestyle":    ("25-34", "Skews female", "Viewers interested in daily routines and self-improvement."),
    "Music":        ("18-24", "Roughly balanced", "Younger viewers discovering new sounds and artists."),
    "Photography":  ("25-34", "Roughly balanced", "Hobbyist creators picking up visual technique tips."),
    "Technology":   ("25-34", "Skews male", "Early adopters curious about gadgets and productivity."),
    "Travel":       ("25-34", "Roughly balanced", "Viewers planning trips or living vicariously through them."),
}


def _score_placeholder(script: str, category: str) -> dict:
    """Deterministic stand-in, heuristic only, NOT a real quality judgment.
    Exists so the product still works end to end with no key configured, or
    if a live call could not complete."""
    words = script.split()
    word_count = len(words)
    has_question = "?" in script
    has_cta = any(kw in script.lower() for kw in
                  ["comment", "follow", "save", "share", "try", "let me know", "tag"])

    def band(base):
        return max(20, min(95, base))

    hook = band(45 + (15 if has_question else 0) + min(15, word_count // 8))
    body = band(40 + min(30, word_count // 4))
    close = band(45 + (25 if has_cta else 0))
    overall = round(WEIGHTS["hook"] * hook + WEIGHTS["body"] * body + WEIGHTS["close"] * close)
    tier = "Low" if overall < TIER_THRESHOLDS["low"] else ("Medium" if overall < TIER_THRESHOLDS["medium"] else "High")

    age_range, gender_skew, persona = _MOCK_AUDIENCE_BY_CATEGORY.get(
        category, ("Broad / all ages", "Not enough signal", "No strong audience signal from this placeholder heuristic.")
    )

    return {
        "hook_score": hook, "body_score": body, "close_score": close,
        "overall_score": overall, "tier": tier,
        "why": f"[Placeholder score, not a real judgment] Heuristic score from script length "
               f"and presence of a question/CTA, for a {category} draft.",
        "improvements": [
            "Add a specific number or timeframe to the hook." if not any(c.isdigit() for c in script) else "Hook already uses a concrete number, good.",
            "End with an explicit ask (comment/save/share)." if not has_cta else "Close already has a clear call-to-action.",
        ],
        "predicted_audience": {
            "age_range": age_range, "gender_skew": gender_skew,
            "persona": f"[Placeholder] {persona}",
        },
        "_mode": "mock", "_model": None,
    }


def analyze_script(script: str, category: str, duration: int, post_hour: int, api_key: str = None) -> dict:
    """Scores one script. api_key is optional and, when present, is used only
    for this one call -- it is never logged, stored, or echoed back."""
    prompt = build_prompt(script, category, duration, post_hour)
    if api_key:
        try:
            result = _score_live(prompt, api_key)
        except Exception as e:
            # Every candidate model failed for this call (bad key, network
            # hiccup, provider outage, etc.). Fall back to the placeholder
            # scorer rather than breaking the rest of the page. Logged
            # server-side (print), AND attached to the result as "_error" so
            # streamlit_app.py can surface it as a small debug caption --
            # useful while a deployment's key setup is still being sorted
            # out, since it saves a trip to the platform's log viewer.
            print(f"[script_scorer] live scoring failed, using placeholder instead: {e}")
            result = _score_placeholder(script, category)
            result["_error"] = f"{type(e).__name__}: {e}"
    else:
        result = _score_placeholder(script, category)
        result["_error"] = (
            "No API key reached the app (get_ai_api_key() returned nothing). "
            "The LLM_API_KEY secret is likely missing, misnamed, or not yet propagated."
        )
    result["category"] = category
    result["duration"] = duration
    result["post_hour"] = post_hour
    return result


if __name__ == "__main__":
    sample_script = (
        "Nobody: ... My brain at 2am: what if the moon is just really far away? "
        "So I spiral into an hour of increasingly unhinged questions, and my roommate "
        "just watches this happen through the wall. Tag someone whose brain does this too."
    )
    mode = "LIVE" if os.environ.get("LLM_API_KEY") else "PLACEHOLDER"
    print(f"Running in {mode} mode\n")
    result = analyze_script(sample_script, "Comedy", 25, 20, os.environ.get("LLM_API_KEY"))
    print(json.dumps(result, indent=2))
