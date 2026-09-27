"""
Objective 4 - prediction logic for the Streamlit app, kept separate from any
Streamlit/UI code so it can be unit-tested directly with plain Python.

AI-Powered Instagram Reels Virality Predictor - Yadunandan N S (24MCMS101224)

Loads the Objective 2 classifiers (both the real-label model and the
disclosed simulated-label model, see build_simulated_target.py and the
project notes for why both exist and what each one honestly shows) and
applies the same one-hot encoding used at training time to a single new
post's pre-upload details.

PURE-PYTHON BUILD: the F1Y build loaded joblib-pickled XGBoost models. Here the
same trained trees live in model_real.py and model_demo.py as plain Python
data, and tree_engine.py walks them (results match the joblib models exactly,
see tools/export_models.py). No joblib, xgboost, scikit-learn, numpy or pandas
is needed to make a prediction. The public functions below keep the same names
and signatures as F1Y, so streamlit_app.py did not need to change.
"""

import importlib
from functools import lru_cache

from tree_engine import SimpleLabelEncoder, TreeEnsembleClassifier

CONTENT_CATEGORIES = [
    "Beauty", "Comedy", "Fashion", "Fitness", "Food",
    "Lifestyle", "Music", "Photography", "Technology", "Travel",
]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ACCOUNT_TYPES = ["creator", "brand"]
TIER_ORDER = ["low", "medium", "high", "viral"]

# Two model sources, deliberately both kept in the app. Which one is "correct"
# to show depends on being honest about what each one is, not on which looks
# better; see each disclaimer below and the Methodology tab in the app.
MODEL_CONFIGS = {
    "simulated": {
        "label": "Demonstration model (responds to your inputs)",
        "module": "model_demo",
        "disclaimer": (
            "Trained on a RESEARCHER-CONSTRUCTED demonstration target built from a "
            "disclosed formula over a subset of the real pre-upload columns (see "
            "build_simulated_target.py for the exact formula; caption length and "
            "hashtag count are deliberately excluded from it). It demonstrates that "
            "the classification pipeline correctly learns structure when structure "
            "exists, achieving 41.6% accuracy against a 25% baseline. It is NOT a "
            "validated real-world virality prediction, and must be labeled as a "
            "demonstration wherever it is shown or discussed."
        ),
    },
    "real": {
        "label": "Real-data model (honest null finding)",
        "module": "model_real",
        "disclaimer": (
            "Trained on the ORIGINAL dataset label. A leakage control test proved this "
            "pipeline works correctly (90.4% accuracy when fed real post-publish "
            "outcomes), and a separate diagnostic proved the real label has no "
            "statistical relationship to any pre-upload feature, individually, jointly, "
            "or through account identity. Predictions below will look close to random "
            "on purpose. That is the honest real-data finding, not a bug in this app."
        ),
    },
}


def build_feature_row(feature_columns, account_type, follower_count, content_category,
                       has_call_to_action, post_hour, day_of_week):
    """
    Build a single feature row (a list of numbers) encoded exactly like the
    training data, in the training-time column order.

    Same one-hot logic as F1Y's pandas get_dummies + reindex(fill_value=0):
    every column starts at 0, then the numeric inputs and the one dummy column
    that matches each categorical input are set. A value that matches no
    training column simply leaves all of that group's dummies at 0.

    caption_length and hashtags_count are deliberately not accepted here
    (product decision, 2026-09-23): neither the app nor the retrained models
    use them any more, see build_simulated_target.py and
    xgboost_virality_classifier.py for the matching change on the training side.
    """
    values = {
        "follower_count": follower_count,
        "has_call_to_action": int(bool(has_call_to_action)),
        "post_hour": post_hour,
        f"account_type_{account_type}": 1,
        f"content_category_{content_category}": 1,
        f"day_of_week_{day_of_week}": 1,
    }
    return [values.get(column, 0) for column in feature_columns]


@lru_cache(maxsize=None)
def _load_cached(source_key):
    cfg = MODEL_CONFIGS[source_key]
    data = importlib.import_module(cfg["module"])
    model = TreeEnsembleClassifier(
        data.TREES, data.BASE_MARGIN, data.FEATURE_COLUMNS, data.CLASSES
    )
    return model, SimpleLabelEncoder(data.CLASSES), list(data.FEATURE_COLUMNS)


def load_model_bundle(model_dir, source_key, algorithm):
    """algorithm: 'xgboost'. model_dir is kept for F1Y compatibility and is not
    used: the models are Python modules next to this file, not files to load.
    The Random Forest was never used by the app and is not part of this build."""
    if algorithm != "xgboost":
        raise ValueError(
            f"Unsupported algorithm {algorithm!r}: this build ships the XGBoost models only."
        )
    if source_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model source {source_key!r}. Valid: {list(MODEL_CONFIGS)}")
    return _load_cached(source_key)


def predict_tier(model, label_encoder, feature_columns, **feature_kwargs):
    """Returns (predicted_tier: str, proba_by_class: dict[str, float])."""
    row = build_feature_row(feature_columns, **feature_kwargs)
    pred_idx = model.predict(row)
    proba = model.predict_proba(row)
    tier = label_encoder.inverse_transform([pred_idx])[0]
    proba_by_class = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}
    # order by the natural low->viral tier order for consistent chart display
    proba_by_class = {t: proba_by_class[t] for t in TIER_ORDER if t in proba_by_class}
    return tier, proba_by_class


# ---------------------------------------------------------------------------
# "Why" explainer for the SIMULATED model only. This intentionally reuses the
# exact disclosed formula from build_simulated_target.py rather than trying
# to explain the model's internals -- for a demonstration target, the true
# generating formula IS the ground truth, so citing it directly is the most
# honest possible explanation. There is no equivalent for the real-data
# model: since no input reliably moves that prediction (that's the proven
# null finding), any per-input explanation of it would be fabricated, so the
# app does not attempt one.
# ---------------------------------------------------------------------------
_CATEGORY_BONUS = {
    "Comedy": 0.60, "Music": 0.50, "Beauty": 0.30, "Fashion": 0.30,
    "Food": 0.20, "Fitness": 0.10, "Travel": 0.10, "Photography": 0.00,
    "Technology": -0.10, "Lifestyle": -0.20,
}


def _peak_hour_bonus(hour):
    if hour in (18, 19, 20, 21, 22):
        return 0.50, "posting in the evening peak window"
    if hour in (12, 13):
        return 0.20, "posting in the lunch peak window"
    if hour in (1, 2, 3, 4):
        return -0.20, "posting in the dead-of-night hours"
    return 0.0, None


def explain_simulated_prediction(account_type, content_category, has_call_to_action,
                                  post_hour, day_of_week):
    """Returns a short list of (effect, label) tuples, sorted strongest-first,
    describing which inputs pushed the SIMULATED score up or down and by how
    much, straight from the disclosed formula's own terms.

    Only five terms remain (2026-09-23): the hashtag-count and caption-length
    sweet-spot terms were removed from build_simulated_target.py's formula
    itself, not just hidden here, so this list still matches the formula
    exactly, every term shown here is a term the label was actually built from."""
    effects = []

    cat_bonus = _CATEGORY_BONUS.get(content_category, 0.0)
    if cat_bonus != 0:
        effects.append((cat_bonus, f"{content_category} category baseline"))

    hour_bonus, hour_label = _peak_hour_bonus(post_hour)
    if hour_label:
        effects.append((hour_bonus, hour_label))

    if day_of_week in ("Saturday", "Sunday"):
        effects.append((0.20, "weekend posting"))

    if has_call_to_action:
        effects.append((0.40, "includes a call-to-action"))
    else:
        effects.append((0.0, "no call-to-action"))

    if account_type == "creator":
        effects.append((0.10, "creator account (small boost over brand)"))

    effects.sort(key=lambda x: -abs(x[0]))
    return effects[:3]  # the three strongest factors, positive or negative
