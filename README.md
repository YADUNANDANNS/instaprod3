# AI-Powered Instagram Reels Virality Predictor

Yadunandan N S (24MCMS101224) &middot; Ramaiah University of Applied Sciences

A single-page product: a shared "Reel details" card feeds two predictors
side by side &mdash; a pre-upload **performance tier** model and an AI-scored
**script readiness** check. This build is deployment-friendly for Streamlit
Community Cloud: no manual API key entry anywhere in the interface, a light-
and-dark-adaptive UI, and a flat file layout (everything sits directly in
this folder, no sub-folders).

## What is in this folder

| File | Role |
|---|---|
| `streamlit_app.py` | The app: layout, styling, and page logic |
| `predictor_utils.py` | Performance-tier prediction and the "key factors" explainer |
| `tree_engine.py` | Small standard-library predictor that walks the exported trees |
| `model_demo.py` | Trained tree data for the performance-tier model (used when the "RI" toggle is on) |
| `model_real.py` | Trained tree data for the performance-tier model (used when the "RI" toggle is off) |
| `script_scorer.py` | AI script scoring, with an automatic placeholder fallback |
| `few_shot_exemplars.json` | Reference examples used to calibrate script scoring |
| `requirements.txt` | What the deployed app installs |

No `xgboost`, `scikit-learn`, `joblib` or `numpy` is needed: the two trained
models live as plain Python data and are walked by a small pure-Python
engine, so there is nothing binary to version-match and nothing heavy to
install.

## Run it locally

Python 3.10 or newer (avoid brand-new versions like 3.14 on Streamlit Cloud,
see Troubleshooting below).

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Then open http://localhost:8501.

## Configure the AI scoring key (do this once)

The script-readiness card calls an AI API behind the scenes to score a
Reel script. The key lives **only on the server side** &mdash; there is no
field for it anywhere in the running app, so nobody using the deployed
product can see it, copy it, or use it themselves. Only you, from your own
deployment's settings, can set or change it.

**On Streamlit Community Cloud, exact template to paste:**

Whether you're setting this during initial deploy (the **Advanced settings**
dialog, which also shows a Python version dropdown) or afterwards (app menu
&rarr; **Settings** &rarr; **Secrets**), it's the same box, and it always
takes the same content. Paste exactly this, with your real key swapped in:

```toml
LLM_API_KEY = "sk-ant-your-real-key-here"
```

Rules for that box:
- The name on the left, `LLM_API_KEY`, must be exactly that, unquoted.
- The value on the right must be in double quotes.
- One line, no extra spaces inside the quotes, no trailing comma.
- Click **Save**. Changes take about a minute to propagate; a manual
  **Reboot app** (app menu) makes it immediate.

**Running locally:** create a `.streamlit/secrets.toml` file next to this
app (not included in this package) with the same line:
```toml
LLM_API_KEY = "sk-ant-your-real-key-here"
```
or export it as an environment variable before running the app:
```bash
export LLM_API_KEY="sk-ant-your-real-key-here"     # macOS/Linux
setx LLM_API_KEY "sk-ant-your-real-key-here"       # Windows (new terminal after this)
```

**If the key is missing, or a live call fails:** the script-readiness card
falls back automatically to a clearly-labelled placeholder score, so the
rest of the product keeps working end to end. Nothing crashes for a visitor
either way.

## Put it on GitHub

Create an empty repository on github.com first (no README, no license, so
nothing conflicts), then, inside this folder:

```bash
git init
git add .
git commit -m "Reels Virality Predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Your API key should never be committed. It only ever lives in Streamlit's
secrets store (or your local `secrets.toml` / environment variable), never
in a file in this repository.

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Choose **Create app**, then **deploy from a GitHub repository**.
3. Repository: the one you just pushed. Branch: `main`. Main file path:
   `streamlit_app.py`.
4. In **Advanced settings**, leave Python version at a stable release
   (3.11 or 3.12), and paste the secrets template above.
5. Deploy.

## Editing the credit footer

`streamlit_app.py` has three lines near the top you can edit directly:

```python
CREATOR_NAME = "Yadunandan N S"
CREATOR_UNIVERSITY = "Ramaiah University of Applied Sciences"
CREATOR_ROLL_NO = "24MCMS101224"
```

## Troubleshooting: still seeing "Placeholder mode"

If the script-readiness card keeps showing the placeholder banner even
though the secret looks right, check these in order:

1. **Billing/credits on your API account.** The single most common cause.
   A freshly created API key with no payment method or credits added will
   authenticate fine but fail on every real request. Check your API
   provider's console under Plans/Billing for a credit balance.
2. **Secret name and quoting.** Must be exactly `LLM_API_KEY = "..."`,
   quotes included, no typos.
3. **Propagation/reboot.** Give it a minute after saving, or use **Reboot
   app** from the app menu to force it immediately.
4. **Check the logs for the real reason.** App menu &rarr; **Manage app**
   opens a live log panel. After clicking **Predict & Analyze** on the live
   app, search the logs for `script_scorer`. A line starting with
   `[script_scorer] live scoring failed, using placeholder instead:` names
   the exact underlying error (never your key itself), which tells you
   precisely what to fix.
5. **Python version.** A very new runtime (e.g. 3.14) can behave
   unpredictably with pinned dependencies. If anything about the deploy
   seems off, redeploy with Python 3.11 or 3.12 in Advanced settings.
