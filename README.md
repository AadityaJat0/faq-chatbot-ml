# ResolveBot — Phone Support Chatbot

**Live demo:** https://faq-chatbot-ml.streamlit.app

A phone customer-support chatbot that replaces keyword matching with a trained
machine-learning intent classifier, keeps each visitor's conversation without
asking anyone to log in, and treats user-supplied answers as suggestions for
review rather than as instant edits to its verified knowledge.

Built as a mini-project combining **Python for Automation** and **Python for
Machine Learning** (GTU Skill Based Training — Spoken Tutorial, EduPyramids,
SINE, IIT Bombay).

---

## What it does

- **Understands rewording.** "my battery dies fast" and "how to improve battery
  life" both reach the same verified answer.
- **Knows when a question is not its business.** A semantic domain gate
  redirects unrelated questions instead of guessing or asking to be taught.
- **Remembers your chat without a login.** An anonymous browser token restores
  the same conversation when you come back, so you can leave the app to follow
  a multi-step instruction and return to it.
- **Collects suggestions safely.** When it has no verified answer for a
  plausible support question, it saves what you suggest for the owner to
  review. It never publishes that answer to other visitors on its own.
- **Shows its working.** Every reply carries a persistent green / amber / red
  badge with the predicted intent, confidence, margin, and in-scope score.

---

## How it works

### Two models, two different jobs

| Component | Question it answers |
|---|---|
| TF-IDF + Logistic Regression | *Which* of my known intents is this closest to? |
| Sentence embeddings + cosine similarity | *Does this belong to phone support at all?* |

Keeping these separate matters. Low classifier confidence means "unsure between
my intents" — it does **not** mean "out of domain". Conflating the two produced
a real bug: the word `swimming?` shares no vocabulary with any training example,
so every intent scored near zero, `greeting` won by accident, the low-confidence
branch fired, and ResolveBot asked a visitor to supply a *phone-support answer*
for a swimming question.

### The semantic domain gate

`semantic_gate.py` embeds every **in-scope** question from `faq_data.json`
(`out_of_scope` rows are excluded on purpose — we measure closeness to what
ResolveBot knows, not to a list of refusals) using
`sentence-transformers/all-MiniLM-L6-v2`. An incoming question is embedded and
compared by cosine similarity against that reference set.

Because embeddings capture meaning rather than shared words, this catches
unrelated questions that share no vocabulary with the training data — precisely
the case that adding more `out_of_scope` examples cannot cover, since you can
only list topics you already thought of. The dataset now carries a much wider
spread of `out_of_scope` examples too (sports, food, weather, school subjects,
animals, entertainment, general chat), but that is the weaker of the two
protections.

### Decision order

```
1. similarity < hard_out_of_scope        -> amber off-topic (overrides a confident classifier)
2. classifier predicts out_of_scope      -> amber off-topic
3. classifier clears confidence + margin -> green verified answer
4. similarity >= suggestion_min          -> red low confidence, suggestion invited
5. otherwise                             -> amber off-topic
```

Step 5 is the fix. Previously this case fell through to the suggestion flow.

### Thresholds

Defined once in `semantic_gate.py` and documented there:

| Threshold | Default | Meaning |
|---|---|---|
| `hard_out_of_scope` | 0.20 | Below this a question is unrelated even if the classifier is confident |
| `suggestion_min_similarity` | 0.35 | The bar a question must clear before a visitor is asked to supply an answer |

Do not take the defaults on trust. Run:

```bash
python3 calibrate_thresholds.py
```

It scores a labelled probe set (ten clearly in-scope questions including ones
the dataset cannot answer, ten clearly unrelated ones), prints every score so
the choice can be justified, and writes `scope_thresholds.json`, which the app
prefers over the defaults when present.

### Graceful fallback

If the embedding model cannot load — no network on first run, a failed install,
or too little memory — the app does **not** crash and does **not** disable the
gate. It falls back to a strict lexical gate that scores vocabulary overlap
against the verified questions plus a device-term lexicon. That fallback is
deliberately stricter than the embedding gate: it may redirect a borderline
genuine question, but it will not let an unrelated one into the suggestion
queue. The About panel in the sidebar shows which mode is active.

---

## Owner-only suggestion review

Suggestions land in the Supabase `knowledge_suggestions` table with status
`pending`. Reviewing them no longer means typing `approved` into the Supabase
Table Editor by hand.

**Where:** `https://<your-app>/Owner_Review`

The page is not linked from the public sidebar (`showSidebarNavigation` is off
in `.streamlit/config.toml`) and is protected by a password held in Streamlit
Secrets and compared in constant time, with a lockout after five failed
attempts. The public chatbot remains completely anonymous and login-free.

What the page gives you:

- Pending, approved, rejected, or all suggestions, newest first, ten per page
- Question, suggested answer, submitted date, and current status for each row
- **Approve** and **Reject** buttons, with an optional reviewer note
- Success and error feedback, and a refreshed list after every action
- Counts per status in the sidebar

### Approval is a decision, not a deployment

**Approving does not retrain the model and does not publish anything to
GitHub.** ResolveBot trains only from `faq_data.json`. To make an approved
answer live:

1. Add the question, a unique intent name, and the answer to `faq_data.json`
   (or add the questions to `build_faq_data.py` and re-run it).
2. Commit and push to `main`.
3. Let Streamlit Cloud redeploy.
4. The model and the gate's reference embeddings are rebuilt from the updated
   verified dataset.

This separation is what keeps every published answer reviewable and reversible
in version history.

---

## Setup

```bash
pip install -r requirements.txt
python3 build_faq_data.py       # regenerate faq_data.json from the source lists
python3 calibrate_thresholds.py # verify and write the domain-gate thresholds
python3 -m unittest discover -s tests -v
streamlit run app.py
```

The first run downloads the MiniLM model (~90 MB) from Hugging Face and caches
it, so expect a slower first start both locally and on the first Streamlit
Cloud deployment after this change.

### Secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` locally,
and paste the same contents into **Streamlit Cloud → your app → Settings →
Secrets**:

```toml
[supabase]
url = "https://YOUR-PROJECT-REF.supabase.co"
service_role_key = "..."

[owner]
review_password = "..."
```

`.streamlit/secrets.toml` must stay out of git. The service-role key bypasses
Row Level Security, so it is server-side only: never in the repository, never
in client-side code, never in a screenshot.

### Database migration

Run `migrations/002_review_metadata.sql` in the Supabase SQL Editor. Every
statement is guarded, so it is safe to run more than once and safe against a
deployment already serving traffic. It adds `reviewed_at` and `reviewer_note`,
defaults `status` to `pending`, constrains it to the three review states,
indexes the review query, and keeps Row Level Security enabled.

---

## Project structure

```
faq-chatbot-ml/
├── faq_data.json               # Verified knowledge base — the only training source
├── build_faq_data.py           # Regenerates faq_data.json (one answer per intent)
├── train_model.py              # TF-IDF + Logistic Regression intent classifier
├── semantic_gate.py            # Embedding domain gate, thresholds, decision logic
├── calibrate_thresholds.py     # Calibrates thresholds against a labelled probe set
├── history_store.py            # Anonymous persistent chat history (Supabase)
├── suggestion_store.py         # Owner-side review reads and status updates
├── chatbot.py                  # Terminal version, for quick model testing
├── app.py                      # Public Streamlit chat interface
├── pages/1_Owner_Review.py     # Password-gated review page
├── migrations/                 # Idempotent SQL migrations
├── tests/                      # Unit tests (no network, no model download)
└── .streamlit/                 # config.toml + secrets.toml.example
```

---

## Privacy and security

- No login, no signup, no personal data. The only identifier is a random token
  in a cookie that lasts about 180 days.
- The raw token never reaches the database; only its SHA-256 hash is stored.
- History returns on the same browser and device only. A different browser,
  incognito mode, or cleared browser data starts a separate conversation. There
  is intentionally no cross-device sync, because there is no login.
- Any visitor can permanently delete their own saved chat, with confirmation.
- The service-role key lives in Streamlit Secrets, never in the repository.
- Row Level Security is enabled; public clients cannot read or write the tables
  directly.
- The interface warns against entering passwords, PINs, or OTPs.

---

## Limitations

- No cross-device history without a login — a deliberate trade-off.
- Clearing browser data starts a new anonymous conversation.
- Approved suggestions still require a manual commit before they go live.
- The embedding model adds a slower cold start on Streamlit Community Cloud.
- The lexical fallback cannot recognise meaning and is stricter than the
  embedding gate by design.

## Future scope

An owner-only **"Approve and create pull request"** action could open a GitHub
Pull Request adding the approved question, intent, and answer to
`faq_data.json`; the owner reviews the diff and merges; Streamlit Cloud
redeploys; ResolveBot retrains from the updated verified dataset.

**This is not implemented.** It is preferable to the alternatives because
writing to Streamlit Cloud's local filesystem is neither durable nor reflected
in GitHub; pushing straight to `main` publishes without review; and storing
official knowledge against a browser ID would fragment shared support advice
across visitors, when the browser token exists only to restore one person's
private chat history. GitHub stays the version-controlled source of truth, and
human review stays mandatory.

Also planned: optional authentication for cross-device history, audit logging
of review decisions, role-based access, and analytics on unanswered questions.
