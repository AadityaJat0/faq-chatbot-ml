# ResolveBot — Phone Support Chatbot

**Live demo:** [Chat with ResolveBot](https://faq-chatbot-ml.streamlit.app)

ResolveBot is a phone customer-support chatbot built as a mini-project for
Python for Automation and Python for Machine Learning. It uses a real machine
learning intent classifier instead of exact keyword matching, and it keeps an
anonymous visitor's chat available after they close and reopen the app.

## What ResolveBot does

- Understands reworded questions: “my battery dies fast” and “how can I
  improve battery life?” route to the same support answer.
- Handles common phone topics including battery, warranty, orders, Wi-Fi,
  software updates, and account access.
- Redirects politely when a question is outside its support scope.
- Saves a conversation automatically in a cloud database, so it can be
  restored after the visitor closes and reopens the app on the same browser and
  device.
- Lets visitors suggest an answer for an unverified question. Suggestions are
  saved for review; they do not immediately alter the model or public answers.

## How it works

1. **TF-IDF** converts each training question into a vector of word-importance
   values.
2. **Logistic Regression** predicts the most likely support intent.
3. ResolveBot returns the verified answer for a confident prediction.
4. An anonymous browser token identifies one saved conversation. The database
   stores only a SHA-256 hash of that token, not the raw browser token.
5. Every user message and bot reply is saved immediately to Supabase Postgres.
   On a later visit, ResolveBot loads that browser's conversation again.

## Anonymous history, not login

ResolveBot intentionally has **no sign-in screen**. It stores a long, random
token in a first-party browser cookie for 180 days. That means:

- Chat history returns on the **same browser and device**.
- Clearing browser/site data, private browsing, or opening the app on another
  device starts a separate chat.
- Anyone sharing the same browser profile could view the same chat, so users
  should not enter passwords, PINs, OTPs, or sensitive personal information.

Adding authentication later would make history portable across devices, but it
is not needed for the same-phone return-to-chat use case.

## Persistent-history setup (Supabase)

The app remains usable without database credentials, but chat history will be
session-only until this setup is complete.

1. Create a free [Supabase](https://supabase.com) project.
2. In its **SQL Editor**, run the complete
   [`supabase_schema.sql`](supabase_schema.sql) script from this repository.
3. In Supabase **Project Settings → API**, copy the Project URL and the
   **service_role** key. The service-role key is server-only; never place it in
   browser code or commit it to GitHub.
4. For local development, copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` and replace the placeholders:

   ```toml
   [supabase]
   url = "https://YOUR-PROJECT-REF.supabase.co"
   service_role_key = "YOUR-SUPABASE-SERVICE-ROLE-KEY"
   ```

5. For Streamlit Community Cloud, open the app's **Settings → Secrets** and
   paste the same TOML values. Do not add this file to GitHub.
6. Redeploy or reboot the app, then test this exact flow: ask a question,
   close the browser completely, reopen the app in the same browser, and check
   that the answer is still visible.

The SQL schema enables Row Level Security and removes public API access to the
chat tables. The Streamlit server accesses them with the private service-role
key stored in Streamlit Secrets.

## Project structure

```text
faq-chatbot-ml/
├── app.py                         # ResolveBot Streamlit interface
├── history_store.py                # Anonymous Supabase history layer
├── faq_data.json                   # Verified model training data
├── train_model.py                  # TF-IDF + Logistic Regression training
├── chatbot.py                      # Terminal test version
├── supabase_schema.sql             # One-time persistent-history schema
├── .streamlit/secrets.toml.example # Safe Secrets template
└── requirements.txt                # Python dependencies
```

## Run locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Supabase Secrets (optional for local chat, required for history)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 4. Launch ResolveBot
streamlit run app.py
```

## Responsible feedback loop

The original project wrote any visitor-provided answer directly into the
running Streamlit container. Cloud filesystem changes are not durable, and a
public visitor should not be able to change verified support advice instantly.

ResolveBot now stores those answers as **pending suggestions** in the database.
Review an entry in the `knowledge_suggestions` table before adding a trusted
question/answer pair to `faq_data.json` and redeploying the model.
