# ResolveBot – FAQ Chatbot — ML-Powered Intent Classification

**Live Demo:** [Click here to chat with the bot!](https://faq-chatbot-ml.streamlit.app)

A phone customer-support chatbot that upgrades keyword-matching with a real machine learning intent classifier. Built as a mini-project combining **Python for Automation** and **Python for Machine Learning** (GTU Skill Based Training, Spoken Tutorial, EduPyramids, SINE, IIT Bombay).

## What it does

- Understands *reworded* questions, not just exact matches — "my battery dies fast" and "how to improve battery life" both route to the same answer.
- Recognizes off-topic questions and redirects politely instead of guessing.
- **Learns live**: if it doesn't know an answer, it asks to be taught, saves the new pair, and retrains instantly — no restart needed.
- Runs as a real chat interface in the browser via Streamlit.

## How it works

1. **TF-IDF** converts each question into a vector based on word importance.
2. A **Logistic Regression** classifier predicts the most likely intent from that vector.
3. Above a confidence threshold, the stored answer for that intent is returned.
4. Below it, the bot asks to be taught — the new pair is appended to `faq_data.json` and the model retrains on the spot.

## Project structure

```text
faq-chatbot-ml/
├── faq_data.json      # Training data: question, intent, answer
├── train_model.py     # Trains the TF-IDF + Logistic Regression model
├── chatbot.py         # Terminal version
├── app.py             # Streamlit web UI (the real product)
└── requirements.txt   # Dependencies list
```

## Running it

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the chat interface
streamlit run app.py
```

The Streamlit app builds its model in memory from the tracked `faq_data.json`
file on every fresh deployment. It never depends on generated `model/` files,
so a fresh Streamlit Cloud container cannot fail because those files are absent.
