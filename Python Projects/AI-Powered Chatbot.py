# ```python
# app.py
"""
AI-Powered Chatbot – FastAPI + Transformers + SQLite
--------------------------------------------------
Single-file implementation for easy deployment or GitHub upload.

Features:
- FastAPI REST API endpoints: /chat, /faq, /faqs, /logs, /health
- SQLite for persistence (sessions, FAQs, interactions)
- FAQ retrieval via embeddings (MiniLM or DistilBERT fallback)
- Generative fallback (FLAN-T5)
- Per-session context memory
"""

import os
import sqlite3
import uuid
import numpy as np
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# ----------------- Config -----------------
DB_PATH = os.getenv("CHATBOT_DB", "chatbot.db")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL_NAME = os.getenv("GEN_MODEL", "google/flan-t5-small")
TOP_K = 3
SIM_THRESHOLD = 0.62
CONTEXT_TURNS = 6

app = FastAPI(title="AI-Powered Chatbot")

# ----------------- Database -----------------

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def create_tables():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS faqs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      question TEXT NOT NULL,
      answer   TEXT NOT NULL,
      embedding BLOB
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      created_at TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      user_text  TEXT,
      bot_text   TEXT,
      retrieved_faq_id INTEGER,
      retrieved_score REAL,
      created_at TEXT,
      FOREIGN KEY(session_id) REFERENCES sessions(id),
      FOREIGN KEY(retrieved_faq_id) REFERENCES faqs(id)
    );
    """)
    conn.commit()
    conn.close()

create_tables()

# ----------------- NLP Setup -----------------
try:
    emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    use_mean_pool = True
except Exception:
    EMBED_MODEL_NAME = "distilbert-base-uncased"
    emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    use_mean_pool = False

generator = pipeline("text2text-generation", model=GEN_MODEL_NAME)

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def normalize(text: str) -> str:
    tokens = word_tokenize(text.lower())
    return " ".join(t for t in tokens if t.isalpha() and t not in STOP_WORDS)

@torch.no_grad()
def embed_texts(texts):
    encoded = emb_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = emb_model(
```
