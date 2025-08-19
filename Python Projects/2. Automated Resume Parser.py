# ```python
# app.py
"""
Automated Resume Parser – Flask + spaCy + PostgreSQL (single-file)
-----------------------------------------------------------------

Upload PDF/DOCX resumes, extract candidate info (name, email, phone, skills,
education, experience) using spaCy + simple rules, and store to a searchable DB.

Features
- POST /upload : multipart file upload (pdf/docx) → JSON + DB insert
- GET  /candidates : list/search candidates (by name/email/skills)
- GET  /candidates/<id> : fetch one
- GET  /health : quick status

Database
- Uses SQLAlchemy ORM. Set DATABASE_URL to a PostgreSQL URI, e.g.:
  postgresql+psycopg2://user:password@localhost:5432/resumes
- Defaults to SQLite (resume.db) for local/dev if DATABASE_URL not set.

Quickstart
---------
1) pip install -r requirements.txt
2) python -m spacy download en_core_web_sm
3) uvicorn app:app --reload   # or: flask run (see bottom)

Notes
- This is intentionally compact and framework-agnostic on the client side.
- You can swap skills list or supply your own via SKILLS_EXTRA env var (comma-separated).
"""

from __future__ import annotations
import io
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify

# Storage
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime
)
from sqlalchemy.orm import sessionmaker, declarative_base

# File parsers
import pdfplumber
from docx import Document as DocxDocument

# NLP
import spacy
from spacy.matcher import Matcher

# ------------------------- Config -------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///resume.db")
MAX_CONTENT_LENGTH_MB = float(os.getenv("MAX_CONTENT_LENGTH_MB", 10))  # file size guard
ALLOWED_EXTS = {"pdf", "docx"}

# Skills dictionary (extend as needed)
DEFAULT_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust", "ruby", "php", "kotlin", "swift", "matlab", "r",
    # Data / ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "nlp", "spacy", "transformers", "opencv", "matplotlib",
    # Web / Backend
    "flask", "fastapi", "django", "spring", "node", "express", "graphql", "rest", "docker", "kubernetes", "nginx",
    # Cloud / DevOps
    "aws", "azure", "gcp", "git", "ci/cd", "jenkins", "terraform", "ansible",
    # Databases
    "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    # Other
    "html", "css", "tailwind", "react", "vue", "angular", "bash", "linux"
}
EXTRA = {s.strip().lower() for s in os.getenv("SKILLS_EXTRA", "").split(",") if s.strip()}
SKILLS = DEFAULT_SKILLS | EXTRA

# Degree & education patterns (simple heuristics)
DEGREE_WORDS = r"(b\.?(tech|e|sc)|m\.?(tech|e|sc|ca)|mba|phd|doctorate|bachelor|master|diploma|degree)"
INSTITUTE_HINTS = r"(university|institute|college|school|iit|nit|iiit|mit|stanford|oxford|cambridge)"

# Email/phone regex
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4})")

# ------------------------- App / DB -------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(MAX_CONTENT_LENGTH_MB * 1024 * 1024)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(64))
    location = Column(String(255))
    summary = Column(Text)
    skills = Column(Text)  # comma-separated lowercased
    education_json = Column(Text)  # JSON list
    experience_json = Column(Text)  # JSON list
    raw_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ------------------------- NLP -------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Lazy instruction message if model missing
    raise SystemExit("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

matcher = Matcher(nlp.vocab)
# Capture degree patterns like "B.Tech in ECE", "Master of Science", etc.
matcher.add("DEGREE", [[{"LOWER": {"REGEX": DEGREE_WORDS}}]])

# ------------------------- Helpers -------------------------

def ext_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower()


def read_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def read_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    doc = DocxDocument(f)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_contact(text: str) -> Dict[str, str | None]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
    }


def best_name(doc) -> str | None:
    # Heuristic: first PERSON entity with > 2 letters, appearing in first 5 lines
    lines = [ln.strip() for ln in doc.text.splitlines() if ln.strip()]
    head = "\n".join(lines[:5])
    head_doc = nlp(head)
    persons = [ent.text.strip() for ent in head_doc.ents if ent.label_ == "PERSON" and len(ent.text.strip()) > 2]
    return persons[0] if persons else None


def guess_location(doc) -> str | None:
    locs = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]
    return locs[0] if locs else None


def extract_skills(text: str) -> List[str]:
    found = set()
    lower = text.lower()
    for s in SKILLS:
        # word boundary-ish containment
        if re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", lower):
            found.add(s)
    return sorted(found)


def extract_education(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    edu_block_indices = [i for i, ln in enumerate(lines) if re.search(r"education|academics|qualifications", ln, re.I)]
    window = lines[min(edu_block_indices)+1:min(edu_block_indices)+15] if edu_block_indices else lines
    # Find degree mentions
    for ln in window:
        if re.search(DEGREE_WORDS, ln, re.I) or re.search(INSTITUTE_HINTS, ln, re.I):
            items.append({"text": ln})
    # Also use spaCy matcher
    doc = nlp("\n".join(window))
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end].text
        items.append({"text": span})
    # uniq while keeping order
    seen = set(); out = []
    for it in items:
        t = it["text"].strip()
        if t.lower() not in seen:
            out.append({"text": t})
            seen.add(t.lower())
    return out[:12]


def extract_experience(text: str) -> List[Dict[str, Any]]:
    # Very light heuristic: capture lines under "Experience/Work/Employment" sections and bullet points
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    idx = next((i for i, ln in enumerate(lines) if re.search(r"experience|employment|work history", ln, re.I)), None)
    block = lines[idx+1: idx+40] if idx is not None else lines
    bullets = [ln for ln in block if ln.startswith("-") or ln.startswith("•") or re.search(r"\d{4}\s*-\s*\d{4}|present", ln, re.I)]
    # Also include some ORG + roles from NER
    doc = nlp("\n".join(block))
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    out = [{"text": b} for b in bullets[:20]] + [{"organization": o} for o in orgs[:10]]
    # Deduplicate
    seen = set(); final = []
    for it in out:
        key = json.dumps(it, sort_keys=True)
        if key not in seen:
            final.append(it); seen.add(key)
    return final[:25]


def parse_resume(file_bytes: bytes, ext: str) -> Dict[str, Any]:
    if ext == "pdf":
        text = read_pdf(file_bytes)
    elif ext == "docx":
        text = read_docx(file_bytes)
    else:
        raise ValueError("Unsupported file type")

    # spaCy doc for NER
    doc = nlp(text)

    contact = extract_contact(text)
    name = best_name(doc)
    location = guess_location(doc)
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)

    # Simple summary: first 3 lines or first sentence
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    summary = " ".join(lines[:3])[:600]

    return {
        "name": name,
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "location": location,
        "summary": summary,
        "skills": skills,
        "education": education,
        "experience": experience,
        "raw_text": text,
    }

# ------------------------- Routes -------------------------
@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/upload')
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No selected file"}), 400
    ext = ext_of(f.filename)
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    file_bytes = f.read()
    try:
        parsed = parse_resume(file_bytes, ext)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store
    db = SessionLocal()
    try:
        cand = Candidate(
            name=parsed.get('name'),
            email=parsed.get('email'),
            phone=parsed.get('phone'),
            location=parsed.get('location'),
            summary=parsed.get('summary'),
            skills=",".join(parsed.get('skills') or []),
            education_json=json.dumps(parsed.get('education') or []),
            experience_json=json.dumps(parsed.get('experience') or []),
            raw_text=parsed.get('raw_text'),
        )
        db.add(cand)
        db.commit()
        db.refresh(cand)
        return jsonify({"id": cand.id, **parsed})
    finally:
        db.close()

@app.get('/candidates')
def list_candidates():
    q_name = request.args.get('name')
    q_email = request.args.get('email')
    q_skill = request.args.get('skill')

    db = SessionLocal()
    try:
        qry = db.query(Candidate)
        if q_name:
            qry = qry.filter(Candidate.name.ilike(f"%{q_name}%"))
        if q_email:
            qry = qry.filter(Candidate.email.ilike(f"%{q_email}%"))
        if q_skill:
            qry = qry.filter(Candidate.skills.ilike(f"%{q_skill.lower()}%"))
        rows = qry.order_by(Candidate.id.desc()).limit(100).all()
        data = [serialize_candidate(c) for c in rows]
        return jsonify(data)
    finally:
        db.close()

@app.get('/candidates/<int:cand_id>')
def get_candidate(cand_id: int):
    db = SessionLocal()
    try:
        c = db.query(Candidate).get(cand_id)
        if not c:
            return jsonify({"error": "Not found"}), 404
        return jsonify(serialize_candidate(c))
    finally:
        db.close()

# ------------------------- Serialization -------------------------

def serialize_candidate(c: Candidate) -> Dict[str, Any]:
    return {
        "id": c.id,
        "name": c.name,
        "email": c.email,
        "phone": c.phone,
        "location": c.location,
        "summary": c.summary,
        "skills": (c.skills or "").split(",") if c.skills else [],
        "education": json.loads(c.education_json or "[]"),
        "experience": json.loads(c.experience_json or "[]"),
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }

# ------------------------- Entrypoint -------------------------
if __name__ == '__main__':
    # Flask dev server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
```

---

### requirements.txt (put alongside `app.py`)

```
flask
sqlalchemy
psycopg2-binary
pdfplumber
python-docx
spacy
uvicorn
```

### Example cURL

```
# Upload a PDF
curl -X POST http://localhost:5000/upload \
  -F "file=@/path/to/resume.pdf"

# List candidates filtered by a skill
curl "http://localhost:5000/candidates?skill=python"

# Fetch one
curl http://localhost:5000/candidates/1
```

### Environment

```
# Default dev (SQLite)
# No env needed

# Production (PostgreSQL)
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/resumes"

# Extra skills
export SKILLS_EXTRA="hadoop,spark,airflow"
```

### Notes & Next Steps

* Improve education parsing with specialized patterns per country.
* Add authentication for candidate browsing.
* Add a simple web UI for uploads & search.
* Consider using a vector DB later for semantic search over raw\_text.
* For DOC (older Word), add `textract` or `mammoth` conversion.
