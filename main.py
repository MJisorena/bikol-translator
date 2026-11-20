import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config import *
from engine import BikolTranslator, expand_triplets

app = FastAPI(title="Bikol Translator API")

# Allow the React website (any localhost port, e.g. 5173 or 8080) to call this API
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])

tr = BikolTranslator()
db = sqlite3.connect(f"{PROJECT}/feedback.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_text TEXT, source_lang TEXT, target_lang TEXT, model_output TEXT,
    suggested_text TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")


class TranslateIn(BaseModel):
    source_text: str
    source_lang: str
    target_lang: str


class FeedbackIn(TranslateIn):
    model_output: str
    suggested_text: str | None = None


class Triplet(BaseModel):
    english: str
    virac: str
    pandan: str


@app.get("/")
def index():
    return FileResponse(f"{PROJECT}/index.html")      # simple built-in page (optional)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "memory_pairs": len(tr.tm.df)}


@app.get("/api/v1/directions")
def directions():
    return {"directions": [list(d) for d in DIRECTIONS], "names": LANG_NAME}


@app.post("/api/v1/translate")
def translate(p: TranslateIn):
    if (p.source_lang, p.target_lang) not in DIRECTIONS or not p.source_text.strip():
        raise HTTPException(status_code=400, detail="direction not supported or empty source_text")
    return tr.translate(p.source_text.strip(), p.source_lang, p.target_lang)


@app.post("/api/v1/feedback")
def add_feedback(f: FeedbackIn):
    db.execute("INSERT INTO feedback (source_text, source_lang, target_lang, model_output, suggested_text) VALUES (?,?,?,?,?)",
               (f.source_text, f.source_lang, f.target_lang, f.model_output, f.suggested_text))
    db.commit()
    return {"status": "success"}


@app.get("/api/v1/feedback")
def list_feedback(limit: int = 200):
    rows = db.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (min(limit, 1000),)).fetchall()
    return {"total": db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], "items": [dict(r) for r in rows]}


def sentence_table():
    # one row per sentence: english / virac / pandan, built from the translation memory
    d = tr.tm.df
    v = d[(d.source_lang == "eng_Latn") & (d.target_lang == "bcl_South")][["source_text", "target_text"]]
    p = d[(d.source_lang == "eng_Latn") & (d.target_lang == "bcl_North")][["source_text", "target_text"]]
    v = v.rename(columns={"source_text": "english", "target_text": "virac"})
    p = p.rename(columns={"source_text": "english", "target_text": "pandan"})
    return v.merge(p, on="english", how="outer").fillna("")


@app.get("/api/v1/sentences")
def sentences(q: str = "", page: int = 1, page_size: int = 50):
    t = sentence_table()
    if q.strip():
        m = t.apply(lambda c: c.str.contains(q.strip(), case=False, regex=False)).any(axis=1)
        t = t[m]
    page_size = max(1, min(page_size, 200))
    start = (max(1, page) - 1) * page_size
    return {"total": len(t), "page": page, "page_size": page_size,
            "items": t.iloc[start:start + page_size].to_dict(orient="records")}


@app.post("/api/v1/memory")
def add_memory(rows: list[Triplet]):
    # RAG: add new sentences (English + Virac + Pandan) with NO retraining
    tr.tm.add(expand_triplets(pd.DataFrame([r.model_dump() for r in rows])))
    tr.tm.save(TM_PATH)
    return {"status": "success", "memory_size": len(tr.tm.df)}
