import os, re, time, torch, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLS = ["source_text", "target_text", "source_lang", "target_lang"]
TRIPLET_COL = {"eng_Latn": "english", "bcl_South": "virac", "bcl_North": "pandan"}

def norm(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()

def expand_triplets(df):
    # df has columns english, virac, pandan -> one row per trained direction
    parts = [pd.DataFrame({"source_text": df[TRIPLET_COL[s]].astype(str).str.strip().values,
                           "target_text": df[TRIPLET_COL[t]].astype(str).str.strip().values,
                           "source_lang": s, "target_lang": t}) for s, t in DIRECTIONS]
    return pd.concat(parts, ignore_index=True)

def load_model(adapter=True, merge=True):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, dtype=dtype)
    except TypeError:   # older transformers
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        if merge:
            model = model.merge_and_unload()   # faster inference, no adapter overhead
    return tok, model.to(DEVICE).eval()

@torch.no_grad()
def translate_batch(tok, model, texts, src, tgt, prefix=True, beams=4, max_new=128):
    # one batch = one direction (src -> tgt). beams must be >= 2 (confidence uses beam scores).
    tok.src_lang = NLLB_TAG[src]
    if prefix:
        texts = [make_input(t, tgt) for t in texts]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
    out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_TAG[tgt]),
                         num_beams=beams, max_length=max_new,
                         output_scores=True, return_dict_in_generate=True)
    hyps = tok.batch_decode(out.sequences, skip_special_tokens=True)
    conf = torch.exp(out.sequences_scores.float()).tolist()   # length-normalised probability
    return hyps, conf

class TranslationMemory:
    def __init__(self):
        self.df = pd.DataFrame(columns=COLS)
        self.vec, self.X, self.vocab = None, None, {}

    def add(self, rows):
        self.df = pd.concat([self.df, rows[COLS]], ignore_index=True).drop_duplicates().reset_index(drop=True)
        keys = self.df["source_text"].map(norm)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        self.X = self.vec.fit_transform(keys)          # rows are L2-normalised -> dot product = cosine
        self.vocab = {lang: {w for s in keys[self.df["source_lang"] == lang] for w in s.split()}
                      for lang in self.df["source_lang"].unique()}
        return self

    def lookup(self, text, src, tgt):
        mask = ((self.df["source_lang"] == src) & (self.df["target_lang"] == tgt)).values
        if not mask.any():
            return 0.0, None
        sims = (self.X[mask] @ self.vec.transform([norm(text)]).T).toarray().ravel()
        i = int(sims.argmax())
        return float(sims[i]), self.df[mask].iloc[i]

    def coverage(self, text, src):
        ws = norm(text).split()
        v = self.vocab.get(src, set())
        return sum(w in v for w in ws) / max(1, len(ws))

    def save(self, path):
        self.df.to_csv(path, index=False)

    @classmethod
    def load(cls, path):
        return cls().add(pd.read_csv(path))

def should_fallback(sim, cov, conf):
    return sim < FUZZY_SIM and (cov < MIN_COVERAGE or conf < MIN_CONF)

class BikolTranslator:
    def __init__(self, tm_path=TM_PATH):
        if not os.path.exists(tm_path):   # fail fast, before downloading the model
            raise FileNotFoundError(f"{tm_path} not found. Copy memory.csv from your Drive folder into this project folder.")
        self.tok, self.model = load_model()
        self.tm = TranslationMemory.load(tm_path)
        translate_batch(self.tok, self.model, ["warm up"], "eng_Latn", "bcl_South", beams=NUM_BEAMS)   # first call is slow

    def translate(self, text, source_lang, target_lang):
        t0 = time.perf_counter()
        sim, row = self.tm.lookup(text, source_lang, target_lang)
        if row is not None and sim >= EXACT_SIM:
            out, conf = row["target_text"], sim
        else:
            hyps, confs = translate_batch(self.tok, self.model, [text], source_lang, target_lang, beams=NUM_BEAMS)
            out, conf = hyps[0], confs[0]
            if should_fallback(sim, self.tm.coverage(text, source_lang), conf):
                out = FALLBACK.get(target_lang) or FALLBACK[source_lang]
        ms = int((time.perf_counter() - t0) * 1000)
        return {"status": "success", "translated_text": out,
                "confidence_score": round(conf, 2), "inference_latency_ms": ms}
