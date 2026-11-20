import argparse, html, json, re, unicodedata
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DIRECTIONS

# Raw file columns: Number, English, Virac (= Southern Bikol), Pandan (= Northern Bikol).
# (The raw CSV also has ~16,000 empty columns; usecols below ignores them.)
RAW_COL = {"eng_Latn": "English", "bcl_South": "Virac", "bcl_North": "Pandan"}
SLASH_KEEP_FIRST = True   # "isisila/lulutuon" -> "isisila" (Bikol columns only)


def clean(text, bikol):
    t = html.unescape(str(text))
    t = re.sub(r"<[^>]+>", " ", t)                       # strip HTML
    t = unicodedata.normalize("NFC", t)                  # standardise diacritics
    for a, b in {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-"}.items():
        t = t.replace(a, b)
    if bikol and SLASH_KEEP_FIRST:
        t = re.sub(r"(\S+?)/\S+", r"\1", t)              # keep first spelling variant
    t = re.sub(r"[^\w\s.,!?;:'\"()\-/]", " ", t)         # stray special characters
    return re.sub(r"\s+", " ", t).strip()                # also fixes \n and double spaces


def norm(t):
    return re.sub(r"[^\w\s]", "", str(t).lower()).strip()


def to_pairs(part):
    frames = []
    for s, t in DIRECTIONS:
        frames.append(pd.DataFrame({
            "source_text": part[RAW_COL[s]].values, "target_text": part[RAW_COL[t]].values,
            "source_lang": s, "target_lang": t, "row_id": part["Number"].values}))
    return pd.concat(frames, ignore_index=True)


def main(a):
    raw = pd.read_csv(a.input, usecols=["Number", "English", "Virac", "Pandan"]).dropna()
    log = {"raw_rows": len(raw)}
    raw["English"] = raw["English"].map(lambda x: clean(x, False))
    raw["Virac"] = raw["Virac"].map(lambda x: clean(x, True))
    raw["Pandan"] = raw["Pandan"].map(lambda x: clean(x, True))
    raw = raw[(raw[["English", "Virac", "Pandan"]] != "").all(axis=1)]
    raw = raw.assign(_k=raw["English"].map(norm)).drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    log["rows_after_cleaning"] = len(raw)

    # Task 1.3: split BY ROW (one row = the same sentence in 3 languages). Every direction of a
    # row therefore lands in the same split, so a sentence can never be seen in training in one
    # direction and tested in another.
    train_r, rest = train_test_split(raw, test_size=0.30, random_state=42)
    val_r, test_r = train_test_split(rest, test_size=0.50, random_state=42)
    train, val, test = to_pairs(train_r), to_pairs(val_r), to_pairs(test_r)

    # Extra leakage guard: drop val/test pairs whose source text (in that source language)
    # also occurs in train (a few Bikol sentences are repeated with different English).
    seen = set(zip(train.source_lang, train.source_text.map(norm)))
    def guard(df, name):
        keep = [(l, norm(s)) not in seen for l, s in zip(df.source_lang, df.source_text)]
        log[f"{name}_removed_for_leakage"] = int(len(df) - sum(keep))
        return df[keep]
    val, test = guard(val, "val"), guard(test, "test")
    train = train.drop_duplicates(subset=["source_lang", "target_lang", "source_text", "target_text"])
    val = val.drop_duplicates(subset=["source_lang", "target_lang", "source_text"])
    test = test.drop_duplicates(subset=["source_lang", "target_lang", "source_text"])
    for part in (val, test):
        assert not set(zip(part.source_lang, part.source_text.map(norm))) & seen

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        part.sample(frac=1, random_state=0).to_csv(out / f"{name}.csv", index=False)
        log[f"{name}_pairs"] = len(part)
        log[f"{name}_by_direction"] = {f"{s}->{t}": int(n) for (s, t), n in part.groupby(["source_lang", "target_lang"]).size().items()}

    # Task 1.4: baseline vocabulary analysis
    stats = {"log": log}
    cols = {"English": raw["English"], "Virac (South)": raw["Virac"], "Pandan (North)": raw["Pandan"]}
    vocab = {}
    for name, ser in cols.items():
        toks = [w for s in ser for w in norm(s).split()]
        lens = ser.map(lambda s: len(norm(s).split()))
        vocab[name] = set(toks)
        stats[name] = {"sentences": int(len(ser)), "tokens": len(toks), "types": len(set(toks)),
                       "TTR": round(len(set(toks)) / len(toks), 4), "len_mean": round(float(lens.mean()), 2),
                       "len_median": float(lens.median()), "len_max": int(lens.max())}
    n, s = vocab["Pandan (North)"], vocab["Virac (South)"]
    stats["Virac_Pandan_vocab_jaccard"] = round(len(n & s) / len(n | s), 4)
    stats["rows_where_Virac_equals_Pandan"] = int((raw["Pandan"].map(norm) == raw["Virac"].map(norm)).sum())
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, ser in cols.items():
        plt.hist(ser.map(lambda x: len(x.split())), bins=range(1, 25), alpha=.5, label=name)
    plt.xlabel("words per sentence"); plt.ylabel("sentences"); plt.legend()
    plt.savefig(out / "length_distribution.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="data/processed")
    main(p.parse_args())
