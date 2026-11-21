# Bikol Translator: English, Virac (Southern) and Pandan (Northern) Catanduanes Bikol

Low-resource neural machine translation capstone. A LoRA adapter on `facebook/nllb-200-distilled-600M`
(Hugging Face: https://huggingface.co/MJCUTIE13/bikol-nllb-lora) is served by a FastAPI backend with a
Bootstrap 5 web page. A translation memory (RAG-style) lets you add new sentences without retraining.

Directions: English -> Virac, English -> Pandan, Virac -> Pandan.

```
 [ Input text ] -> FastAPI (main.py) -> translation memory + NLLB-200 / LoRA adapter (engine.py) -> [ Translated text ]
```

## Repository layout

| File | Purpose |
|---|---|
| `prep_data.py` | Cleaning, de-duplication, leakage-free 70/15/15 split, vocabulary statistics |
| `train.py` | LoRA fine-tuning (run on a Colab T4 GPU) |
| `engine.py` | Model loading, translation, translation memory, "mai ko isi / dai ko isi" fallback |
| `main.py` | FastAPI app (`/api/v1/translate`, `/api/v1/feedback`, `/api/v1/memory`) |
| `index.html` | Bootstrap 5 side-by-side translator page |
| `config.py` | Directions, LoRA settings, thresholds |
| `memory.csv` | Translation memory (included in this repository) |
| `data/processed/` | Cleaned and split corpus (train, validation, test CSVs) |
| `notebooks/Bikol_NMT_Colab.ipynb` | Full Colab pipeline: data, training, evaluation |
| `results/` | Evaluation outputs: metrics, latency, error analysis, loss curve, data statistics |


## Setup and run (VS Code, Windows / macOS / Linux)

Requires Python 3.10+.

1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Terminal > New Terminal) and create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. `memory.csv` is already in this folder, so nothing to copy.
5. Optional: copy the `adapter/` folder from Drive here. If it is missing, the adapter is downloaded from the Hugging Face Hub.
6. Start the app: `python -m uvicorn main:app --port 8000`
7. Open http://127.0.0.1:8000 in your browser.

The first start downloads the base model (about 2.4 GB) and takes a few minutes. A GPU is optional; on CPU only,
translation is slower. Set `NUM_BEAMS = 2` in `config.py` to speed it up slightly.

## API

`POST /api/v1/translate`

```json
{"source_text": "Slice the onions and garlic into small pieces", "source_lang": "eng_Latn", "target_lang": "bcl_North"}
```

Response (example format):

```json
{"status": "success", "translated_text": "...", "confidence_score": 0.89, "inference_latency_ms": 142}
```

Language codes: `eng_Latn` (English), `bcl_South` (Virac), `bcl_North` (Pandan). Supported directions are listed at `GET /api/v1/directions`.

- `POST /api/v1/feedback`: stores a "Report Inaccurate Translation" entry in `feedback.db` (SQLite).
- `POST /api/v1/memory`: adds new sentences as `[{"english": "...", "virac": "...", "pandan": "..."}]` with no retraining.

If the input is not close to anything in the training data, the app returns `mai ko isi` (Pandan target) or `dai ko isi` (Virac target).

## Data sourcing and cleaning

Source: the team's parallel dataset (`Team2_Bikol_Translation_Dataset(1.5k).csv`), 1,505 rows with the same cooking and food sentence in English, Virac (Southern Bikol) and Pandan (Northern Bikol).

Cleaning (`prep_data.py`): HTML tags and entities removed; Unicode normalised (NFC) and curly quotes and dashes standardised; stray special characters removed; extra spaces and line breaks collapsed; where a Bikol cell listed spelling variants separated by "/", the first variant was kept; empty rows and duplicate English sentences removed (1,505 to 1,504 rows).

Splitting: 70/15/15 by row, so every direction of one sentence stays in the same split. As a leakage check, 3 validation and 4 test pairs whose source text also appeared in training were dropped, giving 3,146 / 675 / 673 train / validation / test pairs (one pair per trained direction).

| | Sentences | Types | Type-token ratio | Mean length (words) |
|---|---|---|---|---|
| English | 1,504 | 1,029 | 0.1052 | 6.51 |
| Virac | 1,504 | 1,181 | 0.1192 | 6.59 |
| Pandan | 1,504 | 1,271 | 0.1315 | 6.43 |

Virac and Pandan vocabulary overlap (Jaccard): 0.4449; 77 sentences are identical in both dialects. Overlap against a Tagalog or Central Bikol dataset was not measured.

## Training and evaluation (Google Colab, T4 GPU)

Run `notebooks/Bikol_NMT_Colab.ipynb` from top to bottom. It cleans the data, fine-tunes with LoRA
(r=16, alpha=32, dropout 0.05, lr 3e-4, 20 epochs, AdamW, linear decay with warmup, seed 42), and evaluates with
SacreBLEU, chrF++ and TER against the zero-shot baseline.

## Evaluation results

Evaluated on the held-out test set (673 sentence pairs; the model only, without the translation memory). BLEU and chrF++: higher is better. TER: lower is better.

### Zero-shot baseline vs fine-tuned LoRA

| Model | BLEU | chrF++ | TER |
|---|---|---|---|
| Zero-shot NLLB-200-distilled-600M | 3.09 | 27.53 | 96.60 |
| Fine-tuned LoRA | 45.74 | 66.37 | 37.49 |

The baseline is the un-tuned model with the Tagalog language tag standing in for Bikol and no target-dialect prefix.

### Fine-tuned LoRA by direction

| Direction | n | BLEU | chrF++ | TER |
|---|---|---|---|---|
| Virac to Pandan | 221 | 60.10 | 78.38 | 24.02 |
| Virac to Pandan (non-identical sentences only) | 210 | 58.91 | 77.62 | 24.86 |
| English to Pandan | 226 | 38.56 | 60.26 | 43.52 |
| English to Virac | 226 | 38.17 | 60.72 | 44.62 |

### Latency

Measured on a free Google Colab CPU runtime, on 50 sampled test sentences per setting. Colab's free GPU quota ran out, so the GPU measurement was not possible. The 300 ms target was therefore not met on CPU. Token counts include special tokens, and no sampled sentence exceeded 20 tokens.

| Beams | Tokens | n | Mean (ms) | Median (ms) | Max (ms) |
|---|---|---|---|---|---|
| 4 | 1-10 | 21 | 4627.9 | 4573.9 | 6986.2 |
| 4 | 11-20 | 29 | 5893.6 | 5597.4 | 7915.2 |
| 2 | 1-10 | 21 | 4103.9 | 3932.0 | 6016.1 |
| 2 | 11-20 | 29 | 5604.8 | 5243.4 | 9251.5 |

Sentences found in the translation memory are returned directly without running the model.

## Limitations

- Small, single-domain dataset (about 1,500 cooking sentences): sentences about other topics are translated poorly or return the fallback phrase, and the scores reflect this domain, not general Bikol.
- NLLB-200 has no Bikol language tag; both varieties use the Tagalog tag and the target is selected with a text prefix.
- Only English to Virac, English to Pandan and Virac to Pandan were trained. Bikol to English is not supported.
