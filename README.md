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
| `notebooks/Bikol_NMT_Colab.ipynb` | Full Colab pipeline: data, training, evaluation |
| `memory.csv` | Translation memory (copy it in from the Colab output; see setup) |

## Setup and run (VS Code, Windows / macOS / Linux)

Requires Python 3.10+.

1. Open this folder in VS Code (File > Open Folder).
2. Open a terminal (Terminal > New Terminal) and create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Copy `memory.csv` (from the `bikol_nmt` folder in Google Drive) into this folder.
5. Optional: copy the `adapter/` folder from Drive here. If it is missing, the adapter is downloaded from the Hugging Face Hub.
6. Start the app: `uvicorn main:app --port 8000`
7. Open http://127.0.0.1:8000 in your browser.

The first start downloads the base model (about 2.4 GB) and takes a few minutes. A GPU is optional; on CPU only,
translation is slower. Set `NUM_BEAMS = 2` in `config.py` to speed it up.

## API

`POST /api/v1/translate`

```json
{"source_text": "Slice the onions and garlic into small pieces", "source_lang": "eng_Latn", "target_lang": "bcl_North"}
```

Response:

```json
{"status": "success", "translated_text": "...", "confidence_score": 0.89, "inference_latency_ms": 142}
```

Language codes: `eng_Latn` (English), `bcl_South` (Virac), `bcl_North` (Pandan). Supported directions are listed at `GET /api/v1/directions`.

- `POST /api/v1/feedback`: stores a "Report Inaccurate Translation" entry in `feedback.db` (SQLite).
- `POST /api/v1/memory`: adds new sentences as `[{"english": "...", "virac": "...", "pandan": "..."}]` with no retraining.

If the input is not close to anything in the training data, the app returns `mai ko isi` (Pandan target) or `dai ko isi` (Virac target).

## Training and evaluation (Google Colab, T4 GPU)

Run `notebooks/Bikol_NMT_Colab.ipynb` from top to bottom. It cleans the data, fine-tunes with LoRA
(r=16, alpha=32, dropout 0.05, lr 3e-4, 20 epochs, AdamW, linear decay with warmup, seed 42), and evaluates with
SacreBLEU, chrF++ and TER against the zero-shot baseline.

Data: 1,504 sentences in English, Virac and Pandan (cooking and food domain), expanded to one pair per direction and
split by sentence into 3,146 / 675 / 673 train / validation / test pairs with no overlap between splits.

### Results

(Fill in from `results_metrics.csv`, `results_latency.csv` and `error_analysis.csv`.)

## Limitations

- Small, single-domain dataset: sentences about other topics are translated poorly or return the fallback phrase.
- NLLB-200 has no Bikol language tag; both varieties use the Tagalog tag and the target is selected with a text prefix.
