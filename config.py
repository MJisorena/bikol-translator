import os
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = HERE
DATA_DIR = PROJECT + "/data/processed"
# Use a local ./adapter folder if present, otherwise download the adapter from the Hugging Face Hub
ADAPTER_DIR = PROJECT + "/adapter" if os.path.isdir(PROJECT + "/adapter") else "MJCUTIE13/bikol-nllb-lora"
TM_PATH = PROJECT + "/memory.csv"
TM_TRAIN_ONLY_PATH = PROJECT + "/memory_train_only.csv"
BASE_MODEL = "facebook/nllb-200-distilled-600M"
MAX_LEN = 128

# ---- Translation directions to train (source, target) ----
DIRECTIONS = [
    ("eng_Latn", "bcl_South"),    # English -> Virac (Southern Bikol)
    ("eng_Latn", "bcl_North"),    # English -> Pandan (Northern Bikol)
    ("bcl_South", "bcl_North"),   # Virac -> Pandan
]
# To also train Bikol -> English (the PDF's API example uses bcl_North -> eng_Latn), add:
#   ("bcl_North", "eng_Latn"), ("bcl_South", "eng_Latn")
# then re-run Step 2 (data) and Step 4 (training).

LANG_NAME = {"eng_Latn": "English", "bcl_South": "Virac (Southern Bikol)", "bcl_North": "Pandan (Northern Bikol)"}
TGT_WORD = {"eng_Latn": "english", "bcl_South": "virac", "bcl_North": "pandan"}

# NLLB-200 has no Bikol language tag, so both Bikol varieties borrow the Tagalog tag.
# The real target (virac / pandan / english) is told to the model with a text prefix.
NLLB_TAG = {"eng_Latn": "eng_Latn", "bcl_South": "tgl_Latn", "bcl_North": "tgl_Latn"}

def make_input(text, tgt):
    return ">>" + TGT_WORD[tgt] + "<< " + text

# Fallback phrase. Chosen by the TARGET language (the dialect the system would answer in).
FALLBACK = {"bcl_North": "mai ko isi", "bcl_South": "dai ko isi"}

# LoRA / training (PDF Tasks 2.2 and 2.3)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
EPOCHS = 20
LR = 3e-4

# "Do I know this?" thresholds (tuned in Step 8)
EXACT_SIM = 0.97       # >= this: answer straight from translation memory
FUZZY_SIM = 0.60       # below this the sentence is not close to anything known
MIN_COVERAGE = 0.90    # tuned in Step 8b: catches ~91% of off-topic test sentences (in-domain false alarms ~18%)
MIN_CONF = 0.30        # model confidence

# Beam search width used by the API (must be >= 2). Lower to 2 on a CPU-only laptop for speed.
NUM_BEAMS = 4
