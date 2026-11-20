import glob, json, math, torch, pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType
from config import *

tok = AutoTokenizer.from_pretrained(BASE_MODEL)

def load(split):
    return Dataset.from_pandas(pd.read_csv(f"{DATA_DIR}/{split}.csv"), preserve_index=False)

def encode(ex):
    tok.src_lang = NLLB_TAG[ex["source_lang"]]
    tok.tgt_lang = NLLB_TAG[ex["target_lang"]]
    return tok(make_input(ex["source_text"], ex["target_lang"]), text_target=ex["target_text"],
               max_length=MAX_LEN, truncation=True)

train_ds, val_ds = load("train"), load("validation")
train_ds = train_ds.map(encode, remove_columns=train_ds.column_names)
val_ds = val_ds.map(encode, remove_columns=val_ds.column_names)

# Task 2.1 base model, Task 2.2 LoRA on q_proj / v_proj
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, **LORA))
model.print_trainable_parameters()

# Task 2.3 AdamW + linear decay with warmup, loss monitored every epoch
steps_per_epoch = math.ceil(len(train_ds) / (16 * 2))     # batch 16 x grad-accum 2
WARMUP = int(0.05 * steps_per_epoch * EPOCHS)              # 5% warmup (works on old and new transformers)
args = Seq2SeqTrainingArguments(
    output_dir=f"{PROJECT}/checkpoints",
    num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=16, per_device_eval_batch_size=16, gradient_accumulation_steps=2,
    warmup_steps=WARMUP, lr_scheduler_type="linear", optim="adamw_torch", weight_decay=0.01,
    fp16=True,                                   # T4 has no bf16
    eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
    save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False,
    report_to="none", seed=42,
)
trainer = Seq2SeqTrainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
    processing_class=tok,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)
resume = len(glob.glob(f"{PROJECT}/checkpoints/checkpoint-*")) > 0   # survive Colab disconnects
trainer.train(resume_from_checkpoint=resume)

# Task 2.4 save only the LoRA adapter (small)
model.save_pretrained(ADAPTER_DIR)
json.dump(trainer.state.log_history, open(f"{PROJECT}/train_log.json", "w"))
print("Adapter saved to", ADAPTER_DIR)
