import sys, os, time, torch
sys.path.insert(0, "."); sys.path.insert(0, "src")
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from data import MINDPointwiseSFTDataset
import transformers

tok = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
tok.pad_token = tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("models/Qwen3-1.7B", torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.config.use_cache = False
ds = MINDPointwiseSFTDataset(
    behaviors_path="data/MIND_large/dev/behaviors.tsv",
    news_path="data/MIND_large/dev/news.tsv",
    tokenizer=tok, max_len=2048, sample=2000, seed=42, max_history=30,
    neg_ratio=2.0, hard_neg_ratio=0.5, use_abstract=True, use_chat_template=True)
val = MINDPointwiseSFTDataset(
    behaviors_path="data/MIND_large/dev/behaviors.tsv",
    news_path="data/MIND_large/dev/news.tsv",
    tokenizer=tok, max_len=2048, sample=500, seed=1, max_history=30,
    neg_ratio=2.0, hard_neg_ratio=0.5, use_abstract=True, use_chat_template=True)

args = TrainingArguments(
    output_dir="/tmp/prof_trainer_ddp2",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    max_steps=10,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=1,
    report_to="none",
    save_strategy="steps", save_steps=100000,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    eval_strategy="steps",
    eval_steps=100000,
    warmup_steps=100,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
)
trainer = Trainer(model=model, train_dataset=ds, eval_dataset=val, args=args,
    data_collator=transformers.DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
t0=time.time()
trainer.train()
print(f"10 steps took {time.time()-t0:.1f}s -> {(time.time()-t0)/10:.2f}s/step")
