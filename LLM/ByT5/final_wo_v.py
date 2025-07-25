import pandas as pd 
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Text2TextGenerationPipeline 
)
import torch
device=0 if torch.cuda.is_available else "CPU"

TRAIN_PATH = "Training Data Public Upload/train_data_10k_wo_valves.json"
EVAL_PATH = "Training Data Public Upload/eval_data_1k_wo_valves.json"
TEST_PATH = "Training Data Public Upload/test_data_1k_wo_valves.json"


train_df=pd.read_json(TRAIN_PATH, lines=True).rename(columns={"PFD":"text","PID": "label"})
eval_df=pd.read_json(EVAL_PATH, lines=True).rename(columns={"PFD": "text", "PID":"label"})
test_df=pd.read_json(TEST_PATH, lines=True).rename(columns={"PFD": "text", "PID":"label"})

train_dataset=Dataset.from_pandas(train_df)
eval_dataset=Dataset.from_pandas(eval_df)
test_dataset=Dataset.from_pandas(test_df)

model_load="google/byt5-small"
tokenizer= AutoTokenizer.from_pretrained(model_load)
model=AutoModelForSeq2SeqLM.from_pretrained(model_load)

max_length=512

def tokenize_fn(batch):
    inputs= tokenizer(batch["text"], max_length=max_length,padding="longest",truncation="longest_first")
    labels=tokenizer(batch["label"],max_length=max_length, padding="longest", truncation="longest_first")

    labels["input_ids"]=[
        [token if token != tokenizer.pad_token_id else -100 for token in seq]
        for seq in labels["input_ids"]

    ]
    inputs["labels"]= labels["input_ids"]
    return inputs


tokenized_train=train_dataset.map(tokenize_fn, batched=True, remove_columns=["text","label"])
tokenized_eval=eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text","label"])
tokenized_test=test_dataset.map(tokenize_fn, batched=True, remove_columns=["text","label"])

data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)

training_args=Seq2SeqTrainingArguments(
    output_dir="./byt5-finetuned-wo-v-model-run1",
    eval_strategy="steps",
    save_strategy="steps",
    report_to="wandb",
    run_name="byt5-finetuned-model-run1",
    eval_steps=500,
    save_steps=500,
    logging_steps=500,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

trainer=Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


# Train
trainer.train()

# Save model
model.save_pretrained("./byt5-finetuned-wo-v-model-run1")
tokenizer.save_pretrained("./byt5-finetuned-wo-v-model-run1")

# test
pipe = Text2TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

sample_inputs = [test_df["text"].iloc[i] for i in range(5)]
ground_truths = [test_df["label"].iloc[i] for i in range(5)]

max_new_tokens = 1000

for i, input_text in enumerate(sample_inputs):
    print(f"\nPFD Input {i+1}:\n{input_text}\n")

    outputs = pipe(
        input_text,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True
    )

    for j, output in enumerate(outputs):
        print(f" Prediction {chr(65 + j)}:\n{output['generated_text']}\n")

    print(f" Ground Truth PID:\n{ground_truths[i]}\n")