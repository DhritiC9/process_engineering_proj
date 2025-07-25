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
from difflib import SequenceMatcher

device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "Training Data Public Upload/train_data_10k.json"
EVAL_PATH = "Training Data Public Upload/eval_data_1k.json"
TEST_PATH = "Training Data Public Upload/test_data_1k.json"

train_df = pd.read_json(TRAIN_PATH, lines=True).rename(columns={"PFD": "text", "PID": "label"})
eval_df = pd.read_json(EVAL_PATH, lines=True).rename(columns={"PFD": "text", "PID": "label"})
test_df = pd.read_json(TEST_PATH, lines=True).rename(columns={"PFD": "text", "PID": "label"})

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset=train_dataset.select(range(8000))
model_load = "google/byt5-base"  # Stronger than byt5-small
tokenizer = AutoTokenizer.from_pretrained(model_load)
model = AutoModelForSeq2SeqLM.from_pretrained(model_load)



def tokenize_fn(batch):
    inputs = tokenizer(batch["text"],  padding="longest")
    labels = tokenizer(batch["label"], padding="longest")
    labels["input_ids"] = [[token if token != tokenizer.pad_token_id else -100 for token in seq] for seq in labels["input_ids"]]
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label"])
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./byt5-base-finetuned-model-run1",
    eval_strategy="steps",
    save_strategy="steps",
    report_to="none",
    run_name="byt5-base-finetuned-model-run1",
    eval_steps=500,
    save_steps=1000,
    logging_steps=1000,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    load_best_model_at_end=True,
    predict_with_generate=True,
    label_smoothing_factor=0.1,
    fp16=torch.cuda.is_available(),

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


trainer.train()

model.save_pretrained("./byt5-base-finetuned-model-run2")
tokenizer.save_pretrained("./byt5-base-finetuned-model-run2")

pipe = Text2TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device="cuda" if torch.cuda.is_available() else -1
)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

sample_inputs = [test_df["text"].iloc[i] for i in range(5)]
ground_truths = [test_df["label"].iloc[i] for i in range(5)]

max_new_tokens = 500

def decode_predictions(input_text):
    return pipe(
        input_text,
        max_new_tokens=max_new_tokens,
        num_beams=3,
        num_return_sequences=3,
        early_stopping=True,
        repetition_penalty=1.1,
        length_penalty=1.0,
        diversity_penalty=0.1
    )

for i, input_text in enumerate(sample_inputs):
    print(f"\nPFD Input {i+1}:\n{input_text}\n")
    outputs = decode_predictions(input_text)
    for j, output in enumerate(outputs):
        score = similarity(output['generated_text'], ground_truths[i])
        print(f" Prediction {chr(65 + j)} (Sim: {score:.2f}):\n{output['generated_text']}\n")
    print(f" Ground Truth PID:\n{ground_truths[i]}\n")