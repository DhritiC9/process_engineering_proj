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

tokenizer = AutoTokenizer.from_pretrained("byt5-finetuned-model-run1")
model = AutoModelForSeq2SeqLM.from_pretrained("byt5-finetuned-model-run1")

TEST_PATH = "Training Data Public Upload/test_data_1k.json"
test_df = pd.read_json(TEST_PATH, lines=True).rename(columns={"PFD": "text", "PID": "label"})




pipe = Text2TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

sample_input = "(raw)(v)(hex)<&|(raw)&|[(prod)](mix)<1(r)(v)(splt)[(prod)](v)1"
ground_truth = "(raw)(v)<_1(hex)<&|(raw)&|[(prod)](C){TC}_1(mix)<1(r)[(C){LC}_2](v)<_2(splt)[(prod)](C){FC}_3(v)1<_3"

max_new_tokens = 1000


print(f"\nPFD Input :\n{sample_input}\n")

outputs = pipe(
        sample_input,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True,
        do_sample=False
    )

for j, output in enumerate(outputs):
        print(f" Prediction {chr(65 + j)}:\n{output['generated_text']}\n")

print(f" Ground Truth PID:\n{ground_truth}\n")