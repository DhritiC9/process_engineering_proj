import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from collections import Counter
from tqdm import tqdm

model = AutoModelForSeq2SeqLM.from_pretrained("./byt5-finetuned-model-v2v-run1")
tokenizer = AutoTokenizer.from_pretrained("./byt5-finetuned-model-v2v-run1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

TEST_PATH = "Training Data Public Upload/test_data_1k_wv2v.json"
test_df = pd.read_json(TEST_PATH).rename(columns={"PFD": "text", "PID": "label"})
assert len(test_df) == 1000

def normalize(text):
    return text.replace(" ", "").replace("\n", "").strip()

def generate_beam_predictions(batch_inputs, model, tokenizer, device, num_beams=5, num_return_sequences=5, max_new_tokens=1000):
    inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    assert len(decoded) == len(batch_inputs) * num_return_sequences
    grouped = [decoded[i:i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]
    return grouped

match_distribution = Counter()
batch_inputs = test_df["text"].tolist()
batch_ground_truths = [normalize(x) for x in test_df["label"].tolist()]
batch_size = 8
sample_count = 0
max_examples_to_print = 3

total_samples = 0
total_matches = 0

for i in tqdm(range(0, len(batch_inputs), batch_size), desc="Evaluating"):
    batch = batch_inputs[i:i + batch_size]
    gt_batch = batch_ground_truths[i:i + batch_size]

    outputs = generate_beam_predictions(
        batch_inputs=batch,
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_beams=5,
        num_return_sequences=5,
        max_new_tokens=1000
    )

    for j in range(len(batch)):
        preds = [normalize(pred) for pred in outputs[j]]
        matches = sum([p == gt_batch[j] for p in preds])
        match_distribution[matches] += 1
        total_samples += 1
        total_matches += matches

        if sample_count < max_examples_to_print:
           
            print(f"Input {sample_count+1}:")
            print(batch[j])
            print("Expected Output:")
            print(gt_batch[j])
            print("Model Predictions:")
            for k, p in enumerate(preds):
                status = "Match" if p == gt_batch[j] else "No match"
                print(f"  Beam {k+1}: {p}  --> {status}")
            print(f"Matches Found: {matches} out of 5")
            sample_count += 1

print(f"\nTotal Samples Evaluated: {total_samples}")
print(f"Total Matches Across All Beams: {total_matches} out of {total_samples * 5}")

print("\nMatch Count Distribution (out of 5 beams):")
for i in range(6):
    print(f"{i}/5  matching predictions: {match_distribution[i]} samples")