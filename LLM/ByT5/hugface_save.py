from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load from your local directory (after training)
model = AutoModelForSeq2SeqLM.from_pretrained("./byt5-finetuned-model-run1")
tokenizer = AutoTokenizer.from_pretrained("./byt5-finetuned-model-run1")

# Push both to your repo
model.push_to_hub("dhritic99/byt5_with_valves")
tokenizer.push_to_hub("dhritic99/byt5_with_valves")