
#base T5 code that works 
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google-t5/t5-base"
    )
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google-t5/t5-base",
    torch_dtype=torch.float16,
    device_map="auto"
    )

input_ids = tokenizer("translate English to French: The weather is nice today.", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))

#base T5 code using Quantization which reduces the memory burden of large models by representing the weights in a lower precision. 

#The example below uses torchao to only quantize the weights to int4.

#NOTE - Doesnt run on collab as requires a larger GPU 
# pip install torchao
import torch, transformers
from transformers import TorchAoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5-v1_1-xl",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
input_ids = tokenizer("translate English to French: The weather is nice today.", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
