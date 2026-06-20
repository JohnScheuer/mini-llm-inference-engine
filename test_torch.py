import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16
).cuda()

inp = tok("Hello", return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**inp)

logits = out.logits[0, -1].float().cpu()
print(logits[:10])
