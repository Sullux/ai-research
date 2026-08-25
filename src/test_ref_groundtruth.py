import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "../gemma-4-12B-it"
print(f"Loading reference model from {model_id}...")
tok = AutoTokenizer.from_pretrained(model_id)

with open("tui/PROMPT_KERNEL.md", "r") as f:
    kernel = f.read().strip()

prompt = f"<|turn>system\n<|think|>\n{kernel}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\nThe user is asking \"How are you doing today?\".\nThis is a standard social greeting.\nI should respond politely and informatively, acknowledging my nature as an AI.<channel|>I'm doing well, thank you for asking! I'"
input_ids = tok.encode(prompt, return_tensors="pt")
print(f"Prompt encoded: {input_ids.shape}")

# Inspect tokens for 's' vs 'm'
print(f"Token for 'm': {tok.encode('m', add_special_tokens=False)}")
print(f"Token for 's': {tok.encode('s', add_special_tokens=False)}")
