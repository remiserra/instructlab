#TODORS HERE - Load -fused and resave with model.safetensors.index.json
# Using HF libs
from transformers import AutoModelForCausalLM, AutoTokenizer

# SRC 
# https://huggingface.co/docs/transformers/v4.42.0/en/peft

# setup
model_dir = "instructlab-merlinite-7b-lab-mlx-q-fused"

# load fused model
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# test
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)

output = model.generate(**inputs)

# res-save
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)

