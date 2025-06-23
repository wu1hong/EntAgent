import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from prompt2 import system_prompt
from trl import SFTTrainer

model_name = "/u/luoyajie/llama3.1-instruct"
data_path = "/u/luoyajie/entity_linking/results/train_8B.json"
output_dir = "/u/luoyajie/entity_linking/results/llama3.1-lora-output"


tokenizer = AutoTokenizer.from_pretrained(model_name)
# Use <eos> as pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

# Set up 8-bit quantization config
#bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True

)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.attn_implementation = "flash_attention_2"

# LoRA configuration
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_disable()


with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# using chat template
def formatting_func(example):
    messages = [{"role": "system", "content": system_prompt}] + example["conversations"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )



training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    bf16=True,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    packing=False,
    formatting_func=formatting_func
)
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
