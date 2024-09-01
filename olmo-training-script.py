# Install necessary packages (run these commands separately if not using a Jupyter notebook)
# !pip install transformers datasets accelerate peft bitsandbytes

from huggingface_hub import HfFolder, login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Set your Hugging Face auth token
HUGGINGFACE_TOKEN = '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'
HfFolder.save_token(HUGGINGFACE_TOKEN)
login(token=HUGGINGFACE_TOKEN)

# Load model and tokenizer
model_id = "allenai/OLMo-1B-0724-hf"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HUGGINGFACE_TOKEN
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)

# Prepare the model for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_dataset("prof-freakenstein/medico-preset")

# Preprocess data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="olmo-medical-diagnosis",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    push_to_hub=True,
    hub_model_id="olmo-medical-diagnosis",
    report_to="none"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model()
trainer.save_state()

# Push to Hub
trainer.push_to_hub()
