from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from datasets import Dataset

# ✅ Step 1: Load Dataset from Local CSV
df = pd.read_csv("indian_medicines.csv")  # Ensure your CSV file is in the same directory.

# ✅ Step 2: Preprocess Data
df = df.dropna(subset=["name", "uses"]).sample(n=2000)
queries = ["What is the use of " + n + "?" for n in df["name"]]
gold_uses = df["uses"].tolist()

# ✅ Step 3: Load GPT-2 Model (Hugging Face) and Value Head for PPO
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Use standard CausalLM for model and ref_model
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
# Value head for PPOTrainer
value_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# ✅ Step 4: PPO Configuration
ppo_config = PPOConfig(
    learning_rate=5e-6,
    batch_size=4,
    mini_batch_size=4,
    bf16=False,
    fp16=False,
)

# Minimal HuggingFace Dataset
train_dataset = Dataset.from_dict({"query": queries, "gold": gold_uses})

# Dummy reward model (minimal nn.Module)
class DummyRewardModel(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor([1.0])

dummy_reward_model = DummyRewardModel()

ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer,
    reward_model=dummy_reward_model,
    train_dataset=train_dataset,
    value_model=value_model,
)

# ✅ Step 5: Reward Function
def compute_reward(response_texts, gold_texts):
    rewards = []
    for resp, gold in zip(response_texts, gold_texts):
        r_emb = value_model.transformer.wte(tokenizer.encode(resp, return_tensors="pt"))
        g_emb = value_model.transformer.wte(tokenizer.encode(gold, return_tensors="pt"))
        score = F.cosine_similarity(r_emb.mean(dim=1), g_emb.mean(dim=1), dim=-1)
        rewards.append(score.item())
    return torch.tensor(rewards)

# ✅ Step 6: PPO Training Loop
for i in range(200):
    batch_q = queries[i*4:(i+1)*4]
    batch_gold = gold_uses[i*4:(i+1)*4]

    input_ids = tokenizer(batch_q, return_tensors="pt", padding=True, truncation=True).input_ids
    response_ids = model.generate(input_ids, max_new_tokens=50)
    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_ids]

    rewards = compute_reward(responses, batch_gold)
    # Use PPOTrainer's step method for RL training
    stats = ppo_trainer.step(input_ids, response_ids, rewards)
    # Optionally log stats if needed

print("✅ PPO Training Complete")

# ✅ Step 7: Push to Hugging Face Hub
model.push_to_hub("Mayank-22/gpt2-ppo-indian-medicines")
tokenizer.push_to_hub("Mayank-22/gpt2-ppo-indian-medicines")

