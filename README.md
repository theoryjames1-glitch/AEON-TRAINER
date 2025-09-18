# 📂 AEON TRAINER (AOT) — Repo Layout

```
aeon-trainer/
│
├── aeon/
│   ├── __init__.py
│   ├── trainer.py           # Core AEON Trainer loop
│   ├── reward_factory.py    # Reward definitions + factory
│   ├── optim_factory.py     # Optimizer selector
│   ├── sched_factory.py     # Scheduler selector
│   └── utils.py             # Helpers (logging, etc.)
│
├── examples/
│   ├── run_sft.py           # Supervised fine-tuning demo
│   ├── run_ppo.py           # PPO-only demo
│   ├── run_hybrid.py        # Hybrid SFT + PPO training
│   └── run_classify.py      # Boolean / Yes-No-Maybe classifier demo
│
├── configs/
│   ├── optimizer.json       # Example optimizer configs
│   ├── scheduler.json       # Example scheduler configs
│   └── reward.json          # Example reward configs
│
├── tests/
│   ├── test_rewards.py
│   ├── test_optim.py
│   └── test_trainer.py
│
├── README.md
└── setup.py
```

---

# 🧩 Core Files

### `trainer.py` (AEON Hybrid Trainer)

```python
# aeon/trainer.py
import torch
from trl import (
    SFTTrainer, PPOTrainer, PPOConfig,
    AutoModelForCausalLMWithValueHead, create_reference_model
)
from trl.core import respond_to_batch
from .optim_factory import make_optimizer
from .sched_factory import make_scheduler

class AEONTrainer:
    def __init__(self, model_name, tokenizer, train_dataset, reward_fn,
                 optimizer_cfg=None, scheduler_cfg=None,
                 lr=5e-5, batch_size=1, max_seq_len=512, max_new_tokens=128,
                 num_training_steps=None, num_warmup_steps=None,
                 debug=False):
        self.debug = debug
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reward_fn = reward_fn
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        # Model with LM + Value heads
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.ref_model = create_reference_model(self.model)
        self.device = self.model.device

        # Optimizer + Scheduler
        self.optimizer = make_optimizer(self.model, optimizer_cfg or {"OPTIMIZER": "adamw"}, lr)
        self.scheduler = make_scheduler(
            self.optimizer, scheduler_cfg or {"SCHEDULER": "linear"},
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        # SFT + PPO trainers
        self.sft_trainer = SFTTrainer(
            model=self.model.pretrained_model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            max_seq_length=self.max_seq_len,
            packing=True,
        )
        self.sft_trainer.optimizer = self.optimizer
        self.sft_trainer.lr_scheduler = self.scheduler

        ppo_config = PPOConfig(model_name=model_name, batch_size=batch_size)
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
        )
        self.ppo_trainer.lr_scheduler = self.scheduler

    def sft_step(self, batch):
        loss = self.sft_trainer.compute_loss(self.model.pretrained_model, batch)
        loss.backward()
        self.optimizer.step()
        if self.scheduler: self.scheduler.step()
        self.optimizer.zero_grad()
        return float(loss.item())

    def respond(self, input_ids):
        return respond_to_batch(self.model, input_ids)

    def ppo_step(self, queries, responses, rewards):
        return self.ppo_trainer.step(queries, responses, rewards)
```

---

### `reward_factory.py` (Reward Functions)

```python
# aeon/reward_factory.py
from transformers import pipeline

class Reward:
    def __call__(self, response: str, target: str, **kw): raise NotImplementedError

class SentimentReward(Reward):
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english", target="positive"):
        self.model = pipeline("sentiment-analysis", model=model)
        self.target = target

    def __call__(self, response, target, **_):
        pred = self.model(response)[0]
        label, score = pred["label"].lower(), pred["score"]
        return score if self.target in label else 1 - score

def make_reward(spec: dict) -> Reward:
    t = spec.get("type", "").lower()
    if t == "sentiment":
        return SentimentReward(target=spec.get("target", "positive"))
    raise ValueError(f"Unknown reward type: {t}")
```

---

### `optim_factory.py` (Optimizer Factory)

```python
# aeon/optim_factory.py
from torch.optim import AdamW, SGD, RMSprop

def make_optimizer(model, cfg: dict, lr: float):
    opt = cfg.get("OPTIMIZER", "adamw").lower()
    if opt == "adamw":
        return AdamW(model.parameters(), lr=lr)
    elif opt == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt == "rmsprop":
        return RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")
```

---

### `sched_factory.py` (Scheduler Factory)

```python
# aeon/sched_factory.py
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def make_scheduler(opt, cfg: dict, num_training_steps=None, num_warmup_steps=None):
    name = cfg.get("SCHEDULER", "none").lower()
    if name == "none":
        return None
    elif name == "step":
        return StepLR(opt, step_size=100, gamma=0.5)
    elif name == "cosine":
        return CosineAnnealingLR(opt, T_max=num_training_steps or 1000)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
```

---

# 📖 Example Script

### `examples/run_classify.py`

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from aeon.trainer import AEONTrainer
from aeon.reward_factory import make_reward

# === Dataset ===
dataset = load_dataset("boolq", split="train[:200]")  # Boolean QA dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess(batch):
    text = f"Question: {batch['question']} Context: {batch['passage']}\nAnswer:"
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    enc["labels"] = tokenizer(" yes" if batch["answer"] else " no",
                              truncation=True, padding="max_length", max_length=10)["input_ids"]
    return enc

train_data = dataset.map(preprocess, remove_columns=dataset.column_names)

# === Reward ===
reward_spec = {"type": "sentiment", "target": "positive"}
reward_fn = make_reward(reward_spec)

# === Trainer ===
trainer = AEONTrainer("gpt2", tokenizer, train_data, reward_fn, lr=2e-5, batch_size=1)

# === Training ===
for step, batch in enumerate(train_data.shuffle().select(range(5))):
    batch = {k: torch.tensor(v).unsqueeze(0).to(trainer.device) for k, v in batch.items()}
    loss = trainer.sft_step(batch)
    print(f"[{step}] SFT Loss: {loss:.4f}")
```

---

# 📘 README.md (short version)

````markdown
# AEON TRAINER (AOT)

AEON TRAINER (AOT) = **Adaptive Optimization Trainer**  
A hybrid trainer for LLMs that unifies:

- **Supervised Fine-Tuning (SFT)**
- **Reinforcement Learning with PPO**
- **Modular Reward Functions**
- **Adaptive Optimizer & Scheduler Factories**

## 🚀 Features
- Train with both **labeled data** and **reward signals**
- Plug in custom **rewards** (sentiment, similarity, BLEU, etc.)
- Use **any HuggingFace causal LM** (GPT-2, LLaMA, etc.)
- Extend with custom optimizers and schedulers

## 📂 Structure
- `aeon/` → core trainer, rewards, optimizers, schedulers
- `examples/` → runnable demos
- `configs/` → sample configs for quick setup

## ⚡ Demo
```bash
python examples/run_classify.py
````

## 🔮 Roadmap

* Multi-GPU + DeepSpeed
* More reward types (toxicity, BLEU, ROUGE, etc.)
* Hybrid curriculum (SFT → PPO)
* Direct Preference Optimization (DPO) support
