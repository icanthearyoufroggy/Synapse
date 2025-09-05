# Synapse

## Overview

**Roblox Synapse**, part of the Roblox **Anti-Cheat Toolkit**, is a Python library designed specifically for **realtime detection of extremely rare classes of cheating behavior** by using **contrastive learning principles**.  

While traditional cheat detectors struggle with **highly imbalanced datasets** (most players are legitimate, only a tiny fraction cheat), Synapse excels by:

1. **Collecting recent gameplay events from a single source** (e.g., a player’s recent inputs, stats changes, or actions)  
2. **Calculating individual observation scores using embedding similarity**  
3. **Aggregating these scores using statistical measures like skewness to detect exploit patterns**  

By prioritizing **recall over precision**, Synapse serves as a **high-recall candidate generator** for further cheat investigation. Instead of treating each action in isolation, Synapse analyzes **patterns of suspicious behavior over time**, making it ideal for detecting **rare but critical exploit attempts**.

---

## Terminology

In Synapse’s anti-cheat context:
- **Positive examples** → Examples of **cheating behavior** (e.g., noclipping, teleport spam, impossible stat changes)
- **Negative examples** → Examples of **legitimate gameplay** (e.g., normal walking, regular rebirth, fair combat)

## Installation

```bash
pip install .
```

By default `Synapse` doesn’t pull in all transitive dependencies, specifically avoiding pulling in sentence transformers and its dependencies (torch).  
To pull them in as well, use:

```bash
pip install '.[sbert]'
```

## Quick Start

```python
from Synapse.Synapse_local_index import SynapseLocalIndex

# Load a previously saved index from a local path
index = SynapseLocalIndex.load(path="path/to/local/index")

# Collect recent observations from a single player (e.g., last few actions)
player_recent_actions = [
    "walked normally",
    "jumped",
    "teleported 500 studs instantly",
    "added 1e9 cash in one tick",
    "noclipped through a wall"
]

# Calculate cheat affinity across all observations
result = index.calculate_rare_class_affinity(player_recent_actions)

# Get the overall score (uses skewness by default)
overall_score = result.rare_class_affinity_score
print(f"Overall cheat affinity score: {overall_score:.4f}")

# Examine individual observation scores
for action, score in result.observation_scores.items():
    risk_level = "High" if score > 0.5 else "Medium" if score > 0.1 else "Low"
    print(f"'{action}' - Score: {score:.4f} - Risk: {risk_level}")
```

## Creating a New Index

```python
import torch
from Synapse.Synapse_local_index import SynapseLocalIndex
from Synapse.embeddings.sbert import get_sentence_transformer_and_scaling_fn

# Initialize sentence model and get scaling function
model_name = "all-MiniLM-L6-v2"
model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

# Prepare examples
positive_examples = ["noclip exploit", "teleport hack", "infinite cash injection"]
negative_examples = ["normal movement", "legit rebirth", "standard combat"]

# Encode examples
positive_embeddings = model.encode(positive_examples, normalize_embeddings=True)
negative_embeddings = model.encode(negative_examples, normalize_embeddings=True)

# Create the index
index = SynapseLocalIndex(
    sentence_model=model,
    positive_embeddings=positive_embeddings,
    negative_embeddings=negative_embeddings,
    scale_fn=scale_fn,
    positive_corpus=positive_examples,
    negative_corpus=negative_examples,
)

# Save locally
saved_config = index.save(path="path/to/local/index", encoder_model_name_or_path=model_name)
print(f"Saved index with encoder model: {saved_config.encoder_model_name_or_path}")
```

## How It Works

Synapse uses a two-step process to detect cheats, focusing on **high recall** for realtime anti-cheat applications:

1. **Individual Action Scoring**:  
   - Each player action is compared against both cheat examples and legitimate examples  
   - Using embedding similarity, Synapse calculates how close the action is to known exploit patterns  
   - The action score is the ratio between cheat similarity and legit similarity  
   - Scores > 0.1 indicate closer similarity to cheat patterns  

2. **Pattern Recognition via Skewness**:  
   - Recent action scores from the same player are collected  
   - Skewness measures asymmetry in the distribution of these scores  
   - Positive skewness indicates mostly legit actions, but with spikes of cheat-like behavior  
   - Resistant to variations in activity levels  
   - Reveals exploit patterns missed by analyzing single actions in isolation  

---

## Motivating Use Case

Synapse was developed to detect **extremely rare cheating behaviors** where traditional classifiers fail due to scarcity of examples.  

1. **The Challenge**: Cheating attempts are rare in the playerbase, but harmful when they occur. Traditional classifiers struggle with recall.  
2. **The Approach**:  
   - Collect recent gameplay actions from a single player  
   - Score each action individually using embeddings  
   - Aggregate scores to detect overall exploit patterns  
   - Prioritize recall to avoid missing real cheaters, even at the cost of some false positives  
3. **Real-world Impact**: This methodology makes it possible to surface **rare but dangerous exploiters** for further review, improving platform fairness and safety.  

---

## Storage Options

Synapse supports both local file storage and S3 storage.

- Local: paths starting with `/` or relative paths  
- S3: `s3://bucket-name/path/to/index`  

---

## License

Apache License 2.0
