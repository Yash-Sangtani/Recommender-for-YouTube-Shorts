# Personalized Recommender System for YouTube Short-Form Food Creators 🍽️📹

This repository contains a **creator-facing recommender system** for **YouTube Shorts** in the **food domain**.  
The goal is to help short-form creators answer:

- **What topics should I post more of?**
- **What new topics are promising for my channel?**
- **How should I style my shorts (length, hooks, type of content)?**

It does this by combining:

1. **Global topic trends** from a dataset of food-related YouTube Shorts
2. **Creator-specific performance** within each topic
3. A simple but interpretable **scoring system** for:
   - Topics to **double down** on
   - Topics to **explore** for the first time

---

## 🧱 Core Ideas

- Use the **YouTube Data API v3** to collect **short food videos** (YouTube Shorts–style).
- Engineer:
  - **Engagement features** (likes, comments, normalized by views)
  - **Style features** (duration buckets, tags)
  - **Content features** (TF–IDF on title + description + tags)
- Discover **topics** with **K-Means clustering** on TF–IDF vectors.
- For each **creator × topic**, compute:
  - Creator’s **mean engagement** in that topic
  - **Global mean engagement** for that topic
- Recommend:
  - Topics where the creator is **already strong** → *Double Down*
  - Topics that are **globally strong but unused** → *Explore*

---

## 📂 Project Structure

```text
.
├── dataExtraction.py         # Script to collect YouTube data and save videos_raw.csv
├── recommender_system.py     # Feature engineering, topic modeling, and recommendations
├── videos_raw.csv            # Collected dataset (not committed if large / private)
└── README.md                 # This file
