---

# 🐦 BirdCLEF+ 2026 Competition

## 📌 Overview

The goal of this competition is to develop machine learning frameworks capable of identifying understudied species within continuous audio data from Brazil's Pantanal wetlands. Successful solutions will help advance biodiversity monitoring in the last wild places on Earth.

---

## 📅 Timeline

* **Start Date:** March 11, 2026
* **Entry Deadline:** May 27, 2026
* **Team Merger Deadline:** May 27, 2026
* **Final Submission Deadline:** June 3, 2026

> All deadlines are at **11:59 PM UTC** unless otherwise noted.

---

## 🧠 Description

How do you protect an ecosystem you can’t fully see? One way is to listen.

This competition involves building models that automatically identify wildlife species from their vocalizations in audio recordings collected across the Pantanal wetlands. This work supports biodiversity monitoring in one of the most diverse and threatened ecosystems.

The Pantanal spans over **150,000 km²** across Brazil and neighboring countries and hosts:

* 650+ bird species
* Numerous other wildlife species

Challenges include:

* Seasonal flooding
* Wildfires
* Agricultural expansion
* Climate change

---

## 🎯 Goal of the Competition

Conventional biodiversity monitoring is expensive and difficult in remote regions.

To address this:

* A network of **1,000 acoustic recorders** is deployed
* Continuous audio captures multi-species soundscapes

However:

* Audio volume is massive
* Labeled data is limited

### Objective

Develop machine learning models that:

* Identify species from passive acoustic monitoring (PAM)
* Work across diverse habitats
* Handle noisy, real-world data
* Support conservation decisions

---

## 📊 Evaluation

* Metric: **Macro-averaged ROC-AUC**
* Classes without true positives are excluded from scoring

---

## 📤 Submission Format

* Predict probability of species presence per `row_id`
* Each row = **5-second audio window**
* One column per species
* File name: `submission.csv`

---

## 💰 Prizes

| Rank | Prize   |
| ---- | ------- |
| 1st  | $15,000 |
| 2nd  | $10,000 |
| 3rd  | $8,000  |
| 4th  | $7,000  |
| 5th  | $5,000  |

---

## 📝 Working Note Award (Optional)

Participants are encouraged to submit a **working note** to the CLEF 2026 conference.

### 🏆 Rewards

* 2 winners
* **$2,500 each**

### 📅 Timeline

* Competition Deadline: June 3, 2026
* Working Note Submission: June 17, 2026
* Notification: June 24, 2026
* Camera-ready Deadline: July 6, 2026

---

## 📏 Evaluation Criteria (Working Notes)

### 1. Originality

* Novel ideas or approaches
* Advances knowledge or methods

### 2. Quality

* Technical rigor
* Engineering excellence

### 3. Contribution

* Impact on the field
* Practical usefulness

### 4. Presentation

* Clarity and structure
* Readability and completeness

---

## 📊 Scoring System

Each paper is reviewed by **2 reviewers** (max score: 15)

### a) Work & Contribution

* 5: Excellent
* 4: Good
* 3: Solid
* 2: Marginal
* 1: Poor

### b) Originality

* 5: Trailblazing
* 4: Pioneering
* 3: Slightly novel
* 2: Common
* 1: Repetitive

### c) Readability

* 5: Excellent
* 4: Well written
* 3: Readable
* 2: Needs work
* 1: Poor

---

## 💻 Code Requirements

* Submission via **Notebook**
* Constraints:

  * CPU runtime ≤ 90 minutes
  * GPU submissions effectively disabled (1 min runtime)
  * No internet access
* Allowed:

  * Public external datasets
  * Pre-trained models

---

## 🙏 Acknowledgements

Supported by:

* **Bezos Earth Fund AI for Climate and Nature Grand Challenge**

Key contributors include:

* Chemnitz University of Technology
* Google DeepMind
* iNaturalist
* Instituto Homem Pantaneiro
* INPP
* K. Lisa Yang Center for Conservation Bioacoustics
* LifeCLEF
* UFMS
* Xeno-canto

---

## 📚 Citation

```
Stefan Kahl, Tom Denton, Larissa Sugai, Liliana Piatti, Ryan Holbrook, 
Holger Klinck, and Ashley Oldacre. BirdCLEF+ 2026. 
https://kaggle.com/competitions/birdclef-2026, 2026. Kaggle.
```

---


