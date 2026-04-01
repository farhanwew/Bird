---

# 📊 Dataset Description — BirdCLEF+ 2026

## 🎯 Objective

Your task is to identify which species are calling in audio recordings collected in the Brazilian Pantanal.

### 🐾 Target Species

* Birds
* Amphibians
* Mammals
* Reptiles
* Insects

This task supports biodiversity monitoring for conservation, enabling scientists to better track animal populations.

---

## 🧪 Evaluation Setup

* Uses a **hidden test set**
* Test data is only accessible during notebook submission
* Ensures fair evaluation and prevents data leakage

---

## 📁 Dataset Structure

### 1. `train_audio/`

* Short recordings of individual species
* Sources:

  * Xeno-canto (XC)
  * iNaturalist (iNat)
* Format:

  * Resampled to **32 kHz**
  * Stored as `.ogg`
* Filename format:

  ```
  [collection][file_id].ogg
  ```
* Notes:

  * Covers most relevant species
  * Additional scraping is discouraged

---

### 2. `test_soundscapes/`

* ~600 audio recordings for evaluation
* Each recording:

  * Duration: **1 minute**
  * Format: `.ogg`, 32 kHz
* Filename format:

  ```
  BC2026_Test_<fileID>_<site>_<date>_<time>.ogg
  ```

  Example:

  ```
  BC2026_Test_0001_S05_20250227_010002.ogg
  ```
* Notes:

  * Loaded during submission runtime (~5 minutes)
  * Not all training species appear in test data

---

### 3. `train_soundscapes/`

* Additional real-world recordings
* Same format as test soundscapes
* Some overlap in recording sites (NOT time)

#### 🏷️ Labels (`train_soundscapes_labels.csv`)

* Columns:

  * `filename`
  * `start` (seconds)
  * `end` (seconds)
  * `primary_label` → semicolon-separated species

* Notes:

  * Only partially labeled
  * Useful for weakly supervised learning

---

## ⚠️ Important Notes

* Some species in the test set:

  * May **only appear in train_soundscapes**
  * Not present in `train_audio`
* Not all train species appear in test

---

## 📄 Metadata Files

### 4. `train.csv`

Contains metadata for training audio.

#### Key Columns:

* `primary_label`
  → Main species label (eBird / iNat ID)

* `secondary_labels`
  → Additional species (possibly incomplete)

* `latitude`, `longitude`
  → Recording location (useful for geo-awareness)

* `author`
  → Recording contributor

* `filename`
  → Audio file reference

* `rating`
  → Quality score (1–5)

  * 0 = no rating
  * Penalized if background noise

* `collection`
  → Source: `XC` or `iNat`

---

### 5. `sample_submission.csv`

* Template for submission

#### Format:

* `row_id`:

  ```
  [soundscape_filename]_[end_time]
  ```

  Example:

  ```
  BC2026_Test_0001_S05_20250227_010002_20
  ```

* Species columns:

  * **234 columns**
  * Each represents probability of species presence

---

### 6. `taxonomy.csv`

* Maps species and class information

#### Contains:

* `primary_label` → column name in submission
* `class`:

  * Aves
  * Amphibia
  * Mammalia
  * Insecta
  * Reptilia

#### Notes:

* Some insects are **sonotypes** (e.g., `47158son16`)
* Treated as distinct classes despite no exact species ID

---

### 7. `recording_location.txt`

* High-level location info:

  * Pantanal, Brazil

---

## 🧠 Key Challenges

* Multi-label classification (234 classes)
* Weak labels (incomplete annotations)
* Domain shift:

  * Clean `train_audio` vs noisy `soundscapes`
* Class imbalance
* Temporal segmentation (5-second windows)
* Large-scale inference constraints

---

## 🚀 Practical Insights

* Combine:

  * `train_audio` (clean supervision)
  * `train_soundscapes` (real-world context)
* Use:

  * Spectrogram-based models (CNN / Transformer)
* Consider:

  * Geo + time features
  * Pseudo-labeling
  * Data augmentation (noise, mixup)

---

