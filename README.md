# CSL 7640: Natural Language Understanding

## Assignment-2

**Name:** Anand Mishra
**Date:** February 25, 2026

---

## 📌 Overview

This assignment focuses on two major tasks:

1. **Learning Word Embeddings** using Word2Vec on IIT Jodhpur textual data
2. **Character-Level Name Generation** using RNN-based architectures

The goal is to understand semantic representations of text and compare sequence models for generative tasks.

---

# 🔹 Problem 1: Word Embeddings (IIT Jodhpur Data)

## 📂 Task 1: Dataset Preparation

* Collected text data from:

  * IIT Jodhpur official website (departments, research pages, announcements)
  * Academic regulations (mandatory)
  * Faculty profiles / course syllabus


### 📊 Dataset Statistics

* Total Documents: 1

* Total Tokens: 54,204

* Vocabulary Size: 2,055

* WordCloud generated for most frequent words

---

## ⚙️ Task 2: Model Training

Implemented Word2Vec models:

* **CBOW (Continuous Bag of Words)**
* **Skip-gram with Negative Sampling**

### Hyperparameters explored:

* Embedding Dimension (50, 100, 300)
* Window Size (3, 5, 7)
* Negative Samples (5, 10, 15)

### Output:

* Training time comparison
* Loss comparison
* Vocabulary size

---

## 🔍 Task 3: Semantic Analysis

---

## 📈 Task 4: Visualization

* Used PCA / t-SNE for 2D projection
* Visualized clusters of semantically related words
* Compared CBOW vs Skip-gram embeddings

---

# 🔹 Problem 2: Character-Level Name Generation

## 📂 Task 0: Dataset

* Generated **1000 Indian names** using LLM
* Stored in `TrainingNames.txt`

---

## ⚙️ Task 1: Model Implementation

Implemented from scratch:

### 1. Vanilla RNN

* Sequential processing of characters
* Uses past context only

### 2. Bidirectional LSTM (BLSTM)

* Processes sequence in both directions
* Captures past + future context

### 3. RNN with Attention

* Applies attention over hidden states
* Focuses on important characters

### Hyperparameters:

* Hidden Size: 128
* Layers: 2
* Learning Rate: 0.001

Also reported:

* Trainable parameters
* Training loss

---

## 📊 Task 2: Quantitative Evaluation

Metrics used:

* **Novelty Rate** = % of generated names not in training data
* **Diversity** = Unique names / Total generated names

✔ Compared performance across all models

---

## 🔍 Task 3: Qualitative Analysis

### Observations:

* RNN → simple but less realistic
* BLSTM → better structure
* Attention → more coherent outputs


---


## ✨ Conclusion

This assignment demonstrates:

* Learning semantic embeddings from real-world data
* Understanding differences between CBOW and Skip-gram
* Designing and evaluating sequence models for generation tasks

---
