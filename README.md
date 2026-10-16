# Temporal Fact Verification for Evolving Information

A deep-learning based Temporal Fact Verification system that analyzes the relationship between two sentences and classifies the claim as **Support**, **Invalidate**, or **Neutral**.

The project extends transformer-based fact verification with temporal features to better handle information that may change or become outdated over time.

---

## 📌 Project Overview

Information on the internet is constantly changing. A statement that was correct in the past may become outdated or invalid later.

Traditional fact verification systems mainly focus on whether one statement supports, contradicts, or is unrelated to another statement. However, they may not explicitly consider the **temporal validity** of information.

This project explores a **time-aware transformer-based approach** for fact verification on evolving information.

### Main Objective

To develop a temporal fact verification system that can:

- Analyze relationships between sentence pairs
- Detect whether information supports or invalidates another statement
- Identify neutral relationships
- Incorporate temporal information into transformer-based learning
- Compare standard transformer models with a temporal-enhanced model
- Provide predictions through a web-based interface

---

## 🎯 Classification Task

The system performs three-class classification:

| Label | Meaning |
|---|---|
| `Support` | The second statement supports the first statement |
| `Invalidate` | The second statement contradicts or invalidates the first statement |
| `Neutral` | The relationship between the statements is neither supportive nor invalidating |

---

## 🧠 Models

The project experiments with multiple transformer-based architectures:

- BERT
- DistilBERT
- RoBERTa
- Temporal RoBERTa

The final temporal model incorporates additional temporal features alongside transformer representations.

### Model Comparison

The repository contains experimental results for:

- BERT baseline
- DistilBERT
- RoBERTa
- Temporal RoBERTa
- ## Screenshots

<img width="1917" height="918"  src="Screenshot 2026-10-16 220626.png" />


Results and evaluation metrics are available in:

```text
results/
## 🖥️ Project Demo

![Temporal Fact Verification Demo](images/tfv-demo.png)
