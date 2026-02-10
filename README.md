# MM-NeuroOnco

A large-scale multimodal benchmark for clinically grounded brain tumor MRI understanding.

Official repository for our KDD submission.

---

## 🔥 Highlights

- 📊 24,726 MRI slices aggregated from 20 data sources
- 🧠 ~200K semantically enriched multimodal instructions
- 🏥 Clinically grounded diagnostic reasoning benchmark
- 🚫 Rejection-aware evaluation protocol
- 🤖 Multi-model collaborative medical semantic completion pipeline

---

## 📦 Dataset Overview

MM-NeuroOnco consists of:

- **Closed-Ended VQA**
- **Open-Ended VQA**
- Structured medical attribute annotations
- Chain-of-Thought supervision
- Diagnosis-oriented semantic reasoning

More detailed documentation will be released soon.

---

## 🧩 Multi-Model Semantic Completion Pipeline

We propose a conservative radiologist-inspired multi-model reasoning protocol:

- Omission over fabrication principle
- Default-null initialization
- Structured diagnostic constraints
- Cross-model semantic verification

Pipeline details will be released in `docs/`.

---

## 📊 Benchmark

We evaluate representative LVLMs under both standard and rejection-aware settings.

Full benchmark results and evaluation scripts will be released in `benchmark/` and `evaluation/`.

---

## 🚀 Project Structure

MM-NeuroOnco/
├── assets/
├── benchmark/
├── docs/
├── evaluation/
├── examples/
└── pipeline/

## 📬 Contact

For dataset access or collaboration inquiries, please open an issue.

