# MM-NeuroOnco 🧠

### A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis

[![License: Research Only](https://img.shields.io/badge/License-Research%20Only-red.svg)](./LICENSE)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-MM--NeuroOnco-yellow)](https://huggingface.co/datasets/gfnnnb/MM-NeuroOnco-Images)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()

---

## 📖 Overview

**MM-NeuroOnco** is a large-scale multimodal benchmark and instruction dataset designed for clinically grounded MRI-based brain tumor diagnosis and interpretable reasoning. Unlike traditional classification datasets, this project emphasizes:

* **Clinically Interpretable Reasoning:** Spanning MRI physics, anatomical localization, tumor morphology, and radiological sign extraction.
* **Multimodal Integration:** Utilizing T1, T2, FLAIR, and T1-Contrast modalities.
* **Diverse Task Support:** From closed-ended diagnosis to open-ended Visual Question Answering (VQA).

<p align="center">
  <img src="assets/figure_overview.png" width="850" alt="Overview of MM-NeuroOnco tasks and capabilities">
  <br>
  <em>Figure 1: Overview of MM-NeuroOnco tasks and capabilities.</em>
</p>

---

## 📊 Dataset Statistics

MM-NeuroOnco aggregates data from **20 publicly available medical imaging datasets**, covering **8 tumor types** plus healthy controls.

<p align="center">
  <img src="assets/Figure_3.png" width="800" alt="Distribution of tumor types and question diversity">
  <br>
  <em>Figure 3: Distribution of tumor types and question diversity in the instruction dataset.</em>
</p>

| Metric | Count / Detail |
| :--- | :--- |
| **Total MRI Slices** | **73,226** |
| **Modalities** | T1, T2, FLAIR, T1-Contrast |
| **Open-Ended VQA** | ~70,000 pairs |
| **Closed-Ended VQA** | ~130,000 pairs |
| **Silver-Labeled Slices** | 2,472 (Curated) |
| **Benchmark Set** | 1,000 images & 3,000 VQA pairs |

---

## 🔍 Qualitative Examples

The dataset provides rich Chain-of-Thought (CoT) explanations and grounded diagnosis.

<p align="center">
  <img src="assets/Figure_7.png" width="900" alt="Sample cases showing model reasoning">
  <br>
  <em>Figure 7: Sample cases showing model reasoning, attribute extraction, and diagnosis.</em>
</p>

---

## ⚙️ Pipeline & Methodology

Our **fully reproducible multi-stage label construction pipeline** ensures high-quality silver labels through a rigorous "Skeptic-Conservative" dual-model approach.

<p align="center">
  <img src="assets/Figure_2.png" width="900" alt="The three-stage pipeline">
  <br>
  <em>Figure 2: The three-stage pipeline: (1) Dual-Model Extraction, (2) High-Precision Fusion, and (3) Quality Audit.</em>
</p>

The pipeline consists of:
1.  **Dual-Model Silver Label Extraction:** Utilizing conservative and skeptic extraction modes.
2.  **High-Precision Fusion:** Implementing double-blind agreement filtering and conflict resolution.
3.  **Quality Audit:** External LLM-based auditing and MRI physics consistency checks.

---

## 📦 Data Access & Hosting

Due to licensing restrictions of source datasets (e.g., BraTS), the data is distributed in three parts:

### 1. Benchmark Images (HuggingFace)
Hosted on HuggingFace with **gated access**.
> 👉 **[Go to MM-NeuroOnco-Images](https://huggingface.co/datasets/gfnnnb/MM-NeuroOnco-Images)**

### 2. Instruction Dataset
Requires specific application and approval. Please follow the instructions in the HuggingFace repository to apply.

### 3. Restricted Data (BraTS 2021)
⚠️ **Important:** We do **not** redistribute BraTS 2021 data.
Users must:
1.  Register at the [official BraTS website](https://www.med.upenn.edu/cbica/brats2021/).
2.  Agree to their data usage terms.
3.  Download the data independently.
4.  Use our `data_processing/` scripts to align them with our JSON annotations.

---

## 📂 Repository Structure

```text
MM-NeuroOnco/
├── benchmark/           # Evaluation benchmarks
│   ├── closed/          # Closed-ended JSONs
│   ├── open/            # Open-ended JSONs
│   └── splits/          # Train/Val/Test splits
├── data_processing/     # Metadata extraction & preprocessing scripts
├── pipeline/            # The Multi-model silver labeling pipeline code
├── evaluation/          # Evaluation scripts (Accuracy, BLEU, ROUGE, etc.)
├── docs/                # Documentation & Governance
└── assets/              # Figures for README
```
## ⚖️ Ethics & License
Code & Annotations: Released for Research Use Only.

Medical Images: Governed by the original licenses of their respective source datasets (TCIA, BraTS, etc.).

Disclaimer: This dataset is intended strictly for academic research. It is not for clinical deployment or medical decision-making.

For detailed governance, please refer to docs/ethics_and_governance.md.

## 📝 Citation
If you find this project useful in your research, please cite our paper:





