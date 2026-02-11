# MM-NeuroOnco  
### A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis

[![License](https://img.shields.io/badge/License-Research%20Only-blue.svg)]()


---

## Overview

MM-NeuroOnco is a large-scale multimodal benchmark and instruction dataset designed for clinically grounded MRI-based brain tumor diagnosis and interpretable reasoning over 2D MRI slices.

<p align="center">
  <img src="assets/figure_overview.png" width="900">
</p>

The project provides:

- A closed-ended evaluation benchmark  
- An open-ended reasoning benchmark  
- A large-scale multimodal instruction dataset  
- A fully reproducible multi-stage label construction pipeline  

The benchmark emphasizes clinically interpretable reasoning across MRI physics, anatomical localization, tumor morphology, and radiological sign extraction.

---

## 📦 Data Hosting

The official benchmark images are hosted on HuggingFace:

👉 **HuggingFace Dataset :**  
https://huggingface.co/datasets/gfnnnb/MM-NeuroOnco-Images

- Benchmark images are available via gated access.
- Users must request access and agree to the dataset terms.
- The instruction/training images require approval.

⚠️ BraTS 2021 imaging data are not redistributed in this repository due to licensing restrictions.
Users must obtain the dataset from the official BraTS challenge website
after registration and agreement to the data usage terms:
https://www.med.upenn.edu/cbica/brats2021/

---

## Dataset Statistics

- **20** publicly available medical imaging datasets aggregated  
- **4** MRI modalities (T1, T2, FLAIR, T1-Contrast)  
- **8 tumor types + healthy controls**  
- **73,226** MRI slices  
- **~70K** open-ended VQA pairs  
- **~130K** closed-ended VQA pairs  
- **2,472** curated silver-labeled slices  
- **1K benchmark images & 3K VQA pairs**

---

## Repository Structure

benchmark/
├── closed/ # Closed-ended benchmark JSON
├── open/ # Open-ended benchmark JSON
└── splits/ # Evaluation split definitions

data_processing/ # Metadata extraction & preprocessing scripts
pipeline/ # Multi-model silver labeling & fusion pipeline
docs/ # Documentation and governance statements
assets/ # Figures and visualizations


---

## Reproducible Pipeline

The dataset construction follows a three-stage framework:

### Step 1 — Dual-Model Silver Label Extraction
- Model A: Conservative extraction mode  
- Model B: Skeptic extraction mode  

### Step 2 — High-Precision Fusion
- Double-blind agreement filtering  
- Modality conflict resolution  
- Polarity-aware sign merging  

### Step 3 — Quality Audit
- External LLM-based auditing  
- MRI physics consistency verification  
- Final gold-standard filtering  

All processing scripts are provided in the `pipeline/` directory.

---

## Data Access Policy (Important)

MM-NeuroOnco aggregates data from multiple publicly available medical imaging datasets.

- **Benchmark images:** Hosted on HuggingFace (gated access).
- **Instruction dataset images:** Access requires application and approval.
- **BraTS2021 data:** Must be downloaded independently from the official source.

This repository does **not** redistribute restricted medical datasets.

To reproduce the dataset:

1. Download the original datasets from their official sources.
2. Follow the preprocessing scripts in `data_processing/`.
3. Run the multi-stage labeling pipeline in `pipeline/`.

The directory structure should mirror the `image_path` field defined in the JSON annotations.

---

## Original Dataset Sources

The aggregated datasets include publicly available sources such as:

- BraTS Challenge datasets  
- TCIA (The Cancer Imaging Archive) collections  
- Figshare-hosted medical imaging datasets  
- Other public research datasets  

Please refer to the official websites of each dataset for download and licensing terms.

MM-NeuroOnco does not claim ownership of the underlying medical images.

---

## Evaluation

Evaluation scripts are provided in the `evaluation/` directory.

See `evaluation/README.md` for environment setup and usage instructions.

## License

- **Code & annotations:** Released for research use only.
- **Raw medical images:** Governed by the original licenses of their respective datasets.

Users are responsible for complying with the licensing agreements of each original dataset.

---

## Ethics & Data Governance

This project follows strict data governance principles:

- No redistribution of restricted medical data  
- Respect for original dataset licenses  
- Transparent documentation of data processing  
- No patient-identifiable information included  

For details, see:

`docs/ethics_and_governance.md`

---

## Disclaimer

This dataset and codebase are intended strictly for academic research purposes.  
They are not intended for clinical deployment or medical decision-making.

