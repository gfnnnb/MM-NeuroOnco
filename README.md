# MM-NeuroOnco  
### A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/gfnnnb/MM-NeuroOnco-Images)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![KDD](https://img.shields.io/badge/Conference-KDD%202025-red)]()

Official repository for our KDD submission.

---

## Overview

MM-NeuroOnco is a large-scale multimodal benchmark and instruction dataset designed for clinically grounded MRI-based brain tumor diagnosis.
<p align="center">
  <img src="assets/figure_overview.png" width="900">
</p>

The project consists of:

- A closed-ended evaluation benchmark
- An open-ended reasoning benchmark
- A large-scale instruction dataset
- A complete MRI slice image collection

---
## Dataset Statistics

- 20 public datasets aggregated
- 4 MRI modalities
- 8 tumor types + healthy
- 73,226 MRI slices
- 70K open-ended VQA pairs
- 130K closed-ended VQA pairs
- 2,472 curated slices (silver-labeled)
- 1K benchmark images & 3K VQA pairs

## Repository Structure

benchmark/
├── closed/ # Closed-ended benchmark JSON
├── open/ # Open-ended benchmark JSON
└── splits/ # Evaluation split definitions

docs/ # Documentation


---

## Image Data

The complete MRI slice image collection supporting both benchmark and instruction data is hosted on HuggingFace:

🔗 https://huggingface.co/datasets/gfnnnb/MM-NeuroOnco-Images

The directory structure mirrors the `image_path` field in the JSON files.

---

## License

Image data: CC BY-NC 4.0  
Code & annotations: Released for research purposes.

Users must comply with original licenses of the aggregated public medical datasets.

---
## Ethics & Data Governance

For ethical considerations and data governance policies, see:

[Data Governance & Ethics Statement](docs/ethics_and_governance.md)

## Citation

If you use MM-NeuroOnco, please cite:

MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis
