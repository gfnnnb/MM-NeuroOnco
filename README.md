# MM-NeuroOnco  
### A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis

Official repository for our KDD submission.

---

## Overview

MM-NeuroOnco is a large-scale multimodal benchmark and instruction dataset designed for clinically grounded MRI-based brain tumor diagnosis.

The project consists of:

- A closed-ended evaluation benchmark
- An open-ended reasoning benchmark
- A large-scale instruction dataset
- A complete MRI slice image collection

---

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

## Citation

If you use MM-NeuroOnco, please cite:

MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis
