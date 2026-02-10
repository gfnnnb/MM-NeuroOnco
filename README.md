MM-NeuroOnco
A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis




Official repository for our KDD submission.

Overview

MM-NeuroOnco is a large-scale multimodal benchmark and instruction dataset designed for clinically grounded MRI-based brain tumor diagnosis.

<p align="center"> <img src="assets/figure_overview.png" width="900"> </p>

The project consists of:

A closed-ended evaluation benchmark

An open-ended reasoning benchmark

A large-scale instruction dataset

A reproducible data processing and label construction pipeline

Dataset Statistics

20 public medical imaging datasets aggregated

4 MRI modalities

8 tumor types + healthy

73,226 MRI slices

70K open-ended VQA pairs

130K closed-ended VQA pairs

2,472 curated slices (silver-labeled)

1K benchmark images & 3K VQA pairs

Repository Structure
benchmark/
├── closed/        # Closed-ended benchmark JSON
├── open/          # Open-ended benchmark JSON
└── splits/        # Evaluation split definitions

data_processing/   # Metadata extraction & preprocessing scripts
pipeline/          # Multi-model label extraction pipeline
docs/              # Documentation

⚠️ Data Access (Important)

MM-NeuroOnco aggregates data from multiple publicly available medical imaging datasets.

Raw MRI images are NOT redistributed in this repository due to licensing and data governance constraints.

To reproduce the dataset:

Download the original datasets from their official sources.

Use the provided preprocessing and metadata extraction scripts.

Follow the pipeline instructions in the data_processing/ and pipeline/ directories.

We strictly respect the licensing terms of each original dataset.

Original Dataset Sources

The aggregated datasets include publicly available sources such as:

BraTS Challenge datasets

TCIA (The Cancer Imaging Archive) collections

Figshare-hosted medical imaging datasets

Other publicly accessible research datasets

Please refer to the official websites of each dataset for download and licensing terms.

License

Code & annotations: Released for research purposes.

Raw medical images: Governed by the original licenses of their respective sources.

Users must comply with the licensing agreements of each original dataset.

MM-NeuroOnco does not claim ownership of the underlying medical images.

Ethics & Data Governance

For ethical considerations and data governance policies, see:

Data Governance & Ethics Statement

Citation

If you use MM-NeuroOnco, please cite:

MM-NeuroOnco: A Multimodal Benchmark and Instruction Dataset for MRI-Based Brain Tumor Diagnosis
